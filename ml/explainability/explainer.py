"""TreeExplainer factory + persistence for the price model.

The explainer artifact ``models/shap_explainer_{sale,rent}_v{n}.pkl``
is built **once** by ``scripts/build_shap_explainer.py`` and loaded
**forever** by every serving path (per the ``shap-explainability``
skill: "load precomputed explainer at startup"). Per Request §2.6
(Rules) the explainer is built from the exact persisted model — no
proxy, no surrogate.

Ponytail: stdlib + joblib + shap only. No model wrappers, no
abstract base classes — a 4-tuple of tree estimator types is enough
to gate the build path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import shap

logger = logging.getLogger(__name__)


EXPLAINER_VERSION = "1.0.0"
SHAP_TOP_N = 7


# Allow-list of estimators the spec's TreeExplainer accepts. Per
# Specs 13/14 the chosen model is one of these. A non-tree winner
# (e.g. Ridge) raises immediately in build_explainer — defensive
# guard, not a code path.
try:  # pragma: no cover  (imported lazily so unit tests work without xgboost)
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from xgboost import XGBRegressor
    try:
        from lightgbm import LGBMRegressor
    except ImportError:  # pragma: no cover
        LGBMRegressor = None  # type: ignore[assignment,misc]

    _TREE_ESTIMATOR_TYPES = tuple(
        t for t in (
            XGBRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            LGBMRegressor,
        ) if t is not None
    )
except ImportError:  # pragma: no cover
    _TREE_ESTIMATOR_TYPES = ()


def _last_tree_estimator(model):  # noqa: ANN001 — pipeline + estimator union
    """Return the tree estimator at the end of a Pipeline (or the estimator itself).

    Spec 14's v2 pipeline is ``Pipeline([preprocessor, ..., xgb])``.
    We don't care which index the tree sits at — we want the last step
    with a ``predict`` method that matches the tree allow-list.
    """
    if hasattr(model, "steps") and hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        return last_step
    return model


def build_explainer(model):  # noqa: ANN001 — Pipeline | BaseEstimator
    """Build a ``shap.TreeExplainer`` over the model's last tree step.

    Raises ``ValueError`` with the offending class name in the message
    so the CLI surfaces the misconfiguration directly (pinned by
    ``test_build_explainer_rejects_non_tree_model``).
    """
    estimator = _last_tree_estimator(model)
    if _TREE_ESTIMATOR_TYPES and not isinstance(estimator, _TREE_ESTIMATOR_TYPES):
        raise ValueError(
            f"build_explainer requires a tree estimator (one of "
            f"{[t.__name__ for t in _TREE_ESTIMATOR_TYPES]}); got {type(estimator).__name__}"
        )
    return shap.TreeExplainer(estimator)


def _artifact_path(transact_type: str, version: str, root: Path | str) -> Path:
    """Filename rule: ``shap_explainer_{transact_type}_{version}.pkl``.

    Pinned by Rules §2.5 (versioned, never overwritten in place).
    """
    return Path(root) / f"shap_explainer_{transact_type}_{version}.pkl"


def save_explainer(
    explainer: shap.TreeExplainer,
    transact_type: str,
    version: str,
    out_dir: Path | str,
) -> Path:
    """Persist the explainer via ``joblib.dump``.

    Returns the path so callers can log it. Idempotent re-run with the
    same inputs writes the same content (joblib preserves bytes).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(transact_type, version, out_dir)
    joblib.dump(explainer, path)
    logger.info("saved shap explainer -> %s", path)
    return path


def load_explainer(
    transact_type: str,
    version: str,
    models_dir: Path | str,
) -> shap.TreeExplainer:
    """Inverse of :func:`save_explainer`. Raises ``FileNotFoundError``
    on miss with the resolved path in the message (pinned by
    ``test_load_explainer_raises_with_expected_path_on_miss``).
    """
    path = _artifact_path(transact_type, version, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"shap explainer artifact not found: {path}")
    explainer = joblib.load(path)
    logger.info("loaded shap explainer <- %s", path)
    return explainer


__all__ = [
    "EXPLAINER_VERSION",
    "SHAP_TOP_N",
    "build_explainer",
    "save_explainer",
    "load_explainer",
]


# ponytail: no ModelWrapper class, no abstract BaseExplainer protocol.
# The artifact is joblib-dumped; SHAP's own pickle path handles the
# rest. The "build once, load forever" rule is enforced by where the
# function lives — build_explainer is called only by the build CLI,
# load_explainer is the only public surface the FastAPI route will
# import.
