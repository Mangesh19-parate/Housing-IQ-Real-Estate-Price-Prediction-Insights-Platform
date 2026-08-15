"""Candidate-model factory for the price baseline (Spec 13).

Pins the 6 TRD §10 candidate estimators with sensible default
hyperparameters (``random_state=42`` everywhere — Rules §5.4). No tuning
happens here; tuning is a Week 8 improvement lever. ``make_estimator``
re-instantiates a fresh estimator on every call so candidate state
cannot leak between cross-validation folds.

v2 (Spec 14) extends this with ``V2_CANDIDATE_MODELS`` and
``make_v2_estimator(name, params=...)`` — the same five-model set
(XGB defaults, XGB Optuna, LGBM defaults, LGBM Optuna, stacking) so
the script can sweep the levers against a single preprocessor pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

#: Pinned module-level constant (Rules §2.5). A future retrain bumps to
#: ``"v2"``; old artifacts are never overwritten in place.
PRICE_MODEL_VERSION: Final[str] = "v1"

#: Rent subset minimum row count. Below this the Rent pipeline is
#: skipped (logged + ``metrics_v1.json.rent.skipped = true``).
RENT_MIN_ROWS: Final[int] = 500

#: Pinned SHAP explainer version, paired with ``PRICE_MODEL_VERSION``.
SHAP_EXPLAINER_VERSION: Final[str] = "v1"


def _build_candidates() -> dict[str, BaseEstimator]:
    """Build the 6 TRD §10 candidate estimators with sensible defaults.

    Hyperparameters are pinned (not tuned) — see spec §"New dependencies"
    rationale. ``random_state=42`` everywhere (Rules §5.4). Lasso's
    default convergence tolerance is too tight for a 200k-row frame, so
    ``max_iter`` is raised to 10k.
    """
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.001, random_state=42, max_iter=10_000),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }


# ponytail: built once at import; ``make_estimator`` clones a fresh
# instance on every call so estimator state (XGBoost internal state,
# GBM staged predictions) does not leak between candidates.
CANDIDATE_MODELS: Final[dict[str, BaseEstimator]] = _build_candidates()


def make_estimator(name: str) -> BaseEstimator:
    """Return a fresh instance of the named candidate estimator.

    Unknown names raise ``ValueError`` listing the known candidates.
    Each call returns a *new* object — do not share the returned
    estimator between candidates (XGBoost + GBM hold internal state).
    """
    if name not in CANDIDATE_MODELS:
        raise ValueError(
            f"Unknown candidate: {name!r}. Known: {sorted(CANDIDATE_MODELS)}"
        )
    # ``clone`` from sklearn would work, but sklearn's ``clone`` strips
    # ``n_jobs``/``random_state`` when not in the constructor; building
    # a fresh instance from the pinned default constructor is simpler.
    return _build_candidates()[name]


def candidate_hyperparameters(name: str) -> dict:
    """Return the pinned hyperparameter dict for ``name`` (no clone)."""
    return dict(make_estimator(name).get_params(deep=False))


# ---------------------------------------------------------------------------
# v2 candidates (Spec 14)
# ---------------------------------------------------------------------------


def _build_v2_candidates() -> dict[str, BaseEstimator]:
    """Build the 5 v2 candidate estimators with sensible defaults.

    Names match the order in which the training script will sweep them:

        - ``xgb_v1_defaults`` — XGB with the v1 baseline defaults.
          Reproducibility anchor (must NOT regress vs v1's xgboost row).
        - ``xgb_optuna`` — placeholder; replaced in-place with the
          Optuna-best params at search time (see ``make_v2_estimator``).
        - ``lgbm_v1_defaults`` — LGBM with v1-equivalent defaults.
        - ``lgbm_optuna`` — placeholder, same pattern as xgb_optuna.
        - ``stacking`` — ``StackingRegressor`` with the 5 base learners
          + Ridge meta (see ``levers/stacking.py``).

    The Optuna placeholders use a 1-tree model so the pipeline can
    ``fit`` in case the search is skipped (tests / smoke runs); the
    real training script ALWAYS calls ``make_v2_estimator(name,
    params=...)`` with the Optuna result before fitting.
    """
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "xgb_v1_defaults": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
        "xgb_optuna": XGBRegressor(
            n_estimators=10,  # placeholder; real params injected by make_v2_estimator
            random_state=42,
            verbosity=0,
        ),
        "lgbm_v1_defaults": LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        "lgbm_optuna": LGBMRegressor(
            n_estimators=10,  # placeholder
            random_state=42,
            verbose=-1,
        ),
        # Stacking: real instance built lazily by make_v2_estimator so the
        # base learners can share random_state=42 cleanly.
        "stacking": None,  # type: ignore[dict-item]
    }


#: v2 candidate set (Spec 14). Built lazily so importing this module
#: does not import the StackingRegressor (heavy import) until a caller
#: actually asks for the stacking candidate.
V2_CANDIDATE_MODELS: Final[dict[str, BaseEstimator | None]] = (
    _build_v2_candidates()
)


def make_v2_estimator(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    """Return a fresh v2 estimator.

    Mirrors ``make_estimator`` for the v1 set. ``params`` overrides the
    pinned defaults — used by the script to inject Optuna-best params
    (``{"xgb_optuna": {"max_depth": ..., "learning_rate": ...}}``) and
    by callers that want to tweak a single hyperparameter.

    For ``"stacking"`` we build a fresh ``StackingRegressor`` via the
    lever helper (the v2 module owns the 5 base learners + Ridge meta).
    """
    if name not in V2_CANDIDATE_MODELS:
        raise ValueError(
            f"Unknown v2 candidate: {name!r}. "
            f"Known: {sorted(V2_CANDIDATE_MODELS)}"
        )
    merged: dict[str, Any] = {}
    base = V2_CANDIDATE_MODELS[name]
    if base is not None:
        merged.update(base.get_params(deep=False))
    if params:
        merged.update(params)
    if name == "stacking":
        from ml.training.levers.stacking import make_stacking_regressor

        return make_stacking_regressor(random_state=42)
    # Clone from the canonical constructor for the non-stacking names.
    if name in ("xgb_v1_defaults", "xgb_optuna"):
        from xgboost import XGBRegressor

        return XGBRegressor(**merged)
    if name in ("lgbm_v1_defaults", "lgbm_optuna"):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**merged)
    raise ValueError(f"make_v2_estimator: unhandled name {name!r}")


__all__ = [
    "CANDIDATE_MODELS",
    "PRICE_MODEL_VERSION",
    "RENT_MIN_ROWS",
    "SHAP_EXPLAINER_VERSION",
    "V2_CANDIDATE_MODELS",
    "candidate_hyperparameters",
    "make_estimator",
    "make_v2_estimator",
]
