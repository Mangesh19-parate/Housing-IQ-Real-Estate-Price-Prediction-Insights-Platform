"""Candidate-model factory for the price baseline (Spec 13).

Pins the 6 TRD §10 candidate estimators with sensible default
hyperparameters (``random_state=42`` everywhere — Rules §5.4). No tuning
happens here; tuning is a Week 8 improvement lever. ``make_estimator``
re-instantiates a fresh estimator on every call so candidate state
cannot leak between cross-validation folds.
"""

from __future__ import annotations

import logging
from typing import Final

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


__all__ = [
    "CANDIDATE_MODELS",
    "PRICE_MODEL_VERSION",
    "RENT_MIN_ROWS",
    "SHAP_EXPLAINER_VERSION",
    "candidate_hyperparameters",
    "make_estimator",
]
