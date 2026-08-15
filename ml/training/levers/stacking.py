"""Lever 1 — stacking ensemble (Spec 14).

Builds a ``StackingRegressor`` with 5 base learners per the literature
(B3, B4 — base papers for ensemble stacking in regression):
    - ``Ridge(alpha=1.0)`` — linear baseline, regularized.
    - ``RandomForestRegressor(n_estimators=200, ...)`` — bagged trees.
    - ``GradientBoostingRegressor(n_estimators=200, ...)`` — v1's
      boosting candidate.
    - ``XGBRegressor(...)`` — v1's XGB defaults.
    - ``LGBMRegressor(...)`` — v2's default LGBM.

Meta-learner: ``Ridge(alpha=1.0)`` on out-of-fold base predictions.
``cv=5`` for OOF stacking, ``n_jobs=-1`` on the wrapper.

The factory is the only public symbol — fitting is the caller's job,
so the same factory is reusable for both Sale and Rent subsets.
"""

from __future__ import annotations

import logging
from typing import Final

from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

#: Pinned — number of CV folds for the stacking meta-learner.
STACKING_CV: Final[int] = 5

#: Pinned — meta-learner regularisation.
STACKING_META_ALPHA: Final[float] = 1.0


def make_stacking_regressor(random_state: int = 42) -> StackingRegressor:
    """Return an unfitted ``StackingRegressor`` with 5 base learners.

    Base learners (per spec §"stacking.py"):
        - Ridge
        - RandomForestRegressor
        - GradientBoostingRegressor
        - XGBRegressor (v1 defaults)
        - LGBMRegressor (v2 defaults)

    Meta-learner: ``Ridge(alpha=1.0)`` on out-of-fold base predictions.
    """
    base_learners = [
        (
            "ridge",
            Ridge(alpha=STACKING_META_ALPHA, random_state=random_state),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
            ),
        ),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
            ),
        ),
        (
            "lgbm",
            LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]
    meta = Ridge(alpha=STACKING_META_ALPHA)
    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta,
        cv=STACKING_CV,
        n_jobs=-1,
    )
    logger.info(
        "make_stacking_regressor: built with %d base learners + Ridge meta",
        len(base_learners),
    )
    return stack


__all__ = ["STACKING_CV", "STACKING_META_ALPHA", "make_stacking_regressor"]
