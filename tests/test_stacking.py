"""Tests for the stacking lever (Spec 14).

Pins the structure of ``make_stacking_regressor`` — 5 base learners,
Ridge meta — without running the full StackingRegressor fit (which is
slow with ``cv=5``). The end-to-end stacking pipeline is exercised
implicitly by the v2 script test on a synthetic fixture.
"""

from __future__ import annotations

import pytest
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from ml.training.levers.stacking import (
    STACKING_CV,
    STACKING_META_ALPHA,
    make_stacking_regressor,
)


def test_make_stacking_regressor_returns_stacking_regressor():
    stack = make_stacking_regressor()
    assert isinstance(stack, StackingRegressor)


def test_stacking_has_five_base_learners():
    stack = make_stacking_regressor()
    # sklearn exposes the base learners via .estimators (the fitted
    # attribute is ``estimators_``; pre-fit it's the named list passed
    # to the constructor).
    names = [name for name, _ in stack.estimators]
    assert len(names) == 5
    # Pin the expected base-learner names (literature B3, B4).
    assert set(names) == {
        "ridge",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lgbm",
    }


def test_stacking_meta_learner_is_ridge():
    stack = make_stacking_regressor()
    assert isinstance(stack.final_estimator, Ridge)
    assert abs(stack.final_estimator.alpha - STACKING_META_ALPHA) < 1e-9


def test_stacking_cv_constant_matches_meta_alpha():
    stack = make_stacking_regressor()
    assert stack.cv == STACKING_CV


def test_stacking_factory_returns_unfitted_estimator():
    """Factory must not pre-fit; the caller fits the ensemble."""
    stack = make_stacking_regressor()
    with pytest.raises(Exception):
        # check_is_fitted raises NotFittedError on unfitted estimators.
        check_is_fitted(stack)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
