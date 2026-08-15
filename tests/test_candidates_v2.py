"""Tests for the v2 candidate factory (Spec 14).

Pins:
    - the 5-name v2 candidate set (matches the lever sweep order)
    - ``make_v2_estimator`` returns a fresh instance per call (no
      state leak between folds)
    - ``params=`` override is honored (used by Optuna injection)
    - ``stacking`` returns the lever's ``StackingRegressor``
"""

from __future__ import annotations

import pytest
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from ml.training.candidates import V2_CANDIDATE_MODELS, make_v2_estimator


def test_v2_candidate_set_has_expected_five_names():
    assert set(V2_CANDIDATE_MODELS) == {
        "xgb_v1_defaults",
        "xgb_optuna",
        "lgbm_v1_defaults",
        "lgbm_optuna",
        "stacking",
    }


def test_make_v2_estimator_returns_xgb_for_xgb_names():
    e = make_v2_estimator("xgb_v1_defaults")
    assert isinstance(e, XGBRegressor)
    e2 = make_v2_estimator("xgb_optuna")
    assert isinstance(e2, XGBRegressor)


def test_make_v2_estimator_returns_lgbm_for_lgbm_names():
    e = make_v2_estimator("lgbm_v1_defaults")
    assert isinstance(e, LGBMRegressor)
    e2 = make_v2_estimator("lgbm_optuna")
    assert isinstance(e2, LGBMRegressor)


def test_make_v2_estimator_stacking_returns_stacking_regressor():
    from sklearn.ensemble import StackingRegressor

    e = make_v2_estimator("stacking")
    assert isinstance(e, StackingRegressor)


def test_make_v2_estimator_returns_fresh_instance_each_call():
    """No state leak — each call must produce a distinct object."""
    a = make_v2_estimator("xgb_v1_defaults")
    b = make_v2_estimator("xgb_v1_defaults")
    assert a is not b


def test_make_v2_estimator_params_override_defaults():
    e = make_v2_estimator(
        "xgb_optuna", {"max_depth": 8, "learning_rate": 0.07}
    )
    assert e.max_depth == 8
    assert abs(e.learning_rate - 0.07) < 1e-9


def test_make_v2_estimator_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown v2 candidate"):
        make_v2_estimator("nope")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
