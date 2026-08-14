"""Tests for ml.training.candidates (Spec 13 Phase A)."""

from __future__ import annotations

import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

from ml.training.candidates import (
    CANDIDATE_MODELS,
    PRICE_MODEL_VERSION,
    RENT_MIN_ROWS,
    SHAP_EXPLAINER_VERSION,
    candidate_hyperparameters,
    make_estimator,
)

_EXPECTED_CLASSES = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
}


def test_candidate_models_constant_has_six_entries():
    assert len(CANDIDATE_MODELS) == 6
    assert set(CANDIDATE_MODELS) == set(_EXPECTED_CLASSES)


def test_make_estimator_returns_correct_class():
    for name, expected_cls in _EXPECTED_CLASSES.items():
        est = make_estimator(name)
        assert isinstance(est, expected_cls), (
            f"{name} should produce {expected_cls.__name__}, got "
            f"{type(est).__name__}"
        )


def test_make_estimator_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown candidate"):
        make_estimator("nope_does_not_exist")


def test_all_candidates_have_random_state_42():
    # ``LinearRegression`` and ``Lasso`` don't take ``random_state`` in their
    # constructor — only check the candidates that support seeding.
    seedable = {"ridge", "random_forest", "gradient_boosting", "xgboost"}
    for name in seedable:
        rs = getattr(CANDIDATE_MODELS[name], "random_state", None)
        assert rs == 42, f"{name} has random_state={rs!r}, expected 42"


def test_make_estimator_returns_fresh_instance_each_call():
    """XGBoost + GBM hold internal state; cloning must happen."""
    a = make_estimator("xgboost")
    b = make_estimator("xgboost")
    assert a is not b


def test_candidate_hyperparameters_returns_dict():
    hp = candidate_hyperparameters("random_forest")
    assert hp["n_estimators"] == 200
    assert hp["random_state"] == 42


def test_pinned_version_constants():
    assert PRICE_MODEL_VERSION == "v1"
    assert SHAP_EXPLAINER_VERSION == "v1"
    assert RENT_MIN_ROWS == 500
