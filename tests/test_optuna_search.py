"""Tests for the Optuna search wrappers (Spec 14).

Uses a tiny synthetic regression fixture (sklearn ``make_regression``)
with ``n_trials=3`` so the test stays fast. Pins:
    - returned dict shape (``best_params`` + ``best_value``)
    - XGB vs LGBM search spaces (XGB has ``max_depth``, LGBM has
      ``num_leaves``)
    - determinism under ``random_state=42``
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_regression

from ml.training.levers.optuna_search import (
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    optuna_search_lgbm,
    optuna_search_xgb,
)


@pytest.fixture
def small_reg():
    """200-row linear regression — fast, deterministic."""
    X, y = make_regression(
        n_samples=200, n_features=10, noise=0.1, random_state=42
    )
    return X, y


def test_optuna_search_returns_best_params_dict(small_reg):
    X, y = small_reg
    res = optuna_search_xgb(
        X, y, X, y, n_trials=3, timeout_sec=None, random_state=42
    )
    assert "best_params" in res
    assert "best_value" in res
    # best_value is the negative RMSE from Optuna's "maximize" objective
    assert isinstance(res["best_value"], float)
    # best_params must contain the search-space keys
    assert "max_depth" in res["best_params"]
    assert "learning_rate" in res["best_params"]


def test_optuna_search_respects_random_state(small_reg):
    """Two calls with the same seed should produce identical best_value."""
    X, y = small_reg
    res_a = optuna_search_xgb(
        X, y, X, y, n_trials=3, timeout_sec=None, random_state=42
    )
    res_b = optuna_search_xgb(
        X, y, X, y, n_trials=3, timeout_sec=None, random_state=42
    )
    # best_value should be (numerically) identical — same seed, same
    # TPE sampler, same data -> same study trajectory.
    assert abs(res_a["best_value"] - res_b["best_value"]) < 1e-6


def test_optuna_search_xgb_and_lgbm_have_separate_search_spaces(small_reg):
    X, y = small_reg
    res_xgb = optuna_search_xgb(
        X, y, X, y, n_trials=2, timeout_sec=None, random_state=42
    )
    res_lgbm = optuna_search_lgbm(
        X, y, X, y, n_trials=2, timeout_sec=None, random_state=42
    )
    # XGB search space has max_depth (not num_leaves)
    assert "max_depth" in res_xgb["best_params"]
    assert "num_leaves" not in res_xgb["best_params"]
    # LGBM search space has num_leaves (and max_depth too)
    assert "num_leaves" in res_lgbm["best_params"]
    assert "max_depth" in res_lgbm["best_params"]
    # Fixed LGBM params are present
    assert res_lgbm["best_params"].get("objective") == "regression"
    # Fixed XGB params are present
    assert res_xgb["best_params"].get("objective") == "reg:squarederror"


def test_optuna_constants_pinned():
    assert OPTUNA_N_TRIALS == 40
    assert OPTUNA_TIMEOUT_SEC == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
