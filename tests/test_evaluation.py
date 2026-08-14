"""Tests for ml.training.evaluation (Spec 13 Phase A)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.training.evaluation import (
    SMALL_CITY_TEST_ROWS,
    evaluate_subset,
    per_city_metrics,
    regression_metrics,
)

# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_returns_four_keys():
    y_true = np.log1p([100.0, 200.0, 400.0, 800.0])
    out = regression_metrics(y_true, y_true)
    assert set(out.keys()) == {"r2", "mae", "rmse", "mape"}


def test_regression_metrics_inverse_transforms_from_log():
    """Perfect prediction in log space -> zero error on original scale."""
    y_true = np.log1p([100.0, 200.0, 400.0, 800.0, 1_000_000.0])
    out = regression_metrics(y_true, y_true)
    assert out["mae"] == pytest.approx(0.0, abs=1e-6)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)


def test_regression_metrics_handles_all_zero_y_true():
    """MAPE epsilon guard: y_true=0 must not divide-by-zero."""
    y_true = np.zeros(5)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = regression_metrics(y_true, y_pred)
    # MAE on original scale (expm1): ~exp(1)-1 ~= 1.718 each
    assert np.isfinite(out["mape"])
    assert out["mape"] >= 0


def test_regression_metrics_matches_known_mae():
    """Sanity: y_true=[100,200], y_pred=[110,190] -> |MAE|=10 on original scale."""
    y_true = np.log1p([100.0, 200.0])
    # log1p(110) ≈ 4.7005, log1p(190) ≈ 5.2470
    y_pred = np.log1p([110.0, 190.0])
    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(10.0, abs=1e-4)


# ---------------------------------------------------------------------------
# evaluate_subset
# ---------------------------------------------------------------------------


def _toy_pipeline():
    """Linear regression on a StandardScaler — fits in <10 ms on tiny data."""
    return Pipeline([("s", StandardScaler()), ("e", LinearRegression())])


def test_evaluate_subset_returns_train_val_test_dict():
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # Targets must be strictly > 0 before log1p; use a large intercept
    # so 50 + 30*a + 10*b never goes negative on the random draws below.
    target = 500.0 + 30 * X["a"] + 10 * X["b"] + rng.normal(scale=0.1, size=n)
    y = np.log1p(target.values)

    X_tr, X_va, X_te = X.iloc[:50], X.iloc[50:65], X.iloc[65:]
    y_tr, y_va, y_te = y[:50], y[50:65], y[65:]

    pipe = _toy_pipeline()
    res = evaluate_subset(pipe, X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert set(res.keys()) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert set(res[split].keys()) == {"r2", "mae", "rmse", "mape"}


# ---------------------------------------------------------------------------
# per_city_metrics
# ---------------------------------------------------------------------------


def test_per_city_metrics_warns_on_small_sample(caplog):
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({"a": rng.normal(size=n)})
    # Strictly positive target.
    target = 100.0 + 5 * X["a"]
    y = np.log1p(target.values)
    cities = pd.Series(["BigCity"] * 40 + ["TinyCity"] * 10)
    pipe = _toy_pipeline().fit(X, y)

    with caplog.at_level(logging.WARNING, logger="ml.training.evaluation"):
        out = per_city_metrics(pipe, X, y, cities)

    assert set(out.keys()) == {"BigCity", "TinyCity"}
    warnings = [r for r in caplog.records if "TinyCity" in r.message]
    assert any("10 test rows" in r.message for r in warnings), (
        f"expected a '10 test rows' warning, got: "
        f"{[r.message for r in caplog.records]}"
    )


def test_per_city_metrics_raises_on_non_series():
    pipe = _toy_pipeline()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = np.log1p([10.0, 20.0, 30.0])
    with pytest.raises(TypeError, match="city_test must be a pd.Series"):
        per_city_metrics(pipe, X, y, ["A", "B", "C"])


def test_small_city_test_rows_constant_is_pinned():
    assert SMALL_CITY_TEST_ROWS == 30
