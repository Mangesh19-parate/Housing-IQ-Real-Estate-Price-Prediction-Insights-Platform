"""Tests for ``ml.evaluation.scoring`` — score_predictions + within_tolerance_pct."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation import METRIC_NAMES
from ml.evaluation.scoring import score_predictions, within_tolerance_pct

# ---------------------------------------------------------------------------
# score_predictions
# ---------------------------------------------------------------------------


def test_score_predictions_returns_four_keys_in_pinned_order() -> None:
    rng = np.random.default_rng(42)
    y_true_log = np.log1p(rng.uniform(1e6, 1e8, size=50))
    y_pred_log = y_true_log + rng.normal(0, 0.05, size=50)
    out = score_predictions(y_true_log, y_pred_log)
    assert list(out.keys()) == list(METRIC_NAMES)
    assert set(out) == {"r2", "mae", "rmse", "mape"}


def test_score_predictions_inverts_log_target() -> None:
    """A model whose log predictions equal log targets must score ~0 MAE."""
    y_true_orig = np.array([1.0e6, 5.0e6, 1.0e7, 5.0e7])
    y_true_log = np.log1p(y_true_orig)
    # Perfect log predictions.
    out = score_predictions(y_true_log, y_true_log)
    assert out["mae"] == pytest.approx(0.0, abs=1.0)  # ₹1 absolute noise floor
    assert out["r2"] == pytest.approx(1.0, abs=1e-6)


def test_score_predictions_works_without_invert() -> None:
    """When invert_log=False, scoring runs on the raw input scale."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    out = score_predictions(y_true, y_pred, invert_log=False)
    assert out["mae"] == pytest.approx(0.0, abs=1e-9)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)


def test_score_predictions_returns_floats() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    out = score_predictions(y_true, y_pred, invert_log=False)
    for v in out.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# within_tolerance_pct
# ---------------------------------------------------------------------------


def test_within_tolerance_pct_returns_fraction() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 220.0, 350.0])  # 10%, 10%, ~17% errors
    frac = within_tolerance_pct(y_true, y_pred, tolerance=0.15)
    assert 0.0 <= frac <= 1.0
    # 2 of 3 rows within ±15% (100→110 is 10%, 200→220 is 10%, 300→350 is 16.7%)
    assert frac == pytest.approx(2 / 3, abs=0.01)


def test_within_tolerance_pct_is_zero_when_all_outside_band() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([10.0, 20.0, 30.0])  # 90% errors — all outside ±15%
    frac = within_tolerance_pct(y_true, y_pred, tolerance=0.15)
    assert frac == 0.0


def test_within_tolerance_pct_is_one_when_all_inside_band() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([101.0, 199.0, 301.0])  # all ≤1% errors
    frac = within_tolerance_pct(y_true, y_pred, tolerance=0.15)
    assert frac == 1.0


def test_within_tolerance_pct_handles_empty_input() -> None:
    frac = within_tolerance_pct(np.array([]), np.array([]), tolerance=0.15)
    assert frac == 0.0
