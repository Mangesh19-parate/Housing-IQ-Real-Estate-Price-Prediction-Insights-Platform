"""Tests for ``ml.classification.thresholds`` (Spec 21).

All test names are pinned by the spec's Definition of DoD item 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.classification.thresholds import (
    MIN_CITY_TRAIN_ROWS,
    calibrate_good_deal_thresholds,
)


def _synthetic_labels(
    n: int,
    city: str = "Gurgaon",
    residual_loc: float = 0.0,
    residual_scale: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "listing_id": [f"{city}_L_{i:04d}" for i in range(n)],
            "city": [city] * n,
            "transact_type": ["Sale"] * n,
            "residual_pct": rng.normal(loc=residual_loc, scale=residual_scale, size=n),
            "is_outlier": [False] * n,
        }
    )


def test_min_city_train_rows_is_pinned() -> None:
    """``MIN_CITY_TRAIN_ROWS`` is the spec-pinned threshold."""
    assert MIN_CITY_TRAIN_ROWS == 100


def test_calibrate_good_deal_thresholds_returns_per_city_dict() -> None:
    """4-city frame with 100+ rows each → returned dict has 4 city keys."""
    frames = [
        _synthetic_labels(n=200, city="Gurgaon", seed=1),
        _synthetic_labels(n=200, city="Hyderabad", seed=2),
        _synthetic_labels(n=200, city="Kolkata", seed=3),
        _synthetic_labels(n=200, city="Mumbai", seed=4),
    ]
    df = pd.concat(frames, ignore_index=True)
    is_train_mask = pd.Series([True] * len(df))

    out = calibrate_good_deal_thresholds(df, is_train_mask)
    city_keys = set(out.keys()) - {"_default"}
    assert city_keys == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}


def test_calibrate_good_deal_thresholds_falls_back_to_default_for_small_city(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """City with < min_rows → falls back to defaults + WARNING."""
    frames = [
        _synthetic_labels(n=200, city="Gurgaon", seed=1),
        _synthetic_labels(n=50, city="Kolkata", seed=2),  # too small
    ]
    df = pd.concat(frames, ignore_index=True)
    is_train_mask = pd.Series([True] * len(df))

    import logging

    with caplog.at_level(logging.WARNING, logger="ml.classification.thresholds"):
        out = calibrate_good_deal_thresholds(df, is_train_mask)

    # Kolkata uses defaults.
    assert out["Kolkata"]["low"] == -0.10
    assert out["Kolkata"]["high"] == 0.10
    # WARNING emitted naming the small city.
    assert any("Kolkata" in rec.message for rec in caplog.records)


def test_calibrate_good_deal_thresholds_clamps_pathological_cities() -> None:
    """A city with very high IQR is clamped to [default ± 0.10]."""
    # Build a frame with residuals uniform in [-0.5, 0.5] → IQR ≈ 0.5.
    n = 200
    residuals = np.linspace(-0.5, 0.5, n)
    df = pd.DataFrame(
        {
            "listing_id": [f"L_{i}" for i in range(n)],
            "city": ["Gurgaon"] * n,
            "transact_type": ["Sale"] * n,
            "residual_pct": residuals,
            "is_outlier": [False] * n,
        }
    )
    is_train_mask = pd.Series([True] * n)

    out = calibrate_good_deal_thresholds(df, is_train_mask)
    low, high = out["Gurgaon"]["low"], out["Gurgaon"]["high"]
    # Clamped to [-0.20, 0.20].
    assert low >= -0.20
    assert high <= 0.20


def test_calibrate_good_deal_thresholds_includes_default_block() -> None:
    """Returned dict has a ``_default`` block with rationale."""
    df = _synthetic_labels(n=200, city="Gurgaon", seed=1)
    is_train_mask = pd.Series([True] * len(df))
    out = calibrate_good_deal_thresholds(df, is_train_mask)
    assert "_default" in out
    assert out["_default"]["low"] == -0.10
    assert out["_default"]["high"] == 0.10
    assert isinstance(out["_default"]["rationale"], str)
    assert len(out["_default"]["rationale"]) > 0
