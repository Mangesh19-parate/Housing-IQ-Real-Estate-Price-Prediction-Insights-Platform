"""Tests for ``ml.classification.verdicts`` (Spec 21).

All test names are pinned by the spec's Definition of DoD item 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.classification.verdicts import (
    DEFAULT_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD_LOW,
    VERDICT_LABELS,
    build_good_deal_labels,
)


def _synthetic_clean_listings(
    n: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "listing_id": [f"L_{i:04d}" for i in range(n)],
            "city": ["Gurgaon"] * n,
            "locality": ["Sec_1"] * n,
            "transact_type": ["Sale"] * n,
            "price": rng.integers(5_000_000, 15_000_000, size=n).astype(float),
            "price_per_sqft": rng.integers(3000, 12000, size=n).astype(float),
            "is_outlier": [False] * n,
        }
    )


def test_verdict_labels_are_pinned() -> None:
    """``VERDICT_LABELS`` is the spec-pinned tuple."""
    assert VERDICT_LABELS == ("Good Deal", "Fair Price", "Overpriced")


def test_default_thresholds_match_rules_section_12_2() -> None:
    """Defaults match Rules §12.2 starting point (±10%)."""
    assert DEFAULT_THRESHOLD_LOW == -0.10
    assert DEFAULT_THRESHOLD_HIGH == 0.10


def test_build_good_deal_labels_assigns_three_verdicts() -> None:
    """Synthetic three-row frame with residuals spanning the band → all 3 classes."""
    df = pd.DataFrame(
        {
            "listing_id": ["L1", "L2", "L3"],
            "city": ["Gurgaon", "Gurgaon", "Gurgaon"],
            "transact_type": ["Sale", "Sale", "Sale"],
            "price": [100.0, 100.0, 100.0],
            "is_outlier": [False, False, False],
        }
    )
    # residual = (actual - pred) / pred. −0.20, 0.0, +0.20 → Good Deal, Fair, Overpriced.
    oof = pd.DataFrame(
        {
            "listing_id": ["L1", "L2", "L3"],
            "transact_type": ["Sale", "Sale", "Sale"],
            "oof_predicted_price": [125.0, 100.0, 83.3333],
        }
    )
    out, _ = build_good_deal_labels(df, oof_predictions=oof)
    assert list(out["good_deal_verdict"]) == ["Good Deal", "Fair Price", "Overpriced"]


def test_build_good_deal_labels_uses_per_city_thresholds() -> None:
    """A row whose city has overridden thresholds picks the override, not default."""
    # Two rows in the same city with the same residual_pct, but
    # per-city threshold is tighter than default → one verdict at
    # -0.07 is Good Deal under -0.10 default but Fair Price under
    # -0.05 city override.
    df = pd.DataFrame(
        {
            "listing_id": ["L1", "L2"],
            "city": ["Kolkata", "Kolkata"],
            "transact_type": ["Sale", "Sale"],
            "price": [93.0, 93.0],
            "is_outlier": [False, False],
        }
    )
    oof = pd.DataFrame(
        {
            "listing_id": ["L1", "L2"],
            "transact_type": ["Sale", "Sale"],
            "oof_predicted_price": [100.0, 100.0],
        }
    )
    # Default threshold: residual_pct = -0.07 → Fair Price.
    out_default, _ = build_good_deal_labels(df, oof_predictions=oof)
    assert (out_default["good_deal_verdict"] == "Fair Price").all()

    # Per-city override: low = -0.05 → Good Deal.
    out_override, _ = build_good_deal_labels(
        df,
        oof_predictions=oof,
        thresholds={"Kolkata": {"low": -0.05, "high": 0.05}},
    )
    assert (out_override["good_deal_verdict"] == "Good Deal").all()


def test_build_good_deal_labels_assigns_null_when_oof_missing() -> None:
    """Every row gets verdict=null + reason=missing_oof_prediction when OOF absent."""
    df = _synthetic_clean_listings(n=5)
    out, _ = build_good_deal_labels(df, oof_predictions=None)
    assert out["good_deal_verdict"].isna().all()
    assert (out["reason"] == "missing_oof_prediction").all()


def test_build_good_deal_labels_assigns_null_when_predicted_non_positive() -> None:
    """Row with oof_predicted_price=0 gets verdict=null + reason=non_positive_predicted_price."""
    df = pd.DataFrame(
        {
            "listing_id": ["L1", "L2"],
            "city": ["Gurgaon", "Gurgaon"],
            "transact_type": ["Sale", "Sale"],
            "price": [100.0, 200.0],
            "is_outlier": [False, False],
        }
    )
    oof = pd.DataFrame(
        {
            "listing_id": ["L1", "L2"],
            "transact_type": ["Sale", "Sale"],
            "oof_predicted_price": [0.0, 100.0],
        }
    )
    out, _ = build_good_deal_labels(df, oof_predictions=oof)
    # L1: pred = 0 → null + non_positive.
    assert pd.isna(out.loc[out["listing_id"] == "L1", "good_deal_verdict"].iloc[0])
    assert (
        out.loc[out["listing_id"] == "L1", "reason"].iloc[0]
        == "non_positive_predicted_price"
    )
    # L2: pred = 100, actual = 200 → +100% → Overpriced.
    assert out.loc[out["listing_id"] == "L2", "good_deal_verdict"].iloc[0] == "Overpriced"


def test_build_good_deal_labels_does_not_silently_recompute_oof() -> None:
    """OOF prediction missing for some rows → those rows have reason=missing_oof_prediction."""
    df = _synthetic_clean_listings(n=10)
    # Only provide OOF for 3 of 10 listing_ids.
    oof = pd.DataFrame(
        {
            "listing_id": [
                df["listing_id"].iloc[0],
                df["listing_id"].iloc[1],
                df["listing_id"].iloc[2],
            ],
            "transact_type": ["Sale", "Sale", "Sale"],
            "oof_predicted_price": [100.0, 110.0, 95.0],
        }
    )
    out, _ = build_good_deal_labels(df, oof_predictions=oof)
    n_with_verdict = int(out["good_deal_verdict"].notna().sum())
    n_with_reason = int((out["reason"] == "missing_oof_prediction").sum())
    assert n_with_verdict == 3
    assert n_with_reason == 7
