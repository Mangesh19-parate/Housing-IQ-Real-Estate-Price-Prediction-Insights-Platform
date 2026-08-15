"""Tests for the SectorTargetEncoder lever (Spec 14).

Mirrors the v1 ``LocalityAggregator`` tests in Step 12 — pins LOO
semantics, outlier exclusion, transform-doesn't-refit, and the
city-prior fallback for unseen (city, sector) pairs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.training.levers.target_encoding import (
    SECTOR_OUTPUT_COLUMN,
    SECTOR_SMOOTHING_PRIOR_WEIGHT,
    SectorTargetEncoder,
)


def _make_frame(n_per: int = 5) -> pd.DataFrame:
    """Build a tiny 3-city × 2-sector fixture with stable prices."""
    rows = []
    for city in ("Gurgaon", "Mumbai", "Kolkata"):
        for sector in ("sector 36", "sector 50"):
            for i in range(n_per):
                rows.append(
                    {
                        "city": city,
                        "sector": sector,
                        "price_inr": 1_000_000.0,
                        "price_per_sqft": 5_000.0
                        + (100.0 if city == "Mumbai" else 0.0)
                        + (50.0 if sector == "sector 50" else 0.0),
                        "is_outlier": False,
                    }
                )
    return pd.DataFrame(rows)


def test_sector_target_encoder_fit_computes_group_means():
    df = _make_frame(n_per=4)
    enc = SectorTargetEncoder().fit(df)
    # 3 cities × 2 sectors = 6 groups
    assert enc.n_groups_ == 6
    # Each (city, sector) group mean is recoverable from the frame.
    g = (
        df[~df["is_outlier"]]
        .groupby(["city", "sector"])["price_per_sqft"]
        .mean()
    )
    for (city, sector), expected in g.items():
        rows = enc.fitted_aggregates_[
            (enc.fitted_aggregates_["city"] == city)
            & (enc.fitted_aggregates_["sector"] == sector)
        ]
        # Fitted sums + counts should reproduce the mean.
        sum_v = float(rows["sum_psf"].iloc[0])
        cnt_v = int(rows["count"].iloc[0])
        assert cnt_v == 4
        assert abs(sum_v / cnt_v - expected) < 1e-6


def test_sector_target_encoder_excludes_outliers_from_fit():
    df = _make_frame(n_per=4)
    # Mark 2 of 4 Gurgaon/sector 36 rows as outliers with bogus prices.
    mask_idx = df.index[
        (df["city"] == "Gurgaon") & (df["sector"] == "sector 36")
    ].tolist()
    df.loc[mask_idx[:2], "is_outlier"] = True
    df.loc[mask_idx[:2], "price_per_sqft"] = 99_999.0

    enc = SectorTargetEncoder().fit(df)
    g = enc.fitted_aggregates_
    row = g[(g["city"] == "Gurgaon") & (g["sector"] == "sector 36")].iloc[0]
    # The 2 outliers' prices must not be in the sum / count.
    assert int(row["count"]) == 2
    # Original group: 4 rows × 5000.0 (Gurgaon, sector 36) = 20000.
    # The 2 outliers were replaced with 99_999; after exclusion the
    # sum should be 2 × 5000 = 10000 (NOT contaminated by 99_999).
    assert float(row["sum_psf"]) == pytest.approx(10_000.0, abs=1e-6)


def test_sector_target_encoder_leave_one_out_semantics():
    """Pin the LOO semantic — own contribution excluded from the mean."""
    # Single (city, sector) group with N=5 rows, known prices.
    df = pd.DataFrame(
        {
            "city": ["Gurgaon"] * 5,
            "sector": ["sector 36"] * 5,
            "price_inr": [1_000_000.0] * 5,
            "price_per_sqft": [1.0, 2.0, 3.0, 4.0, 5.0],
            "is_outlier": [False] * 5,
        }
    )
    enc = SectorTargetEncoder().fit(df)
    out = enc.transform(df)
    # Row with price_per_sqft=3.0 (index 2): LOO mean over [1, 2, 4, 5]
    # = 12 / 4 = 3.0. The smoothed value blends toward the city mean
    # (= 3.0 here), so it equals 3.0 exactly when the LOO mean equals
    # the city prior.
    val_3 = out[SECTOR_OUTPUT_COLUMN].iloc[2]
    expected_loo = 12.0 / 4.0
    w = SECTOR_SMOOTHING_PRIOR_WEIGHT
    # city prior = mean([1,2,3,4,5]) = 3.0
    smoothed = (4.0 * expected_loo + w * 3.0) / (4.0 + w)
    assert abs(val_3 - smoothed) < 1e-6


def test_sector_target_encoder_transform_does_not_refit():
    df_a = _make_frame(n_per=4)
    df_b = pd.concat(
        [_make_frame(n_per=2), pd.DataFrame(
            [{
                "city": "Gurgaon",
                "sector": "sector_NEW",
                "price_inr": 1.0,
                "price_per_sqft": 7_777.0,
                "is_outlier": False,
            }]
        )],
        ignore_index=True,
    )
    enc = SectorTargetEncoder().fit(df_a)
    out = enc.transform(df_b)
    # The unseen (Gurgaon, sector_NEW) row should fall back to the
    # Gurgaon city prior, NOT to the mean over df_b's rows
    # (which would include itself).
    unseen = out[out["sector"] == "sector_NEW"].iloc[0]
    city_prior = enc.city_priors_["Gurgaon"]
    # LOO with n=0 + smoothing: num = w*prior, denom = w -> equal to prior
    assert abs(unseen[SECTOR_OUTPUT_COLUMN] - city_prior) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
