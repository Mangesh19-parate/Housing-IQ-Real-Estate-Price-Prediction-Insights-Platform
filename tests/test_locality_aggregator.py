"""Tests for ml.features.locality_aggregator (Phase 2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.features.locality_aggregator import (
    SMOOTHING_PRIOR_WEIGHT,
    LocalityAggregator,
)


def _make_frame() -> pd.DataFrame:
    """3 cities x 2 localities, 5 rows each. Deterministic prices.

    Total: 30 rows. None are flagged as outliers.
    """
    rows = []
    for city, base in [("Gurgaon", 10_000), ("Hyderabad", 8_000), ("Mumbai", 12_000)]:
        for loc, mult in [("A", 1.0), ("B", 1.5)]:
            for i in range(5):
                psf = base * mult + i  # price_per_sqft varies per row
                price_inr = psf * 1000  # area = 1000 sqft each
                rows.append(
                    {
                        "city": city,
                        "locality": loc,
                        "price_per_sqft": float(psf),
                        "price_inr": float(price_inr),
                        "is_outlier": False,
                    }
                )
    return pd.DataFrame(rows)


def _make_lo_frame() -> pd.DataFrame:
    """Single (city, locality) group with prices 1..10 (LOO math fixture)."""
    return pd.DataFrame(
        {
            "city": ["Gurgaon"] * 10,
            "locality": ["S1"] * 10,
            "price_per_sqft": list(range(1, 11)),  # 1..10
            "price_inr": list(range(1, 11)),
            "is_outlier": [False] * 10,
        }
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def test_locality_aggregator_fit_computes_group_means() -> None:
    df = _make_frame()
    agg = LocalityAggregator().fit(df)
    assert agg.fitted_aggregates_ is not None
    # 3 cities x 2 localities = 6 groups.
    assert len(agg.fitted_aggregates_) == 6
    # All groups have count == 5 (5 rows each).
    assert (agg.fitted_aggregates_["count"] == 5).all()
    # city_priors_ has 3 cities.
    assert set(agg.city_priors_.keys()) == {"Gurgaon", "Hyderabad", "Mumbai"}


def test_locality_aggregator_excludes_outlier_rows_from_fit() -> None:
    """Rows with is_outlier=True must not contribute to group sums/counts."""
    df = _make_frame().copy()
    # Flag all Gurgaon-B rows as outliers (5 rows).
    mask = (df["city"] == "Gurgaon") & (df["locality"] == "B")
    df.loc[mask, "is_outlier"] = True
    agg = LocalityAggregator().fit(df)
    # Gurgaon-B group should have count == 0 (all 5 rows excluded).
    row = agg.fitted_aggregates_[
        (agg.fitted_aggregates_["city"] == "Gurgaon")
        & (agg.fitted_aggregates_["locality"] == "B")
    ].iloc[0]
    assert int(row["count"]) == 0
    # Other groups unaffected: count == 5.
    others = agg.fitted_aggregates_[
        ~(
            (agg.fitted_aggregates_["city"] == "Gurgaon")
            & (agg.fitted_aggregates_["locality"] == "B")
        )
    ]
    assert (others["count"] == 5).all()


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_locality_aggregator_transform_joins_by_city_locality() -> None:
    """New (city, locality) at transform time falls back to the city mean."""
    train = _make_frame()
    agg = LocalityAggregator().fit(train)
    # Transform a row from a never-seen locality.
    new_row = pd.DataFrame(
        {
            "city": ["Gurgaon"],
            "locality": ["NEW_LOCALITY"],
            "price_per_sqft": [99_999.0],
            "price_inr": [99_999_000.0],
            "is_outlier": [False],
        }
    )
    out = agg.transform(new_row)
    # locality_avg_price_sqft should equal the Gurgaon city prior, not
    # the row's own price (which would be leakage).
    ggn_prior = agg.city_priors_["Gurgaon"]
    assert math.isclose(
        float(out["locality_avg_price_sqft"].iloc[0]),
        ggn_prior,
        rel_tol=1e-9,
    )
    assert int(out["locality_listing_count"].iloc[0]) == 0


def test_locality_aggregator_transform_does_not_refit() -> None:
    """Fitting on frame A, transforming frame B with a new combo does not
    recompute group means from B's rows.
    """
    train = _make_frame()
    agg = LocalityAggregator().fit(train)
    # Build a transform frame with a brand-new locality + a price that
    # would obviously shift a naive refit.
    new_frame = pd.DataFrame(
        {
            "city": ["Gurgaon"] * 2,
            "locality": ["FAKE_LOCALITY", "FAKE_LOCALITY"],
            "price_per_sqft": [99_999.0, 99_999.0],
            "price_inr": [99_999_000.0, 99_999_000.0],
            "is_outlier": [False, False],
        }
    )
    out = agg.transform(new_frame)
    # Both rows should get the city prior (the new locality was unseen).
    ggn_prior = agg.city_priors_["Gurgaon"]
    assert all(
        math.isclose(float(out["locality_avg_price_sqft"].iloc[i]), ggn_prior, rel_tol=1e-9)
        for i in range(2)
    )


def test_locality_aggregator_leave_one_out_semantics() -> None:
    """Row #5 of [1..10] -> locality_avg_price_sqft = mean([1,2,3,4,5,7,8,9,10]).

    Specifically, NOT mean([1..10]) = 5.5.
    """
    df = _make_lo_frame()
    agg = LocalityAggregator().fit(df)
    out = agg.transform(df)
    # Row #5 (index 5) had price_per_sqft == 6.
    row_5_avg = float(out["locality_avg_price_sqft"].iloc[5])
    expected = np.mean([1, 2, 3, 4, 5, 7, 8, 9, 10])
    assert math.isclose(row_5_avg, expected, rel_tol=1e-9)
    # And NOT the full group mean (which would be 5.5).
    assert not math.isclose(row_5_avg, 5.5, rel_tol=1e-9)


def test_locality_aggregator_smoothed_price_blends_with_prior() -> None:
    """Bayesian smoother: at the locality level the formula is

        (n * mean + w * city_prior) / (n + w)

    For a 1-row group with self excluded (n=0), the result is
    city_prior * w / w = city_prior.
    """
    df = pd.DataFrame(
        {
            "city": ["Gurgaon", "Gurgaon"],
            "locality": ["L1", "L2"],
            "price_per_sqft": [10_000.0, 12_000.0],
            "price_inr": [10_000_000.0, 12_000_000.0],
            "is_outlier": [False, False],
        }
    )
    agg = LocalityAggregator().fit(df)
    out = agg.transform(df)
    # city_priors_ for Gurgaon is mean([10_000, 12_000]) = 11_000.
    ggn_prior = agg.city_priors_["Gurgaon"]
    assert math.isclose(ggn_prior, 11_000.0, rel_tol=1e-9)
    # Each locality has only 1 row in training; LOO weight = 0, so
    # smoothed = (0 + w * prior) / (0 + w) = prior. The value equals
    # the city prior exactly.
    for i in range(2):
        val = float(out["locality_smoothed_price"].iloc[i])
        assert math.isclose(val, ggn_prior, rel_tol=1e-9)
    # locality_listing_count reflects the LOO-adjusted count (n=1 - 1 = 0).
    # We don't expose this as LOO here — we report the raw group count
    # (which is 1) so downstream consumers can spot rare localities.
    for i in range(2):
        assert int(out["locality_listing_count"].iloc[i]) == 1


def test_locality_aggregator_idempotent_fit() -> None:
    """Calling fit twice overwrites prior state without leaking rows."""
    df1 = _make_frame()
    agg = LocalityAggregator().fit(df1)
    groups_after_first = agg.fitted_aggregates_.copy()
    # Refit on a smaller frame.
    df2 = df1[df1["city"] == "Gurgaon"].reset_index(drop=True)
    agg.fit(df2)
    # Should have only Gurgaon groups (3 -> 2 localities = 2 groups).
    assert len(agg.fitted_aggregates_) == 2
    assert set(agg.fitted_aggregates_["city"]) == {"Gurgaon"}
    # Prior state should not bleed through.
    assert groups_after_first is not agg.fitted_aggregates_


def test_locality_aggregator_transform_before_fit_raises() -> None:
    """Calling transform before fit raises RuntimeError."""
    df = _make_frame()
    agg = LocalityAggregator()
    with pytest.raises(RuntimeError, match="before fit"):
        agg.transform(df)


def test_locality_aggregator_smoothing_prior_weight_pinned() -> None:
    """The smoothing constant is a pinned module-level value."""
    assert SMOOTHING_PRIOR_WEIGHT == 20.0
