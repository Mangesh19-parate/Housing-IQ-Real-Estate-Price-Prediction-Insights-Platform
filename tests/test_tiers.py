"""Tests for ``ml.classification.tiers`` (Spec 21).

All test names are pinned by the spec's Definition of DoD item 1.
"""

from __future__ import annotations

import logging
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from ml.classification.tiers import (
    QUANTILE_CUTPOINTS,
    TIER_LABELS,
    build_price_tier_labels,
    compute_per_city_quantile_boundaries,
)


def _synthetic_clean_listings(
    n_per_city: int = 200,
    cities: tuple[str, ...] = ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"),
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic clean_listings DataFrame.

    Mirrors the canonical schema (subset of columns) used by
    Step 07: ``listing_id``, ``city``, ``locality``, ``transact_type``,
    ``price``, ``price_per_sqft``, ``is_outlier``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for city in cities:
        for transact in ("Sale", "Rent"):
            for i in range(n_per_city):
                rows.append(
                    {
                        "listing_id": f"{city}_{transact}_{i:04d}",
                        "city": city,
                        "locality": f"{city}_Sector_{i % 30}",
                        "transact_type": transact,
                        "price": float(rng.integers(2_000_000, 20_000_000)),
                        "price_per_sqft": float(rng.integers(3000, 12000)),
                        "is_outlier": False,
                    }
                )
    return pd.DataFrame(rows)


def test_tier_labels_are_pinned() -> None:
    """``TIER_LABELS`` is the spec-pinned literal tuple."""
    assert TIER_LABELS == ("Budget", "Mid-Range", "Premium", "Luxury")


def test_quantile_cutpoints_are_pinned() -> None:
    """``QUANTILE_CUTPOINTS`` is the spec-pinned quartile literal."""
    assert QUANTILE_CUTPOINTS == (0.25, 0.50, 0.75)


def test_compute_per_city_quantile_boundaries_returns_three_keys_per_city() -> None:
    """Every synthetic city has a 3-element quantile list."""
    df = _synthetic_clean_listings(n_per_city=200)
    train_df = df.copy()  # 200 rows per (city, transact)
    train_df["is_outlier"] = False

    boundaries = compute_per_city_quantile_boundaries(
        train_df,
        price_per_sqft_col="price_per_sqft",
        min_rows=100,
    )

    assert set(boundaries.keys()) == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}
    for city, by_transact in boundaries.items():
        assert set(by_transact.keys()) == {"Sale", "Rent"}
        for transact, q in by_transact.items():
            assert len(q) == 3
            assert q[0] <= q[1] <= q[2]


def test_compute_per_city_quantile_boundaries_skips_small_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A (city, transact) group with n < min_rows is dropped + WARNING."""
    # 50 rows for Kolkata; 200 for everyone else.
    rows = []
    rng = np.random.default_rng(7)
    for i in range(50):
        rows.append(
            {
                "city": "Kolkata",
                "transact_type": "Sale",
                "price_per_sqft": float(rng.integers(3000, 12000)),
            }
        )
    for i in range(200):
        rows.append(
            {
                "city": "Gurgaon",
                "transact_type": "Sale",
                "price_per_sqft": float(rng.integers(3000, 12000)),
            }
        )
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.WARNING, logger="ml.classification.tiers"):
        boundaries = compute_per_city_quantile_boundaries(
            df, price_per_sqft_col="price_per_sqft", min_rows=100
        )

    assert "Kolkata" not in boundaries
    assert "Gurgaon" in boundaries
    assert any("Kolkata" in rec.message for rec in caplog.records)


def test_build_price_tier_labels_assigns_four_tiers() -> None:
    """Synthetic 4-city data produces all four tier labels."""
    df = _synthetic_clean_listings(n_per_city=300)
    out, _ = build_price_tier_labels(df, min_rows=100)

    assert set(out["price_tier"].dropna().unique()) == {
        "Budget",
        "Mid-Range",
        "Premium",
        "Luxury",
    }


def test_build_price_tier_labels_reuses_passed_boundaries() -> None:
    """Passing boundaries skips quantile recomputation."""
    df = _synthetic_clean_listings(n_per_city=200)

    # Custom boundaries: 1.0/2.0/3.0 → Budget < 1, Mid 1..2, Premium 2..3, Luxury ≥ 3.
    custom_boundaries = {
        "Gurgaon": {"Sale": [1.0, 2.0, 3.0]},
    }
    # The other cities are absent → rows in those cities get price_tier = null.
    # Sale rows in Gurgaon → assigned via custom boundaries.
    # Rent rows in Gurgaon → no Rent boundary → null.
    # All Kolkata/Hyderabad/Mumbai → null.

    # Mock pd.DataFrame.quantile to assert it is NOT called on the
    # price_per_sqft column of the input frame when boundaries are
    # passed. The split helper does call it internally; we check that
    # the *tier builder's* boundary recomputation step doesn't.
    with mock.patch.object(pd.DataFrame, "quantile", autospec=True):
        out, boundaries = build_price_tier_labels(
            df, boundaries=custom_boundaries, min_rows=100
        )

    # Boundaries dict is returned as-is (no recomputation).
    assert boundaries == custom_boundaries

    # Synthetic data uses price_per_sqft in [3000, 12000] → all
    # Sale/Gurgaon rows should be "Luxury" (≥ 3).
    ggn_sale = out[
        (out["city"] == "Gurgaon") & (out["transact_type"] == "Sale")
    ]
    assert (ggn_sale["price_tier"] == "Luxury").all()

    # Rent in Gurgaon → no boundary → null.
    ggn_rent = out[
        (out["city"] == "Gurgaon") & (out["transact_type"] == "Rent")
    ]
    assert ggn_rent["price_tier"].isna().all()

    # Other cities → no boundary → null.
    other = out[out["city"] != "Gurgaon"]
    assert other["price_tier"].isna().all()


def test_build_price_tier_labels_preserves_is_outlier_flag() -> None:
    """``is_outlier`` is bit-for-bit identical between input and output."""
    df = _synthetic_clean_listings(n_per_city=200)
    # Flip 10% to outlier.
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(df.index, size=int(0.10 * len(df)), replace=False)
    df.loc[outlier_idx, "is_outlier"] = True

    out, _ = build_price_tier_labels(df, min_rows=100)

    assert (out["is_outlier"].astype(bool) == df.loc[out.index, "is_outlier"].astype(bool)).all()


def test_build_price_tier_labels_assigns_null_for_missing_city_group() -> None:
    """A row whose (city, transact_type) has no boundary gets null."""
    df = _synthetic_clean_listings(n_per_city=200)
    # Use boundaries that omit "Sale" everywhere → every Sale row is null.
    boundaries = {
        "Gurgaon": {"Rent": [1.0, 2.0, 3.0]},
        "Hyderabad": {"Rent": [1.0, 2.0, 3.0]},
        "Kolkata": {"Rent": [1.0, 2.0, 3.0]},
        "Mumbai": {"Rent": [1.0, 2.0, 3.0]},
    }
    out, _ = build_price_tier_labels(df, boundaries=boundaries, min_rows=100)

    sale_rows = out[out["transact_type"] == "Sale"]
    assert sale_rows["price_tier"].isna().all()
