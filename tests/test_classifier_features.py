"""Tests for ``ml.classification.features`` (Spec 22)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.schemas.predict_v3 import INPUT_FIELDS_V3
from ml.classification.features import (
    DROPPED_FOR_LEAKAGE,
    RETAINED_LOCALITY_COLUMN,
    build_classifier_feature_frame,
)

# ----- synthetic fixture helpers -----


def _make_clean_listings(
    n_per_city: int = 50,
    seed: int = 42,
    n_outlier_frac: float = 0.10,
) -> pd.DataFrame:
    """Build a small synthetic ``clean_listings``-shaped frame.

    Columns mirror the real parquet plus the 16 ``INPUT_FIELDS_V3``
    contract fields with valid values, ``is_outlier`` (bool),
    ``was_missing_bedRoom`` (so ``build_feature_frame`` validation
    passes), and the locality columns the price model would add.
    Deterministic via ``np.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    localities = ["Sector 1", "Sector 2", "Bandra", "Andheri", "Salt Lake", "Jubilee Hills"]
    rows = []
    idx = 0
    for city in cities:
        for _ in range(n_per_city):
            for _ in range(2):  # 2 transact types per (city, locality) pair
                locality = localities[idx % len(localities)]
                idx += 1
                price_inr = float(rng.uniform(50_000_000.0, 200_000_000.0))
                area = float(rng.uniform(800.0, 3000.0))
                price_per_sqft = price_inr / area
                is_outlier = bool(rng.random() < n_outlier_frac)
                rows.append(
                    {
                        "listing_id": f"L{idx:06d}",
                        "city": city,
                        "locality": locality,
                        "transact_type": "Sale" if idx % 2 == 0 else "Rent",
                        "price_inr": price_inr,
                        "price_per_sqft": price_per_sqft,
                        "is_outlier": is_outlier,
                        "property_type": "flat",
                        "sector": locality,
                        "bedRoom": 3,
                        "bathroom": 2,
                        "balcony": "2",
                        "agePossession": "Relatively New",
                        "built_up_area": area,
                        "servant_room": False,
                        "store_room": False,
                        "furnishing_type": "Semifurnished",
                        "luxury_category": "Medium",
                        "floor_category": "Mid Floor",
                        "facing": "North",
                        "amenities": ["Clubhouse", "Swimming Pool"],
                        "amenities_list": ["Clubhouse", "Swimming Pool"],
                        "features_list": ["Power Backup"],
                        "floor_num": 5.0,
                        "total_floor": 10.0,
                        "was_missing_bedRoom": False,
                    }
                )
    return pd.DataFrame(rows)


class _SentinelAggregator:
    """A fake LocalityAggregator that injects a sentinel column on
    ``transform``. Used by tests that need to verify the function
    does NOT refit a passed-in aggregator.
    """

    def __init__(self) -> None:
        self.fit_called = False
        self.transform_calls = 0

    def fit(self, df: pd.DataFrame) -> "_SentinelAggregator":
        self.fit_called = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.transform_calls += 1
        out = df.copy()
        out["locality_sentinel"] = 42
        return out


# ----- tests -----


def test_dropped_for_leakage_is_pinned() -> None:
    assert DROPPED_FOR_LEAKAGE == frozenset(
        {"price", "price_per_sqft", "locality_avg_price_sqft", "locality_smoothed_price"}
    )


def test_retained_locality_column_is_pinned() -> None:
    assert RETAINED_LOCALITY_COLUMN == "locality_listing_count"


def test_build_classifier_feature_frame_drops_all_price_derived_columns() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    leak = [c for c in DROPPED_FOR_LEAKAGE if c in out.columns]
    assert leak == [], f"Leakage columns survived: {leak}"


def test_build_classifier_feature_frame_keeps_locality_listing_count() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    assert RETAINED_LOCALITY_COLUMN in out.columns


def test_build_classifier_feature_frame_keeps_all_16_input_fields() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    for field in INPUT_FIELDS_V3:
        assert field in out.columns, f"Missing contract field: {field}"


def test_build_classifier_feature_frame_adds_split_column() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    assert "split" in out.columns
    assert set(out["split"].astype(str).unique()) <= {"train", "val", "test"}
    n_train = int((out["split"] == "train").sum())
    n_val = int((out["split"] == "val").sum())
    n_test = int((out["split"] == "test").sum())
    n_total = len(out)
    # 70/15/15 with 1-row tolerance.
    assert abs(n_train / n_total - 0.70) < 0.02
    assert abs(n_val / n_total - 0.15) < 0.02
    assert abs(n_test / n_total - 0.15) < 0.02


def test_build_classifier_feature_frame_preserves_is_outlier_flag() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    # is_outlier must be present in the output (re-attached after build_feature_frame).
    assert "is_outlier" in out.columns
    # And the values must be bit-identical to the input (matched on listing_id).
    expected = (
        df[["listing_id", "is_outlier"]]
        .set_index("listing_id")
        .loc[out["listing_id"].values]
        ["is_outlier"]
        .values
    )
    assert (out["is_outlier"].values == expected).all()


def test_build_classifier_feature_frame_preserves_listing_id() -> None:
    df = _make_clean_listings()
    out = build_classifier_feature_frame(df)
    # Same set of listing_ids.
    assert set(out["listing_id"]) == set(df["listing_id"])
    # Same length (no rows dropped).
    assert len(out) == len(df)


def test_build_classifier_feature_frame_reuses_passed_locality_aggregator_without_refit() -> None:
    df = _make_clean_listings()
    sent = _SentinelAggregator()
    out = build_classifier_feature_frame(df, locality_aggregator=sent)
    # The sentinel column the fake aggregator injected must appear.
    assert "locality_sentinel" in out.columns
    # And `fit` must not have been called.
    assert sent.fit_called is False
    assert sent.transform_calls == 1


def test_build_classifier_feature_frame_column_order_is_deterministic() -> None:
    df = _make_clean_listings()
    out1 = build_classifier_feature_frame(df)
    out2 = build_classifier_feature_frame(df)
    assert list(out1.columns) == list(out2.columns)
