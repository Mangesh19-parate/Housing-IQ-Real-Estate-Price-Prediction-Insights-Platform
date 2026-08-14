"""Tests for ml.features.feature_frame (Phase 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from api.schemas.predict_v3 import INPUT_FIELDS_V3
from ml.features.feature_frame import (
    AGE_BUCKET_ORDINAL,
    BALCONY_ORDINAL,
    ENGINEERED_COLUMNS,
    FLOOR_CATEGORY_ORDINAL,
    FURNISHING_TYPE_ORDINAL,
    LUXURY_CATEGORY_ORDINAL,
    build_feature_frame,
    derive_row_features,
    select_top_amenities,
    slugify_amenity,
)

# 16 contract fields per INPUT_FIELDS_V3, plus is_outlier + a was_missing_*.
_BASE_COLUMNS = list(INPUT_FIELDS_V3) + ["is_outlier", "was_missing_bedRoom"]
_RENAMES = {"amenities_list": "amenities"}  # see build_feature_frame comment


def _make_minimal_frame(n: int = 3) -> pd.DataFrame:
    """Build a minimal valid input frame with the 16 contract fields.

    The cleaning layer emits ``amenities_list`` (list-valued), not the API
    wire-format ``amenities``. We mirror that here.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "property_type": "flat",
                "sector": f"sector {i}",
                "city": "Gurgaon",
                "transact_type": "Sale",
                "bedRoom": 3,
                "bathroom": 3,
                "balcony": "2",
                "agePossession": "Relatively New",
                "built_up_area": 1450.0,
                "servant_room": False,
                "store_room": False,
                "furnishing_type": "Semifurnished",
                "luxury_category": "Medium",
                "floor_category": "Mid Floor",
                "facing": "North",
                "amenities_list": ["Swimming Pool", "Club House"],
                "is_outlier": False,
                "was_missing_bedRoom": False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constant pinning
# ---------------------------------------------------------------------------


def test_engineered_columns_constant_has_expected_entries() -> None:
    """Pins the 11-entry ENGINEERED_COLUMNS tuple."""
    assert isinstance(ENGINEERED_COLUMNS, tuple)
    assert len(ENGINEERED_COLUMNS) == 11
    assert ENGINEERED_COLUMNS == (
        "price_per_sqft",
        "n_amenities",
        "n_features",
        "floor_ratio",
        "age_bucket_ord",
        "bath_bed_ratio",
        "area_per_bedroom",
        "locality_avg_price_sqft",
        "locality_listing_count",
        "locality_smoothed_price",
        "top_amenities_count",
    )


def test_age_bucket_ordinal_mapping_is_pinned() -> None:
    """Exact equality against the documented dict."""
    assert AGE_BUCKET_ORDINAL == {
        "New Property": 0,
        "Under Construction": 1,
        "Relatively New": 2,
        "Moderately Old": 3,
        "Old Property": 4,
    }
    # All FloorCategory, LuxuryCategory, FurnishingType, Balcony dicts
    # are pinned too — quick check on a couple.
    assert FLOOR_CATEGORY_ORDINAL == {"Low Floor": 0, "Mid Floor": 1, "High Floor": 2}
    assert LUXURY_CATEGORY_ORDINAL == {"Low": 0, "Medium": 1, "High": 2}
    assert FURNISHING_TYPE_ORDINAL == {
        "Unfurnished": 0,
        "Semifurnished": 1,
        "Furnished": 2,
    }
    assert BALCONY_ORDINAL == {"0": 0, "1": 1, "2": 2, "3": 3, "3+": 4}


# ---------------------------------------------------------------------------
# Row-feature derivations
# ---------------------------------------------------------------------------


def test_derive_row_features_adds_expected_columns() -> None:
    """Every non-locality ENGINEERED_COLUMNS entry appears in output."""
    df = _make_minimal_frame()
    out = derive_row_features(df)
    # Locality cols come from LocalityAggregator; here we only verify
    # the 8 row-level ones are present.
    for col in (
        "price_per_sqft",
        "n_amenities",
        "n_features",
        "floor_ratio",
        "age_bucket_ord",
        "bath_bed_ratio",
        "area_per_bedroom",
        "top_amenities_count",
    ):
        assert col in out.columns, f"missing derived column: {col}"


def test_price_per_sqft_is_within_row_ratio() -> None:
    """price_inr=10_000_000, built_up_area=1000 -> price_per_sqft=10_000.0."""
    df = _make_minimal_frame(n=1)
    df["price_inr"] = 10_000_000
    df["built_up_area"] = 1000.0
    out = derive_row_features(df)
    assert float(out["price_per_sqft"].iloc[0]) == 10_000.0


def test_n_amenities_counts_list_length() -> None:
    """amenities_list of length 3 -> n_amenities=3."""
    df = _make_minimal_frame(n=1)
    df["amenities_list"] = [["A", "B", "C"]]
    out = derive_row_features(df)
    assert int(out["n_amenities"].iloc[0]) == 3


def test_floor_ratio_is_floor_over_total() -> None:
    """floor_num=7, total_floor=14 -> floor_ratio=0.5."""
    df = _make_minimal_frame(n=1)
    df["floor_num"] = 7
    df["total_floor"] = 14
    out = derive_row_features(df)
    assert float(out["floor_ratio"].iloc[0]) == 0.5


def test_bath_bed_ratio_handles_zero_bedrooms() -> None:
    """bedRoom=0 -> bath_bed_ratio is NaN, no raise."""
    df = _make_minimal_frame(n=1)
    df["bedRoom"] = 0
    out = derive_row_features(df)
    val = out["bath_bed_ratio"].iloc[0]
    assert pd.isna(val) or val != val  # NaN sentinel


def test_area_per_bedroom_handles_zero_bedrooms() -> None:
    """bedRoom=0 -> area_per_bedroom is NaN, no raise."""
    df = _make_minimal_frame(n=1)
    df["bedRoom"] = 0
    out = derive_row_features(df)
    val = out["area_per_bedroom"].iloc[0]
    assert pd.isna(val) or val != val


def test_top_amenities_count_equals_10_when_data_has_at_least_10() -> None:
    """With >=10 unique amenities in the corpus, top_amenities_count flags 10 per row.

    Construct a frame where 12 amenities each appear once, and one row
    has all of them -> top_amenities_count should be 10 (capped at K).
    """
    amenities = [f"amenity_{i}" for i in range(12)]
    rows = [{"amenities_list": amenities, **{
        k: v for k, v in _make_minimal_frame(n=1).iloc[0].items() if k != "amenities_list"
    }}]
    df = pd.DataFrame(rows)
    out = derive_row_features(df)
    assert int(out["top_amenities_count"].iloc[0]) == 10


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


def test_slugify_amenity_normalizes_punctuation() -> None:
    """Punctuation normalized to underscores."""
    assert slugify_amenity("Swimming Pool") == "swimming_pool"
    assert slugify_amenity("Club House / Lounge") == "club_house_lounge"
    assert slugify_amenity("  24/7 Security  ") == "24_7_security"
    assert slugify_amenity("Power-Backup!") == "power_backup"


# ---------------------------------------------------------------------------
# Top-K amenity selection
# ---------------------------------------------------------------------------


def test_select_top_amenities_returns_top_k_by_frequency() -> None:
    df = pd.DataFrame(
        {
            "amenities_list": [
                ["A", "B", "C"],
                ["A", "B"],
                ["A"],
                ["D"],
            ]
        }
    )
    top = select_top_amenities(df, k=3)
    assert top == ["A", "B", "C"]


def test_select_top_amenities_handles_missing_column() -> None:
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    assert select_top_amenities(df, k=10) == []


# ---------------------------------------------------------------------------
# build_feature_frame — top-level
# ---------------------------------------------------------------------------


def test_build_feature_frame_raises_on_missing_input_field() -> None:
    """Omitting built_up_area raises ValueError."""
    df = _make_minimal_frame()
    df = df.drop(columns=["built_up_area"])
    with pytest.raises(ValueError, match="built_up_area"):
        build_feature_frame(df)


def test_build_feature_frame_column_order_is_deterministic() -> None:
    """Final column order = INPUT_FIELDS_V3 + ENGINEERED_COLUMNS."""
    df = _make_minimal_frame()
    out = build_feature_frame(df)
    expected = list(INPUT_FIELDS_V3) + list(ENGINEERED_COLUMNS)
    assert list(out.columns) == expected


def test_build_feature_frame_drops_is_outlier_and_was_missing() -> None:
    """is_outlier and was_missing_* are dropped from the output."""
    df = _make_minimal_frame()
    out = build_feature_frame(df)
    assert "is_outlier" not in out.columns
    assert "was_missing_bedRoom" not in out.columns
    # No was_missing_* at all.
    assert not any(c.startswith("was_missing_") for c in out.columns)


def test_feature_frame_excludes_contact_fields() -> None:
    """No column name matches the PII regex."""
    df = _make_minimal_frame()
    out = build_feature_frame(df)
    bad = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.I)
    for c in out.columns:
        assert not bad.search(c), f"PII-like column in feature frame: {c}"


def test_feature_frame_excludes_price_from_inputs() -> None:
    """``price`` is not in any preprocessor input tuple.

    (We don't have direct access to those constants from the feature
    frame test; instead we document the invariant here: build_feature_frame
    does not surface ``price_inr`` as a feature column. It is a training
    target only.)
    """
    df = _make_minimal_frame(n=1)
    df["price_inr"] = 5_000_000
    out = build_feature_frame(df)
    # price_inr is not in INPUT_FIELDS_V3 and not in ENGINEERED_COLUMNS,
    # so it should NOT appear in the output.
    assert "price_inr" not in out.columns


# Late import to satisfy regex symbol at top
import re  # noqa: E402
