"""Tests for ``ml.cleaning.outliers`` — Step 06 outlier flagging layer."""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from ml.cleaning.outliers import (
    IQR_MULTIPLIER,
    OUTLIER_DOMAIN_RULES,
    OUTLIER_NUMERIC_COLUMNS,
    OUTLIER_PROPERTY_TYPE_EXEMPTIONS,
    OUTLIER_REASON_COLUMN,
    PERCENTILE_LOWER,
    PERCENTILE_UPPER,
    flag_all_outliers,
    flag_domain_rule_outliers,
    flag_iqr_outliers,
    flag_percentile_outliers,
)
from tests.fixtures.dedup_outlier_fixtures import (
    CANONICAL_SUBSET,
    NORMAL_ROW,
    make_frame,
)

# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------

def test_outlier_numeric_columns_constant_matches_trd() -> None:
    """OUTLIER_NUMERIC_COLUMNS == ("price_inr", "area_sqft", "price_per_sqft")."""
    assert OUTLIER_NUMERIC_COLUMNS == ("price_inr", "area_sqft", "price_per_sqft")


def test_percentile_bounds_constants() -> None:
    assert PERCENTILE_LOWER == 0.01
    assert PERCENTILE_UPPER == 0.99


def test_iqr_multiplier_constant() -> None:
    assert IQR_MULTIPLIER == 1.5


def test_outlier_reason_column_constant() -> None:
    assert OUTLIER_REASON_COLUMN == "outlier_reasons"


def test_outlier_property_type_exemptions_contains_villa_farmhouse() -> None:
    assert "villa" in OUTLIER_PROPERTY_TYPE_EXEMPTIONS
    assert "farmhouse" in OUTLIER_PROPERTY_TYPE_EXEMPTIONS
    assert "independent house" in OUTLIER_PROPERTY_TYPE_EXEMPTIONS


def test_outlier_domain_rules_target_bedroom_and_bathroom() -> None:
    assert "bedRoom" in OUTLIER_DOMAIN_RULES
    assert "bathroom" in OUTLIER_DOMAIN_RULES
    assert OUTLIER_DOMAIN_RULES["bedRoom"]["max"] == 15
    assert OUTLIER_DOMAIN_RULES["bathroom"]["max"] == 15


# ---------------------------------------------------------------------------
# B. flag_percentile_outliers
# ---------------------------------------------------------------------------

def test_flag_percentile_outliers_returns_bool_series() -> None:
    df = make_frame([{"city": "Gurgaon", "price_inr": 1.0}])
    out = flag_percentile_outliers(df, "price_inr")
    assert out.dtype == "bool"
    assert len(out) == len(df)


def test_flag_percentile_outliers_uses_per_city_bounds() -> None:
    """City A: row above A's 99th → True; row inside → False.
    City B: row below B's 1st → True; row inside → False."""
    # Build two distinct distributions per city.
    df = pd.DataFrame({
        "city": ["A"] * 100 + ["B"] * 100,
        "price_inr": (
            [1.0] * 50 + [50.0] * 49 + [1_000_000.0]   # A: extreme above 99th
            + [10.0] * 50 + [20.0] * 49 + [-1_000.0]  # B: extreme below 1st
        ),
    })
    out = flag_percentile_outliers(df, "price_inr")
    # Last row of A is the 100th → flagged True.
    assert bool(out.iloc[99]) is True
    # Last row of B is the 200th → flagged True.
    assert bool(out.iloc[199]) is True
    # A row inside A's distribution is False.
    assert bool(out.iloc[0]) is False
    # B row inside B's distribution is False.
    assert bool(out.iloc[100]) is False


# ---------------------------------------------------------------------------
# C. flag_iqr_outliers
# ---------------------------------------------------------------------------

def test_flag_iqr_outliers_returns_bool_series() -> None:
    df = make_frame([{"city": "Gurgaon", "price_inr": 1.0}])
    out = flag_iqr_outliers(df, "price_inr")
    assert out.dtype == "bool"
    assert len(out) == len(df)


def test_flag_iqr_outliers_uses_per_city_bounds() -> None:
    """Same shape as percentile test, with Tukey fence."""
    df = pd.DataFrame({
        "city": ["A"] * 100 + ["B"] * 100,
        "price_inr": (
            [1.0] * 50 + [50.0] * 49 + [10_000.0]
            + [10.0] * 50 + [20.0] * 49 + [-1_000.0]
        ),
    })
    out = flag_iqr_outliers(df, "price_inr")
    assert bool(out.iloc[99]) is True
    assert bool(out.iloc[199]) is True
    assert bool(out.iloc[0]) is False
    assert bool(out.iloc[100]) is False


# ---------------------------------------------------------------------------
# D. flag_domain_rule_outliers
# ---------------------------------------------------------------------------

def test_flag_domain_rule_outliers_flags_high_bedroom_count() -> None:
    """bedRoom=20 + property_type='flat' → flagged."""
    df = make_frame([{"city": "Gurgaon", "property_type": "flat", "bedRoom": 20, "bathroom": 3}])
    out = flag_domain_rule_outliers(df)
    assert bool(out.iloc[0]) is True


def test_flag_domain_rule_outliers_does_not_flag_villa_with_high_bedroom() -> None:
    """bedRoom=20 + property_type='villa' → NOT flagged."""
    df = make_frame([{"city": "Gurgaon", "property_type": "villa", "bedRoom": 20, "bathroom": 3}])
    out = flag_domain_rule_outliers(df)
    assert bool(out.iloc[0]) is False


def test_flag_domain_rule_outliers_does_not_flag_farmhouse() -> None:
    """property_type='farmhouse' + bedRoom=25 → NOT flagged."""
    df = make_frame([
        {"city": "Gurgaon", "property_type": "farmhouse", "bedRoom": 25, "bathroom": 5},
    ])
    out = flag_domain_rule_outliers(df)
    assert bool(out.iloc[0]) is False


def test_flag_domain_rule_outliers_does_not_flag_independent_house() -> None:
    """property_type='independent house' → NOT flagged."""
    df = make_frame([{
        "city": "Gurgaon",
        "property_type": "independent house",
        "bedRoom": 30,
        "bathroom": 6,
    }])
    out = flag_domain_rule_outliers(df)
    assert bool(out.iloc[0]) is False


# ---------------------------------------------------------------------------
# E. flag_all_outliers (top-level)
# ---------------------------------------------------------------------------

def test_flag_all_outliers_adds_is_outlier_column() -> None:
    df = make_frame([{**NORMAL_ROW}])
    out = flag_all_outliers(df)
    assert "is_outlier" in out.columns
    assert out["is_outlier"].dtype == "bool"


def test_flag_all_outliers_adds_outlier_reasons_column() -> None:
    df = make_frame([{**NORMAL_ROW}])
    out = flag_all_outliers(df)
    assert OUTLIER_REASON_COLUMN in out.columns
    assert out[OUTLIER_REASON_COLUMN].dtype == "object"


def test_flag_all_outliers_row_not_flagged_has_empty_reason_list() -> None:
    """Clean row has outlier_reasons == [] (empty list, not None, not NaN)."""
    df = make_frame([{**NORMAL_ROW}])
    out = flag_all_outliers(df)
    val = out[OUTLIER_REASON_COLUMN].iloc[0]
    assert val == []
    assert isinstance(val, list)


def test_flag_all_outliers_row_flagged_for_two_reasons_has_both() -> None:
    """A row triggering both percentile + IQR on the same column has both reasons."""
    # Need a denser frame so per-city bounds produce a meaningful distribution.
    rows: list[dict] = []
    # 100 normal Gurgaon rows to anchor the distribution.
    for i in range(100):
        rows.append({
            "city": "Gurgaon",
            "property_type": "flat",
            "bedRoom": 3,
            "bathroom": 3,
            "price_inr": 1.0e7 + i * 1.0e5,
            "area_sqft": 1500.0,
            "price_per_sqft": (1.0e7 + i * 1.0e5) / 1500.0,
        })
    # The outlier row: extreme price + tiny area → triggers both percentile
    # and IQR on price_inr AND price_per_sqft.
    rows.append({
        "city": "Gurgaon",
        "property_type": "flat",
        "bedRoom": 3,
        "bathroom": 3,
        "price_inr": 1.0e10,
        "area_sqft": 100.0,
        "price_per_sqft": 1.0e8,
    })
    df = make_frame(rows)
    out = flag_all_outliers(df)
    flagged = out[out["is_outlier"]]
    assert len(flagged) >= 1
    # Last row is the outlier.
    reasons_for_outlier = flagged[OUTLIER_REASON_COLUMN].iloc[-1]
    # At least one percentile AND one IQR reason present.
    assert any(r.startswith("percentile_") for r in reasons_for_outlier)
    assert any(r.startswith("iqr_") for r in reasons_for_outlier)


def test_flag_all_outliers_reason_strings_are_from_documented_set() -> None:
    """Every reason in the column is from the documented reason set."""
    df = make_frame([
        # Domain-rule outlier + numeric outlier in the same row.
        {"city": "Gurgaon", "property_type": "flat", "bedRoom": 25, "bathroom": 3,
         "price_inr": 1.0e9, "area_sqft": 100.0, "price_per_sqft": 1.0e7},
        {**NORMAL_ROW},
    ])
    out = flag_all_outliers(df)
    documented = {
        f"percentile_{c}" for c in OUTLIER_NUMERIC_COLUMNS
    } | {
        f"iqr_{c}" for c in OUTLIER_NUMERIC_COLUMNS
    } | {f"domain_{c}" for c in OUTLIER_DOMAIN_RULES}
    all_reasons: set[str] = set()
    for r in out[OUTLIER_REASON_COLUMN]:
        all_reasons.update(r)
    assert all_reasons.issubset(documented), (
        f"unexpected reasons: {all_reasons - documented}"
    )


def test_flag_all_outliers_is_idempotent() -> None:
    """flag_all_outliers(flag_all_outliers(df)) matches flag_all_outliers(df)."""
    df = make_frame([
        {**NORMAL_ROW},
        {"city": "Gurgaon", "property_type": "flat", "bedRoom": 25, "bathroom": 3,
         "price_inr": 1.0e9, "area_sqft": 100.0, "price_per_sqft": 1.0e7},
    ])
    once = flag_all_outliers(df)
    twice = flag_all_outliers(once)
    # is_outlier + outlier_reasons must match; other columns untouched.
    pd.testing.assert_series_equal(once["is_outlier"], twice["is_outlier"], check_names=False)
    for r1, r2 in zip(once[OUTLIER_REASON_COLUMN], twice[OUTLIER_REASON_COLUMN]):
        assert sorted(r1) == sorted(r2)


def test_flag_all_outliers_logs_per_city_summary(caplog: pytest.LogCaptureFixture) -> None:
    """caplog captures one log line per city with row count and flagged count."""
    df = make_frame([
        {**NORMAL_ROW, "city": "Gurgaon"},
        {**NORMAL_ROW, "city": "Gurgaon", "listing_id": "G00000002"},
        {**NORMAL_ROW, "city": "Mumbai", "listing_id": "M00000001"},
    ])
    with caplog.at_level(logging.INFO, logger="ml.cleaning.outliers"):
        flag_all_outliers(df)
    summary = [r for r in caplog.records if "outliers.summary" in r.message]
    assert summary, "expected an outliers.summary log line"
    msg = summary[0].message
    # Per-city tokens appear.
    assert "Gurgaon" in msg
    assert "Mumbai" in msg


def test_outlier_reasons_column_is_json_serializable() -> None:
    """json.dumps(df["outlier_reasons"].iloc[0].tolist()) round-trips."""
    df = make_frame([
        {"city": "Gurgaon", "property_type": "flat", "bedRoom": 3, "bathroom": 3,
         "price_inr": 1.0e9, "area_sqft": 100.0, "price_per_sqft": 1.0e7},
    ])
    out = flag_all_outliers(df)
    val = out[OUTLIER_REASON_COLUMN].iloc[0]
    serialized = json.dumps(val.tolist() if hasattr(val, "tolist") else list(val))
    deserialized = json.loads(serialized)
    assert isinstance(deserialized, list)


def test_flag_all_outliers_preserves_other_columns() -> None:
    """Columns other than is_outlier / outlier_reasons are unchanged."""
    df = make_frame([{**NORMAL_ROW}, {**NORMAL_ROW, "listing_id": "G00000002"}])
    out = flag_all_outliers(df)
    for col in CANONICAL_SUBSET:
        if col in df.columns:
            pd.testing.assert_series_equal(
                df[col].reset_index(drop=True),
                out[col].reset_index(drop=True),
                check_names=False,
            )


def test_flag_all_outliers_overwrites_existing_is_outlier_column() -> None:
    """If is_outlier already exists (from Step 05), flag_all_outliers replaces it."""
    df = make_frame([{**NORMAL_ROW}])
    df["is_outlier"] = True  # simulate Step 05's column override
    out = flag_all_outliers(df)
    # After flagging, the value reflects the NEW calculation, not the seed.
    assert bool(out["is_outlier"].iloc[0]) is False  # normal row shouldn't be flagged


# ---------------------------------------------------------------------------
# F. Real-data smoke
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_real_outlier_flagging_against_assembled_frame() -> None:
    """Real-data smoke: outlier flagging produces a non-empty flagged set."""
    from pathlib import Path

    from ml.cleaning.assemble import assemble_cleaned_frame
    raw = Path("data/raw")
    df = assemble_cleaned_frame(raw, raw / "facets")
    assert "is_outlier" in df.columns
    assert int(df["is_outlier"].sum()) > 0, "real data should produce at least one outlier"
    # All reason codes in the flagged set must be subset of documented set.
    documented = (
        {f"percentile_{c}" for c in OUTLIER_NUMERIC_COLUMNS}
        | {f"iqr_{c}" for c in OUTLIER_NUMERIC_COLUMNS}
        | {f"domain_{c}" for c in OUTLIER_DOMAIN_RULES}
    )
    found = set()
    for r in df[OUTLIER_REASON_COLUMN]:
        found.update(r)
    assert found.issubset(documented)
