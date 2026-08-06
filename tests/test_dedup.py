"""Tests for ``ml.cleaning.dedup`` — Step 06 deduplication layer.

Mirrors the structure of ``tests/test_canonical_mapping.py`` (sections A–H,
literal DataFrames, exact test names per the spec).
"""

from __future__ import annotations

import inspect
import logging

import pandas as pd
import pytest

from ml.cleaning.dedup import (
    CONFLICT_TIEBREAKER_ORDER,
    DEDUP_KEY_COLUMN,
    compute_nonnull_field_count,
    deduplicate_listings,
)
from tests.fixtures.dedup_outlier_fixtures import NORMAL_ROW, make_frame, make_multi_city_frame

# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------

def test_dedup_key_column_constant_is_listing_id() -> None:
    """DEDUP_KEY_COLUMN == "listing_id" per spec."""
    assert DEDUP_KEY_COLUMN == "listing_id"


def test_dedup_tiebreaker_order_is_three_levels() -> None:
    """CONFLICT_TIEBREAKER_ORDER documents the 3-level policy."""
    assert CONFLICT_TIEBREAKER_ORDER == ("nonnull_fields_count", "register_date", "row_order")


def test_dedup_does_not_import_app_or_api() -> None:
    """Module-level imports do not include anything from app.* or api.*."""
    import ml.cleaning.dedup as dedup_mod
    src = inspect.getsource(dedup_mod)
    assert "from app" not in src
    assert "from api" not in src
    assert "import app" not in src
    assert "import api" not in src


def test_dedup_does_not_write_to_disk() -> None:
    """AST scan: no to_parquet/to_csv/to_json/open( literals in module source."""
    import ast

    import ml.cleaning.dedup as dedup_mod
    tree = ast.parse(inspect.getsource(dedup_mod))
    forbidden = {"to_parquet", "to_csv", "to_json"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                pytest.fail(f"dedup.py contains forbidden write call: {node.func.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                pytest.fail("dedup.py contains forbidden open() call")


# ---------------------------------------------------------------------------
# B. Null / empty / whitespace listing_id handling
# ---------------------------------------------------------------------------

def test_dedup_drops_rows_with_null_listing_id(caplog: pytest.LogCaptureFixture) -> None:
    """5 rows, 2 with null listing_id → output has 3 rows, drops logged."""
    df = make_frame([
        {"listing_id": "A", "city": "Gurgaon"},
        {"listing_id": None, "city": "Gurgaon"},
        {"listing_id": "C", "city": "Gurgaon"},
        {"listing_id": pd.NA, "city": "Mumbai"},
        {"listing_id": "E", "city": "Mumbai"},
    ])
    with caplog.at_level(logging.INFO, logger="ml.cleaning.dedup"):
        out = deduplicate_listings(df)
    assert len(out) == 3
    assert set(out["listing_id"]) == {"A", "C", "E"}
    # Drop log line present.
    assert any("no_listing_id" in r.message for r in caplog.records)


def test_dedup_drops_rows_with_empty_string_listing_id(caplog: pytest.LogCaptureFixture) -> None:
    """Whitespace-only listing_id treated as null and dropped."""
    df = make_frame([
        {"listing_id": "A", "city": "Gurgaon"},
        {"listing_id": "   ", "city": "Gurgaon"},
        {"listing_id": "B", "city": "Gurgaon"},
    ])
    with caplog.at_level(logging.INFO, logger="ml.cleaning.dedup"):
        out = deduplicate_listings(df)
    assert len(out) == 2
    assert any("no_listing_id" in r.message for r in caplog.records)


def test_dedup_strips_whitespace_on_listing_id() -> None:
    """"  ABC123  " dedupes against "ABC123"."""
    df = make_frame([
        {"listing_id": "  ABC123  ", "city": "Gurgaon", "price_inr": 1.0},
        {"listing_id": "ABC123", "city": "Gurgaon", "price_inr": 2.0},
        {"listing_id": "ABC123  ", "city": "Gurgaon", "price_inr": 3.0},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 1
    assert out["listing_id"].iloc[0] == "ABC123"


def test_dedup_casts_listing_id_to_string() -> None:
    """Integer PROP_ID coerced to str before dedup."""
    df = make_frame([
        {"listing_id": 12345, "city": "Gurgaon", "price_inr": 1.0},
        {"listing_id": 12345, "city": "Gurgaon", "price_inr": 2.0},
        {"listing_id": 67890, "city": "Mumbai", "price_inr": 1.0},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 2
    # All listing_ids in the output should be unique strings.
    assert out[DEDUP_KEY_COLUMN].is_unique


# ---------------------------------------------------------------------------
# C. Tiebreaker policy
# ---------------------------------------------------------------------------

def test_dedup_keeps_one_row_per_duplicate_listing_id() -> None:
    """3 rows with same listing_id → output has exactly 1."""
    df = make_frame([
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 2.0},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 3.0},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 1


def test_dedup_keeps_row_with_most_nonnull_fields() -> None:
    """Two rows, same listing_id; A has 5 populated, B has 3 → A wins."""
    df = make_frame([
        # Row A: more populated.
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0, "area_sqft": 100.0,
         "bedRoom": 2, "bathroom": 1, "property_type": "flat", "register_date": "Jan 2024"},
        # Row B: sparser.
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 2.0, "area_sqft": 100.0},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 1
    # A's price_inr=1.0 should win (it's the more complete row).
    assert out["price_inr"].iloc[0] == 1.0
    # And the additional populated fields should match row A.
    assert out["bedRoom"].iloc[0] == 2
    assert out["property_type"].iloc[0] == "flat"


def test_dedup_breaks_ties_by_most_recent_register_date() -> None:
    """Same nonnull count; A=2024-01-01, B=2025-06-01 → B wins."""
    df = make_frame([
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0, "area_sqft": 100.0,
         "register_date": "1st Jan, 2024"},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0, "area_sqft": 100.0,
         "register_date": "1st Jun, 2025"},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 1
    assert out["register_date"].iloc[0] == "1st Jun, 2025"


def test_dedup_breaks_final_ties_by_input_order() -> None:
    """Same nonnull + same register_date (or both NaN) → first input row wins."""
    df = make_frame([
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0, "area_sqft": 100.0,
         "register_date": "Jan 2024"},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 2.0, "area_sqft": 100.0,
         "register_date": "Jan 2024"},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 3.0, "area_sqft": 100.0,
         "register_date": pd.NA},
        {"listing_id": "X", "city": "Gurgaon", "price_inr": 4.0, "area_sqft": 100.0,
         "register_date": pd.NA},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 1
    # First row wins (price_inr=1.0).
    assert out["price_inr"].iloc[0] == 1.0


def test_dedup_does_not_modify_non_listing_id_columns() -> None:
    """Non-duplicate row with extra fields preserved verbatim."""
    df = make_frame([
        {**NORMAL_ROW, "listing_id": "X1"},
        {**NORMAL_ROW, "listing_id": "X2", "price_inr": 99_999_999.0},
    ])
    out = deduplicate_listings(df)
    assert len(out) == 2
    # Both rows preserved with their full field set.
    assert set(out["listing_id"]) == {"X1", "X2"}
    assert out.set_index("listing_id").loc["X2", "price_inr"] == 99_999_999.0


# ---------------------------------------------------------------------------
# D. compute_nonnull_field_count
# ---------------------------------------------------------------------------

def test_compute_nonnull_field_count_uses_canonical_columns() -> None:
    """Counts non-null across CANONICAL_COLUMNS, not all DataFrame columns."""
    df = pd.DataFrame({
        "listing_id": ["A", "B"],
        "city": ["Gurgaon", None],
        # extra column not in CANONICAL_COLUMNS → must NOT be counted.
        "RANDOM_EXTRA": [1, 2],
    })
    counts = compute_nonnull_field_count(df)
    # Row A: listing_id + city + (Gurgaon not in CANONICAL_COLUMNS) → wait, city IS
    # in CANONICAL_COLUMNS. Both listing_id and city populated → 2.
    # Row B: listing_id populated, city NaN → 1.
    # RANDOM_EXTRA ignored (not in CANONICAL_COLUMNS).
    assert counts.tolist() == [2, 1]


def test_compute_nonnull_field_count_returns_int_series() -> None:
    df = make_frame([{"listing_id": "A", "city": "Gurgaon", "price_inr": 1.0}])
    out = compute_nonnull_field_count(df)
    assert out.dtype.kind in ("i", "u")
    assert len(out) == len(df)


# ---------------------------------------------------------------------------
# E. Logging + summary
# ---------------------------------------------------------------------------

def test_dedup_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    """Summary log line includes rows_in / dropped / rows_out counts."""
    df = make_frame([
        {"listing_id": "A", "city": "Gurgaon"},
        {"listing_id": None, "city": "Gurgaon"},
        {"listing_id": "B", "city": "Gurgaon"},
        {"listing_id": "B", "city": "Gurgaon"},
    ])
    with caplog.at_level(logging.INFO, logger="ml.cleaning.dedup"):
        deduplicate_listings(df)
    summary = [r for r in caplog.records if "dedup.summary" in r.message]
    assert summary, "expected a dedup.summary log line"
    msg = summary[0].message
    assert "rows_in=4" in msg
    assert "rows_out=2" in msg


# ---------------------------------------------------------------------------
# F. Idempotency + uniqueness
# ---------------------------------------------------------------------------

def test_dedup_is_idempotent() -> None:
    """deduplicate_listings(deduplicate_listings(df)) == deduplicate_listings(df)."""
    df = make_multi_city_frame()
    once = deduplicate_listings(df)
    twice = deduplicate_listings(once)
    pd.testing.assert_frame_equal(once, twice)


def test_dedup_preserves_listing_id_uniqueness_on_output() -> None:
    """Every listing_id in the output is unique."""
    df = make_multi_city_frame()
    out = deduplicate_listings(df)
    assert out["listing_id"].is_unique


# ---------------------------------------------------------------------------
# G. Real-data smoke (mark as realdata, opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_real_dedup_against_gurgaon() -> None:
    """Real-data smoke: deduplicate_listings on the Gurgaon canonical frame."""
    from pathlib import Path

    from ml.cleaning.canonical_mapping import map_city
    from ml.cleaning.facet_decoders import load_facet_frames
    raw = Path("data/raw")
    facets = load_facet_frames(raw / "facets")
    df = map_city("Gurgaon", raw / "gurgaon_10k.csv", facets)
    out = deduplicate_listings(df)
    assert out["listing_id"].is_unique
    # Output rows ≤ input rows (dedup never grows the frame).
    assert len(out) <= len(df)
