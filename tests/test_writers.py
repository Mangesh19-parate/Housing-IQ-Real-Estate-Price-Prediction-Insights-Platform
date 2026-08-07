"""Tests for ``ml.cleaning.writers`` — Step 07 Parquet writer layer."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")

from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS  # noqa: E402
from ml.cleaning.writers import (  # noqa: E402
    CLEAN_LISTINGS_DATASET_VERSION,
    CLEAN_LISTINGS_PARQUET_PATH,
    build_clean_listings_columns_order,
    read_clean_listings_parquet,
    verify_clean_listings_parquet,
    write_clean_listings_parquet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer_frame() -> pd.DataFrame:
    """Minimal canonical-shaped frame covering every column the writer expects.

    Uses sensible default values (not ``pd.NA``) for every canonical column
    so the Parquet round-trip is identity-preserving. pyarrow reads back
    ``None`` for ``pd.NA``-filled object columns, which is not equivalent
    under ``pd.testing.assert_frame_equal``.
    """
    n = 5
    data: dict[str, list[object]] = {}
    # Per-column defaults — keyed by CANONICAL_COLUMNS name. Anything not
    # in this dict gets a typed sensible default via the dispatch below.
    defaults: dict[str, list[object]] = {
        "listing_id": [f"G{i:05d}" for i in range(n)],
        "city": ["Gurgaon"] * n,
        "sector": ["Sector 1"] * n,
        "locality": ["DLF Phase 1"] * n,
        "transact_type": ["sale"] * n,
        "ownership_type": ["freehold"] * n,
        "property_type": ["flat"] * n,
        "bedrooms": [2] * n,
        "bathrooms": [2] * n,
        "balconies": [1] * n,
        "bedRoom": ["2"] * n,
        "bathroom": ["2"] * n,
        "balcony": ["1"] * n,
        "servant_room": [False] * n,
        "store_room": [False] * n,
        "furnish": ["semi"] * n,
        "furnishing_type": ["Semi-Furnished"] * n,
        "facing": ["east"] * n,
        "age_bucket": ["1-3"] * n,
        "agePossession": ["1-3 years"] * n,
        "floor_num": ["2 of 5"] * n,
        "total_floor": [5] * n,
        "floor_category": ["Low"] * n,
        "luxury_category": ["Standard"] * n,
        "area_sqft": [float(1000 + i * 50) for i in range(n)],
        "built_up_area": [float(950 + i * 50) for i in range(n)],
        "price_inr": [float(1_000_000 + i * 100_000) for i in range(n)],
        "price_per_sqft": [float(1000 + i * 50) for i in range(n)],
        "floor_ratio": [0.4] * n,
        "features_list": [["parking"]] * n,
        "amenities_list": [["gym"]] * n,
        "n_amenities": [1] * n,
        "n_features": [1] * n,
        "building_name": ["Tower A"] * n,
        "building_id": ["B1"] * n,
        "latitude": [28.4] * n,
        "longitude": [77.1] * n,
        "description_clean": ["Nice flat"] * n,
        "register_date": ["2024-01-01"] * n,
        "is_outlier": [False] * n,
    }
    for col in CANONICAL_COLUMNS:
        data[col] = defaults[col]
    # Derived column (not in CANONICAL_COLUMNS).
    data["outlier_reasons"] = [[] for _ in range(n)]
    # Ensure was_missing_* flags are explicitly created for the
    # round-trip test.
    data["was_missing_price_inr"] = [i == 0 for i in range(n)]
    data["was_missing_area_sqft"] = [i == 1 for i in range(n)]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------


def test_clean_listings_parquet_path_constant_default() -> None:
    """Default output path is data/processed/clean_listings.parquet."""
    assert CLEAN_LISTINGS_PARQUET_PATH == Path("data/processed/clean_listings.parquet")


def test_clean_listings_parquet_columns_order_is_deterministic() -> None:
    """``build_clean_listings_columns_order`` puts canonical → ``is_outlier``
    (already in canonical) → sorted ``was_missing_*`` → ``outlier_reasons``.
    """
    df = _make_writer_frame()
    ordered = build_clean_listings_columns_order(df)
    assert isinstance(ordered, tuple)
    # All canonical columns must be at the start, in canonical order.
    for col, expected in zip(ordered, CANONICAL_COLUMNS):
        assert col == expected
    # Sorted was_missing_* after the canonical block.
    was_missing_in_ordered = [c for c in ordered if c.startswith("was_missing_")]
    was_missing_in_df = sorted(c for c in df.columns if c.startswith("was_missing_"))
    assert was_missing_in_ordered == was_missing_in_df
    # outlier_reasons is the last column.
    assert ordered[-1] == "outlier_reasons"


# ---------------------------------------------------------------------------
# B. write_clean_listings_parquet
# ---------------------------------------------------------------------------


def test_write_clean_listings_parquet_creates_file(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    assert target.exists()
    assert target.stat().st_size > 0


def test_write_clean_listings_parquet_creates_sidecar_meta_json(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    # Sidecar lives at <file>.parquet.meta.json (two-dot extension).
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in (
        "dataset_version",
        "row_count",
        "column_count",
        "columns",
        "computed_at_utc",
        "source_raw_files",
    ):
        assert key in payload, f"missing key: {key}"
    assert payload["dataset_version"] == CLEAN_LISTINGS_DATASET_VERSION
    assert payload["row_count"] == 5
    assert payload["column_count"] == len(df.columns)


def test_write_clean_listings_parquet_writes_in_canonical_column_order(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    read_back = pd.read_parquet(target)
    expected = list(build_clean_listings_columns_order(df))
    assert read_back.columns.tolist() == expected


def test_write_clean_listings_parquet_round_trip_outlier_reasons_list(tmp_path: Path) -> None:
    """A frame with a non-empty list in ``outlier_reasons`` round-trips intact."""
    data: dict[str, list[object]] = {}
    for col in CANONICAL_COLUMNS:
        data[col] = [pd.NA] * 3
    data["listing_id"] = ["G1", "G2", "G3"]
    data["city"] = ["Gurgaon", "Mumbai", "Kolkata"]
    data["property_type"] = ["flat", "villa", "flat"]
    data["price_inr"] = [1_000_000.0, 5_000_000.0, 2_500_000.0]
    data["area_sqft"] = [1000.0, 3000.0, 1500.0]
    data["is_outlier"] = [True, False, True]
    data["outlier_reasons"] = [
        ["percentile_price_inr", "iqr_area_sqft"],
        [],
        ["domain_bedRoom"],
    ]
    df = pd.DataFrame(data)
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    read_back = read_clean_listings_parquet(target)
    # Index alignment: rows may be reordered by pyarrow; match by listing_id.
    lookup = dict(zip(read_back["listing_id"], read_back["outlier_reasons"]))
    assert sorted(lookup["G1"]) == sorted(["percentile_price_inr", "iqr_area_sqft"])
    assert list(lookup["G2"]) == []
    assert sorted(lookup["G3"]) == sorted(["domain_bedRoom"])


def test_write_clean_listings_parquet_round_trip_preserves_was_missing_flags(
    tmp_path: Path,
) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    read_back = read_clean_listings_parquet(target)
    # True/False distribution survives.
    flag_cols = [c for c in df.columns if c.startswith("was_missing_")]
    assert flag_cols, "test fixture should include was_missing_* columns"
    for col in flag_cols:
        original = list(df[col])
        round_tripped = list(read_back[col])
        assert original == round_tripped, f"flag {col} mismatch"


def test_write_clean_listings_parquet_returns_path(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "x.parquet"
    result = write_clean_listings_parquet(df, target)
    assert result == target


# ---------------------------------------------------------------------------
# C. read + verify
# ---------------------------------------------------------------------------


def test_read_clean_listings_parquet_round_trip(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    read_back = read_clean_listings_parquet(target)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        read_back.reset_index(drop=True)[df.columns.tolist()],
        check_dtype=False,
    )


def test_verify_clean_listings_parquet_passes_after_write(tmp_path: Path) -> None:
    df = _make_writer_frame()
    target = tmp_path / "clean_listings.parquet"
    write_clean_listings_parquet(df, target)
    result = verify_clean_listings_parquet(target)
    assert result["exists"] is True
    assert result["listing_id_unique"] is True
    assert result["has_is_outlier"] is True
    assert result["has_was_missing_columns"] is True
    assert result["columns_match_canonical_order"] is True
    assert result["row_count"] == 5


def test_verify_clean_listings_parquet_fails_for_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "does_not_exist.parquet"
    result = verify_clean_listings_parquet(target)
    assert result["exists"] is False


# ---------------------------------------------------------------------------
# D. AST scan
# ---------------------------------------------------------------------------


def test_writers_module_does_not_touch_data_raw() -> None:
    """writers.py must not contain a Path("data/raw" literal or import ingest writers."""
    import ml.cleaning.writers as mod
    src = inspect.getsource(mod)
    assert 'Path("data/raw"' not in src, "writers.py references data/raw path"
    # AST scan for forbidden tokens.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith("ml.cleaning.ingest"), (
                f"unexpected import from ml.cleaning.ingest: {node.module}"
            )
