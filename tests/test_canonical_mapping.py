"""Unit tests for ``ml.cleaning.canonical_mapping`` (Step 05).

Test names mirror the spec's "Definition of done" #1 exactly — they're
the contract. No real-data dependency: every test uses literal
``pd.DataFrame``s from ``tests/fixtures/canonical_mapping_fixtures.py``
or inlined literals, and the synthetic-dir helper writes to ``tmp_path``.

The whole suite runs under ``pytest -m "not realdata"``.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
import pytest

from ml.cleaning import canonical_mapping as cm_module
from ml.cleaning.canonical_mapping import (
    CANONICAL_COLUMNS,
    CITY_COLUMN_ALIASES,
    CITY_FRAME_LOADERS,
    UNSAFE_COLUMNS,
    clean_description,
    map_city,
    map_gurgaon,
    map_hyderabad,
    map_kolkata,
    map_mumbai,
    normalize_columns,
)
from ml.cleaning.facet_decoders import decode_furnish
from ml.cleaning.ingest import _snapshot_raw_files
from tests.fixtures.canonical_mapping_fixtures import (
    GURGAON_RAW_DF,
    HYDERABAD_RAW_DF,
    KOLKATA_RAW_DF,
    MUMBAI_RAW_DF,
    build_synthetic_raw_dir,
    facet_frames_for_tests,
)

# ===========================================================================
# Synthetic per-city CSV writer (for tests that need a real on-disk file).
# ===========================================================================


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ===========================================================================
# A. Constants — 5 tests
# ===========================================================================


def test_canonical_columns_constant_matches_backend_schema() -> None:
    """CANONICAL_COLUMNS is a tuple[str, ...] containing every documented
    column from 05-BACKEND-SCHEMA.md §2 + §U-SCHEMA-5.
    """
    assert isinstance(CANONICAL_COLUMNS, tuple)
    assert all(isinstance(c, str) for c in CANONICAL_COLUMNS)

    canonical_set = set(CANONICAL_COLUMNS)

    # 16 input-contract fields with their exact canonical names.
    required_16 = {
        "city", "sector", "property_type", "transact_type",
        "bedRoom", "bathroom", "balcony",
        "agePossession", "built_up_area",
        "servant_room", "store_room",
        "furnishing_type", "luxury_category", "floor_category",
        "facing", "amenities_list",
    }
    missing_16 = required_16 - canonical_set
    assert not missing_16, f"CANONICAL_COLUMNS missing input-contract fields: {missing_16}"

    # Extended-schema fields from §2.
    required_extended = {
        "listing_id", "locality", "ownership_type",
        "bedrooms", "bathrooms", "balconies",
        "furnish", "age_bucket", "floor_num", "total_floor",
        "floor_ratio", "area_sqft", "price_inr", "price_per_sqft",
        "features_list", "n_amenities", "n_features",
        "building_name", "building_id", "latitude", "longitude",
        "description_clean", "register_date", "is_outlier",
    }
    missing_extended = required_extended - canonical_set
    assert not missing_extended, (
        f"CANONICAL_COLUMNS missing extended-schema fields: {missing_extended}"
    )


def test_city_column_aliases_has_four_cities() -> None:
    """CITY_COLUMN_ALIASES has exactly 4 city keys."""
    assert set(CITY_COLUMN_ALIASES.keys()) == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}


def test_city_column_aliases_canonical_to_raw_is_nonempty_per_city() -> None:
    """Each city's alias dict contains at least 12 canonical->raw mappings
    covering the documented core fields.
    """
    required_canonical = {
        "listing_id", "city", "bedrooms", "price_inr", "area_sqft",
        "latitude", "longitude",
    }
    for city, alias in CITY_COLUMN_ALIASES.items():
        assert len(alias) >= 12, f"{city} alias dict has only {len(alias)} entries"
        missing = required_canonical - set(alias.keys())
        assert not missing, f"{city} alias missing required canonicals: {missing}"


def test_city_frame_loaders_has_four_entries() -> None:
    """CITY_FRAME_LOADERS has 4 city keys and each value is callable."""
    assert set(CITY_FRAME_LOADERS.keys()) == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}
    for loader in CITY_FRAME_LOADERS.values():
        assert callable(loader)


def test_bedroom_canonical_name_uses_camelcase() -> None:
    """The canonical column name for bedrooms is exactly 'bedRoom' (the
    reference-project contract spelling). The module also exposes
    'bedrooms' as the extended-schema name.
    """
    assert "bedRoom" in CANONICAL_COLUMNS
    assert "bedrooms" in CANONICAL_COLUMNS
    # NOT the snake_case or title-case variants.
    assert "bed_room" not in CANONICAL_COLUMNS
    assert "BedRoom" not in CANONICAL_COLUMNS


# ===========================================================================
# B. map_city dispatcher — 1 test (and 1 negative test in section H)
# ===========================================================================


def test_map_unknown_city_raises_value_error() -> None:
    """Unknown city name raises ValueError."""
    fake_path = Path("/nonexistent")
    fake_frames: dict[str, pd.DataFrame] = {}
    with pytest.raises(ValueError, match="Unknown city"):
        map_city("Atlantis", fake_path, fake_frames)


# ===========================================================================
# C. normalize_columns — 3 tests
# ===========================================================================


def test_normalize_columns_drops_unsafe_columns() -> None:
    """All UNSAFE_COLUMNS entries that appear in the input are dropped."""
    df = pd.DataFrame(
        {
            "PROP_ID": ["X1"],
            "CITY": ["Gurgaon"],
            "PHOTO_URL": ["u1"],
            "DEALER_NAME": ["dn"],
            "DEALER_PHONE": ["9999"],
            "CONTACT_EMAIL": ["a@b.c"],
            "PROP_DETAILS_URL": ["u2"],
            "SPID": ["S1"],
            "PROP_URL": ["u3"],
            "FURNISH": [4],
        }
    )
    out = normalize_columns(df, "Gurgaon")
    for unsafe in ("PHOTO_URL", "DEALER_NAME", "DEALER_PHONE", "CONTACT_EMAIL",
                   "PROP_DETAILS_URL", "SPID", "PROP_URL"):
        assert unsafe not in out.columns, f"unsafe col {unsafe} survived normalize"


def test_normalize_columns_renames_via_alias_dict() -> None:
    """normalize_columns renames at least 5 documented raw columns per city."""
    for city, df in [
        ("Gurgaon", GURGAON_RAW_DF.iloc[:1]),
        ("Hyderabad", HYDERABAD_RAW_DF.iloc[:1]),
        ("Kolkata", KOLKATA_RAW_DF.iloc[:1]),
        ("Mumbai", MUMBAI_RAW_DF.iloc[:1]),
    ]:
        out = normalize_columns(df, city)
        alias = CITY_COLUMN_ALIASES[city]
        # Verify at least 5 renames actually happened in the output.
        expected_canonicals = set(alias.values())
        actually_renamed = expected_canonicals & set(out.columns)
        assert len(actually_renamed) >= 5, (
            f"{city}: only {len(actually_renamed)} canonical columns present"
        )


def test_normalize_columns_passes_through_unknown_columns_as_is() -> None:
    """A raw column not in the alias dict is preserved under its raw name."""
    df = pd.DataFrame(
        {
            "PROP_ID": ["X1"],
            "CITY": ["Gurgaon"],
            "FURNISH": [4],
            "SOME_GURGAON_SPECIFIC_FIELD": ["survives"],
        }
    )
    out = normalize_columns(df, "Gurgaon")
    assert "SOME_GURGAON_SPECIFIC_FIELD" in out.columns


# ===========================================================================
# D. clean_description — 7 tests
# ===========================================================================


def test_clean_description_lowercases() -> None:
    s = pd.Series(["Hello World"])
    assert clean_description(s).tolist() == ["hello world"]


def test_clean_description_strips_html_tags() -> None:
    s = pd.Series(["<p>3BHK with <b>clubhouse</b></p>"])
    assert clean_description(s).tolist() == ["3bhk with clubhouse"]


def test_clean_description_drops_urls() -> None:
    s = pd.Series(["Visit https://example.com for details"])
    assert clean_description(s).tolist() == ["visit for details"]


def test_clean_description_drops_emails() -> None:
    s = pd.Series(["Contact agent@gmail.com today"])
    assert clean_description(s).tolist() == ["contact today"]


def test_clean_description_collapses_whitespace() -> None:
    s = pd.Series(["  multiple    spaces  here  "])
    assert clean_description(s).tolist() == ["multiple spaces here"]


def test_clean_description_passes_nan_through() -> None:
    s = pd.Series([pd.NA, float("nan"), None])
    out = clean_description(s)
    assert len(out) == 3
    assert all(pd.isna(v) for v in out.tolist())


def test_clean_description_is_vectorized() -> None:
    """Helper accepts a Series and returns a Series of equal length."""
    s = pd.Series(["hello", "world", pd.NA, "Visit https://x.com y"])
    out = clean_description(s)
    assert isinstance(out, pd.Series)
    assert len(out) == len(s)
    assert out.iloc[0] == "hello"
    assert out.iloc[1] == "world"
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == "visit y"


# ===========================================================================
# E. Per-city mappers — write synthetic CSVs, run mappers, assert
# ===========================================================================


@pytest.fixture()
def ff():
    return facet_frames_for_tests()


# --- Gurgaon (10 tests) ---


def test_map_gurgaon_emits_all_canonical_columns(ff, tmp_path) -> None:
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    assert out.columns.tolist() == list(CANONICAL_COLUMNS)
    assert len(out) == len(GURGAON_RAW_DF)


def test_map_gurgaon_decodes_furnish_via_step04(ff, tmp_path) -> None:
    """Gurgaon row with FURNISH=4 produces furnishing_type = decode_furnish(4, ff)."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    expected = decode_furnish(4, ff["FURNISH"])
    # Row 0 has FURNISH=4 -> "Semifurnished"
    assert out.iloc[0]["furnishing_type"] == expected == "Semifurnished"


def test_map_gurgaon_decodes_amenities_as_list(ff, tmp_path) -> None:
    """Gurgaon row with AMENITIES='20,21,32' produces a list[str] of 3 labels."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    amenities = out.iloc[0]["amenities_list"]
    assert isinstance(amenities, list)
    assert len(amenities) == 3
    assert all(isinstance(a, str) for a in amenities)


def test_map_gurgaon_parses_price_via_step03(ff, tmp_path) -> None:
    """Gurgaon row with PRICE='3.5 Cr' produces price_inr=35_000_000.0."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    assert float(out.iloc[0]["price_inr"]) == 35_000_000.0


def test_map_gurgaon_parses_area_via_step03(ff, tmp_path) -> None:
    """Gurgaon row with AREA='1450 sq.ft.' produces area_sqft=1450.0."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    assert float(out.iloc[0]["area_sqft"]) == 1450.0


def test_map_gurgaon_parses_map_details(ff, tmp_path) -> None:
    """Gurgaon row MAP_DETAILS dict-string produces (latitude, longitude) floats."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    assert float(out.iloc[0]["latitude"]) == 28.4065
    assert float(out.iloc[0]["longitude"]) == 76.9628


def test_map_gurgaon_cleans_description(ff, tmp_path) -> None:
    """Gurgaon DESCRIPTION with HTML + uppercase -> description_clean lowercased + tags stripped."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    # Row 0: "<p>Spacious 3BHK with CLUBHOUSE</p>" -> "spacious 3bhk with clubhouse"
    assert out.iloc[0]["description_clean"] == "spacious 3bhk with clubhouse"


def test_map_gurgaon_idempotent(ff, tmp_path) -> None:
    """Calling map_gurgaon twice on identical inputs yields equal DataFrames."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out1 = map_gurgaon(csv, ff)
    out2 = map_gurgaon(csv, ff)
    pd.testing.assert_frame_equal(out1, out2)


def test_map_gurgaon_drops_all_unsafe_columns(ff, tmp_path) -> None:
    """None of the UNSAFE_COLUMNS names survive in the output."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    for unsafe in UNSAFE_COLUMNS:
        assert unsafe not in out.columns, f"unsafe column {unsafe!r} survived map_gurgaon"


def test_map_gurgaon_does_not_write_to_data_raw_or_data_processed(ff, tmp_path) -> None:
    """map_gurgaon reads but never writes to data/raw or data/processed.

    Uses the same _snapshot_raw_files immutability primitive as Step 04.
    """
    data_dir = build_synthetic_raw_dir(tmp_path)
    raw_csv = data_dir / "raw" / "gurgaon_10k.csv"

    before = _snapshot_raw_files(data_dir)
    _ = map_gurgaon(raw_csv, ff)
    after = _snapshot_raw_files(data_dir)
    assert before == after, "map_gurgaon modified data/raw/ files"


def test_luxury_category_left_as_nan(ff, tmp_path) -> None:
    """luxury_category is NaN for every row (Rules §10.2)."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    for v in out["luxury_category"]:
        assert pd.isna(v)


def test_transact_type_preserved_as_string(ff, tmp_path) -> None:
    """TRANSACT_TYPE codes 1.0/2.0 are decoded to 'Sale'/'Rent' strings."""
    csv = _write_csv(GURGAON_RAW_DF, tmp_path, "gurgaon.csv")
    out = map_gurgaon(csv, ff)
    # Rows 0,2 are TRANSACT_TYPE=1.0 (Sale); row 1 is 2.0 (Rent).
    assert out.iloc[0]["transact_type"] == "Sale"
    assert out.iloc[1]["transact_type"] == "Rent"
    assert out.iloc[2]["transact_type"] == "Sale"


# --- Hyderabad (2 tests) ---


def test_map_hyderabad_uses_value_label_for_ownership(ff, tmp_path) -> None:
    """Hyderabad ownership_type comes from VALUE_LABEL (string pass-through)."""
    csv = _write_csv(HYDERABAD_RAW_DF, tmp_path, "hyderabad.csv")
    out = map_hyderabad(csv, ff)
    assert out.iloc[0]["ownership_type"] == "Freehold"
    assert out.iloc[1]["ownership_type"] == "Freehold"
    assert out.iloc[2]["ownership_type"] == "Leasehold"


def test_map_hyderabad_extracts_locality_from_nested_dict(ff, tmp_path) -> None:
    """Hyderabad locality comes from the nested location dict (LOCALITY_NAME)."""
    csv = _write_csv(HYDERABAD_RAW_DF, tmp_path, "hyderabad.csv")
    out = map_hyderabad(csv, ff)
    assert out.iloc[0]["locality"] == "Banjara Hills"
    assert out.iloc[1]["locality"] == "Jubilee Hills"
    assert out.iloc[2]["locality"] == "Gachibowli"


# --- Kolkata (1 test) ---


def test_map_kolkata_register_date_is_nan(ff, tmp_path) -> None:
    """Kolkata has no REGISTER_DATE column -> register_date is NaN for every row."""
    csv = _write_csv(KOLKATA_RAW_DF, tmp_path, "kolkata.csv")
    out = map_kolkata(csv, ff)
    for v in out["register_date"]:
        assert pd.isna(v)


# --- Mumbai (2 tests) ---


def test_map_mumbai_emits_all_canonical_columns(ff, tmp_path) -> None:
    csv = _write_csv(MUMBAI_RAW_DF, tmp_path, "mumbai.csv")
    out = map_mumbai(csv, ff)
    assert out.columns.tolist() == list(CANONICAL_COLUMNS)
    assert len(out) == len(MUMBAI_RAW_DF)


def test_map_mumbai_does_not_drop_local_rows(ff, tmp_path) -> None:
    """All Mumbai-only rows survive the mapper (no city-filtering at this step)."""
    csv = _write_csv(MUMBAI_RAW_DF, tmp_path, "mumbai.csv")
    out = map_mumbai(csv, ff)
    assert len(out) == len(MUMBAI_RAW_DF)
    for v in out["city"]:
        assert v == "Mumbai"


# ===========================================================================
# F. Dispatcher + caplog — 1 test each
# ===========================================================================


def test_map_city_dispatches_correctly(ff, tmp_path) -> None:
    """map_city(name, ...) returns the same DataFrame as map_<city>(...) for each city."""
    cases = [
        ("Gurgaon", map_gurgaon, GURGAON_RAW_DF, "gurgaon.csv"),
        ("Hyderabad", map_hyderabad, HYDERABAD_RAW_DF, "hyderabad.csv"),
        ("Kolkata", map_kolkata, KOLKATA_RAW_DF, "kolkata.csv"),
        ("Mumbai", map_mumbai, MUMBAI_RAW_DF, "mumbai.csv"),
    ]
    for name, mapper, df, filename in cases:
        csv = _write_csv(df, tmp_path, filename)
        via_dispatcher = map_city(name, csv, ff)
        via_direct = mapper(csv, ff)
        pd.testing.assert_frame_equal(via_dispatcher, via_direct)


def test_map_city_emits_log_warning_for_unknown_facet_id(ff, tmp_path, caplog) -> None:
    """A row with FURNISH=999 (unmapped) produces a warning containing 'furnish'."""
    csv = _write_csv(HYDERABAD_RAW_DF, tmp_path, "hyderabad.csv")
    # HYDERABAD row 0 has FURNISH=999, an unmapped id.
    caplog.set_level(logging.WARNING, logger="ml.cleaning.parsing")
    _ = map_hyderabad(csv, ff)
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("furnish" in r.getMessage() for r in warn_records), (
        f"expected 'furnish' in warning records, got: "
        f"{[r.getMessage() for r in warn_records]}"
    )


# ===========================================================================
# G. Isolation (static checks against module source) — 2 tests
# ===========================================================================


def test_map_does_not_import_app_or_api() -> None:
    """ml.cleaning.canonical_mapping has no imports from app.* or api.*."""
    source_path = Path(cm_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported.append(f"{module}.{alias.name}" if module else alias.name)
    flat = " ".join(imported).lower()
    # Any token starting with app. or api. (followed by anything) is a violation.
    for imp in flat.split():
        if imp.startswith("app.") or imp.startswith("api."):
            pytest.fail(f"forbidden import detected: {imp!r}")


def test_canonical_mapping_does_not_touch_filesystem_outside_data_raw() -> None:
    """Module source contains no write-path literals (open, to_parquet, etc.)."""
    source_path = Path(cm_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    forbidden = [
        "open(", "to_parquet", "to_csv", "to_json",
        'Path("data/processed', "Path('data/processed",
        'Path("data/raw', "Path('data/raw",
    ]
    for f in forbidden:
        assert f not in source, f"forbidden literal {f!r} found in canonical_mapping source"


# ===========================================================================
# H. Real-data smoke (the only test that touches real data; gated)
# ===========================================================================


@pytest.mark.realdata
def test_real_gurgaon_csv_loads_and_has_canonical_columns(ff) -> None:
    """Real Gurgaon CSV (~10.7k rows) loads via map_gurgaon and emits canonical cols.

    Opt-in only — deselected by default via ``pytest -m 'not realdata'``.
    """
    real_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "gurgaon_10k.csv"
    if not real_csv.exists():
        pytest.skip("real data not present (CI environment)")
    out = map_gurgaon(real_csv, ff)
    assert out.columns.tolist() == list(CANONICAL_COLUMNS)
    assert len(out) > 10000  # ~10.7k rows in gurgaon_10k.csv
