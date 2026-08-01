"""Step 02 — Raw ingestion + schema inventory tests.

Anchored to the spec at ``.claude/specs/02-raw-data-ingestion-and-schema-inventory.md``.
Tests that read the real ~182k-row CSVs are guarded with ``@pytest.mark.realdata``
so the fast path (CI without the dataset) excludes them via ``-m "not realdata"``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import ml.cleaning.ingest as ingest
from tests.fixtures.raw_snapshot_fixture import build_synthetic_pair

# ===========================================================================
# Realdata tests — only run when the actual data/raw/ CSVs are present.
# ===========================================================================


@pytest.mark.realdata
def test_load_raw_listings_returns_four_cities():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    dfs = ingest.load_raw_listings(data_dir)
    assert set(dfs.keys()) == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}
    for city, df in dfs.items():
        assert not df.empty, f"{city} DF is empty"


@pytest.mark.realdata
def test_load_raw_listings_does_not_modify_files():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    snap = ingest._snapshot_raw_files(data_dir)
    ingest.load_raw_listings(data_dir)
    after = ingest._snapshot_raw_files(data_dir)
    assert snap == after, "data/raw/ was modified during ingestion"


@pytest.mark.realdata
def test_load_facets_returns_fifteen_files():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    facet_dfs = ingest.load_facets(data_dir)
    assert set(facet_dfs.keys()) == set(ingest.FACET_NAMES)


@pytest.mark.realdata
def test_inventory_marks_pii_columns():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    dfs = ingest.load_raw_listings(data_dir)
    inventory = ingest.build_inventory(dfs)
    # Gurgaon has the rich schema (photo URLs, CONTACT_NAME, DEALER_PHOTO_URL, ...).
    # Other cities (Hyderabad/Kolkata/Mumbai) have no PII-named columns at all —
    # the spec acknowledges this with "others may not". The assertion is that
    # the regex fires where the data has the marks, not that every city has one.
    gurgaon = inventory["cities"]["Gurgaon"]["columns"]
    pii_cols = [c for c in gurgaon if c["pii_risk"]]
    assert pii_cols, "Gurgaon should have at least one PII-flagged column"
    # Spot-check a few well-known PII columns.
    pii_names = {c["name"] for c in pii_cols}
    for must_have in ("PHOTO_URL", "CONTACT_NAME", "PROP_DETAILS_URL"):
        assert must_have in pii_names, f"expected {must_have} in PII-flagged columns"


# ===========================================================================
# Fixture-based tests — run in fast path (no real data needed).
# ===========================================================================


def test_inventory_per_city_shape(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    dfs = ingest.load_raw_listings(data_dir)
    inventory = ingest.build_inventory(dfs)
    for city, info in inventory["cities"].items():
        assert "rows" in info
        assert "total_columns" in info
        assert isinstance(info["columns"], list)
        assert len(info["columns"]) == info["total_columns"]
        for col in info["columns"]:
            for key in (
                "name",
                "dtype",
                "null_count",
                "null_pct",
                "n_unique",
                "sample_values",
                "pii_risk",
            ):
                assert key in col, f"inventory column missing key {key} for {city}"


def test_inventory_sample_values_bounded(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    dfs = ingest.load_raw_listings(data_dir)
    inventory = ingest.build_inventory(dfs)
    for city, info in inventory["cities"].items():
        for col in info["columns"]:
            assert len(col["sample_values"]) <= ingest.SAMPLE_VALUES_MAX


def test_facet_join_coverage_emits_one_row_per_coded_column(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    dfs = ingest.load_raw_listings(data_dir)
    facets = ingest.load_facets(data_dir)
    coverage = ingest.compute_facet_join_coverage(dfs, facets)
    # Spec: 15 facets × 4 cities = 60 rows expected.
    assert len(coverage) == 15 * 4, f"expected {15 * 4} coverage rows, got {len(coverage)}"
    # Every (city, facet_name) pair is present exactly once.
    expected_pairs = {
        (city, facet_name) for facet_name in ingest.CODED_COLUMNS_BY_FACET for city in dfs
    }
    actual_pairs = {
        (row["city"], row["facet_name"]) for row in coverage.to_dict(orient="records")
    }
    assert expected_pairs == actual_pairs


def test_schema_map_references_canonical_names(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    dfs = ingest.load_raw_listings(data_dir)
    facets = ingest.load_facets(data_dir)
    inventory = ingest.build_inventory(dfs)
    coverage = ingest.compute_facet_join_coverage(dfs, facets)
    schema_map = ingest.build_schema_map(coverage, inventory)

    # Build the set of canonical names that the spec doc says we should use.
    canonicals = set()
    for doc in ("docs/10-FINALIZED-INPUT-SCHEMA.md", "docs/05-BACKEND-SCHEMA.md"):
        doc_path = Path(__file__).resolve().parents[1] / doc
        if not doc_path.exists():
            continue
        for line in doc_path.read_text(encoding="utf-8").splitlines():
            # Pipe-tabulated markdown tables — grab the backticked identifiers.
            for match in re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", line):
                canonicals.add(match)
    assert canonicals, "could not parse canonical field names from docs"

    for city, mapping in schema_map["cities"].items():
        for raw_col, info in mapping.items():
            canon = info["canonical"]
            if canon in {"DROP", "UNMAPPED"}:
                continue
            assert canon in canonicals, (
                f"{city}.{raw_col} maps to canonical {canon!r} but it does not "
                "appear in 10-FINALIZED-INPUT-SCHEMA.md or 05-BACKEND-SCHEMA.md"
            )


def test_unmapped_columns_listed_for_review(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    dfs = ingest.load_raw_listings(data_dir)
    facets = ingest.load_facets(data_dir)
    inventory = ingest.build_inventory(dfs)
    coverage = ingest.compute_facet_join_coverage(dfs, facets)
    schema_map = ingest.build_schema_map(coverage, inventory)

    pending = {(p["city"], p["raw_column"]) for p in schema_map["pending_review"]}
    for city, mapping in schema_map["cities"].items():
        for raw_col, info in mapping.items():
            if info["canonical"] == "UNMAPPED":
                assert (city, raw_col) in pending, (
                    f"{city}.{raw_col} is UNMAPPED but missing from pending_review"
                )


def test_snapshot_manifest_is_sha256(tmp_path):
    data_dir, _ = build_synthetic_pair(tmp_path)
    snap = ingest.snapshot_raw(data_dir)
    for rel_path, info in snap.items():
        assert isinstance(info["sha256"], str)
        assert len(info["sha256"]) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
    sv1 = ingest._manifest_hash(snap)
    sv2 = ingest._manifest_hash(snap)
    assert sv1 == sv2
    assert len(sv1) == 64


def test_meta_file_has_required_keys(tmp_path):
    data_dir, output_dir = build_synthetic_pair(tmp_path)
    result = ingest.run_ingestion(data_dir, output_dir)
    meta = json.loads((output_dir / ingest.OUTPUT_FILES["meta"]).read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "git_commit",
        "python_version",
        "pandas_version",
        "numpy_version",
        "generated_at",
        "source_version",
        "spec_version",
    ):
        assert key in meta, f"meta missing required key: {key}"
    assert meta["source_version"] == result["source_version"]
    assert meta["spec_version"] == ingest.SPEC_VERSION


def test_ingestion_is_idempotent(tmp_path):
    data_dir, output_dir = build_synthetic_pair(tmp_path)
    r1 = ingest.run_ingestion(data_dir, output_dir)
    # The _meta/ingest_v1.json intentionally carries a fresh run_id + timestamp
    # per invocation — exclude it from the byte-equality check. Every other
    # JSON output must be byte-identical across runs.
    files = sorted(
        f for f in output_dir.rglob("*.json") if "_meta" not in f.parts
    )
    before = {f: f.read_bytes() for f in files}
    r2 = ingest.run_ingestion(data_dir, output_dir)
    assert r1["source_version"] == r2["source_version"]
    for f, content in before.items():
        assert f.read_bytes() == content, f"{f} changed between runs (non-idempotent)"


def test_run_pipeline_imports_ingest():
    import scripts.run_pipeline as rp

    assert callable(rp.main)


def test_fixture_ingestion_end_to_end(tmp_path):
    data_dir, output_dir = build_synthetic_pair(tmp_path)
    result = ingest.run_ingestion(data_dir, output_dir)

    # All 6 outputs must exist and parse.
    for fname in ingest.OUTPUT_FILES.values():
        path = output_dir / fname
        assert path.exists(), f"missing output: {path}"
    inv = json.loads(
        (output_dir / ingest.OUTPUT_FILES["raw_inventory"]).read_text(encoding="utf-8")
    )
    assert inv["cities"]
    schema_map = json.loads(
        (output_dir / ingest.OUTPUT_FILES["schema_map"]).read_text(encoding="utf-8")
    )
    assert "cities" in schema_map and "pending_review" in schema_map
    coverage = pd.read_csv(output_dir / ingest.OUTPUT_FILES["coverage"])
    assert len(coverage) == 15 * 4
    manifest = json.loads(
        (output_dir / ingest.OUTPUT_FILES["snapshot_manifest"]).read_text(encoding="utf-8")
    )
    assert "source_version" in manifest
    assert manifest["source_version"] == result["source_version"]
    meta = json.loads((output_dir / ingest.OUTPUT_FILES["meta"]).read_text(encoding="utf-8"))
    assert meta["source_version"] == manifest["source_version"]
