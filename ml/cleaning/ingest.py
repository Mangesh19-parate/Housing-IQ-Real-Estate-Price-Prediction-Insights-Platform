"""Step 02 — Raw Data Ingestion and Schema Inventory.

The first concrete pipeline stage. Loads the four city CSVs and the fifteen
facet CSVs under ``data/raw/``, then writes a reproducible inventory +
schema mapping + facet-join coverage report under ``data/processed/``.

Per Rules doc §1.1 (binding): ``data/raw/`` is immutable. We never write to it
and assert mtimes+sizes are unchanged after every run. Idempotent: identical
inputs produce byte-identical outputs (the SHA256-of-the-manifest hash is
the ``source_version`` downstream stages stamp on their own artifacts).

Public API (per spec): ``load_raw_listings``, ``load_facets``,
``build_inventory``, ``build_facet_inventory``, ``compute_facet_join_coverage``,
``build_schema_map``, ``snapshot_raw``, ``write_meta``, ``run_ingestion``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

logger = logging.getLogger("ml.cleaning.ingest")

# ---------------------------------------------------------------------------
# Constants — single source of truth for what files exist on disk.
# ---------------------------------------------------------------------------

# Filing-name → canonical city label (matches the CITY facet label `Gurgaon`).
# Per spec: cities are keyed by the CITY facet label, not the filename.
RAW_FILE_TO_CITY: Final[dict[str, str]] = {
    "gurgaon_10k.csv": "Gurgaon",
    "hyderabad.csv": "Hyderabad",
    "kolkata.csv": "Kolkata",
    "mumbai.csv": "Mumbai",
}

# 15 facet lookup files (Step 01 spec'd this exact set; do not reorder casually).
FACET_NAMES: Final[tuple[str, ...]] = (
    "AGE",
    "AMENITIES",
    "BATHROOM_NUM",
    "BEDROOM_NUM",
    "BUILDING_ID",
    "CITY",
    "FACING_DIRECTION",
    "FEATURES",
    "FLOOR_NUM",
    "FURNISH",
    "LOCALITY_ID",
    "OWNERSHIP_TYPE",
    "PROPERTY_TYPE",
    "SUB_AVAILABILITY",
    "TOTAL_FLOOR",
)

# For each facet, the raw column name(s) likely to hold its code. The
# coverage pass emits exactly one row per (city, facet) pair — 60 total
# (15 facets × 4 cities). For CITY and LOCALITY_ID, multiple candidate raw
# columns may exist; the dict preserves the ordering so the caller can
# surface which one was actually used.
CODED_COLUMNS_BY_FACET: Final[dict[str, list[str]]] = {
    "AGE": ["AGE"],
    "AMENITIES": ["AMENITIES"],  # multi-value; coverage = unique-code coverage
    "BATHROOM_NUM": ["BATHROOM_NUM"],
    "BEDROOM_NUM": ["BEDROOM_NUM"],
    "BUILDING_ID": ["BUILDING_ID"],
    "CITY": ["CITY", "CITY_ID"],
    "FACING_DIRECTION": ["FACING"],
    "FEATURES": ["FEATURES"],  # multi-value
    "FLOOR_NUM": ["FLOOR_NUM"],
    "FURNISH": ["FURNISH"],
    "LOCALITY_ID": ["LOCALITY_ID", "location"],
    "OWNERSHIP_TYPE": ["OWNTYPE"],
    "PROPERTY_TYPE": ["PROPERTY_TYPE"],
    "SUB_AVAILABILITY": ["SUB_AVAILABILITY"],
    "TOTAL_FLOOR": ["TOTAL_FLOOR"],
}

# Per-city raw column → canonical field name (the draft mapping this spec ships).
# `DROP` columns are removed at the cleaning stage (spec rule: PII / media).
# `UNMAPPED` columns are listed in `pending_review` so Step 03 sees them.
# Anything not in this dict is also `UNMAPPED` + added to pending_review.
_DRAFT_CANONICAL_MAP: Final[dict[str, str]] = {
    # Identity / core
    "PROP_ID": "listing_id",
    "SPID": "listing_id",  # Hyderabad/Mumbai use SPID; PROP_ID is also there
    "CITY": "city",
    "CITY_ID": "city",  # Gurgaon code; facet decode handled at cleaning stage
    "LOCALITY": "sector",
    "LOCALITY_WO_CITY": "sector",  # Gurgaon: locality name without city suffix
    "PROPERTY_TYPE": "property_type",
    "PROPERTY_TYPE__U": "property_type",  # Hyderabad/Mumbai duplicate
    "TRANSACT_TYPE": "transact_type",
    "OWNTYPE": "ownership_type",
    "VALUE_LABEL": "ownership_type",  # Hyderabad: alternate ownership column
    # Numeric core
    "BEDROOM_NUM": "bedRoom",
    "BATHROOM_NUM": "bathroom",
    "BALCONY_NUM": "balcony",
    "FURNISH": "furnishing_type",
    "FACING": "facing",
    "AGE": "agePossession",
    "FLOOR_NUM": "floor_num",
    "TOTAL_FLOOR": "total_floor",
    "AREA": "built_up_area",
    "BUILTUP_SQFT": "built_up_area",
    "SUPER_SQFT": "built_up_area",
    "SUPERBUILTUP_SQFT": "built_up_area",
    "SUPER_AREA": "built_up_area",
    "CARPET_SQFT": "built_up_area",
    "MIN_AREA_SQFT": "built_up_area",
    "MAX_AREA_SQFT": "built_up_area",
    # Price
    "PRICE": "price",
    "MIN_PRICE": "price",
    "MAX_PRICE": "price",
    "PRICE_SQFT": "price_per_sqft",
    "PRICE_PER_UNIT_AREA": "price_per_sqft",
    "BROKERAGE": "price",  # Gurgaon: brokerage column
    # Text / lists
    "DESCRIPTION": "description_clean",
    "FEATURES": "features_list",
    "AMENITIES": "amenities_list",
    "TOP_USPS": "top_usps",
    # Geo / metadata
    "MAP_DETAILS": "map_details_raw",  # parsed into latitude/longitude at cleaning stage
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "location": "location_raw",
    "REGISTER_DATE": "register_date",
    "REGISTER_DATE__U": "register_date",
    "POSTING_DATE": "register_date",
    "UPDATE_DATE": "register_date",
    "EXPIRY_DATE": "expiry_date",
    "PROP_HEADING": "prop_heading",
    "PROP_NAME": "building_name",
    "SOCIETY_NAME": "building_name",
    "BUILDING_NAME": "building_name",
    "BUILDING_ID": "building_id",
    "PROP_DETAILS_URL": "DROP",  # navigation URL — not a feature
    "PD_URL": "DROP",  # backward compat name
    "PREFERENCE": "preference",  # S/R/A flag, not currently a model feature
    "REGISTERED_DAYS": "registered_days",
    "RES_COM": "res_com",
    "PRODUCT_TYPE": "product_type",
    "CLASS": "class",
    "CLASS_HEADING": "class_heading",
    "CLASS_LABEL": "class_label",
    "PROPERTY_NUMBER": "property_number",
    "PROJ_ID": "project_id",
    "VERIFIED": "verified",
    "SECONDARY_TAGS": "secondary_tags",
    "PRIMARY_TAGS": "primary_tags",
    "TOTAL_LANDMARK_COUNT": "total_landmark_count",
    "FORMATTED_LANDMARK_DETAILS": "formatted_landmark_details",
    "ALT_TAG": "alt_tag",
    "SECONDARY_AREA": "secondary_area",
    "FSL_Data": "fsl_data",
    "profile": "profile",
    "xid": "xid",
    "metadata": "metadata",
    "LISTING": "listing",
    "GROUP_NAME": "group_name",
    "FORMATTED": "formatted",
    "COMMON_FURNISHING_ATTRIBUTES": "common_furnishing_attributes",
    "QUALITY_SCORE": "quality_score",
    "FURNISHING_ATTRIBUTES": "furnishing_attributes",
}

# Anything containing any of these tokens is PII / media / contact and must never
# reach the UI (Rules §1.1). Matches case-insensitively. "Token" boundary:
# not preceded/followed by [A-Za-z0-9]. Crucially, `_` is treated as a
# separator (not a continuation char) so `CONTACT_NAME` matches `CONTACT`,
# `PHOTO_URL` matches `PHOTO`, etc. — column names in the raw data use
# snake_case, and the spec's `test_inventory_marks_pii_columns` requires
# these to be flagged.
PII_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?i:phone|tel|mobile|contact|dealer|agent|email|url|link|photo|image|img|src|media|whats?app)(?![A-Za-z0-9])"
)

# Bounded sample values per column — keeps the inventory JSON small.
SAMPLE_VALUES_MAX: Final[int] = 5

# Coverage join uses at most this many rows per city — keeps it fast on 67-col Gurgaon.
COVERAGE_SAMPLE_SIZE: Final[int] = 10_000

# Manifest hash method published in the manifest JSON so downstream stages
# can recompute it for cross-checking.
MANIFEST_HASH_METHOD: Final[str] = (
    "sha256(canonical_json({sorted_path: sha256, sorted_path: sha256, ...}))"
)

# Output filenames (single source of truth — referenced by tests too).
OUTPUT_FILES: Final[dict[str, str]] = {
    "raw_inventory": "raw_inventory.json",
    "facet_inventory": "facet_inventory.json",
    "coverage": "facet_join_coverage.csv",
    "schema_map": "raw_schema_map.json",
    "snapshot_manifest": "raw_snapshot_manifest.json",
    "meta": "_meta/ingest_v1.json",
}

SPEC_VERSION: Final[str] = "02-raw-data-ingestion-v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    """Stream SHA256 of *path* (files can be large; 8 KB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_stats(path: Path) -> dict[str, Any]:
    """Return {sha256, size_bytes, mtime_iso} for a single file."""
    stat = path.stat()
    return {
        "sha256": _sha256_of_file(path),
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _snapshot_raw_files(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {rel_path_str: {sha256, size_bytes, mtime_iso}} for every file under data/raw/.

    Sorted by path for deterministic JSON output.
    """
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw data dir not found: {raw_dir}")
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file():
            entries[str(path.relative_to(raw_dir))] = _file_stats(path)
    return entries


def _manifest_hash(per_file: dict[str, dict[str, Any]]) -> str:
    """Compute the canonical source_version hash over per-file SHA256s.

    Hashes just the {path: sha256} dict, NOT the full file stats — keeps the
    hash stable when mtimes change (e.g. touching a file). Documented in
    MANIFEST_HASH_METHOD.
    """
    slim = {p: v["sha256"] for p, v in sorted(per_file.items())}
    canonical = json.dumps(slim, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write payload as deterministic JSON (sort_keys, indent=2, ensure_ascii)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _git_commit(repo_root: Path) -> str:
    """Return current HEAD commit hash, or '' if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _sample_values(series: pd.Series, max_n: int = SAMPLE_VALUES_MAX) -> list[str]:
    """First N unique non-null values, sorted (string-coerced for stability)."""
    non_null = series.dropna()
    if non_null.empty:
        return []
    as_str = non_null.astype(str)
    unique_sorted = sorted(as_str.unique())
    return list(unique_sorted[:max_n])


def _column_record(name: str, series: pd.Series) -> dict[str, Any]:
    """Build one inventory entry for a single column."""
    n = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / n) if n > 0 else 0.0, 4)
    return {
        "name": name,
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_pct": null_pct,
        "n_unique": int(series.nunique(dropna=True)),
        "sample_values": _sample_values(series),
        "pii_risk": bool(PII_PATTERN.search(name)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw_listings(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the 4 city CSVs into a {city: DataFrame} dict.

    City keys are the CITY facet labels (e.g. ``Gurgaon``, not the filename).
    Reads in read-only mode and asserts ``data/raw/`` was not modified.
    """
    raw_dir = data_dir / "raw"
    before = _snapshot_raw_files(data_dir)
    out: dict[str, pd.DataFrame] = {}
    for filename, city in RAW_FILE_TO_CITY.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"raw city CSV missing: {path}")
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            raise RuntimeError(f"raw city CSV is empty: {path}")
        out[city] = df
        logger.info("ingest.city_loaded city=%s rows=%d cols=%d", city, len(df), len(df.columns))
    after = _snapshot_raw_files(data_dir)
    if before != after:
        raise RuntimeError(
            "data/raw/ was modified during ingestion — raw immutability violated (Rules §1.1)"
        )
    return dict(sorted(out.items()))


def assert_raw_readonly(data_dir: Path) -> None:
    """Public immutability gate. Raises RuntimeError if ``data/raw/`` changed.

    Takes two snapshots of every file under ``data_dir/raw/`` (SHA256 + size +
    mtime, per ``_snapshot_raw_files``) and compares them. If they differ,
    raises the same ``RuntimeError`` as the inline check inside
    ``load_raw_listings`` — Rules §1.1 is binding.

    Used by Step 06's ``assemble_cleaned_frame`` before reading any raw data.
    Cheap to call: hashes 15 small facet CSVs + 4 large city CSVs once each.
    """
    before = _snapshot_raw_files(data_dir)
    after = _snapshot_raw_files(data_dir)
    if before != after:
        raise RuntimeError(
            "data/raw/ was modified during ingestion — raw immutability violated (Rules §1.1)"
        )


def load_raw_city_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Public alias of ``load_raw_listings``. Same return contract.

    Exposed for Step 06 (assemble.py) which expects a city-keyed dict and a
    raw-readonly gate named ``load_raw_city_frames``. Delegates to
    ``load_raw_listings`` to keep the inline immutability check working
    unchanged — both names run the same gate.
    """
    return load_raw_listings(data_dir)


def load_facets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the 15 facet CSVs into a {facet_name: DataFrame} dict."""
    facet_dir = data_dir / "raw" / "facets"
    if not facet_dir.exists():
        raise FileNotFoundError(f"facet dir missing: {facet_dir}")
    out: dict[str, pd.DataFrame] = {}
    for name in FACET_NAMES:
        path = facet_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"facet file missing: {path}")
        df = pd.read_csv(path)
        out[name] = df
        logger.info("ingest.facet_loaded name=%s rows=%d", name, len(df))
    return dict(out)


def build_inventory(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Per-city column inventory: dtype, null stats, cardinality, sample values, pii_risk."""
    inventory: dict[str, Any] = {}
    for city, df in dfs.items():
        inventory[city] = {
            "rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": [_column_record(str(col), df[col]) for col in df.columns],
        }
        logger.info(
            "ingest.inventory_built city=%s rows=%d cols=%d",
            city,
            len(df),
            len(df.columns),
        )
    # No `generated_at` here — the run timestamp lives in _meta/ingest_v1.json,
    # and excluding it makes this output byte-identical across runs.
    return {"spec_version": SPEC_VERSION, "cities": inventory}


def build_facet_inventory(facet_dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Per-facet-file inventory with primary_key_candidate."""
    inv: dict[str, Any] = {}
    for name, df in facet_dfs.items():
        cols = list(df.columns)
        pkc = next((c for c in ("id", "code") if c in cols), None)
        inv[name] = {
            "path": f"data/raw/facets/{name}.csv",
            "rows": int(len(df)),
            "columns": [
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isna().sum()),
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "sample_values": _sample_values(df[col]),
                }
                for col in df.columns
            ],
            "primary_key_candidate": pkc,
        }
    return {"spec_version": SPEC_VERSION, "facets": inv}


def compute_facet_join_coverage(
    listing_dfs: dict[str, pd.DataFrame],
    facet_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """For each (city, facet) compute join match rate.

    Emits exactly 15 facets × 4 cities = 60 rows. Both sides are cast to
    str before comparison so ``8`` (int) and ``"008"`` (str) are compared
    honestly and the mismatch is *reported*, not silently coerced.
    Multi-value columns (FEATURES, AMENITIES) split on commas first.

    For facets whose mapping lists multiple candidate columns (CITY,
    LOCALITY_ID), the first one that actually exists in the city's schema
    is used; ``raw_column`` in the output records which one was used.
    """
    rows: list[dict[str, Any]] = []
    for city, df in listing_dfs.items():
        for facet_name, raw_columns in CODED_COLUMNS_BY_FACET.items():
            facet = facet_dfs[facet_name]
            facet_keys = set(facet["id"].astype(str).unique()) if "id" in facet.columns else set()

            # Pick the first candidate column that actually exists in this city.
            raw_col = next((c for c in raw_columns if c in df.columns), None)
            if raw_col is None:
                rows.append(
                    {
                        "city": city,
                        "raw_column": "(none)",
                        "facet_name": facet_name,
                        "facet_path": f"data/raw/facets/{facet_name}.csv",
                        "join_match_rate": 0.0,
                        "null_in_facet_rate": 0.0,
                        "mismatched_code_count": 0,
                        "note": "no matching raw column in this city's schema",
                    }
                )
                continue

            sample = df[raw_col].dropna()
            if sample.empty:
                rows.append(
                    {
                        "city": city,
                        "raw_column": raw_col,
                        "facet_name": facet_name,
                        "facet_path": f"data/raw/facets/{facet_name}.csv",
                        "join_match_rate": 0.0,
                        "null_in_facet_rate": 0.0,
                        "mismatched_code_count": 0,
                        "note": "no non-null values in sample",
                    }
                )
                continue
            # Subsample to COVERAGE_SAMPLE_SIZE max for speed, but never
            # request more rows than we have (pandas refuses when replace=False).
            if len(sample) > COVERAGE_SAMPLE_SIZE:
                sample = sample.sample(n=COVERAGE_SAMPLE_SIZE, random_state=42)

            # Multi-value encoded fields: split on commas, explode unique codes.
            if facet_name in {"AMENITIES", "FEATURES"}:
                exploded = (
                    sample.astype(str).str.split(",").explode().str.strip().dropna()
                )
                exploded = exploded[exploded != ""]
                codes = exploded.unique().tolist()
                n_total = len(codes)
                hits = sum(1 for c in codes if c in facet_keys)
                mr = (hits / n_total) if n_total else 0.0
                mismatched = n_total - hits
                note = "multi-value: rate is unique-code coverage"
            else:
                codes = sample.astype(str).unique().tolist()
                n_total = len(codes)
                hits = sum(1 for c in codes if c in facet_keys)
                mr = (hits / n_total) if n_total else 0.0
                mismatched = n_total - hits
                note = "unique-code match rate over 10k-row sample"

            rows.append(
                {
                    "city": city,
                    "raw_column": raw_col,
                    "facet_name": facet_name,
                    "facet_path": f"data/raw/facets/{facet_name}.csv",
                    "join_match_rate": round(mr, 4),
                    "null_in_facet_rate": round(mismatched / max(n_total, 1), 4),
                    "mismatched_code_count": int(mismatched),
                    "note": note,
                }
            )
    out = pd.DataFrame(rows)
    logger.info("ingest.coverage_computed rows=%d", len(out))
    return out


def build_schema_map(
    coverage_df: pd.DataFrame,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Draft raw→canonical column map per city, plus a pending_review list.

    Uses ``_DRAFT_CANONICAL_MAP`` for known names; PII-marked columns are
    ``DROP``; everything else is ``UNMAPPED`` and added to ``pending_review``.
    """
    cities_block: dict[str, Any] = {}
    pending: list[dict[str, str]] = []
    for city, info in inventory["cities"].items():
        city_map: dict[str, dict[str, str]] = {}
        for col in info["columns"]:
            name = col["name"]
            if col["pii_risk"]:
                city_map[name] = {
                    "canonical": "DROP",
                    "reason": "PII / media URL — Rules §1.1 (dropped at cleaning)",
                }
                continue
            if name in _DRAFT_CANONICAL_MAP:
                canon = _DRAFT_CANONICAL_MAP[name]
                if canon == "DROP":
                    city_map[name] = {
                        "canonical": "DROP",
                        "reason": "internal URL / not a feature",
                    }
                else:
                    city_map[name] = {
                        "canonical": canon,
                        "reason": "draft mapping — Step 03 finalizes",
                    }
            else:
                city_map[name] = {
                    "canonical": "UNMAPPED",
                    "reason": "not in draft canonical map — needs Step 03 decision",
                }
                pending.append({"city": city, "raw_column": name})
        cities_block[city] = city_map
    # De-duplicate pending entries (multiple appear if same column across cities)
    pending = sorted(
        {(p["city"], p["raw_column"]): p for p in pending}.values(),
        key=lambda p: (p["city"], p["raw_column"]),
    )
    return {
        "spec_version": SPEC_VERSION,
        "note": "Draft map. Step 03 (cleaning) finalizes UNMAPPED entries.",
        "cities": cities_block,
        "pending_review": pending,
    }


def snapshot_raw(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash every file under data/raw/. Returns per-file {sha256, size, mtime}."""
    snap = _snapshot_raw_files(data_dir)
    logger.info("ingest.snapshot_taken files=%d", len(snap))
    return snap


def write_meta(
    snapshot: dict[str, dict[str, Any]],
    run_id: str,
    source_version: str,
    output_dir: Path,
    repo_root: Path,
) -> Path:
    """Write ``data/processed/_meta/ingest_v1.json`` with run provenance."""
    meta = {
        "run_id": run_id,
        "git_commit": _git_commit(repo_root),
        "python_version": _python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "generated_at": _now_iso(),
        "source_version": source_version,
        "spec_version": SPEC_VERSION,
        "raw_files_count": len(snapshot),
    }
    out_path = output_dir / OUTPUT_FILES["meta"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, meta)
    logger.info("ingest.meta_written path=%s", out_path)
    return out_path


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def run_ingestion(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Top-level orchestrator. Returns ``{"source_version": ..., "outputs": {...}}``.

    Idempotent: identical inputs produce byte-identical outputs (same
    ``source_version``).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    logger.info("ingest.start data_dir=%s output_dir=%s", data_dir, output_dir)
    snap = snapshot_raw(data_dir)
    source_version = _manifest_hash(snap)

    listing_dfs = load_raw_listings(data_dir)
    facet_dfs = load_facets(data_dir)

    inventory = build_inventory(listing_dfs)
    facet_inventory = build_facet_inventory(facet_dfs)
    coverage_df = compute_facet_join_coverage(listing_dfs, facet_dfs)
    schema_map = build_schema_map(coverage_df, inventory)

    # Manifest: hash + per-file stats + method documentation.
    manifest = {
        "spec_version": SPEC_VERSION,
        "manifest_hash_method": MANIFEST_HASH_METHOD,
        "source_version": source_version,
        "files": snap,
    }

    # Write outputs (deterministic JSON / CSV).
    outputs: dict[str, Path] = {}
    p = output_dir / OUTPUT_FILES["raw_inventory"]
    _write_json(p, inventory)
    outputs["raw_inventory"] = p

    p = output_dir / OUTPUT_FILES["facet_inventory"]
    _write_json(p, facet_inventory)
    outputs["facet_inventory"] = p

    p = output_dir / OUTPUT_FILES["coverage"]
    coverage_df.to_csv(p, index=False, quoting=csv.QUOTE_MINIMAL)
    outputs["facet_join_coverage"] = p

    p = output_dir / OUTPUT_FILES["schema_map"]
    _write_json(p, schema_map)
    outputs["raw_schema_map"] = p

    p = output_dir / OUTPUT_FILES["snapshot_manifest"]
    _write_json(p, manifest)
    outputs["raw_snapshot_manifest"] = p

    outputs["meta"] = write_meta(
        snap,
        run_id=uuid.uuid4().hex,
        source_version=source_version,
        output_dir=output_dir,
        repo_root=repo_root,
    )

    logger.info("ingest.done source_version=%s outputs=%d", source_version, len(outputs))
    return {"source_version": source_version, "outputs": {k: str(v) for k, v in outputs.items()}}


__all__ = [
    "RAW_FILE_TO_CITY",
    "FACET_NAMES",
    "CODED_COLUMNS_BY_FACET",
    "PII_PATTERN",
    "OUTPUT_FILES",
    "assert_raw_readonly",
    "load_raw_city_frames",
    "load_raw_listings",
    "load_facets",
    "build_inventory",
    "build_facet_inventory",
    "compute_facet_join_coverage",
    "build_schema_map",
    "snapshot_raw",
    "write_meta",
    "run_ingestion",
]
