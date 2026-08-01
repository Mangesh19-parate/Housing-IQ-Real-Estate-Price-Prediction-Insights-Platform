# Spec: Raw Data Ingestion And Schema Inventory

## Overview
Load the four raw city CSVs (`gurgaon_10k.csv`, `hyderabad.csv`, `kolkata.csv`, `mumbai.csv`) and the 15 facet lookup CSVs under `data/raw/facets/` into a single, reproducible inventory that downstream cleaning/feature-engineering steps can build on. This spec produces a deterministic, content-hashed snapshot of the raw data (read-only, never writes to `data/raw/` per the immutable-raw rule), an inventory of every column per city with dtype/null-count/cardinality stats, a per-city facet-join coverage report, and a documented column-to-canonical mapping skeleton that the next spec (`03-data-cleaning-and-canonical-schema-build`) will fill in. Module: **foundation**. This is Step 02 — Step 01 (repo scaffolding) is done; this is the first concrete data-pipeline step of the 7-week roadmap and unblocks every downstream cleaning/EDA/training/analytics task.

## Depends on
- Step 01 — `01-repo-scaffolding-and-environment-setup` (provides `app/`, `api/`, `ml/`, `scripts/`, working `pytest` harness, `python-dotenv` config, ruff config).

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — a Python pipeline stage invoked from `scripts/run_pipeline.py` (or directly via `python -m ml.cleaning.ingest`). It produces files under `data/processed/` that later specs and FastAPI routes read, but does not touch the Flask or FastAPI HTTP layers.

## Data / Schema changes

**No schema changes to the application DB (`data/app.db`)** — no new operational tables; existing `prediction_log` / `recommendation_log` / `classification_log` / `model_registry` stay as-is per Step 01.

**New files under `data/processed/`** (all written by this spec, all include a `generated_at` and `source_version` metadata header):

- `data/processed/raw_inventory.json` — per-city inventory: `{city: {rows, columns: [{name, dtype, null_count, null_pct, n_unique, sample_values}], total_rows, total_columns}}`. Keyed by city; consumed by `03-data-cleaning` to plan per-city cleaning.
- `data/processed/facet_inventory.json` — per-facet-file inventory: `{facet_name: {path, rows, columns: [{name, dtype, null_count, n_unique, sample_values], primary_key_candidate}}`. Used to validate that every coded column in the raw listings has a matching facet lookup (gaps get flagged).
- `data/processed/facet_join_coverage.csv` — one row per (city, raw_coded_column, facet_name), with columns `city, raw_column, facet_path, join_match_rate, null_in_facet_rate, mismatched_code_count`. Surfaces columns whose facet decode is incomplete (e.g. codes present in the data but absent from the facet file).
- `data/processed/raw_schema_map.json` — the column-mapping skeleton. Per-city raw column → canonical field name (or `DROP` with reason). This is a *draft* that Step 03 finalizes; any column whose canonical name is not yet agreed is flagged `UNMAPPED` and listed under `pending_review` so reviewers can see what's left to decide. Final canonical names must follow `10-FINALIZED-INPUT-SCHEMA.md` §1+§2 (the 16-field input contract) and `05-BACKEND-SCHEMA.md` §2 / §U-SCHEMA-5 (the canonical Listing schema).
- `data/processed/raw_snapshot_manifest.json` — content hashes + sizes for every file under `data/raw/` at the time of ingestion (SHA256 of each file). Lets later steps detect "the raw data changed since this inventory was generated" without re-ingesting. Combined with the manifest, this is the `source_version` value that all downstream derived tables stamp in their metadata.
- `data/processed/_meta/ingest_v1.json` — run metadata: `run_id` (uuid4), `git_commit`, `python_version`, `pandas_version`, `numpy_version`, `generated_at`, `source_version` (= the SHA256-of-manifest hash), `spec_version: "02-raw-data-ingestion-v1"`. Same provenance pattern that all later derived tables use.

**No writes to `data/raw/`** — Rule §1.1 (raw CSVs immutable) and Rule §1.2 (cleaning writes to `/processed`) are binding. The ingestion step opens files in read-only mode and verifies no write happened (via a snapshot of `data/raw/` mtimes before/after).

**No model artifacts touched** — that is Step 04+ territory.

## Templates / UI
None. This spec produces no Flask templates, no CSS, no JS, no static assets. All output is JSON/CSV consumed by other Python modules.

## Files to change / Files to create

**Create:**
- `ml/cleaning/__init__.py` — already empty stub from Step 01; no changes needed if Step 01 landed the empty package marker.
- `ml/cleaning/ingest.py` — the ingestion entry point. Public API:
  - `load_raw_listings(data_dir: Path) -> dict[str, pd.DataFrame]` — returns `{city: df}` for all 4 CSVs; deterministic ordering (sorted city names).
  - `load_facets(data_dir: Path) -> dict[str, pd.DataFrame]` — returns `{facet_name: df}` for the 15 facet CSVs.
  - `build_inventory(dfs: dict[str, pd.DataFrame]) -> dict` — produces the per-city inventory JSON (rows, columns with dtype/null_count/null_pct/n_unique/sample_values).
  - `build_facet_inventory(facet_dfs: dict[str, pd.DataFrame]) -> dict` — produces the per-facet inventory JSON.
  - `compute_facet_join_coverage(listing_dfs, facet_dfs) -> pd.DataFrame` — for each (city, coded_raw_column, facet_name), compute join match rate and null-in-facet rate.
  - `build_schema_map(coverage_df, inventory) -> dict` — produces the raw-to-canonical column map; flags `UNMAPPED` and `DROP` columns with reasons.
  - `snapshot_raw(data_dir: Path) -> dict` — returns `{path: {sha256, size_bytes, mtime_iso}}` for every file under `data/raw/`.
  - `write_meta(snapshot: dict, run_id: str) -> Path` — writes `data/processed/_meta/ingest_v1.json`.
  - `run_ingestion(data_dir: Path, output_dir: Path) -> dict` — top-level orchestrator that returns the final `source_version` (SHA256 of the manifest JSON, used by all downstream tables). Idempotent: running twice with identical raw data produces identical output (manifest hash stable).
- `scripts/ingest_raw.py` — thin CLI wrapper around `ml.cleaning.ingest.run_ingestion()` for one-command local execution. Calls the orchestrator, prints a short summary (rows per city, facet coverage % per city, # UNMAPPED columns), exits non-zero if any city is empty or if any facet file is unreadable.
- `scripts/run_pipeline.py` — Step 01 created a placeholder. This spec extends it to call the ingestion stage first: `def main(): ingest_raw.main(); print("ingest done — see data/processed/raw_inventory.json")`. Keeps the single-entry-point contract from TRD §13.
- `tests/test_raw_ingestion.py` — pytest tests (see "Definition of done" for the full list).
- `tests/fixtures/raw_snapshot_fixture.py` — small fixture builder for tests that need a synthetic `data/raw/` subtree: writes 1–2 minimal CSVs (1 row each) to a tmp dir, exercises the ingestion functions, asserts the expected JSON output shape. Avoids depending on the real ~180k-row CSVs in CI.
- `data/processed/_meta/.gitkeep` — keep the metadata dir in git.

**Modify:**
- `.gitignore` — add `data/processed/_meta/` exception if blanket-ignored (verify current state; the processed dir is currently tracked, but `_meta/` shouldn't be). Actually: `data/processed/.gitkeep` and `data/processed/analytics_cache/.gitkeep` already exist; confirm `_meta/` is untracked and add `.gitkeep` so the directory itself is in git while leaving the run files (`ingest_v1.json`, etc.) gitignored.
- `requirements.txt` — no changes needed. `pandas` and `numpy` are already pinned.

## New dependencies
No new dependencies. `pandas` (CSV/parquet IO, dtype inspection) and `numpy` (deterministic hashing) are already in `requirements.txt` per Step 01. `hashlib` (SHA256), `uuid` (run id), `json` (manifest write), and `pathlib` (file traversal) are Python stdlib.

## Rules for implementation
- **No SQLAlchemy/ORM.** N/A for this spec — no SQL is written; if anything ever does use SQLite for inventory lookup later, it must still be `?`-parameterized per CLAUDE.md.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** This spec never reaches a UI. But: the column-inventory step MUST scan each city's columns for any field whose name suggests PII or media (regex pattern: `(?i)\b(phone|tel|mobile|contact|dealer|agent|email|url|link|photo|image|img|src|media|whats?app)\b`) and flag it in the inventory as `pii_risk: true`. Step 03 must drop them; this spec just makes them visible early so Step 03's drop list is fully justified, not silent.
- **CSS variables only.** N/A — no templates or styles in this spec.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifacts produced.
- **Raw data is immutable.** The ingestion step must not write any file under `data/raw/`. A defensive check (`os.access(path, os.W_OK)` returns False for raw files; or simpler: compute mtime+sha256 snapshot before and after, assert unchanged) is mandatory. If it ever changes, raise and abort — that's a sign the environment is misconfigured (writable raw dir) or something is corrupting the data.
- **Determinism.** All JSON output must be `sort_keys=True`, `indent=2`, UTF-8, no trailing whitespace. `json.dumps(..., ensure_ascii=False)` is forbidden — `ensure_ascii=True` is required for stable byte-level hashing of the manifest. Sample values in the inventory are sorted and deduplicated before inclusion so two ingestion runs over identical raw produce identical JSON.
- **Sample values are bounded.** At most 5 sample values per column per city (first 5 non-null unique values, sorted). Never dump full columns into inventory JSON — `gurgaon_10k.csv` has 67 columns × ~45k rows; the inventory must stay small (target < 1 MB total).
- **Reproducibility of the manifest hash.** `raw_snapshot_manifest.json` itself is included in the hash that becomes `source_version`? No — that would create a circular hash. The hash is computed over a sorted, canonicalized concatenation of the per-file hashes (NOT the JSON file containing them). Documented in `raw_snapshot_manifest.json` itself under a `manifest_hash_method` field: `"sha256(canonical_json({sorted_path: sha256, sorted_path: sha256, ...}))"`. The same method must be used by every downstream stage that wants to stamp `source_version` on its own output.
- **Schema map is a draft, not a contract.** `raw_schema_map.json` is the *starting point* for `03-data-cleaning-and-canonical-schema-build`. Any field whose canonical name is not yet locked (per `10-FINALIZED-INPUT-SCHEMA.md`) is mapped to `UNMAPPED` and listed in `pending_review`. Step 03 resolves these. This keeps this spec's blast radius small: even if Step 03 disagrees with a mapping, only the `raw_schema_map.json` needs re-emitting (or even just patched in Step 03).
- **Facet join coverage is computed, not assumed.** For every (city, coded_column, facet_name) candidate, the function computes an actual join match rate on a 10k-row sample of the listing (full-dataset for small cities). If a city has a coded column with no matching facet file (e.g. Kolkata missing `FLOOR_NUM` facet entries for some codes), the coverage CSV surfaces it — Step 03 then decides whether to (a) drop those rows, (b) keep the raw code, or (c) extend the facet map.
- **All randomness is seeded.** Not directly used here, but if any sampling is needed (e.g., the 10k sample for join coverage), `random_state=42` per CLAUDE.md.
- **Logging.** `logging` module at each stage (`ingest.start`, `ingest.city_loaded`, `ingest.facet_loaded`, `ingest.coverage_computed`, `ingest.manifest_written`, `ingest.done`). No `print()` in library code; the CLI script (`scripts/ingest_raw.py`) is allowed to print a final summary, but internal stages log via the `logging` module.
- **Test fixtures must not touch the real raw data.** `tests/fixtures/raw_snapshot_fixture.py` builds a synthetic `data/raw/` subtree in `tmp_path` (1-row CSVs) and exercises the ingestion end-to-end against it. Tests that *do* read the real `data/raw/` must be guarded by a pytest marker (`@pytest.mark.realdata`) so they're opt-in (CI without the dataset skips them).

## Definition of done
A specific, testable checklist verifiable by running the app or the test suite.

1. `python -m pytest tests/test_raw_ingestion.py -v` from repo root runs and passes. Tests required (exact names):
   - `test_load_raw_listings_returns_four_cities` — loads real (or fixture) CSVs, asserts the dict has exactly 4 keys: `Gurgaon`, `Hyderabad`, `Kolkata`, `Mumbai` (matching the `CITY` facet labels, not the filenames — `gurgaon_10k.csv` decodes to `Gurgaon`).
   - `test_load_raw_listings_does_not_modify_files` — snapshots `data/raw/` mtimes+sizes before and after, asserts identical.
   - `test_load_facets_returns_fifteen_files` — loads all 15 facet CSVs, asserts the dict has exactly 15 keys matching the documented facet names (`AGE`, `AMENITIES`, `BATHROOM_NUM`, `BEDROOM_NUM`, `BUILDING_ID`, `CITY`, `FACING_DIRECTION`, `FEATURES`, `FLOOR_NUM`, `FURNISH`, `LOCALITY_ID`, `OWNERSHIP_TYPE`, `PROPERTY_TYPE`, `SUB_AVAILABILITY`, `TOTAL_FLOOR`).
   - `test_inventory_per_city_shape` — for each city, `inventory[city]['columns']` is a list of `{name, dtype, null_count, null_pct, n_unique, sample_values, pii_risk}` entries; total entries equals the raw column count.
   - `test_inventory_marks_pii_columns` — verifies at least one column per city is flagged `pii_risk: true` (Gurgaon has photo URLs, CONTACT_NAME-style fields; others may not, but the check is on regex match, not on count).
   - `test_inventory_sample_values_bounded` — every `sample_values` list has length ≤ 5.
   - `test_facet_join_coverage_emits_one_row_per_coded_column` — for each coded column per city (CITIES join, FURNISH, FACING, AGE, PROPERTY_TYPE, OWNERSHIP_TYPE, FLOOR_NUM, TOTAL_FLOOR, BEDROOM_NUM, BATHROOM_NUM, BUILDING_ID, LOCALITY_ID, FEATURES, AMENITIES, SUB_AVAILABILITY — 15 facets × 4 cities = 60 rows expected), the CSV has a row. Any missing row fails the test (catches a typo in facet-name lookup).
   - `test_schema_map_references_canonical_names` — every non-`DROP` non-`UNMAPPED` mapping in `raw_schema_map.json` uses a canonical field name that exists in `10-FINALIZED-INPUT-SCHEMA.md` or `05-BACKEND-SCHEMA.md` §U-SCHEMA-5 (test loads both docs as text, parses out the listed canonical field names, asserts overlap).
   - `test_unmapped_columns_listed_for_review` — every column not mapped to a canonical name appears in `pending_review` (not silently dropped).
   - `test_snapshot_manifest_is_sha256` — every entry in `raw_snapshot_manifest.json` is a 64-char hex SHA256 digest; the file is valid JSON; loading it twice yields identical `source_version`.
   - `test_meta_file_has_required_keys` — `data/processed/_meta/ingest_v1.json` has `run_id`, `git_commit`, `python_version`, `pandas_version`, `numpy_version`, `generated_at`, `source_version`, `spec_version`.
   - `test_ingestion_is_idempotent` — running `run_ingestion()` twice on the same `data/raw/` produces identical `source_version` and byte-identical JSON outputs (diff is empty).
   - `test_run_pipeline_imports_ingest` — `python -c "import scripts.run_pipeline as rp; assert callable(rp.main)"` works.
   - `test_fixture_ingestion_end_to_end` — using the synthetic fixture, the full pipeline writes all 5 JSON files + the coverage CSV, and the `_meta/` JSON points at the manifest.
2. `python scripts/ingest_raw.py` from repo root runs end-to-end against the real `data/raw/`, prints a summary line per city (`Gurgaon: 44890 rows, 67 cols, 18 pii-flagged, 95% facet coverage`), and exits 0. Verifiable by running it manually.
3. `python scripts/run_pipeline.py` from repo root runs and exits 0 (still prints "ingest done — see data/processed/raw_inventory.json" since later stages aren't implemented yet).
4. `python -m pytest -m "not realdata"` from repo root (the fast path that excludes real-data tests) still passes — confirms the suite runs in CI without the 180k-row CSVs.
5. `ruff check .` from repo root reports zero issues on `ml/cleaning/ingest.py`, `scripts/ingest_raw.py`, `scripts/run_pipeline.py`, `tests/test_raw_ingestion.py`, `tests/fixtures/raw_snapshot_fixture.py`.
6. `git status` clean after a fresh `git add . && git commit` — all 5 JSON outputs and the coverage CSV are gitignored (they're regeneratable from raw); only the source `.py` files, `tests/`, `.gitkeep`, and any spec updates are committed. Verifiable by inspecting the diff before commit.
7. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes. Step 03+ will start filling in routes. (If a future spec changes the table, that's its job.)
8. `data/processed/raw_inventory.json`, `data/processed/facet_inventory.json`, `data/processed/facet_join_coverage.csv`, `data/processed/raw_schema_map.json`, `data/processed/raw_snapshot_manifest.json`, `data/processed/_meta/ingest_v1.json` all exist after a successful run; each is valid JSON (or CSV) and parses without error.
9. `07-TRACKER.md` "Week 1 — Data Understanding & Cleaning" Day 1 row's status is updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.
