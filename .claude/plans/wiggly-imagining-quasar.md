# Implementation Plan: Step 02 — Raw Data Ingestion and Schema Inventory

## Context

HousingIQ's first concrete data-pipeline step. Step 01 landed the repo scaffolding (Flask + FastAPI stubs, SQLite app DB, scripts entry point). Step 02 builds the foundation that every downstream cleaning/EDA/training/analytics task will read from: a deterministic, content-hashed snapshot of the 4 raw city CSVs + 15 facet CSVs, plus a column inventory, facet-join coverage report, and a draft raw→canonical column map.

Why now: the Cleaning spec (Step 03) needs to know **exactly which columns exist per city**, **which are PII** (and must be dropped per Rules §1.1), **which coded columns have facet lookups**, and **which canonical names are still TBD**. This spec produces all of that — read-only against `data/raw/`, writes only to `data/processed/`.

Module: `foundation`. Depends on Step 01 only.

## Critical files (to be created)

- `ml/cleaning/ingest.py` — single file containing all 8 public functions + the orchestrator
- `scripts/ingest_raw.py` — CLI wrapper
- `scripts/run_pipeline.py` — extend the Step 01 placeholder to call the ingestion stage
- `tests/test_raw_ingestion.py` — 13 tests named in spec DoD §1
- `tests/fixtures/__init__.py` — empty package marker
- `tests/fixtures/raw_snapshot_fixture.py` — synthetic `data/raw/` builder for fixture-based tests
- `data/processed/_meta/.gitkeep` — directory marker (matches existing `data/processed/.gitkeep` style)

## Critical files (to be modified)

- `.gitignore` — add `data/processed/_meta/*.json` (run files gitignored; the `.gitkeep` keeps the dir tracked)
- `pyproject.toml` — register `realdata` marker in `[tool.pytest.ini_options]` so `-m "not realdata"` works as required by spec DoD §4

Reused from Step 01:
- `pytest.ini` already has `pythonpath = .` and `testpaths = tests` — no changes
- `pyproject.toml` already has ruff config (E/F/W/I) — new code must pass
- `app/database/db.py`, `app/config.py` — not touched (no DB writes this spec)

Reused from docs (read at runtime by tests):
- `docs/10-FINALIZED-INPUT-SCHEMA.md` — canonical field names parsed by `test_schema_map_references_canonical_names`
- `docs/05-BACKEND-SCHEMA.md` — same parser

## Implementation outline

### 1. `ml/cleaning/ingest.py` (single file, ~300-400 LOC)

Constants near top:
- `RAW_FILE_TO_CITY: dict[str, str]` — maps `gurgaon_10k.csv → Gurgaon`, `hyderabad.csv → Hyderabad`, `kolkata.csv → Kolkata`, `mumbai.csv → Mumbai` (matching `CITY` facet labels per spec).
- `FACET_NAMES: tuple[str, ...]` — 15-element tuple in spec-defined order: `AGE, AMENITIES, BATHROOM_NUM, BEDROOM_NUM, BUILDING_ID, CITY, FACING_DIRECTION, FEATURES, FLOOR_NUM, FURNISH, LOCALITY_ID, OWNERSHIP_TYPE, PROPERTY_TYPE, SUB_AVAILABILITY, TOTAL_FLOOR`.
- `CODED_COLUMNS_BY_FACET: dict[str, list[str]]` — per facet, the raw column names likely to hold its code, for the coverage join. Built from the 4 city headers × each facet (CITY facet joins on `CITY` column where it's a label and `CITY_ID` where it's a code; FURNISH facet joins on `FURNISH`; etc.).
- `PII_PATTERN: re.Pattern` — compiled `(?i)\b(phone|tel|mobile|contact|dealer|agent|email|url|link|photo|image|img|src|media|whats?app)\b`.
- `SAMPLE_VALUES_MAX: int = 5` — bound on per-column sample values.
- `logger = logging.getLogger("ml.cleaning.ingest")` — module-level logger per spec rules.

Functions (public API as specified):

1. **`load_raw_listings(data_dir: Path) -> dict[str, pd.DataFrame]`**
   - For each filename in `RAW_FILE_TO_CITY`: `pd.read_csv(path, low_memory=False)`.
   - Return dict sorted by city key.
   - Verify `data/raw/` not modified: snapshot mtime+size of each `*.csv` before, after, assert equal.

2. **`load_facets(data_dir: Path) -> dict[str, pd.DataFrame]`**
   - For each name in `FACET_NAMES`: `pd.read_csv(data_dir / "facets" / f"{name}.csv")`.
   - Return dict in facet-name order.

3. **`build_inventory(dfs: dict[str, pd.DataFrame]) -> dict`**
   - Per city, per column: compute `name, dtype, null_count, null_pct (rounded 4 dp), n_unique, sample_values (first 5 non-null unique, sorted), pii_risk (bool)` via `PII_PATTERN.search(col_name)`.
   - Output shape: `{city: {"rows": int, "total_columns": int, "columns": [{...}, ...]}}`.
   - Deterministic: `json.dumps(..., sort_keys=True, indent=2, ensure_ascii=True)` (per spec rule).

4. **`build_facet_inventory(facet_dfs: dict[str, pd.DataFrame]) -> dict`**
   - Per facet: `{path, rows, columns: [{name, dtype, null_count, n_unique, sample_values}], primary_key_candidate}`.
   - `primary_key_candidate` = first column named `id` (or `code`); if none, `None`.

5. **`compute_facet_join_coverage(listing_dfs, facet_dfs) -> pd.DataFrame`**
   - For every `(city, raw_column, facet_name)` from the mapping, take a 10k-row sample of the listing (`random_state=42`) and:
     - Count rows where raw column is non-null.
     - For each such code, attempt to find a matching row in the facet (compare on string-form of both sides after `astype(str)` — handles `8` vs `"008"` mismatch).
     - Compute `join_match_rate = matches / non_null_count`, `null_in_facet_rate = unmatched / non_null_count`, `mismatched_code_count` = absolute number of unmatched unique codes (capped at report size).
   - 60 rows expected: 15 facets × 4 cities.

6. **`build_schema_map(coverage_df, inventory) -> dict`**
   - Hard-coded initial mapping skeleton: e.g. `CITY → city`, `LOCALITY → sector`, `PROPERTY_TYPE → property_type`, `BEDROOM_NUM → bedRoom`, `BATHROOM_NUM → bathroom`, `BALCONY_NUM → balcony`, `FURNISH → furnishing_type` (raw code, encoded later), `FACING → facing`, `AGE → agePossession`, `FLOOR_NUM → floor_num`, `TOTAL_FLOOR → total_floor`, `AREA → built_up_area`, `PRICE → price`, `MIN_PRICE / MAX_PRICE / PRICE_SQFT → price (fallback)`, `MAP_DETAILS → latitude/longitude (split at parse time)`, `DESCRIPTION → description_clean`, `REGISTER_DATE / POSTING_DATE / UPDATE_DATE → register_date`.
   - Drop list: every column whose `pii_risk` is true → `{"canonical": "DROP", "reason": "PII / media URL — see Rules §1.1"}`.
   - Anything not in either bucket → `{"canonical": "UNMAPPED", "reason": "..."}` and added to `pending_review` with the city + column name.
   - Output: `{city: {raw_col: {"canonical": str, "reason": str}}, pending_review: [{city, raw_col}, ...], generated_at: iso8601}`.

7. **`snapshot_raw(data_dir: Path) -> dict`**
   - Walk `data_dir / "raw"`, for each regular file: SHA256 via `hashlib.sha256` (read in 8KB chunks — files are large), `size_bytes` via `path.stat().st_size`, `mtime_iso` via `datetime.fromtimestamp(mtime).isoformat()`.
   - Returns `{relative_path_str: {sha256, size_bytes, mtime_iso}, ...}` sorted by path.
   - The canonical hash method: `hashlib.sha256(json.dumps(per_file_hashes, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()`. This is what downstream stages stamp as `source_version`.
   - Documented in the manifest JSON under `manifest_hash_method`.

8. **`write_meta(snapshot: dict, run_id: str, source_version: str) -> Path`**
   - Write `data/processed/_meta/ingest_v1.json` with `{run_id, git_commit, python_version, pandas_version, numpy_version, generated_at, source_version, spec_version: "02-raw-data-ingestion-v1"}`.
   - `git_commit` via `subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()` — empty string if not a git repo (defensive).

9. **`run_ingestion(data_dir: Path, output_dir: Path) -> dict`**
   - Orchestrator. Order (matches spec activity diagram):
     1. `snapshot_raw(data_dir)` → `snap`
     2. `source_version = sha256(json.dumps({k: v["sha256"] for k, v in snap.items()}, sort_keys=True))`
     3. `load_raw_listings(data_dir)` → `listing_dfs` (assert non-modified)
     4. `load_facets(data_dir)` → `facet_dfs`
     5. `inventory = build_inventory(listing_dfs)` → write `raw_inventory.json`
     6. `facet_inv = build_facet_inventory(facet_dfs)` → write `facet_inventory.json`
     7. `coverage_df = compute_facet_join_coverage(listing_dfs, facet_dfs)` → write `facet_join_coverage.csv`
     8. `schema_map = build_schema_map(coverage_df, inventory)` → write `raw_schema_map.json`
     9. `manifest = {**snap, "manifest_hash_method": "sha256(canonical_json({path: sha256, ...}))", "source_version": source_version}` → write `raw_snapshot_manifest.json`
     10. `write_meta(snap, run_id=uuid4().hex, source_version=source_version)` → write `_meta/ingest_v1.json`
   - Returns `{"source_version": source_version, "outputs": {path: ...}}`.
   - **Idempotent**: identical inputs → byte-identical outputs.

### 2. `scripts/ingest_raw.py` (CLI wrapper)

- `argparse`: `--data-dir` (default `data/`), `--output-dir` (default `data/processed/`).
- Call `ml.cleaning.ingest.run_ingestion(...)`.
- Print summary: per-city line (`Gurgaon: 44890 rows, 67 cols, 18 pii-flagged, 95% facet coverage`), then total. Exits non-zero on `RuntimeError` (empty city, unreadable facet).
- No `print()` inside `ml/cleaning/ingest.py` — logging only.

### 3. `scripts/run_pipeline.py` (extend Step 01 placeholder)

- `def main():` calls `ingest_raw.main()` first; on success prints `"ingest done — see data/processed/raw_inventory.json"`.
- Future specs (03 cleaning, 04 EDA) extend this function by adding calls before/after.
- Imports `scripts.ingest_raw` (relative import not needed since `pytest.ini` puts repo root on path; `scripts/` is already importable).

### 4. `.gitignore` (one-line addition)

Append:
```
# Pipeline run metadata (regenerable from raw)
data/processed/_meta/*.json
!data/processed/_meta/.gitkeep
```

### 5. `pyproject.toml` (one section addition)

Add `[tool.pytest.ini_options]` with `markers = ["realdata: tests that read real data/raw/ CSVs (deselect with '-m \"not realdata\"')"]`. Required so `--strict-markers` (already set in `pytest.ini`) doesn't error on `@pytest.mark.realdata`.

### 6. `tests/test_raw_ingestion.py` (13 tests, one per spec DoD §1 bullet)

Tests requiring the real 182k-row dataset are guarded with `@pytest.mark.realdata`. Tests using the fixture are unmarked (run in fast path).

| Test | Type | What it asserts |
|---|---|---|
| `test_load_raw_listings_returns_four_cities` | realdata | Dict has keys `Gurgaon, Hyderabad, Kolkata, Mumbai` (verified against `CITY.csv` label `Gurgaon`, not the filename). |
| `test_load_raw_listings_does_not_modify_files` | realdata | Before/after mtime+size of `data/raw/*.csv` and `data/raw/facets/*.csv` is identical. |
| `test_load_facets_returns_fifteen_files` | realdata | Dict has all 15 documented facet names. |
| `test_inventory_per_city_shape` | fixture | `inventory[city]['columns']` is a list of `{name, dtype, null_count, null_pct, n_unique, sample_values, pii_risk}`; total length = raw column count. |
| `test_inventory_marks_pii_columns` | realdata | At least one `pii_risk: true` per city (regex match). |
| `test_inventory_sample_values_bounded` | fixture + realdata | Every `sample_values` list has `len ≤ 5`. |
| `test_facet_join_coverage_emits_one_row_per_coded_column` | fixture | 60 rows (15 facets × 4 cities) present; missing row fails. |
| `test_schema_map_references_canonical_names` | fixture | Every non-DROP, non-UNMAPPED mapping uses a canonical name appearing in `10-FINALIZED-INPUT-SCHEMA.md` or `05-BACKEND-SCHEMA.md` §U-SCHEMA-5 (parsed via simple markdown table-row extraction). |
| `test_unmapped_columns_listed_for_review` | fixture | Every non-mapped column appears in `pending_review`. |
| `test_snapshot_manifest_is_sha256` | fixture | Every entry is 64-char hex SHA256; loading twice yields identical `source_version`. |
| `test_meta_file_has_required_keys` | fixture | `_meta/ingest_v1.json` has all 8 required keys. |
| `test_ingestion_is_idempotent` | fixture | Two back-to-back runs produce identical `source_version` and byte-identical JSON outputs. |
| `test_run_pipeline_imports_ingest` | fast | `import scripts.run_pipeline as rp; assert callable(rp.main)` works. |
| `test_fixture_ingestion_end_to_end` | fixture | Full `run_ingestion()` on fixture subtree writes all 5 JSON + 1 CSV + 1 `_meta/*.json`. |

### 7. `tests/fixtures/raw_snapshot_fixture.py` (~80 LOC)

- `def build_synthetic_raw(tmp_path: Path, *, with_facets: bool = True) -> Path`
  - Creates `tmp_path / "raw" / "gurgaon_10k.csv"` with 2 rows, columns including 1 PII-ish (`PHOTO_URL`) and 1 coded (`FURNISH=4`) column.
  - Same for `hyderabad.csv`, `kolkata.csv`, `mumbai.csv` with smaller shapes.
  - Creates `tmp_path / "raw" / "facets/"` with 15 minimal 2-row CSVs (`id,label`).
  - Returns the synthetic `data_dir`.
- `def build_synthetic_processed_dir(tmp_path: Path) -> Path` — same but yields the `output_dir`.
- Pytest fixture `synthetic_data_dir` and `synthetic_processed_dir` in `conftest.py` (or local to the fixture file).

### 8. `tests/conftest.py` (extend Step 01)

Add 2 fixtures:
- `synthetic_data_dir(tmp_path)` — wraps `build_synthetic_raw(tmp_path)` and yields the path.
- `synthetic_processed_dir(tmp_path)` — pairs the two.

## Key design decisions

- **City keys = facet labels**, not filenames. `gurgaon_10k.csv` decodes to `Gurgaon` (the only CITY facet row), the others by name. This matches the spec's `test_load_raw_listings_returns_four_cities` requirement and aligns with how every downstream stage will reference city.
- **Coverage join uses string-coerced matching**. Facet `008` (string) and raw `8` (int) won't match as `int`/`str` comparison — both are cast to `str` before matching. The resulting non-match is reported honestly (Gurgaon's `CITY_ID=8` vs `CITY.csv`'s `008,Gurgaon` is exactly the kind of gap the coverage report surfaces for Step 03 to resolve).
- **Idempotency** is anchored on the SHA256-of-the-manifest-hash being stable for identical raw bytes. Every output JSON is `sort_keys=True, indent=2, ensure_ascii=True` (spec rule).
- **Sample values are sorted strings** so byte-identical output across runs.
- **No write to `data/raw/`** is enforced by an in-orchestrator guard: snapshot mtimes before any read, after all reads, assert equal. If anything mutates, abort with `RuntimeError`.

## Verification

Run from repo root:

1. **Fixture-only fast path** (no real CSVs):
   ```bash
   python -m pytest tests/test_raw_ingestion.py -v -m "not realdata"
   ```
   Expected: 8–9 tests pass in <10s.

2. **Full suite including realdata** (requires the 4 CSVs on disk):
   ```bash
   python -m pytest tests/test_raw_ingestion.py -v
   ```
   Expected: all 14 tests pass. Realdata tests take ~1 min (10k-row coverage join × 60 combos).

3. **CLI smoke test**:
   ```bash
   python scripts/ingest_raw.py
   ```
   Expected: prints 4 city summary lines + total, exits 0.

4. **Pipeline entry point**:
   ```bash
   python scripts/run_pipeline.py
   ```
   Expected: prints `"ingest done — see data/processed/raw_inventory.json"`, exits 0.

5. **Output files exist and parse**:
   ```bash
   ls data/processed/raw_inventory.json data/processed/facet_inventory.json \
       data/processed/facet_join_coverage.csv data/processed/raw_schema_map.json \
       data/processed/raw_snapshot_manifest.json data/processed/_meta/ingest_v1.json
   python -c "import json; [json.load(open(p)) for p in [...]]"  # all parse
   ```

6. **Lint clean**:
   ```bash
   ruff check ml/cleaning/ingest.py scripts/ingest_raw.py scripts/run_pipeline.py \
              tests/test_raw_ingestion.py tests/fixtures/raw_snapshot_fixture.py
   ```
   Expected: zero issues.

7. **Raw immutability check** (manual or test):
   ```bash
   git status data/raw/   # must show no changes after a run
   ```
   The `test_load_raw_listings_does_not_modify_files` test enforces this in CI.

8. **Tracker update** (per spec DoD §9 — separate step):
   ```bash
   /update-tracker  # mark Week 1 Day 1 as Done with date and link to this spec
   ```

## Notes on scope (what this plan does NOT do)

- No actual column renaming or cleaning — Step 03's job. The `raw_schema_map.json` is a draft; Step 03 may add/change mappings.
- No parquet output, no model artifacts, no Flask/FastAPI route changes — those are downstream specs.
- No new pip dependencies — `pandas`, `numpy`, `hashlib`, `uuid`, `json`, `pathlib`, `subprocess`, `re`, `logging` cover everything.
- No documentation updates beyond what this spec itself dictates (`.gitignore` + `pyproject.toml` markers).
