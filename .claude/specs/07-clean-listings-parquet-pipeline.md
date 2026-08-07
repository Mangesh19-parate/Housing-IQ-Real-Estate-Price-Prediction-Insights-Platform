# Spec: Clean Listings Parquet Pipeline

## Overview
Build the missing-value imputation + Parquet-writer layer that consumes the
deduplicated, outlier-flagged DataFrame emitted by Step 06's
`assemble_cleaned_frame()` and turns it into the canonical
`data/processed/clean_listings.parquet` — the single training-and-serving
artifact referenced by every downstream consumer (feature engineering, price
regression, classification, recommender, analytics, insights). This is Step 07
of the foundation module, Week 1 Day 6 + Day 7 (checkpoint) of the
Implementation Plan, and it executes `02-TRD.md` §5 (4-tier missing-value
strategy) plus the canonical-schema artifact contract from
`05-BACKEND-SCHEMA.md` §1 and §2. Module: **foundation**.

## Depends on
- **Step 06** — `06-data-deduplication-and-outlier-flagging` —
  `ml/cleaning/assemble.py` provides `assemble_cleaned_frame()`, which
  returns the deduped + outlier-flagged canonical DataFrame this spec
  consumes.
- **Step 05** — `05-canonical-schema-mapping-per-city` —
  `ml/cleaning/canonical_mapping.py` provides `CANONICAL_COLUMNS`
  (single source of truth for column order) and the per-city mappers
  that feed the assembler.
- **Step 03** — `03-price-and-area-parsing-utilities` — `parse_price` /
  `parse_area` are needed by Step 06, which we consume; the imputation
  layer trusts their outputs.
- **Step 04** — `04-facet-decoding-joins` — `load_facet_frames`,
  `DEFAULT_UNKNOWN_LABEL`; needed by Step 06, used transitively.
- **Step 02** — `02-raw-data-ingestion-and-schema-inventory` —
  `assert_raw_readonly()`, `load_raw_listings`; needed by Step 06 and
  re-checked here for symmetry.
- **Step 01** — `01-repo-scaffolding-and-environment-setup` —
  `pytest.ini`, `ruff.toml`, `tests/conftest.py`, `scripts/run_pipeline.py`,
  `.gitignore` (already excludes the generated Parquet).

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python cleaning
utilities + pytest coverage. The Parquet is a data artifact, not an API.
Loading of the Parquet by FastAPI / Flask happens in later specs
(Week 4+).

## Data / Schema changes
- **Write** to `data/processed/clean_listings.parquet` (the Parquet is
  already excluded from git via `.gitignore` — see line 55 — so this is
  the first spec that produces a tracked artifact even though the file
  itself is gitignored; CI regenerates it). This is the **only** spec
  allowed to write this file.
- **No writes to `data/raw/`** — Rules §1.1, §1.2 binding. The writer
  re-invokes `assert_raw_readonly()` even though it never touches raw
  CSVs directly — symmetry gate.
- **No writes to `data/processed/analytics_cache/`** — that lives in
  Week 6 (Step 36).
- **No application DB changes** — `data/app.db` is untouched.
- **No new model artifacts** — the Parquet is the dataset artifact, not
  a model.
- **No new tables or columns beyond the canonical schema** — every
  output column is in `CANONICAL_COLUMNS` (Step 05) + `is_outlier`
  (Step 06) + the `was_missing_<field>` columns this spec adds. Column
  order in the Parquet matches `CANONICAL_COLUMNS` ordering with the
  derived/flag columns appended in a documented order.

## Templates / UI
None.

## Files to change / Files to create

**Create:**
- `ml/cleaning/imputation.py` — the missing-value imputation layer.
  Public API:
  - `MISSINGNESS_LOW_THRESHOLD: float = 0.05`,
    `MISSINGNESS_MEDIUM_THRESHOLD: float = 0.40`,
    `MISSINGNESS_HIGH_THRESHOLD: float = 0.70` — the four TRD §5
    missingness-tier boundaries as documented constants (single
    source of truth; re-referenced by tests).
  - `IMPUTATION_LOW_TIER_COLUMNS: tuple[str, ...]` — the columns
    imputed with global median (numeric) / mode (categorical) under
    the <5% tier. Derived as the subset of `CANONICAL_COLUMNS` whose
    corpus missingness is < 5% at runtime; the literal tuple is
    produced by `classify_missingness_tiers(df)` and stored on the
    returned dict so tests can pin it.
  - `IMPUTATION_NUMERIC_LOW: tuple[str, ...] = ("balconies", "floor_num_int", "total_floor", "area_sqft")` — the numeric columns eligible for global-median imputation under <5%. Columns not present (e.g. `floor_num_int` if absent in a city) are skipped silently.
  - `IMPUTATION_CATEGORICAL_LOW: tuple[str, ...] = ("furnish", "facing", "ownership_type", "age_bucket", "property_type")` — categorical columns eligible for global-mode imputation under <5%.
  - `IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS: tuple[str, ...] = ("price_inr", "price_per_sqft", "bedrooms", "bathrooms")` — the 5–40% tier; group-wise median/mode by `(city, locality, property_type)` triples.
  - `IMPUTATION_HIGH_TIER_COLUMNS: tuple[str, ...]` — 40–70% tier; filled with a literal `"Unknown"` category for categoricals, or left NaN with a `was_missing_*` flag for numerics (configurable per column; default = "Unknown" for strings, NaN-flag for numerics).
  - `IMPUTATION_DROP_THRESHOLD: float = 0.70` — columns above this missingness are dropped and logged (NOT silently).
  - `classify_missingness_tiers(df: pd.DataFrame) -> dict[str, list[str]]` — returns `{ "low": [...], "medium": [...], "high": [...], "drop": [...] }` keyed by the canonical column name, given a frame. The classification is computed against the **input** frame, not the post-imputation frame.
  - `add_was_missing_flags(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame` — adds one `was_missing_<column>` bool column per imputed column (only for columns that actually had NaNs in the input). Returns a new frame.
  - `impute_low_tier(df: pd.DataFrame) -> pd.DataFrame` — applies global median (numeric) / mode (categorical) for the columns listed in `IMPUTATION_NUMERIC_LOW` / `IMPUTATION_CATEGORICAL_LOW` that exist on the input frame and are in the `<5%` tier. Returns a new frame.
  - `impute_medium_tier(df: pd.DataFrame) -> pd.DataFrame` — applies group-wise median/mode by `(city, locality, property_type)` for the columns in `IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS` that are in the `5–40%` tier. Falls back to global median/mode when a group has zero rows. Returns a new frame.
  - `impute_high_tier(df: pd.DataFrame) -> pd.DataFrame` — fills 40–70% tier strings with `"Unknown"`; for numerics, leaves NaN and lets the flag column carry the signal. Returns a new frame.
  - `drop_high_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` — drops columns above `IMPUTATION_DROP_THRESHOLD`. Returns `(df_after_drop, dropped_columns)`. Dropped columns are logged.
  - `impute_missing_values(df: pd.DataFrame) -> pd.DataFrame` — top-level helper. Steps:
    1. `tiers = classify_missingness_tiers(df)`.
    2. `df, dropped = drop_high_missing_columns(df)` — drops >70% columns.
    3. Re-classify missingness on the slimmed frame (so dropped columns don't distort the percentages).
    4. `df = add_was_missing_flags(df, columns=tiers["medium"] + tiers["high"])` — flags created BEFORE imputation, so they remain `True` for rows that were imputed.
    5. `df = impute_low_tier(df)`.
    6. `df = impute_medium_tier(df)`.
    7. `df = impute_high_tier(df)`.
    8. Return the imputed frame. Pure function; no I/O.
  - `_log_imputation_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, dropped: list[str], tiers: dict) -> None` — stdlib `logging` helper. One summary line: tier counts, dropped column names, `was_missing_*` flag count, total NaNs before/after.
  - Module docstring referencing `02-TRD.md` §5 verbatim and noting that this spec is **training-set shaping only** — `log1p(price_inr)` and the actual feature engineering are downstream (Week 3).

- `ml/cleaning/writers.py` — the Parquet-writer layer. Public API:
  - `CLEAN_LISTINGS_PARQUET_PATH: Path = Path("data/processed/clean_listings.parquet")` — single source of truth for the output path (referenced by tests and by `scripts/run_pipeline.py`).
  - `CLEAN_LISTINGS_DATASET_VERSION: str = "v1"` — version tag (used in the sidecar metadata file, below).
  - `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER: tuple[str, ...]` — the deterministic column order written to the Parquet. Built from `CANONICAL_COLUMNS` (Step 05) + `is_outlier` (Step 06) + `was_missing_<...>` (this spec, alphabetically sorted) + `outlier_reasons` (Step 06).
  - `write_clean_listings_parquet(df: pd.DataFrame, output_path: Path | None = None) -> Path` — public function. Steps:
    1. Assert `output_path` (or the default `CLEAN_LISTINGS_PARQUET_PATH`) parent directory exists; `mkdir(parents=True, exist_ok=True)` if missing.
    2. Reorder columns to `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER`; drop any columns not in that order (with a logged warning — defensive; should be a no-op in practice).
    3. Cast `outlier_reasons` column from `object` of Python lists to a Parquet-friendly representation (lists round-trip natively in pyarrow ≥ 8, but verify in the writer).
    4. `df.to_parquet(output_path, index=False, engine="pyarrow")`.
    5. Write a sidecar metadata file at `output_path.with_suffix(output_path.suffix + ".meta.json")` containing `{ "dataset_version": CLEAN_LISTINGS_DATASET_VERSION, "row_count": int, "column_count": int, "columns": [...], "computed_at_utc": "<ISO8601>", "source_raw_files": [...] }`. The metadata satisfies the Rules §1.5 ("every derived table … states its computation date and source dataset version").
    6. Return the path written.
  - `read_clean_listings_parquet(path: Path | None = None) -> pd.DataFrame` — round-trip reader. Used by tests to verify the writer. Defaults to `CLEAN_LISTINGS_PARQUET_PATH`.
  - `verify_clean_listings_parquet(path: Path | None = None) -> dict` — sanity check: returns `{ "exists": bool, "row_count": int, "column_count": int, "columns_match_canonical_order": bool, "listing_id_unique": bool, "has_is_outlier": bool, "has_was_missing_columns": bool }`. Used by `scripts/run_pipeline.py` after the write.
  - Module docstring referencing `05-BACKEND-SCHEMA.md` §1 (clean_listings.parquet is the canonical dataset) and §2 (canonical schema + `is_outlier` + `was_missing_<field>` columns).

- `ml/cleaning/pipeline.py` — the small orchestrator that ties
  imputation + Parquet write into one end-to-end function, so the
  pipeline runner and tests have a single entry point. Public API:
  - `run_clean_listings_pipeline(raw_dir: Path, facet_dir: Path, output_path: Path | None = None, persist: bool = True) -> pd.DataFrame` — the one function the pipeline runner calls. Steps:
    1. `df_before = assemble_cleaned_frame(raw_dir, facet_dir)` — Step 06 entry point.
    2. `df_imputed = impute_missing_values(df_before)` — this spec.
    3. If `persist`: `written_path = write_clean_listings_parquet(df_imputed, output_path)` — this spec.
    4. Return `df_imputed` regardless of `persist` (so tests can call with `persist=False` and avoid touching the actual Parquet path).
  - `PIPELINE_REPORT_FIELDS: tuple[str, ...]` — what the pipeline logs at the end (rows_in, rows_dropped_dedup, rows_dropped_outlier_flag, rows_dropped_high_missing_columns, rows_in_after_imputation, parquet_path, dataset_version, computed_at_utc).

- `tests/test_imputation.py` — pytest unit tests for `ml/cleaning/imputation.py`, all using literal DataFrames (no real-data dependency). Exact test names listed in "Definition of done".

- `tests/test_writers.py` — pytest unit tests for `ml/cleaning/writers.py`, using `tmp_path` fixtures for Parquet I/O. Exact test names listed in "Definition of done".

- `tests/test_pipeline.py` — pytest unit tests for `ml/cleaning/pipeline.py`, stubbing `raw_dir` and `facet_dir` paths and using `tmp_path` for the output Parquet. Exact test names listed in "Definition of done".

**Modify:**
- `ml/cleaning/__init__.py` — re-export `MISSINGNESS_LOW_THRESHOLD`, `MISSINGNESS_MEDIUM_THRESHOLD`, `MISSINGNESS_HIGH_THRESHOLD`, `IMPUTATION_DROP_THRESHOLD`, `IMPUTATION_NUMERIC_LOW`, `IMPUTATION_CATEGORICAL_LOW`, `IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS`, `classify_missingness_tiers`, `impute_missing_values`, `add_was_missing_flags`, `drop_high_missing_columns`, `CLEAN_LISTINGS_PARQUET_PATH`, `CLEAN_LISTINGS_DATASET_VERSION`, `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER`, `write_clean_listings_parquet`, `read_clean_listings_parquet`, `verify_clean_listings_parquet`, `run_clean_listings_pipeline`, `PIPELINE_REPORT_FIELDS` so the API layer can import them by short name.
- `scripts/run_pipeline.py` — add a `run_clean_listings_pipeline(...)` invocation after the existing Step 06 wiring. One-line comment explaining why this is the canonical Parquet producer. Same `noqa: F401` pattern as the prior wiring.

**No changes** to:
- `requirements.txt` — `pandas` and `pyarrow` are already pinned. (If `pyarrow` is missing, the spec flags it explicitly as a new dependency and the test for the writer will skip with a clear message — but it's expected to be present.)
- `.gitignore`, `pytest.ini`, `ruff.toml`, `app/`, `api/`, `models/`, `notebooks/`, `data/`, `tests/conftest.py`, prior cleaning modules (`ingest.py`, `parsing.py`, `facet_decoders.py`, `canonical_mapping.py`, `dedup.py`, `outliers.py`, `assemble.py`).

## New dependencies
None expected. `pandas` and `pyarrow` are already in `requirements.txt` per
Step 01. If `pyarrow` is missing for any reason, the writer test will skip
with `pytest.importorskip("pyarrow")` — the spec does not require adding a
new package, but flags this for the implementer to verify before merging.

## Rules for implementation
- **No SQLAlchemy/ORM.** N/A — no SQL is written.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** This spec inherits the Step 05 unsafe-column drop because the dedup+outlier assembler consumes Step 05's output — those columns are already gone before imputation/writing run. Do not reintroduce them. The `was_missing_*` flags this spec adds apply to canonical columns only, never to unsafe ones.
- **Raw data is immutable.** `assert_raw_readonly()` is re-invoked inside `run_clean_listings_pipeline` before reading raw (via `assemble_cleaned_frame` which already gates), and again inside the writer for symmetry. `data/raw/` is never written.
- **CSS variables only.** N/A — no templates or styles.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifact produced. (The Parquet is the *training input*; its evaluation protocol is the Step 26+ model's concern.)
- **Pure functions, deterministic, no side effects beyond logging and the optional Parquet write.** Each public function is a pure function of its input DataFrame / paths. Calling `impute_missing_values(same_df)` twice returns equal DataFrames (column-dtypes + values + ordering). Tested via `test_imputation_is_idempotent`.
- **`was_missing_*` flags are set BEFORE imputation.** The flag must capture "this row's value was missing at imputation time," not "is missing now." Imputation fills the value; the flag stays `True`. This is the Step 03 pattern applied to multiple columns.
- **Group-wise imputation falls back gracefully.** When `(city, locality, property_type)` group has zero non-null values for a column, fall back to global median/mode; never propagate NaN silently. Logged once per fallback via `_log_imputation_summary`.
- **Dropped columns are logged, not silently buried.** Any column above `IMPUTATION_DROP_THRESHOLD` (70%) is dropped AND logged with its name + missingness percentage, per Rules §10.4 ("Drop column entirely — documented explicitly, not silently").
- **Tier boundaries are constants, not magic numbers.** `MISSINGNESS_LOW_THRESHOLD` / `MEDIUM` / `HIGH` / `DROP` are the single place the policy lives. Tests assert the constants match `02-TRD.md` §5.
- **Column order is deterministic and pinned.** The Parquet column order is `CANONICAL_COLUMNS` (Step 05) → `is_outlier` (Step 06) → `was_missing_*` (alphabetical, this spec) → `outlier_reasons` (Step 06). Pinned via `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER` and asserted by the writer test.
- **Parquet sidecar metadata is written alongside the dataset.** Per Rules §1.5, every derived artifact states its computation date and source dataset version. The sidecar `<path>.meta.json` satisfies this and is what `verify_clean_listings_parquet` reads.
- **Parquet is gitignored.** `data/processed/clean_listings.parquet` is already in `.gitignore` line 55; the writer respects this (writes to disk; never `git add`-ed). Tests use `tmp_path` so they never touch the real path.
- **`outlier_reasons` list dtype survives the round-trip.** pyarrow ≥ 8 round-trips Python lists in object columns natively, but the writer test explicitly asserts `df["outlier_reasons"].iloc[0] == [original_list_value]` after a write + read cycle.
- **No SQL row I/O.** N/A.
- **Logging uses stdlib `logging`, not `print()`.** Reuse the Step 03/06 `ml.cleaning.parsing` pattern. Each public function emits one summary log line; per-row drops are aggregated and logged once.
- **Idempotency.** `impute_missing_values(df) == impute_missing_values(impute_missing_values(df))` — tested. Same for the writer (writing the same DataFrame twice produces identical bytes, modulo mtime).
- **No ML model evaluation, no SHAP, no insights templating, no recommender work.** All out of scope for this foundation step.
- **No DataFrame writes from `impute_missing_values`.** The pure imputation function never calls `to_parquet` / `to_csv` / `to_json`. The writer is the only writer; `run_clean_listings_pipeline(persist=False)` exercises the pure path.
- **`is_outlier` rows are retained, never dropped.** This is consistent with Rules §1.4 (flagged, never deleted) and with Step 06. The Parquet contains all rows; downstream training scripts filter on `is_outlier == False` at modeling time.
- **`transact_type` is preserved.** Sale/Rent strings pass through verbatim; the imputation layer never splits by it.

## Definition of done
A specific, testable checklist verifiable by running the test suite.

1. `python -m pytest tests/test_imputation.py tests/test_writers.py tests/test_pipeline.py -v` from repo root runs and passes. Tests required (exact names):

   **tests/test_imputation.py** (each test name is exact, must pass):
   - `test_missingness_threshold_constants_match_trd` — all four `MISSINGNESS_*` / `IMPUTATION_DROP_THRESHOLD` constants match TRD §5 (0.05 / 0.40 / 0.70 / 0.70).
   - `test_classify_missingness_tiers_returns_four_keys` — output dict has exactly the keys `low`, `medium`, `high`, `drop`, all lists.
   - `test_classify_missingness_tiers_low_under_5pct` — a synthetic frame where one column is 4% missing classifies it under `low`.
   - `test_classify_missingness_tiers_medium_between_5_and_40pct` — 20% missing → `medium`.
   - `test_classify_missingness_tiers_high_between_40_and_70pct` — 50% missing → `high`.
   - `test_classify_missingness_tiers_drop_above_70pct` — 80% missing → `drop`.
   - `test_classify_missingness_tiers_uses_input_frame_not_imputed` — the classifier is called on the input frame, NOT after imputation (asserted by giving it an input where a column is 60% missing and confirming it's classified `high`, not `low` even though it would become 0% missing after a fill).
   - `test_impute_low_tier_numeric_uses_global_median` — a numeric column with <5% NaN is filled with the global median.
   - `test_impute_low_tier_categorical_uses_global_mode` — same for a categorical column.
   - `test_impute_low_tier_no_op_for_columns_not_in_tier` — a column with 50% NaN is NOT touched by `impute_low_tier` (still has NaNs afterwards).
   - `test_impute_medium_tier_uses_groupwise_median` — for a 5–40% missing numeric column, values are filled using `(city, locality, property_type)` group medians, not global medians (verified by constructing a frame where the global median would be a wrong answer and asserting the group median is used).
   - `test_impute_medium_tier_falls_back_to_global_when_group_empty` — when a `(city, locality, property_type)` group has zero non-null values for the column, the global median is used and the fact is logged.
   - `test_impute_high_tier_categorical_filled_with_unknown` — a 40–70% missing string column is filled with the literal `"Unknown"` value.
   - `test_impute_high_tier_numeric_left_nan_with_flag` — a 40–70% missing numeric column is left NaN and a `was_missing_<col>` flag carries the signal.
   - `test_drop_high_missing_columns_drops_above_70pct` — a column at 80% missing is removed; the returned `dropped` list contains it.
   - `test_drop_high_missing_columns_logs_dropped` — `caplog` captures the dropped column name and its missingness percentage.
   - `test_add_was_missing_flags_creates_one_flag_per_imputed_column` — given 3 columns with NaNs, 3 `was_missing_*` columns are added.
   - `test_add_was_missing_flags_are_set_before_imputation` — running `add_was_missing_flags` then `impute_low_tier` leaves the flag `True` for the row whose value was filled.
   - `test_add_was_missing_flags_does_not_create_flag_for_column_without_nans` — a column with zero NaNs does NOT get a `was_missing_*` sibling.
   - `test_impute_missing_values_is_idempotent` — `impute_missing_values(impute_missing_values(df)) == impute_missing_values(df)` under `pd.testing.assert_frame_equal`.
   - `test_impute_missing_values_logs_summary` — `caplog.set_level(logging.INFO)` captures one summary line with tier counts, dropped column names, and total NaNs before/after.
   - `test_impute_does_not_write_to_disk` — module source string contains no `to_parquet`, `to_csv`, `to_json`, or `open(` literals (ast scan).

   **tests/test_writers.py** (uses `tmp_path` fixture):
   - `test_clean_listings_parquet_path_constant_default` — `CLEAN_LISTINGS_PARQUET_PATH` is `Path("data/processed/clean_listings.parquet")` exactly.
   - `test_clean_listings_parquet_columns_order_is_deterministic` — the constant has the documented order (canonical → `is_outlier` → `was_missing_*` alphabetical → `outlier_reasons`).
   - `test_write_clean_listings_parquet_creates_file` — given a minimal valid frame and `tmp_path`, the Parquet file exists after `write_clean_listings_parquet`.
   - `test_write_clean_listings_parquet_creates_sidecar_meta_json` — alongside the Parquet, a `.meta.json` file exists with `dataset_version`, `row_count`, `column_count`, `columns`, `computed_at_utc`, `source_raw_files` keys.
   - `test_write_clean_listings_parquet_writes_in_canonical_column_order` — `pd.read_parquet(written_path).columns.tolist()` matches `list(CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER)`.
   - `test_write_clean_listings_parquet_round_trip_outlier_reasons_list` — after write + read, `df["outlier_reasons"].iloc[0]` equals the original list.
   - `test_write_clean_listings_parquet_round_trip_preserves_was_missing_flags` — `was_missing_*` bool columns survive the round-trip with correct True/False distribution.
   - `test_write_clean_listings_parquet_returns_path` — return value equals the path that was written.
   - `test_read_clean_listings_parquet_round_trip` — read after write returns the same DataFrame (modulo pyarrow dtype coercions, which the test tolerates via `check_dtype=False` on the assertion).
   - `test_verify_clean_listings_parquet_passes_after_write` — `verify_clean_listings_parquet(tmp_path_parquet)` returns a dict with `exists=True`, `listing_id_unique=True`, `has_is_outlier=True`, `has_was_missing_columns=True`, `columns_match_canonical_order=True`.
   - `test_verify_clean_listings_parquet_fails_for_missing_file` — given a non-existent path, returns `exists=False`.
   - `test_writers_module_does_not_touch_data_raw` — module-level imports do not include `ml.cleaning.ingest` write paths; no `Path("data/raw"` literal in source (ast scan).

   **tests/test_pipeline.py** (uses `tmp_path` + stubbed `raw_dir`/`facet_dir`):
   - `test_pipeline_report_fields_constant_has_expected_keys` — `PIPELINE_REPORT_FIELDS` includes `rows_in`, `rows_in_after_imputation`, `parquet_path`, `dataset_version`, `computed_at_utc` at minimum.
   - `test_run_clean_listings_pipeline_returns_dataframe` — calling it returns a DataFrame.
   - `test_run_clean_listings_pipeline_persist_false_does_not_write` — calling with `persist=False` and the default `output_path` does NOT touch `data/processed/clean_listings.parquet` (verified by checking the file's mtime before/after).
   - `test_run_clean_listings_pipeline_persist_true_writes_parquet` — calling with `persist=True` and a `tmp_path` output writes a Parquet at that path with the expected row count.
   - `test_run_clean_listings_pipeline_logs_report_fields` — `caplog` captures one summary log line containing every key in `PIPELINE_REPORT_FIELDS`.
   - `test_run_clean_listings_pipeline_asserts_raw_readonly` — given a `raw_dir` where `assert_raw_readonly` would fail, the pipeline raises the gate's exception.
   - `test_run_clean_listings_pipeline_is_pure_no_io_when_persist_false` — called twice with the same args (and `persist=False`) returns equal DataFrames; no module-level state is stashed.
   - `test_run_clean_listings_pipeline_does_not_import_app_or_api` — `ml.cleaning.pipeline` module-level imports do not include anything from `app.*` or `api.*` (ast scan / `sys.modules` check).
   - `test_run_clean_listings_pipeline_handles_already_imputed_input` — calling it twice in sequence (both with `persist=True` to `tmp_path1` and `tmp_path2`) produces the same row count and column set; idempotent end-to-end.

2. `python -m pytest -m "not realdata"` from repo root still passes — confirms no real-data dependency was accidentally introduced into the test suite.
3. `ruff check ml/cleaning/imputation.py ml/cleaning/writers.py ml/cleaning/pipeline.py tests/test_imputation.py tests/test_writers.py tests/test_pipeline.py` reports zero issues.
4. `python -c "from ml.cleaning.imputation import MISSINGNESS_LOW_THRESHOLD, MISSINGNESS_MEDIUM_THRESHOLD, MISSINGNESS_HIGH_THRESHOLD, IMPUTATION_DROP_THRESHOLD, impute_missing_values; from ml.cleaning.writers import CLEAN_LISTINGS_PARQUET_PATH, CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER, write_clean_listings_parquet; from ml.cleaning.pipeline import run_clean_listings_pipeline; print(MISSINGNESS_LOW_THRESHOLD, MISSINGNESS_MEDIUM_THRESHOLD, MISSINGNESS_HIGH_THRESHOLD, IMPUTATION_DROP_THRESHOLD, CLEAN_LISTINGS_PARQUET_PATH, len(CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER))"` from repo root prints `0.05 0.4 0.7 0.7 data/processed/clean_listings.parquet <n>` where `<n>` is the documented column count (≥ 28).
5. `python -c "from pathlib import Path; from ml.cleaning.pipeline import run_clean_listings_pipeline; df = run_clean_listings_pipeline(Path('data/raw'), Path('data/raw/facets'), persist=False); print(df.shape, df['is_outlier'].sum(), df['listing_id'].is_unique, df.filter(like='was_missing_').shape[1])"` from repo root, when run against the real 4-city dataset (~182k rows), completes without error, prints a shape, an `is_outlier` count, `True` for `listing_id.is_unique`, and the count of `was_missing_*` columns produced. (This is the **only** DoD item that touches real data — gated on the previous tests passing, so it doesn't slow CI.)
6. `git status` after committing the spec, the three new modules, the three new test files, and the modified `scripts/run_pipeline.py` + `ml/cleaning/__init__.py` shows only those files changed (plus this spec). The Parquet itself (`data/processed/clean_listings.parquet`) and its sidecar (`.meta.json`) MUST remain gitignored — verified by `git status` not listing them. No accidental additions to `data/raw/`, `models/`, `notebooks/`, or `app/`.
7. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes.
8. `07-TRACKER.md` "Week 1 — Data Understanding & Cleaning" Day 6 and Day 7 rows' statuses are updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.