# Spec: Data Deduplication and Outlier Flagging

## Overview
Build the deduplication + outlier-flagging layer that turns the per-city canonical frames emitted by Step 05 (`ml/cleaning/canonical_mapping.py`) into one consolidated, outlier-flagged, training-ready DataFrame that is the input to Step 07 (missing-value imputation) and ultimately written to `data/processed/clean_listings.parquet` in Step 07. This is Step 06 of the foundation module, Week 2 Day 12 of the Implementation Plan — it executes `02-TRD.md` §4.9 (dedup by `PROP_ID`) and §6 (outlier detection via 1st/99th percentile + IQR + domain rules), and produces the `is_outlier` column promised in `05-BACKEND-SCHEMA.md` §2. Module: **foundation**.

## Depends on
- **Step 05** — `05-canonical-schema-mapping-per-city` (`ml/cleaning/canonical_mapping.py`) — provides `CITY_FRAME_LOADERS`, `map_city`, `CANONICAL_COLUMNS`, `CITY_COLUMN_ALIASES`, `normalize_columns`, `clean_description`, and the per-city `map_<city>` functions.
- **Step 02** — `02-raw-data-ingestion-and-schema-inventory` (`ml/cleaning/ingest.py`) — provides `load_raw_city_frames()`, `assert_raw_readonly()`, and the read-only data/raw gate that this spec must re-use on every read.
- **Step 01** — `01-repo-scaffolding-and-environment-setup` — `pytest.ini`, `ruff.toml`, `tests/conftest.py`, `scripts/run_pipeline.py`.
- **Step 03** — `03-price-and-area-parsing-utilities` — `parse_price`/`parse_area` output is what feeds the outlier thresholds.

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python cleaning utilities + pytest coverage. Nothing in this spec is exposed via FastAPI or Flask.

## Data / Schema changes
- **No writes to `data/raw/`** — Rules §1.1, §1.2 are binding. `assert_raw_readonly()` (Step 02) gates every read path that touches the raw CSVs.
- **No writes to `data/processed/clean_listings.parquet` either** — that write is Step 07's responsibility. This spec only:
  - Reads from `data/raw/*.csv` via the Step 02 helpers.
  - Calls the Step 05 per-city mappers.
  - Concatenates, deduplicates, and flags outliers in-memory.
  - Returns a single `pd.DataFrame` from the public API.
  Step 07 picks up the output DataFrame, runs imputation, then writes the Parquet.
- **No new model artifacts.**
- **No application DB changes** — `data/app.db` (Step 01) is unchanged.
- **No new files under `tests/fixtures/`** — test suite uses literal DataFrames / dictionaries per the Step 02/03/04/05 pattern.

## Templates / UI
None.

## Files to change / Files to create

**Create:**
- `ml/cleaning/dedup.py` — the deduplication layer. Public API:
  - `DEDUP_KEY_COLUMN: str = "listing_id"` — single source of truth for which column drives dedup. All 4 city frames already carry a `listing_id` populated from raw `PROP_ID`/`PROPHEADID`/`PROPERTY_ID` by the Step 05 mapper.
  - `CONFLICT_TIEBREAKER_ORDER: tuple[str, ...] = ("nonnull_fields_count", "register_date", "row_order")` — when two rows share a `listing_id`, the row with the most non-null canonical fields wins; ties broken by most-recent `register_date`; final ties broken by stable row order (preserve first-seen).
  - `compute_nonnull_field_count(df: pd.DataFrame) -> pd.Series` — vectorized per-row count of non-null values across the documented canonical columns (not all 28 — uses the `CANONICAL_COLUMNS` tuple from `ml.cleaning.canonical_mapping`). Returned as an int Series aligned to `df.index`.
  - `deduplicate_listings(df: pd.DataFrame) -> pd.DataFrame` — main public function. Steps:
    1. Drop rows where `listing_id` is null/empty/NaN — log the count via `_log_dedup_drop`.
    2. Strip whitespace on `listing_id`, cast to str.
    3. Sort by `CONFLICT_TIEBREAKER_ORDER` (descending nnull, descending register_date, ascending row_order) within each `listing_id` group.
    4. `groupby("listing_id").first()` — keep the winning row.
    5. Log summary: input rows, dropped (no listing_id), dropped (duplicate listing_id), output rows.
  - `_log_dedup_drop(reason: str, count: int, total: int) -> None` — module-level stdlib `logging` helper (consistent with `ml.cleaning.parsing._log_unparseable` from Step 03).
  - Module docstring that references `02-TRD.md` §4.9 as the source for the dedup rule.

- `ml/cleaning/outliers.py` — the outlier-flagging layer. Public API:
  - `OUTLIER_NUMERIC_COLUMNS: tuple[str, ...] = ("price_inr", "area_sqft", "price_per_sqft")` — the three columns TRD §6 names for IQR + percentile capping.
  - `OUTLIER_DOMAIN_RULES: dict[str, dict]` — domain-rule caps keyed by column, e.g. `{"bedRoom": {"max": 15, "note": "unless property_type in (villa, farmhouse)"}, "bathroom": {"max": 15, "note": "same"}}`.
  - `OUTLIER_PROPERTY_TYPE_EXEMPTIONS: frozenset[str] = frozenset({"villa", "farmhouse", "independent house"})` — property types exempt from the bed/bath cap (per TRD §6.3).
  - `PERCENTILE_LOWER: float = 0.01`, `PERCENTILE_UPPER: float = 0.99` — per-city percentile bounds from TRD §6.1.
  - `IQR_MULTIPLIER: float = 1.5` — standard Tukey fence per TRD §6.2.
  - `flag_percentile_outliers(df: pd.DataFrame, column: str) -> pd.Series[bool]` — returns a bool Series aligned to `df.index`, `True` if the value is outside the per-city `[1st, 99th]` percentile. Uses `groupby("city")` so bounds are city-relative.
  - `flag_iqr_outliers(df: pd.DataFrame, column: str) -> pd.Series[bool]` — same shape, uses `Q1 − 1.5×IQR` / `Q3 + 1.5×IQR` per city group.
  - `flag_domain_rule_outliers(df: pd.DataFrame) -> pd.Series[bool]` — applies `OUTLIER_DOMAIN_RULES` per column; bedroom/bathroom cap is overridden for rows whose `property_type_label` is in `OUTLIER_PROPERTY_TYPE_EXEMPTIONS`.
  - `flag_all_outliers(df: pd.DataFrame) -> pd.DataFrame` — top-level helper. Returns the input frame with two new columns:
    - `is_outlier: bool` — `True` if any of the three flag types is True for that row.
    - `outlier_reasons: list[str]` — list of strings from `{"percentile_price_inr", "percentile_area_sqft", "percentile_price_per_sqft", "iqr_price_inr", "iqr_area_sqft", "iqr_price_per_sqft", "domain_bedRoom", "domain_bathroom"}` documenting why the row was flagged. Stored as a Python list per row (so the column dtype is `object`; an explicit `dtype=object` cast is applied).
  - `OUTLIER_REASON_COLUMN: str = "outlier_reasons"` — single source of truth for the reason-list column name.
  - `_log_outlier_summary(df: pd.DataFrame) -> None` — logs per-city outlier counts: `{city: {n_rows: int, n_outliers: int, pct: float}}`.
  - Module docstring that references `02-TRD.md` §6 verbatim and calls out that log1p target transformation is **out of scope** here (it's a Week 3 training-time concern per Day 13 of `06-IMPLEMENTATION-PLAN.md`).

- `ml/cleaning/assemble.py` — the small orchestrator that ties dedup + outlier flagging together, so Step 07 has a single entry point. Public API:
  - `assemble_cleaned_frame(raw_dir: Path, facet_dir: Path) -> pd.DataFrame` — the one function Step 07 will call. Steps:
    1. `assert_raw_readonly(raw_dir)` — Step 02 helper.
    2. `facet_frames = load_facet_frames(facet_dir)` — Step 04 helper.
    3. For each city in `CITY_FRAME_LOADERS` keys (Step 05): call `map_city(name, raw_dir / <filename>, facet_frames)`.
    4. `pd.concat([...], ignore_index=True)` — single canonical frame.
    5. `df = deduplicate_listings(df)` (dedup.py).
    6. `df = flag_all_outliers(df)` (outliers.py).
    7. Return the assembled DataFrame. Does NOT write to `data/processed/`.
  - `ASSEMBLE_CITY_FILES: dict[str, str]` — `{ "Gurgaon": "gurgaon_10k.csv", "Hyderabad": "hyderabad.csv", "Kolkata": "kolkata.csv", "Mumbai": "mumbai.csv" }` — same filenames the Step 02 inventory already documented.
  - `ASSEMBLE_REPORT_FIELDS: tuple[str, ...]` — what `assemble_cleaned_frame` logs at the end (rows_in, rows_dropped_no_listing_id, rows_dropped_duplicate, rows_in_after_dedup, rows_flagged_outlier, rows_in_after_outlier_flag, per_city_breakdown).

- `tests/test_dedup.py` — pytest unit tests for `ml/cleaning/dedup.py`, all using literal DataFrames (no real-data dependency). Exact test names listed in "Definition of done".

- `tests/test_outliers.py` — pytest unit tests for `ml/cleaning/outliers.py`, same pattern.

- `tests/test_assemble.py` — pytest unit tests for `ml/cleaning/assemble.py`, same pattern.

**Modify:**
- `ml/cleaning/__init__.py` — re-export `DEDUP_KEY_COLUMN`, `deduplicate_listings`, `OUTLIER_NUMERIC_COLUMNS`, `OUTLIER_DOMAIN_RULES`, `PERCENTILE_LOWER`, `PERCENTILE_UPPER`, `IQR_MULTIPLIER`, `flag_all_outliers`, `OUTLIER_REASON_COLUMN`, `assemble_cleaned_frame` so Step 07 and the API can import them by short name.
- `scripts/run_pipeline.py` — add a no-op wiring placeholder: `from ml.cleaning import assemble  # noqa: F401`. Same pattern as the Step 04/05 wiring. One-line comment explaining why.

**No changes** to:
- `requirements.txt` — `pandas` is already pinned; the new modules use only stdlib + pandas + Step 02/03/04/05 helpers.
- `.gitignore`, `pytest.ini`, `ruff.toml`, `app/`, `api/`, `data/`, `models/`, `notebooks/`, `scripts/ingest_raw.py`, `tests/conftest.py`, `ml/cleaning/parsing.py`, `ml/cleaning/facet_decoders.py`, `ml/cleaning/ingest.py`, `ml/cleaning/canonical_mapping.py`.

## New dependencies
No new dependencies. `pandas` (already in `requirements.txt`) and Python stdlib (`logging`, `dataclasses`) are the only imports.

## Rules for implementation
- **No SQLAlchemy/ORM.** N/A — no SQL is written.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** This spec inherits the Step 05 unsafe-column drop because it consumes Step 05's output — those columns are already gone before dedup/outliers run. Do not reintroduce them. The Step 05 `UNSAFE_COLUMNS` list (in `canonical_mapping.py`) is the source of truth; this spec does not redefine it.
- **Raw data is immutable.** `assert_raw_readonly()` is called inside `assemble_cleaned_frame` before any read; dedup/outliers modules never touch `data/raw/` directly — they receive a DataFrame as input. `data/processed/` is not written by this spec.
- **CSS variables only.** N/A — no templates or styles.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifacts produced. (Outlier flagging affects which rows the *future* model trains on, but no model is trained in this spec.)
- **Pure functions, deterministic, no side effects beyond logging.** Each public function is a pure function of its input DataFrame / paths. Calling `deduplicate_listings(same_df)` twice returns equal DataFrames (column-dtypes + values + ordering). Tested explicitly via `test_dedup_is_idempotent` and `test_flag_all_outliers_is_idempotent`.
- **City routing is by string lookup, not by column sniffing.** The orchestrator iterates `CITY_FRAME_LOADERS.keys()`; per-city logic is in Step 05, not here.
- **Per-city bounds, not global bounds.** Both `flag_percentile_outliers` and `flag_iqr_outliers` use `groupby("city")` so a Mumbai luxury flat and a Kolkata budget flat are evaluated against their own city's distribution, not a global one. This matches TRD §6.1 and §6.2.
- **Outlier rows are flagged, never deleted** — Rules §1.4 is binding. The `is_outlier` column is set on the row; the row is **retained** in the returned DataFrame. The training-time exclusion is Step 07+'s responsibility (or, more precisely, the modeling step in Week 3-4).
- **`outlier_reasons` is a list, not a delimited string.** A Python list per row keeps it grep-able, testable, and JSON-serializable for the eventual `data/processed/` Parquet writer. Stored under the `object` dtype.
- **Deduplication tiebreaker is documented, not magic.** The `CONFLICT_TIEBREAKER_ORDER` constant is the single place the policy lives. Tests assert that the order of inputs to `deduplicate_listings` is irrelevant — sorting is internal.
- **All randomness is seeded.** N/A — no randomness used (sorting is stable; groupby preserves order).
- **`luxury_category` and `was_missing_*` are NOT touched by this spec.** They remain NaN per Step 05's contract; Step 07 sets `was_missing_*` flags during imputation, and `luxury_category` is derived at feature-engineering time per Rules §10.2.
- **`transact_type` is preserved.** Sale/Rent strings pass through verbatim; this spec does not split into separate frames — that's the FastAPI routing concern per `02-TRD.md` §U-TRD-4.
- **No DataFrame writes.** `deduplicate_listings`, `flag_all_outliers`, and `assemble_cleaned_frame` return DataFrames; they do not call `to_parquet`, `to_csv`, `to_json`, or any other writer. Step 07 owns the first write.
- **No SQL row I/O.** N/A.
- **Logging uses the stdlib `logging` module, not `print()`.** Reuse the Step 03/`ml.cleaning.parsing` pattern. Each public function emits exactly one summary log line; per-row drops are aggregated and logged once at the end of the function.
- **Idempotency.** `deduplicate_listings(df) == deduplicate_listings(deduplicate_listings(df))` (no further changes on the second call) — tested. Same for `flag_all_outliers`.
- **No ML model evaluation, no SHAP, no insights templating, no recommender work.** All out of scope for this foundation step.
- **Reason-list column is JSON-safe.** Each reason in `outlier_reasons` is a member of a fixed set of lowercase_snake_case strings (the keys above). No free-form text. This guarantees the column round-trips through Parquet + any future JSON export without loss.

## Definition of done
A specific, testable checklist verifiable by running the test suite.

1. `python -m pytest tests/test_dedup.py tests/test_outliers.py tests/test_assemble.py -v` from repo root runs and passes. Tests required (exact names):

   **tests/test_dedup.py** (each test name is exact, must pass):
   - `test_dedup_key_column_constant_is_listing_id` — `DEDUP_KEY_COLUMN == "listing_id"`.
   - `test_dedup_drops_rows_with_null_listing_id` — given 5 rows, 2 with null `listing_id`, output has 3 rows and the null ones are logged.
   - `test_dedup_drops_rows_with_empty_string_listing_id` — whitespace-only `listing_id` is treated as null and dropped.
   - `test_dedup_strips_whitespace_on_listing_id` — `"  ABC123  "` is deduped against `"ABC123"`.
   - `test_dedup_casts_listing_id_to_string` — integer PROP_ID values are coerced to str before dedup.
   - `test_dedup_keeps_one_row_per_duplicate_listing_id` — given 3 rows with `listing_id="X"`, output has exactly 1.
   - `test_dedup_keeps_row_with_most_nonnull_fields` — two rows with same `listing_id`, row A has 5 populated fields, row B has 3 → output matches A.
   - `test_dedup_breaks_ties_by_most_recent_register_date` — same nonnull count, A has 2024-01-01, B has 2025-06-01 → B wins.
   - `test_dedup_breaks_final_ties_by_input_order` — same nonnull, same register_date (or both NaN), first input row wins.
   - `test_dedup_does_not_modify_non_listing_id_columns` — given a non-duplicate `listing_id="X"` row with extra fields, all extra fields are preserved verbatim.
   - `test_dedup_logs_summary` — using `caplog.set_level(logging.INFO)`, one log line with the input/dropped/output counts is emitted.
   - `test_dedup_is_idempotent` — `deduplicate_listings(deduuplicate_listings(df)) == deduplicate_listings(df)` under `pd.testing.assert_frame_equal`.
   - `test_dedup_preserves_listing_id_uniqueness_on_output` — every `listing_id` in the output is unique (`df["listing_id"].is_unique == True`).
   - `test_dedup_does_not_write_to_disk` — module source string contains no `to_parquet`, `to_csv`, `to_json`, or `open(` literals (ast scan).

   **tests/test_outliers.py:**
   - `test_outlier_numeric_columns_constant_matches_trd` — `OUTLIER_NUMERIC_COLUMNS == ("price_inr", "area_sqft", "price_per_sqft")` exactly, in that order.
   - `test_percentile_bounds_constants` — `PERCENTILE_LOWER == 0.01`, `PERCENTILE_UPPER == 0.99`.
   - `test_iqr_multiplier_constant` — `IQR_MULTIPLIER == 1.5`.
   - `test_flag_percentile_outliers_returns_bool_series` — output dtype is bool, length matches input index.
   - `test_flag_percentile_outliers_uses_per_city_bounds` — synthetic frame with 2 cities where only the city-A row is above A's 99th percentile and only the city-B row is below B's 1st: output is `[True, True, False, False]` in input order.
   - `test_flag_iqr_outliers_returns_bool_series` — same shape contract.
   - `test_flag_iqr_outliers_uses_per_city_bounds` — same per-city test shape.
   - `test_flag_domain_rule_outliers_flags_high_bedroom_count` — a row with `bedRoom=20` and `property_type_label="flat"` is flagged.
   - `test_flag_domain_rule_outliers_does_not_flag_villa_with_high_bedroom` — a row with `bedRoom=20` and `property_type_label="villa"` is NOT flagged.
   - `test_flag_domain_rule_outliers_does_not_flag_farmhouse` — `property_type_label="farmhouse"` with `bedRoom=25` is NOT flagged.
   - `test_flag_all_outliers_adds_is_outlier_column` — input frame without `is_outlier` gets a bool `is_outlier` column.
   - `test_flag_all_outliers_adds_outlier_reasons_column` — input frame gets an `outlier_reasons` column of dtype `object` containing lists.
   - `test_flag_all_outliers_row_not_flagged_has_empty_reason_list` — a clean row has `outlier_reasons == []` (empty list, not None, not NaN).
   - `test_flag_all_outliers_row_flagged_for_two_reasons_has_both` — a row triggering both percentile + IQR on the same column has both reason strings in its list.
   - `test_flag_all_outliers_reason_strings_are_from_documented_set` — every reason in the column is a member of the documented set (no free-form strings).
   - `test_flag_all_outliers_is_idempotent` — `flag_all_outliers(flag_all_outliers(df))` matches `flag_all_outliers(df)`.
   - `test_flag_all_outliers_logs_per_city_summary` — `caplog` captures one log line per city with the row count and flagged count.
   - `test_outlier_reasons_column_is_json_serializable` — `json.dumps(df["outlier_reasons"].iloc[0].tolist())` round-trips without error (verifies the column is JSON-safe for Parquet/JSON consumers).
   - `test_flag_all_outliers_preserves_other_columns` — every input column that is not `is_outlier`/`outlier_reasons` is unchanged in the output (asserted by `pd.testing.assert_frame_equal` on the slice).

   **tests/test_assemble.py:**
   - `test_assemble_city_files_constant_has_four_entries` — `ASSEMBLE_CITY_FILES` has exactly the 4 expected keys/values.
   - `test_assemble_cleaned_frame_does_not_write_to_data_processed` — given a stubbed raw_dir + facet_dir, the function returns a DataFrame but `data/processed/` (the real one) is untouched (verified by checking that no new files appear in it via `os.listdir` before/after).
   - `test_assemble_cleaned_frame_asserts_raw_readonly` — given a raw_dir where `assert_raw_readonly` would fail, the function raises the same exception the gate raises (per Step 02 contract).
   - `test_assemble_cleaned_frame_returns_dataframe_with_all_canonical_columns` — output `df.columns.tolist()` equals `list(CANONICAL_COLUMNS)` (in the canonical order from Step 05), no missing, no extras.
   - `test_assemble_cleaned_frame_has_listing_id_unique` — `df["listing_id"].is_unique == True` after dedup runs inside the assembler.
   - `test_assemble_cleaned_frame_has_is_outlier_column` — `is_outlier` column exists, dtype bool, contains both True and False values in a synthetic multi-city input.
   - `test_assemble_cleaned_frame_has_outlier_reasons_column` — `outlier_reasons` column exists, dtype object, contains a mix of empty and non-empty lists.
   - `test_assemble_cleaned_frame_logs_summary` — `caplog` captures the summary line with `rows_in`, `rows_dropped_no_listing_id`, `rows_dropped_duplicate`, `rows_in_after_dedup`, `rows_flagged_outlier`, `rows_in_after_outlier_flag`, and a per-city breakdown.
   - `test_assemble_cleaned_frame_is_pure_no_io` — called twice with the same args returns equal DataFrames; no state is stashed on the module.
   - `test_assemble_does_not_import_app_or_api` — `ml.cleaning.assemble` module-level imports do not include anything from `app.*` or `api.*` (asserted via `sys.modules` introspection or a static `ast` scan).
   - `test_assemble_canonical_mapping_does_not_touch_filesystem_outside_data_raw` — module source string contains no `to_parquet`, `to_csv`, `to_json`, or `Path("data/processed"` literals (ast scan).

2. `python -m pytest -m "not realdata"` from repo root still passes — confirms no real-data dependency was accidentally introduced.
3. `ruff check ml/cleaning/dedup.py ml/cleaning/outliers.py ml/cleaning/assemble.py tests/test_dedup.py tests/test_outliers.py tests/test_assemble.py` reports zero issues.
4. `python -c "from ml.cleaning.dedup import DEDUP_KEY_COLUMN, deduplicate_listings; from ml.cleaning.outliers import OUTLIER_NUMERIC_COLUMNS, flag_all_outliers, OUTLIER_REASON_COLUMN; from ml.cleaning.assemble import assemble_cleaned_frame, ASSEMBLE_CITY_FILES; print(DEDUP_KEY_COLUMN, len(OUTLIER_NUMERIC_COLUMNS), OUTLIER_REASON_COLUMN, len(ASSEMBLE_CITY_FILES))"` from repo root prints `listing_id 3 outlier_reasons 4` — confirms the public API imports cleanly.
5. `python -c "import pandas as pd; from pathlib import Path; from ml.cleaning.assemble import assemble_cleaned_frame; df = assemble_cleaned_frame(Path('data/raw'), Path('data/raw/facets')); print(df.shape, df['is_outlier'].sum(), df['listing_id'].is_unique, set(df['outlier_reasons'].explode().dropna().unique()))"` from repo root, when run against the real 4-city dataset (~182k rows), completes without error, prints a shape whose row count equals the input count minus dedup drops, prints an `is_outlier` count greater than 0, prints `True` for listing_id uniqueness, and prints a set of reason strings that is a subset of the documented set. (This is the **only** DoD item that touches real data — gated on the previous tests passing, so it doesn't slow CI.)
6. `git status` after committing the spec, the three new modules, the three new test files, and the modified `scripts/run_pipeline.py` + `ml/cleaning/__init__.py` shows only those files changed (plus this spec). No accidental additions to `data/processed/`, `data/raw/`, `models/`, or `notebooks/`.
7. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes.
8. `07-TRACKER.md` "Week 2 — EDA & Outlier Handling" Day 12 row's status is updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.