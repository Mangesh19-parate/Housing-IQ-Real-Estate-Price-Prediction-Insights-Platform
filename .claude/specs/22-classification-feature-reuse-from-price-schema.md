# Spec: Classification Feature Reuse From Price Schema

## Overview
Bridge the gap between the price-prediction feature pipeline (Step 12) and
the Classification module (Week 8) by defining a **pure, leakage-safe
classifier feature builder** that reuses the price model's feature frame,
**minus the price-derived columns** (`price`, `price_per_sqft`,
`locality_*`), per `08-RULES.md` §8.1 / §12.4 and the
`model-training-classification` skill. This spec ships no classifier and
no trained model — it ships the deterministic feature DataFrame that
**both** upcoming specs (Day 52: classifier training, Day 53: SHAP on the
winning classifier) consume. The output is a single module
`ml/classification/features.py` exposing `build_classifier_feature_frame`
plus a CLI `scripts/build_classifier_features.py` that materializes the
canonical `data/processed/classifier/feature_frame.parquet` and a
companion `feature_frame_report.md` (column list, dropped-column
rationale, per-city non-outlier row counts) so Day 52 has one fixed path
to read from and a reviewer can audit what was kept vs. dropped.

Module: **classification**.

## Depends on
- **Step 21** — `21-classification-target-definition` — supplies the
  classification target files this spec consumes (and adds a
  `listing_id`-aligned feature frame alongside):
  - `data/processed/classifier/price_tier_labels.parquet`
  - `data/processed/classifier/good_deal_labels.parquet`
  Both have a `listing_id` primary key that this spec joins onto
  `build_feature_frame`'s output so Day 52 can do a single
  `feature_frame.parquet ⊕ price_tier_labels.parquet ⊕
  good_deal_labels.parquet` merge.
- **Step 12** — `12-feature-engineering-price-model` — supplies the
  canonical price feature pipeline this spec reuses verbatim:
  - `ml.features.feature_frame.build_feature_frame` — produces the
    16-contract-field + 11-engineered-column DataFrame.
  - `ml.features.locality_aggregator.LocalityAggregator` — fits
    leakage-safe locality aggregates on `is_outlier == False` train
    rows, applies (does not refit) on val/test/serving.
  - `ml.features.split.split_train_val_test(target="price",
    random_state=42)` — the fixed 70/15/15 split. This spec calls it
    **once** on the same input DataFrame so its train mask matches the
    price model's train mask byte-equivalently — a reviewer auditing
    "did the classifier see the same training subset as the regression?"
    gets the same answer either side.
  - `ml.features.preprocessor.make_preprocessor` — the price
    `ColumnTransformer` factory. This spec reuses it without
    modification; the leakage-ban is enforced **upstream**, by
    dropping `price`/`price_per_sqft`/`locality_*` before the
    preprocessor runs.
- **Step 11** — `11-price-prediction-input-schema-v3` — supplies the
  pinned 16-field input contract `INPUT_FIELDS_V3` (the spec the
  cleaned DataFrame's columns match) and the Pydantic enums the
  feature frame mirrors.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` —
  supplies the `is_outlier` flag. The classifier feature frame
  preserves the flag on every row; the `is_outlier == False`
  filter happens at training time in Day 52, not here.
- **`02-TRD.md` §U-TRD-1 + §U-TRD-6** — original Classification
  technical spec: classifier consumes the same input schema as the
  price model; "good_deal_flag" (now `good_deal_verdict`) requires
  out-of-fold price predictions.
- **`05-BACKEND-SCHEMA.md` §U-SCHEMA-8** — final target field names
  (`price_tier`, `good_deal_verdict`).
- **`08-RULES.md` §8.1** — leakage ban: `price_tier` and
  `good_deal_verdict` must never feed back into the price
  regression as inputs (this spec does not violate that rule — it
  drops price-derived features, not classifier outputs).
- **`08-RULES.md` §8.4 + §12.4** — neither `price_tier` nor
  `good_deal_verdict` is a regression input feature; the
  classifier's input is the **same as the regression's minus
  price-derived columns**.
- **`08-RULES.md` §2.1 + §5.4** — fixed 70/15/15 split,
  `random_state=42`.
- **`08-RULES.md` §2.3** — leakage-safe aggregation; locality
  aggregates are fit train-only via `LocalityAggregator`.

## Routes / Endpoints
No new routes/endpoints. This spec is offline tooling — a pure
feature builder + CLI + tests. The FastAPI `/classify` route
remains a Stub; the trained classifier that consumes this feature
frame lands in a follow-on spec (Day 52).

## Data / Schema changes

**Read:**
- `data/processed/clean_listings.parquet` (Step 07).
- `data/processed/classifier/price_tier_labels.parquet` (Step 21;
  used to confirm `listing_id` join keys align, and to count
  per-(city × tier) cell sizes for the report).
- `data/processed/classifier/good_deal_labels.parquet` (Step 21;
  same — used for `listing_id` alignment + per-(city × verdict)
  cell sizes).

**Write (new directory already created by Spec 21 — appending):**
- `data/processed/classifier/feature_frame.parquet` — one row per
  listing in `clean_listings.parquet` (incl. outliers — `is_outlier`
  preserved). Columns:
  - `listing_id` (str; PK; matches Step 07's `listing_id`).
  - `split` (str; `train` / `val` / `test` from
    `split_train_val_test(..., random_state=42)`).
  - `is_outlier` (bool; copied from Step 06).
  - The 16 input-contract fields from `INPUT_FIELDS_V3` (in the
    pinned order), EXCLUDING `luxury_category` from the JSON
    payload-side dump (it stays in the feature frame because
    the classifier **does** use it as a feature — server-derived
    per Rules §10.2 means the price regression's preprocessor
    resolves it server-side, not that the classifier can't
    consume it). The Decision Log entry for this spec documents
    this distinction.
  - The 8 non-locality engineered columns from `ENGINEERED_COLUMNS`
    that are not price-derived: `n_amenities`, `n_features`,
    `floor_ratio`, `age_bucket_ord`, `bath_bed_ratio`,
    `area_per_bedroom`, `top_amenities_count`, plus the 10
    `has_<amenity>` flags resolved at fit time.
  - **DROPPED** (leakage ban, Rules §8.1 + §12.4):
    - `price` — the regression target.
    - `price_per_sqft` — price-derived ratio.
    - `locality_avg_price_sqft` — locality aggregate of
      `price_per_sqft`.
    - `locality_listing_count` — locality aggregate (kept by
      this spec — it's a **count**, not a price-derived
      statistic; counts don't leak the target. Decision Log
      entry documents this).
    - `locality_smoothed_price` — locality aggregate of
      `price_inr`.
  - `dataset_version` (str; parquet filename + sha1 of first
    1 MB, per Rules §1.3).
- `data/processed/classifier/feature_frame_report.md` — append-
  only, one "Run YYYY-MM-DD HH:MM:SS" section per CLI run.
  Mirrors the style of `label_construction_report.md`:
  - Header listing every column kept + every column dropped,
    with a one-line rationale per dropped column.
  - Per-city non-outlier row counts (so the reviewer sees the
    same train-set shape the price model saw).
  - Per-(city × `price_tier`) row counts joined from
    `price_tier_labels.parquet` (so the reviewer sees e.g.
    "Kolkata-Luxury = 4 rows" before Day 52 has to decide
    whether to drop that cell).
  - Per-(city × `good_deal_verdict`) row counts joined from
    `good_deal_labels.parquet`.
  - `dataset_version` and `git_commit` of the run.
  - A "Leakage audit" line stating: "Price-derived columns
  dropped: {price, price_per_sqft, locality_avg_price_sqft,
  locality_smoothed_price}. `locality_listing_count` retained
  (count, not price-derived). Verified `feature_frame.parquet`
  contains none of {price, price_per_sqft, locality_avg_price_sqft,
  locality_smoothed_price} via `pyarrow.Schema` introspection."
  Pinned by
  `test_feature_frame_drops_all_price_derived_columns`.

**No writes** to:
- `data/raw/` (immutable, Rules §1.2).
- `data/processed/clean_listings.parquet` (immutable input, Rules
  §1.2 + Step 07).
- `data/processed/classifier/{price_tier_labels,good_deal_labels,
  price_tier_quantile_boundaries,good_deal_thresholds}.parquet/json`
  (Spec 21 owns them; this spec reads, doesn't touch).
- `data/processed/feature_selection_report.md` (this spec doesn't
  amend the price model's feature selection — it writes its own
  `feature_frame_report.md`).
- `data/processed/analytics_cache/`, `data/app.db`.
- `models/` (no new model artifacts — only feature frame files;
  the trained classifier `*.pkl` lands in Spec 23 / Day 52).

## Templates / UI
None. This spec is offline tooling — no Flask templates, no static
assets, no HTML. The UI that surfaces classification results
(VerdictBadge, AffordabilityChip, tier filter in the Recommender)
is `07-TRACKER.md` Day 55 — out of scope here.

## Files to change / Files to create

**Create:**
- `ml/classification/features.py` — the classifier feature
  builder. Public API:
  - `DROPPED_FOR_LEAKAGE: frozenset[str] = frozenset({"price",
    "price_per_sqft", "locality_avg_price_sqft",
    "locality_smoothed_price"})` — pinned literal. The
    classification feature frame **never** contains any of these
    column names (verified by a test against the produced
    parquet's schema).
  - `RETAINED_LOCALITY_COLUMN: str = "locality_listing_count"` —
    the one locality aggregate the spec keeps. Count is not
    price-derived; it doesn't leak the target. Documented in
    the module docstring with rationale.
  - `build_classifier_feature_frame(clean_listings: pd.DataFrame,
    locality_aggregator: LocalityAggregator | None = None,
    split_col: str = "split", processed_dir: Path | str | None
    = None, split_fn: Callable | None = None,
    fit_aggregator_fn: Callable | None = None) -> pd.DataFrame`
    — the public entry point. Steps:
    1. Read `clean_listings.parquet` from `processed_dir` (or
       accept a pre-loaded DataFrame).
    2. Call `split_train_val_test(clean_listings,
       target="price", random_state=42)` (the default
       `split_fn`) — or accept a pre-split / pre-loaded frame.
       Tag rows with `split` column. This is the **same call**
       Step 12 makes, so the train mask is byte-equivalent.
    3. If `locality_aggregator is None`, instantiate one and
       call `fit` on
       `clean_listings[is_outlier == False AND split ==
       "train"]` (Rules §2.3 + §8.4), then `transform` on the
       full frame. If `locality_aggregator` is passed (the
       price model's fitted aggregator from Spec 20),
       reuse it without refitting — `transform` only.
       `fit_aggregator_fn` lets tests inject a fake.
    4. Call `build_feature_frame(df)` (Step 12) on the
       full frame — produces the 16-contract + 11-engineered
       DataFrame.
    5. Drop the leakage columns: any of `DROPPED_FOR_LEAKAGE`
       that are present. The drop is **idempotent** (a
       column already absent stays absent), and the function
       verifies the produced DataFrame has **zero** columns in
       `DROPPED_FOR_LEAKAGE` before returning (raises
       `ValueError` otherwise — caught by
       `test_build_classifier_feature_frame_drops_all_price_derived_columns`).
    6. Append `listing_id` (copied from `clean_listings`) if
       not already present.
    7. Reorder columns to: `[listing_id, split, is_outlier] +
       INPUT_FIELDS_V3 + retained engineered + locality_*` —
       deterministic order, pinned by a test.
    8. Returns the new frame. Pure function with respect to
       its inputs; persistence is the caller's job (the CLI
       below).
    Module docstring cites the leakage rule sources (RULES
    §8.1, §12.4) and explicitly notes that
    `luxury_category` is **kept** because the classifier
    consumes it as a feature — the server-derived rule
    applies to the API surface, not to the feature frame.
- `ml/classification/feature_report.py` — the report writer.
  Public API:
  - `write_feature_frame_report(*, output_dir: Path,
    feature_frame: pd.DataFrame,
    price_tier_labels: pd.DataFrame,
    good_deal_labels: pd.DataFrame,
    dataset_version: str, git_commit: str) -> Path` —
    appends one "Run YYYY-MM-DD HH:MM:SS" section to
    `data/processed/classifier/feature_frame_report.md` (or
    writes the header on first call). Includes:
    - The full kept-column list + the dropped-column list with
      one-line rationales.
    - Per-city non-outlier row counts.
    - Per-(city × `price_tier`) row counts joined from
      `price_tier_labels`.
    - Per-(city × `good_deal_verdict`) row counts joined from
      `good_deal_labels`.
    - The "Leakage audit" line.
    - `dataset_version` and `git_commit`.
  - `write_feature_frame_artifact(df: pd.DataFrame,
    output_dir: Path) -> Path` — writes
    `feature_frame.parquet` at the canonical path.
- `scripts/build_classifier_features.py` — the CLI entry
  point. Invoked as
  `python scripts/build_classifier_features.py
  [--processed-dir data/processed]
  [--output-dir data/processed/classifier]`. Steps:
  1. Parse args (one argparse group; reuses the same arg
     style as `scripts/build_classification_labels.py`).
  2. Call `build_classifier_feature_frame(...)` to get
     the feature DataFrame.
  3. Load `price_tier_labels.parquet` and
     `good_deal_labels.parquet` (Step 21's outputs) for
     the report's per-cell counts.
  4. Call `write_feature_frame_artifact(df, output_dir)`.
  5. Call
     `write_feature_frame_report(output_dir=output_dir,
     feature_frame=df, price_tier_labels=...,
     good_deal_labels=..., dataset_version=...,
     git_commit=...)`.
  6. Print a one-line stdout summary: `"[OK] feature_frame
     rows=N1 (outliers=N2); dropped columns:
     {price, price_per_sqft, locality_avg_price_sqft,
     locality_smoothed_price}; retained:
     locality_listing_count; per-city non-outlier rows:
     {city: N, ...}; written to <output_dir>"`.
  7. Exit 0. Idempotent — a re-run overwrites
     `feature_frame.parquet`; a re-run with different
     upstream data (new clean_listings.parquet) writes
     new content but same filename.

- `tests/test_classifier_features.py` — pytest tests for
  `build_classifier_feature_frame`. Required tests (exact
  names):
  - `test_dropped_for_leakage_is_pinned` —
    `DROPPED_FOR_LEAKAGE == frozenset({"price",
    "price_per_sqft", "locality_avg_price_sqft",
    "locality_smoothed_price"})`.
  - `test_retained_locality_column_is_pinned` —
    `RETAINED_LOCALITY_COLUMN == "locality_listing_count"`.
  - `test_build_classifier_feature_frame_drops_all_price_derived_columns`
    — synthetic frame that **does** contain all four
    price-derived columns; assert the output frame's
    `.columns` does not contain any name in
    `DROPPED_FOR_LEAKAGE`.
  - `test_build_classifier_feature_frame_keeps_locality_listing_count`
    — synthetic frame with `locality_listing_count` set;
    assert that column survives the drop.
  - `test_build_classifier_feature_frame_keeps_all_16_input_fields`
    — synthetic frame with all 16 contract fields; assert
    every `INPUT_FIELDS_V3` name appears in the output.
  - `test_build_classifier_feature_frame_adds_split_column`
    — synthetic frame; assert the output has a `split`
    column with values in `{train, val, test}` and exactly
    the 70/15/15 proportions (within 1 row tolerance).
  - `test_build_classifier_feature_frame_preserves_is_outlier_flag`
    — synthetic frame with 10% outliers; assert the output
    `is_outlier` is bit-identical to the input.
  - `test_build_classifier_feature_frame_preserves_listing_id`
    — synthetic frame; assert the output's `listing_id`
    column equals the input's `listing_id` column row-
    for-row.
  - `test_build_classifier_feature_frame_reuses_passed_locality_aggregator_without_refit`
    — pass a fitted `LocalityAggregator` whose `transform`
    adds a sentinel column; assert the sentinel is in the
    output (proves the function did **not** call `fit`
    on the passed aggregator).
  - `test_build_classifier_feature_frame_column_order_is_deterministic`
    — two calls on the same input; assert
    `list(df1.columns) == list(df2.columns)` exactly
    (no set equality — order matters).

- `tests/test_classifier_feature_report.py` — pytest tests
  for the report writer. Required tests (exact names):
  - `test_write_feature_frame_report_writes_header_on_first_run`
    — empty `output_dir`; assert the report file exists
    with a header.
  - `test_write_feature_frame_report_appends_runs_not_overwrites`
    — call `write_feature_frame_report` twice with
    different `git_commit` values; assert both "Run"
    sections are present (append-only).
  - `test_write_feature_frame_report_includes_leakage_audit_line`
    — assert the written "Run" section contains the
    string "Leakage audit" and lists every column in
    `DROPPED_FOR_LEAKAGE`.
  - `test_write_feature_frame_report_includes_per_city_tier_counts`
    — synthetic 4-city frame; assert each city name
    appears with row counts.

- `tests/test_build_classifier_features_cli.py` — pytest
  test for the CLI. Required tests (exact names):
  - `test_cli_runs_end_to_end_on_synthetic_artifacts` —
    build tiny synthetic `clean_listings.parquet` +
    tiny synthetic `price_tier_labels.parquet` +
    `good_deal_labels.parquet` in `tmp_path`, invoke the
    CLI via `subprocess.run`, assert exit code 0 +
    `feature_frame.parquet` exists + the report file
    exists.
  - `test_cli_writes_deterministic_artifacts_on_rerun` —
    run the CLI twice with the same inputs; assert
    `feature_frame.parquet` is byte-identical (the
    schema is deterministic — no timestamp / run-id
    embedded in the data).
  - `test_cli_does_not_log_contact_fields` — grep the
    captured stdout for any column name matching the
    regex `(contact|dealer|phone|email|photo|url|spid)`
    — must be absent.
  - `test_cli_output_parquet_contains_no_price_derived_columns`
    — read the produced `feature_frame.parquet` schema
    with `pyarrow`; assert no column name is in
    `DROPPED_FOR_LEAKAGE`. Pins the leakage ban at the
    artifact level.

**Modify:**
- `ml/classification/__init__.py` — add `from
  ml.classification import features  # noqa: F401` (the new
  submodule) and re-export the new public symbols:
  `build_classifier_feature_frame`,
  `DROPPED_FOR_LEAKAGE`, `RETAINED_LOCALITY_COLUMN`. Keeps the
  package's public API consistent with Spec 21's exports.
- `ml/__init__.py` — no change. The `from ml import
  classification` line added by Spec 21 covers the new
  submodule automatically.
- `scripts/run_pipeline.py` — append one CLI line after the
  current classification-label-builder line (Spec 21's
  `scripts/build_classification_labels.py` invocation):
  ```python
  subprocess.run(
      [sys.executable, "scripts/build_classifier_features.py"],
      check=False,
  )
  ```
  Same non-fatal pattern as Spec 21: pipeline prints the
  CLI's stdout summary regardless and proceeds to the next
  stage. Exit code surfaced in the pipeline's final
  summary.
- `requirements.txt` — no new packages; this spec only uses
  pandas / numpy / pyarrow (already pinned) + stdlib.
  Verify with `pip freeze | grep -E "(numpy|pandas|pyarrow)"`
  and flag explicitly per CLAUDE.md "no new packages
  without checking first."

**No changes** to:
- `app/`, `api/`, `migrations/` (no Flask/FastAPI code in
  this spec; routes land in follow-on specs).
- `data/raw/`, `data/processed/clean_listings.parquet`,
  `data/processed/feature_selection_report.md`,
  `data/processed/analytics_cache/`, `data/app.db`,
  `data/model_registry.csv` (this spec writes its own
  artifacts under `data/processed/classifier/` and never
  touches the others).
- `notebooks/`, `tests/conftest.py` (existing fixtures
  remain; new fixtures live in the new test files).
- `ml/features/*` (read-only — this spec consumes
  `build_feature_frame`, `LocalityAggregator`,
  `split_train_val_test`, `INPUT_FIELDS_V3` without
  modification. Any change to those would be a Spec-12+
  concern, not a Spec-22 concern.)
- `ml/training/`, `ml/evaluation/`, `ml/cleaning/` (no
  read-write interaction; this spec does not train or
  evaluate anything.)
- `CLAUDE.md`'s "Implemented vs stub routes" table — this
  spec adds **no routes**. The `POST /classify` FastAPI
  route stays a Stub until a follow-on spec wires the
  classifier trained from this feature frame.

## New dependencies
None. The spec uses pandas / numpy / pyarrow (already pinned)
+ stdlib (`argparse`, `json`, `logging`, `pathlib`,
`subprocess`, `datetime`). **No** new pip/npm packages, **no**
Flask/FastAPI route additions, **no** new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The spec reads +
  writes Parquet + Markdown files only.
- **No dealer/contact/media-URL fields ever reach the
  UI or an export.** The feature builder does not log
  any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. Pinned by
  a test that greps the CLI's captured stdout for the
  regex — must be absent. The CLI also never logs the
  values of any contact-typed column, only aggregate row
  counts.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no
  templates.
- **Model evaluation must reference the fixed
  evaluation protocol (Rules §2.1).** This spec ships
  no model, but it ships the train/val/test split that
  every downstream classifier will see. The split helper
  it wraps (`split_train_val_test(target="price",
  random_state=42)`) is the same helper the price model
  uses, so a reviewer auditing "what train set did the
  classifier see?" gets the same answer as "what train
  set did the price model see?" — bit-equivalent.
- **Leakage ban (Rules §8.1 + §12.4).** The spec drops
  every price-derived column from the classifier feature
  frame. `DROPPED_FOR_LEAKAGE` is the pinned literal the
  spec promises to drop; the function asserts the
  post-drop DataFrame has zero of those columns before
  returning. The CLI's end-of-run artifact schema check
  (`test_cli_output_parquet_contains_no_price_derived_columns`)
  re-verifies this on disk.
- **`locality_listing_count` is kept (not dropped).**
  Counts don't leak the target — they describe the
  popularity of a locality, not its price level. Keeping
  it lets Day 52 use locality popularity as a classifier
  feature. The Decision Log entry for this spec
  documents this exception; the "Leakage audit" line in
  the report names it explicitly.
- **`luxury_category` is kept (not dropped).** The
  classifier consumes `luxury_category` as a feature
  (it's a useful ordinal signal — `Low` / `Medium` /
  `High` correlates strongly with `price_tier`). Rules
  §10.2's "server-derived, not self-reported" rule
  applies to the **API surface** (the client never sends
  it; the server resolves it from the amenity checklist).
  It does **not** apply to the offline feature frame,
  which has access to the resolved value via
  `clean_listings.parquet`. Documented in the module
  docstring + the Decision Log entry.
- **`transact_type` is a routing key (Rules §10.3),
  not a classifier feature.** The classifier feature
  frame includes `transact_type` as a column (it is one
  of the 16 contract fields), but Day 52's training
  script will route on it the same way the price model
  does — train one classifier per `(transact_type)` (or
  include it as a feature and let the model learn it
  — implementation choice in Day 52, not here). This
  spec just preserves the column.
- **Reuse `LocalityAggregator`, do not refit.** The
  function accepts a `LocalityAggregator` instance; if
  passed, it `transform`s only (no `fit`). If the
  caller wants a fresh aggregator (e.g., a unit test),
  they instantiate one themselves and pass it. The
  default path (no aggregator passed) fits a new one
  on `is_outlier == False AND split == "train"` rows.
  This mirrors the price model's behavior and keeps
  the leakage-safe "apply, don't recompute" rule.
- **`split_train_val_test` called once, not per-module.**
  The function calls `split_train_val_test(...)` once
  on the same input DataFrame the price model sees.
  Day 52 reads the `split` column from
  `feature_frame.parquet` — it does not re-split.
- **No SQL queries.** All joins are pandas left-joins on
  `listing_id`.
- **No FastAPI imports in `ml/classification/`.** The
  feature builder is pure-Python; no FastAPI / httpx
  client imports. Keeps the feature builder runnable in
  offline CI without a live inference service.
- **Logging uses stdlib `logging` only.** One module-
  level logger per file (`logger =
  logging.getLogger(__name__)`). INFO for stage
  boundaries; WARNING for expected-but-noteworthy
  conditions (per-cell row count < 10); ERROR for hard
  failures (no `clean_listings.parquet` at the
  configured path, schema mismatch).
- **CLI never aborts the pipeline on non-zero exit.**
  The pipeline calls the CLI with `check=False`. The
  CLI's stdout summary is printed regardless; the
  pipeline proceeds. The CLI exits 0 on success and 1
  only on a hard failure — missing input parquet,
  schema mismatch.
- **No notebook-only steps (Rules §5.3).** Everything
  in this spec is reproducible via
  `python scripts/build_classifier_features.py`. No
  Jupyter cell computes a feature the script can't
  reproduce.
- **Single source of truth for the dropped column
  set.** `DROPPED_FOR_LEAKAGE` is the only place the
  list of columns to drop lives. The CLI's stdout
  summary reads from it; the report writer reads from
  it; the test asserts on it. If a future spec
  discovers another leakage vector, add it to
  `DROPPED_FOR_LEAKAGE` once and it propagates.

## Definition of done

1. `python -m pytest
   tests/test_classifier_features.py
   tests/test_classifier_feature_report.py
   tests/test_build_classifier_features_cli.py -v` from
   repo root runs and passes. Tests required (exact
   names):
   - **Feature builder**
     (`tests/test_classifier_features.py`):
     - `test_dropped_for_leakage_is_pinned`
     - `test_retained_locality_column_is_pinned`
     - `test_build_classifier_feature_frame_drops_all_price_derived_columns`
     - `test_build_classifier_feature_frame_keeps_locality_listing_count`
     - `test_build_classifier_feature_frame_keeps_all_16_input_fields`
     - `test_build_classifier_feature_frame_adds_split_column`
     - `test_build_classifier_feature_frame_preserves_is_outlier_flag`
     - `test_build_classifier_feature_frame_preserves_listing_id`
     - `test_build_classifier_feature_frame_reuses_passed_locality_aggregator_without_refit`
     - `test_build_classifier_feature_frame_column_order_is_deterministic`
   - **Report writer**
     (`tests/test_classifier_feature_report.py`):
     - `test_write_feature_frame_report_writes_header_on_first_run`
     - `test_write_feature_frame_report_appends_runs_not_overwrites`
     - `test_write_feature_frame_report_includes_leakage_audit_line`
     - `test_write_feature_frame_report_includes_per_city_tier_counts`
   - **CLI**
     (`tests/test_build_classifier_features_cli.py`):
     - `test_cli_runs_end_to_end_on_synthetic_artifacts`
     - `test_cli_writes_deterministic_artifacts_on_rerun`
     - `test_cli_does_not_log_contact_fields` — grep
       the captured stdout for any column name
       matching
       `(contact|dealer|phone|email|photo|url|spid)`
       — must be absent.
     - `test_cli_output_parquet_contains_no_price_derived_columns`
2. `python -m pytest -m "not realdata"` from repo root
   still passes (no real-data dependency introduced
   by this spec).
3. `ruff check ml/classification/features.py
   ml/classification/feature_report.py
   ml/classification/__init__.py
   scripts/build_classifier_features.py
   tests/test_classifier_features.py
   tests/test_classifier_feature_report.py
   tests/test_build_classifier_features_cli.py` reports
   zero issues.
4. `python -c "from ml.classification import
   build_classifier_feature_frame, DROPPED_FOR_LEAKAGE,
   RETAINED_LOCALITY_COLUMN; print(sorted(DROPPED_FOR_LEAKAGE),
   RETAINED_LOCALITY_COLUMN)"` from repo root prints
   `['locality_avg_price_sqft', 'locality_smoothed_price',
   'price', 'price_per_sqft'] locality_listing_count`
   without error — public API imports cleanly.
5. `python scripts/build_classifier_features.py` from
   repo root exits 0 and prints the `[OK]` summary line
   — manual smoke test of the CLI.
6. After running step 5,
   `data/processed/classifier/feature_frame.parquet`
   exists and its `pyarrow.Schema` contains **no**
   column in `DROPPED_FOR_LEAKAGE`. And
   `data/processed/classifier/feature_frame_report.md`
   contains a "Run YYYY-MM-DD HH:MM:SS" section with:
   - The full kept-column list + the dropped-column
     list with one-line rationales.
   - Per-city non-outlier row counts.
   - Per-(city × `price_tier`) row counts.
   - Per-(city × `good_deal_verdict`) row counts.
   - The "Leakage audit" line.
   - `dataset_version` and `git_commit`.
7. The artifact file does **not** contain any column
   whose name matches
   `(contact|dealer|phone|email|photo|url|spid)` —
   spot-check by reading the parquet schema and
   grepping the column names.
8. The `git_commit` recorded in the run's report
   section matches `git rev-parse HEAD` at the time
   the CLI was invoked.
9. `git status` after committing shows only the new
   files listed above, the modified
   `scripts/run_pipeline.py`, and the modified
   `ml/classification/__init__.py`. No accidental
   additions to `app/`, `api/`,
   `data/processed/clean_listings.parquet`,
   `data/processed/feature_selection_report.md`,
   `data/raw/`, `notebooks/`, `data/processed/analytics_cache/`,
   or `data/processed/classifier/{price_tier_labels,
   good_deal_labels,price_tier_quantile_boundaries,
   good_deal_thresholds}.parquet/json`.
10. `CLAUDE.md`'s "Implemented vs stub routes" table
    is unchanged — this spec adds **no routes**. The
    `POST /classify` FastAPI route stays a Stub until
    a follow-on spec (Day 52: classifier training +
    serving) wires the classifier trained from the
    feature frame this spec produces.
11. `07-TRACKER.md` is updated via `/update-tracker`
    to mark Day 51 (Classification feature set +
    Logistic Regression baseline, the **first half**)
    as **Done** with the actual measured per-city
    non-outlier row counts from this spec's report.
    The **second half** (Logistic Regression baseline
    + Day 52 Random Forest / XGBoost classifiers) stays
    "Not Started" with a `← needs Spec 22 feature
    frame` note so the next session picks up from a
    clean handoff.
12. The Decision Log in `07-TRACKER.md` gains two
    new entries dated today with the rationale for:
    (a) keeping `locality_listing_count` despite the
    locality-aggregate drop (count is not
    price-derived), and (b) keeping
    `luxury_category` in the offline feature frame
    despite the Rules §10.2 server-derived rule (the
    rule applies to the API surface, not to offline
    consumption of the resolved value). Both
    decisions are auditable from the report file in
    step 6 without rerunning any code.