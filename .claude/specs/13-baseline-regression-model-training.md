# Spec: Baseline Regression Model Training

## Overview
Train the **v1 baseline price regression model** end-to-end so the FastAPI
`/predict` route (a later spec) has a real, evaluated artifact to serve. This
spec implements `02-TRD.md` §10 (Model Selection & Productionization) at its
**minimum acceptable level**: a single global model per `transact_type` (Sale
+ Rent → two trained pipelines), using the feature frame and fitted
`ColumnTransformer` produced by Step 12, evaluated against the fixed
70/15/15 protocol (`random_state=42`) and the four headline metrics (R²,
MAE, RMSE, MAPE on the original ₹ scale). Candidate set: **Linear
Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost** —
selected per the TRD §10 candidate list. The winner is wrapped in an
`sklearn.Pipeline(preprocessor → estimator)`, serialized via `joblib` to
`models/price_model_sale_v1.pkl` / `models/price_model_rent_v1.pkl`, and
paired with `models/metrics_v1.json` containing per-model train/val/test
metrics + the chosen model name + git commit + dataset version (the
lightweight `model_registry` row, per Backend Schema §U-SCHEMA-13). This
is the **baseline** in the "30–35% MAE/RMSE reduction" tracking series —
later specs (Week 4 improvement levers, Week 8 final tuning) reduce against
these numbers. Module: **price-prediction**.

## Depends on
- **Step 12** — `12-feature-engineering-price-model` — produces
  `models/feature_pipeline_v1.pkl` (the fitted `ColumnTransformer` +
  `LocalityAggregator` tuple), `models/feature_list_v1.json`, and the
  reproducible `python scripts/build_features.py` entry point. This spec
  consumes those artifacts and adds the estimator step on top.
- **Step 07** — `07-clean-listings-parquet-pipeline` — produces
  `data/processed/clean_listings.parquet`. The training script reads
  this directly (not the feature frame) so the outlier filter and
  the canonical column set are applied consistently.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` — provides
  the `is_outlier` flag the training subset filters on
  (`is_outlier == False` rows only).
- **Step 11** — `11-price-prediction-input-schema-v3` — locks the 16
  input fields and `INPUT_FIELDS_V3`; the model card documents
  exactly which fields it consumes.
- **`ml/features/split.py`** — the 70/15/15 split helper with
  `random_state=42` (Step 12). The training script reuses it
  unchanged; no second split definition.
- **`02-TRD.md` §10** — candidate model list + evaluation protocol +
  productionization checklist.
- **`05-BACKEND-SCHEMA.md` §6** — model artifact filenames
  (`price_model_v{n}.pkl`, `metrics_v{n}.json`, `feature_list_v{n}.json`)
  + §U-SCHEMA-13 — `model_registry` table fields.
- **`08-RULES.md` §2.1–§2.5** — fixed evaluation protocol, train-only
  feature aggregation, versioned artifacts, paired metrics JSON.
- **`08-RULES.md` §5.4** — `random_state=42` everywhere.
- **`10-FINALIZED-INPUT-SCHEMA.md`** — input field set the model
  consumes.
- **`12-feature-engineering-price-model.md` §"Rules for
  implementation"** — the split helper, leakage rules, and
  `transact_type`-as-routing-key conventions this spec inherits.

## Routes / Endpoints
No new routes/endpoints. This spec is offline model training +
serialization only. FastAPI wiring of `price_model_sale_v1.pkl` /
`price_model_rent_v1.pkl` into `POST /predict` is a separate
spec (Week 4 follow-on: "FastAPI /predict route + smoke test").

## Data / Schema changes
- **Read** `data/processed/clean_listings.parquet` (Step 07 output).
- **Read** `models/feature_pipeline_v1.pkl` (Step 12 output — the
  fitted preprocessor + locality aggregator).
- **Read** `models/feature_list_v1.json` (Step 12 — final ordered
  feature list).
- **Write** `models/price_model_sale_v1.pkl` — `joblib.dump` of the
  full `sklearn.Pipeline` for the Sale subset. Contains the
  preprocessor + the chosen estimator (preprocessor is re-loaded
  from Step 12's artifact, not re-fit; estimator is fit here).
  Filename pinned by Rules §2.5.
- **Write** `models/price_model_rent_v1.pkl` — same, for the Rent
  subset. Skipped with a logged INFO if Rent rows < a minimum
  threshold (e.g., n < 500), per TRD §U-TRD-4's per-pipeline caveat.
- **Write** `models/metrics_v1.json` — single JSON containing
  per-model train/val/test metrics for all candidates on both
  Sale and Rent (so future specs can compare without re-training),
  plus the chosen model's version tag, dataset version, git commit,
  hyperparameters, feature hash, and per-city R²/MAE. Schema:
  ```json
  {
    "version": "v1",
    "created_at": "<ISO timestamp>",
    "dataset_version": "clean_listings.parquet",
    "git_commit": "<sha>",
    "split": {"train": 0.70, "val": 0.15, "test": 0.15, "random_state": 42},
    "sale": {
      "candidates": {
        "linear":   {"train": {...}, "val": {...}, "test": {...}},
        "ridge":    {"train": {...}, "val": {...}, "test": {...}},
        ...
        "xgboost":  {"train": {...}, "val": {...}, "test": {...}}
      },
      "chosen_model": "<name>",
      "chosen_metrics": {"train": {...}, "val": {...}, "test": {...}},
      "per_city_test": {"Gurgaon": {...}, "Hyderabad": {...}, ...}
    },
    "rent": { <same shape, or {"skipped": true, "reason": "n=X < 500"}> }
  }
  ```
  Each metrics dict: `{r2, mae, rmse, mape}` — MAE/RMSE/MAPE on the
  **original ₹ price scale** (inverse `expm1` from log target).
  MAPE is `mean(|y_true - y_pred| / y_true) * 100`, with the
  conventional `epsilon=1.0` guard to avoid div-by-zero on ₹0 rows
  (none expected post-cleaning, but the guard is pinned).
- **Write** `data/processed/feature_selection_report.md` —
  **append** Round 2 (RF importance, GB importance, Permutation
  importance) + Round 3 (SHAP ranking) sections to the file Step 12
  wrote. Step 12 explicitly defers these because they need a fitted
  model — this spec lands them. Sections added:
  - "Round 2 — tree-based + permutation importance" — top-N feature
    tables per model that produced them (RF, GB, XGB).
  - "Round 3 — SHAP ranking" — top-N `mean |SHAP value|` features
    from a `shap.TreeExplainer` fit on the chosen model.
  - "Final feature list & rationale" — explicit keep/drop decisions
    with citations to which methods agreed (the TRD §9 "keep if ≥2
    of 4 methods agree" rule). Anything kept without an explicit
    justification is a defect (Rules §2.2).
- **Write** `data/model_registry.csv` — one row per trained model
  per `transact_type`, append-only. Matches Backend Schema §U-SCHEMA-13
  columns: `model_name, version, training_dataset_version, git_commit,
  training_date, rmse, mae, r2, hyperparameters, feature_hash`. The
  training script appends; no separate service.
- **No writes to `data/raw/`** (Rules §1.2).
- **No writes to `data/processed/clean_listings.parquet`** — Step 07
  owns it; this spec is read-only on the cleaning artifact.
- **No writes to `data/processed/analytics_cache/`** — Week 6.
- **No application DB changes** — `data/app.db` is untouched. The
  `model_registry.csv` is a flat file, not a SQL table; a future
  spec can migrate it to a real table, but Rules §5.3 + §13 (no
  MLOps tooling) keep it as a CSV for now.

## Templates / UI
None. This spec is offline model training — no Flask templates, no
static assets, no HTML. The FastAPI route that consumes the trained
model is a later spec.

## Files to change / Files to create

**Create:**
- `ml/training/__init__.py` — empty; re-exports below.
- `ml/training/candidates.py` — the candidate-model factory.
  Public API:
  - `CANDIDATE_MODELS: dict[str, BaseEstimator]` — the 6 candidate
    estimators from TRD §10: `"linear"` (LinearRegression),
    `"ridge"` (Ridge, default `alpha=1.0`),
    `"lasso"` (Lasso, default `alpha=0.001`),
    `"random_forest"` (RandomForestRegressor, `n_estimators=200`,
    `max_depth=None`, `n_jobs=-1`, `random_state=42`),
    `"gradient_boosting"` (GradientBoostingRegressor,
    `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`,
    `random_state=42`),
    `"xgboost"` (XGBRegressor, `n_estimators=300`, `max_depth=6`,
    `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`,
    `tree_method="hist"`, `n_jobs=-1`, `random_state=42`).
    Hyperparameters are **sensible defaults**, not tuned — tuning
    is a Week 8 improvement lever, not this baseline spec. Pinned
    defaults + logged in `metrics_v1.json`.
  - `make_estimator(name: str) -> BaseEstimator` — factory from
    the dict. Unknown name raises `ValueError`.
- `ml/training/evaluation.py` — metric computation + per-model
  evaluation loop. Public API:
  - `regression_metrics(y_true, y_pred) -> dict[str, float]` —
    returns `{r2, mae, rmse, mape}` on the original price scale.
    Uses `sklearn.metrics.r2_score`, `mean_absolute_error`,
    `mean_squared_error(squared=False)`, and a numpy MAPE with
    `epsilon=1.0` guard. Pure function.
  - `evaluate_model(pipeline, X_train, y_train, X_val, y_val,
    X_test, y_test) -> dict[str, dict[str, float]]` — fits the
    pipeline on train, returns
    `{"train": metrics, "val": metrics, "test": metrics}` where
    each metrics dict is the output of `regression_metrics`. Pure
    function (no I/O).
  - `per_city_metrics(pipeline, X_test, y_test, city_series) ->
    dict[str, dict[str, float]]` — same metric dict, sliced by
    city. Logs a WARNING for any city with n < 30 test rows.
- `ml/training/persistence.py` — write the versioned artifacts.
  Public API:
  - `PRICE_MODEL_VERSION: str = "v1"` — pinned module constant.
  - `save_price_model(pipeline: Pipeline, transact_type: str,
    version: str = "v1") -> Path` — `joblib.dump` to
    `models/price_model_{transact_type.lower()}_{version}.pkl`.
    Filename rules per Rules §2.5: versioned, never overwritten in
    place (a `v2` training writes a new file; `v1` is preserved).
  - `save_metrics(metrics: dict, version: str = "v1") -> Path` —
    `json.dump` to `models/metrics_{version}.json`, indented=2,
    deterministic key order. Uses `default=str` to handle numpy
    scalars + datetimes. Returns the path.
  - `append_model_registry(row: dict, csv_path: Path = ...) ->
    None` — appends one row to `data/model_registry.csv`. Header
    is written if the file doesn't exist. Field order matches
    Backend Schema §U-SCHEMA-13. Idempotent re-run: if the same
    `(model_name, version, git_commit)` triple already appears in
    the CSV, the row is **not** appended again (Rules §2.5 — never
    overwrite; also never duplicate).
- `ml/training/selection.py` — pick the winner from the candidate
  evaluation. Public API:
  - `select_winner(candidate_results: dict[str, dict],
    primary_metric: str = "val_rmse") -> str` — returns the name
    of the candidate with the lowest `primary_metric` on the
    validation set. Tie-break: lowest `val_mae`, then shortest
    training time (logged but not enforced as a column in
    metrics_v1.json — just a documented rule). Documented in the
    module docstring. Pure function.
- `scripts/train_price_model.py` — the one entry point. Invoked
  as `python scripts/train_price_model.py` from repo root.
  Idempotent (re-running with same git commit + same data appends
  nothing to `model_registry.csv`). Steps:
  1. Load `data/processed/clean_listings.parquet`.
  2. Filter to `is_outlier == False`.
  3. Split into Sale / Rent subsets per `transact_type` (Rent
     subset may be near-empty or empty — handle with INFO log).
  4. For each subset with n >= 500:
     a. Call `split_train_val_test(df, target="price")` from
        `ml.features.split` to get the 70/15/15 split (reuses
        Step 12's split helper, same seed → identical boundaries
        if the input is identical).
     b. Build the feature frame via
        `ml.features.feature_frame.build_feature_frame(df)`,
        passing the cleaned DF (which already has the 16 contract
        fields + `is_outlier` + `was_missing_*`).
     c. Apply the fitted locality aggregator + preprocessor loaded
        from `models/feature_pipeline_v1.pkl` via
        `ml.features.persistence.load_feature_artifacts()`. Log
        the loaded artifact version.
     d. For each candidate name in `CANDIDATE_MODELS`:
        - Build `Pipeline([("preproc", preproc), ("est",
          make_estimator(name))])`.
        - Call `evaluate_model(...)` to get train/val/test metrics
          + `per_city_metrics(...)` for the test slice.
        - Append to `candidate_results[name]`.
     e. Run `select_winner(candidate_results)` to pick the name.
     f. Call `save_price_model(winner_pipeline, transact_type)`.
     g. Append Sale (or Rent) rows to `candidate_results`,
        `chosen_model`, `chosen_metrics`, `per_city_test` to a
        per-transact-type dict. Accumulate into the final
        `metrics_v1.json` payload.
  5. Build the `metrics_v1.json` payload (with `git_commit`,
     `dataset_version`, `created_at`, `split`, `sale`, `rent`).
  6. Call `save_metrics(payload)`.
  7. Compute SHAP values for the winner (if it's a tree model —
     RF, GB, XGB) via `shap.TreeExplainer`. Compute
     `mean |SHAP value|` per feature, write Round 3 section of
     `feature_selection_report.md`. For non-tree winners, log a
     WARNING explaining SHAP ranking is skipped and the report's
     Round 3 section is left as the TRD §9 "n/a — non-tree model"
     placeholder. Append the file; don't overwrite Step 12's
     content.
  8. Compute RF/GB/XGB impurity-based importance from the
     candidate models that were tree-based (already trained in
     step 4d). Write Round 2 section of the report. For the
     Permutation importance column, run
     `sklearn.inspection.permutation_importance` on the
     validation slice of the winner (cheaper than the full
     train set, per TRD §9 method note) with
     `n_repeats=10`, `random_state=42`.
  9. Append `data/model_registry.csv` with one row per
     `transact_type` (or skip the row if the subset was too
     small).
  10. Log a summary line at INFO: chosen model per transact
      type + test R²/MAE/RMSE + "artifacts at <path>".
- `tests/test_candidates.py` — pytest tests for the candidate
  factory + estimator defaults. Pure-Python; tiny synthetic
  regression fixture.
- `tests/test_evaluation.py` — pytest tests for
  `regression_metrics` + `evaluate_model` + `per_city_metrics`
  on a synthetic frame.
- `tests/test_selection.py` — pytest tests for `select_winner`'s
  tie-break logic.
- `tests/test_training_persistence.py` — pytest tests for
  `save_price_model` / `save_metrics` / `append_model_registry`
  using `tmp_path` + a synthetic metrics dict + a synthetic CSV.
- `tests/test_train_price_model_script.py` — pytest test that
  runs `scripts/train_price_model.py` against a tiny synthetic
  `clean_listings.parquet` (built in `tmp_path` by the test, with
  both Sale and Rent rows so both pipelines train), asserts:
  - both `price_model_sale_v1.pkl` and `price_model_rent_v1.pkl`
    exist after the run,
  - `metrics_v1.json` parses and contains both `sale` and `rent`
    blocks with the expected candidate keys,
  - `feature_selection_report.md` contains a "Round 2" section
    header (Step 12's report is preserved above it),
  - `model_registry.csv` has exactly 2 new rows appended (one
    per transact type) and no duplicates on re-run.

**Modify:**
- `scripts/run_pipeline.py` — append one line:
  `subprocess.run([sys.executable, "scripts/train_price_model.py"],
  check=True)` after the Step 12 feature-build line, so the
  training is reproducible end-to-end via `make pipeline`
  (TRD §13). No logic change; just sequence registration.
- `requirements.txt` — verify `xgboost` is already pinned
  (Step 01 baseline). If absent, add with a pinned minor
  version. Flag the addition explicitly per CLAUDE.md
  "no new packages without checking first." SHAP is also
  already in Step 01's stack (TRD §1); verify.
- `ml/__init__.py` — add `from ml import training  # noqa: F401`
  so the new submodule is importable consistently with
  `ml.features` / `ml.cleaning`.

**No changes** to:
- `app/`, `api/`, `data/raw/`,
  `data/processed/clean_listings.parquet`,
  `data/processed/feature_selection_report.md`'s Round 1 content
  (this spec **appends**; never overwrites Step 12's content),
  `notebooks/`, `migrations/`, `tests/conftest.py` (existing
  fixtures remain; new fixtures live in the new test files).
- `CLAUDE.md`'s "Implemented vs stub routes" table — this spec
  adds **no routes**. The `POST /predict` FastAPI route stays
  a Stub until a follow-on spec wires the trained model.

## New dependencies
- `xgboost` — for the `XGBRegressor` candidate. Already in the
  Step 01 stack per `02-TRD.md` §1 ("scikit-learn, XGBoost /
  LightGBM, SHAP"). Verify `pip freeze | grep -i xgboost`
  returns a pinned version; add with a pinned minor version if
  absent.
- `shap` — for `TreeExplainer` + mean |SHAP| ranking. Already in
  Step 01 stack per TRD §1. Verify the same way.
- `joblib` — already in Step 12's stack. Verify it's importable.
- **No** new npm packages, **no** Flask/FastAPI route additions,
  **no** new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The `model_registry.csv`
  is a flat file append, not a DB write.
- **No dealer/contact/media-URL fields ever reach the UI or an
  export.** The training script is offline-only; it does not log
  any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. Pinned by
  `test_training_script_does_not_log_contact_fields`.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol
  (Rules §2.1).** All six candidates are trained on the same
  70/15/15 split from `ml.features.split.split_train_val_test`
  (Step 12 — single source of truth for split boundaries).
  `random_state=42` is pinned both in the split helper and in
  every estimator's constructor. No re-sampling, no per-model
  split.
- **Log-transform the target (TRD §6.4, Step 12 contract).**
  Models train on `y = log1p(price)`. Metrics are reported on
  the original ₹ scale via `expm1` on predictions. The
  `Pipeline` does **not** include the inverse-transform — that
  lives in the FastAPI route layer (a later spec) so the same
  trained artifact can serve both the point estimate and the
  SHAP-on-log-scale explanation. The training script's
  `evaluate_model` handles the inverse explicitly before
  computing metrics.
- **Outliers excluded from training (Rules §1.4).** Step 4 of
  the training script filters to `is_outlier == False` before
  splitting. Outlier rows are present in the input parquet and
  remain in the analytics store; they are not in the training
  subset.
- **`transact_type` is a routing key (TRD §U-TRD-4, Rules
  §10.3).** The training script fits **two separate
  pipelines** — one for Sale, one for Rent. A single model
  trained on both is a hard rule violation. The Rent pipeline
  is skipped (with a logged INFO + `metrics_v1.json.rent.skipped
  = true`) if the Rent subset has fewer than 500 rows after
  outlier filtering — empirically the Rent volume is small in
  this dataset, and a model trained on n < 500 is unlikely to
  beat the Sale-only model on the val set, so the script
  surfaces this transparently rather than silently shipping a
  weak artifact.
- **The preprocessor is loaded, not refit (Rules §2.4).**
  Step 12's `feature_pipeline_v1.pkl` is the fitted
  preprocessor. The training script loads it via
  `load_feature_artifacts("v1")` and reuses the fitted
  instance — never calls `.fit()` again. This is the single
  most common source of train/serve skew and is explicitly
  disallowed (Rules §2.4).
- **All randomness is seeded (Rules §5.4).** Estimator
  constructors all carry `random_state=42`. Permutation
  importance uses `random_state=42`. SHAP `TreeExplainer` is
  deterministic for tree models given a fixed model + data.
- **Config values live in module constants.** `PRICE_MODEL_
  VERSION = "v1"`, the 6 candidate hyperparameters, the
  Rent-minimum threshold (`RENT_MIN_ROWS = 500`), and the
  `primary_metric = "val_rmse"` selection rule are all module
  constants. Future tuning edits the constant and logs the
  change in the Decision Log (not this spec's scope).
- **Versioned artifacts, never overwritten in place (Rules
  §2.5).** `save_price_model` writes
  `price_model_{transact}_v1.pkl`; a v2 training (future
  spec) writes `price_model_{transact}_v2.pkl` without
  touching v1. Same rule for `metrics_v1.json`.
- **`model_registry.csv` append is idempotent on re-run.**
  `append_model_registry` checks for an existing row with the
  same `(model_name, version, git_commit)` and skips the
  append if found. Re-running the training script with no
  data change produces no new CSV rows (deterministic +
  traceable).
- **SHAP only for tree-model winners (TRD §9).** Linear /
  Ridge / Lasso winners get a logged WARNING + the report's
  Round 3 section is filled with "n/a — non-tree model; SHAP
  ranking deferred to a tree-based winner." This is
  documented, not silently skipped.
- **Per-city test metrics are required (Rules §8.5).** The
  training script always computes `per_city_test` for the
  chosen model and includes it in `metrics_v1.json`. Cities
  with < 30 test rows get a logged WARNING but the row is
  still included (zero metrics would be more misleading than
  small-sample metrics).
- **No FastAPI imports in `ml/training/`.** This spec is
  offline-only. Wiring the trained model into the FastAPI
  route is a follow-on spec.
- **Logging uses stdlib `logging` only.** One module-level
  logger per file (`logger = logging.getLogger(__name__)`).
  INFO-level for stage boundaries; WARNING for
  expected-but-noteworthy conditions (Rent subset skipped,
  SHAP skipped for non-tree winner, city with n < 30 test
  rows); ERROR for hard failures.
- **No notebook-only steps (Rules §5.3).** Everything in this
  spec is reproducible via `python scripts/train_price_model.py`.
  No Jupyter cell fits an estimator or computes a metric that
  the script can't reproduce.

## Definition of done

1. `python -m pytest tests/test_candidates.py tests/test_evaluation.py
   tests/test_selection.py tests/test_training_persistence.py
   tests/test_train_price_model_script.py -v` from repo root runs
   and passes. Tests required (exact names):
   - **Candidates** (`test_candidates.py`):
     - `test_candidate_models_constant_has_six_entries` — pins
       `len(CANDIDATE_MODELS) == 6` and the names.
     - `test_make_estimator_returns_correct_class` — each name
       maps to the expected sklearn/XGBoost class.
     - `test_all_candidates_have_random_state_42` — for tree
       models, asserts `getattr(est, "random_state", None) == 42`.
   - **Evaluation** (`test_evaluation.py`):
     - `test_regression_metrics_returns_four_keys` — output
       dict has exactly `{r2, mae, rmse, mape}`.
     - `test_regression_metrics_inverse_transforms_from_log` —
       if `y_true` is in log space and `y_pred` matches, the
       output MAE on the original scale equals the original-
       scale MAE (catches a missed `expm1`).
     - `test_evaluate_model_returns_train_val_test_dict` —
       `evaluate_model` output has the three expected keys.
     - `test_per_city_metrics_warns_on_small_sample` — synthetic
       city with n=10 test rows triggers a logged WARNING.
   - **Selection** (`test_selection.py`):
     - `test_select_winner_returns_lowest_val_rmse` — synthetic
       candidate results, picks the one with the smallest
       `val_rmse`.
     - `test_select_winner_tie_breaks_on_val_mae` — same
       `val_rmse`, different `val_mae`, picks the lower MAE.
   - **Persistence** (`test_training_persistence.py`):
     - `test_save_price_model_writes_versioned_filename` — saves
       to `tmp_path`, asserts `price_model_sale_v1.pkl` exists
       in the resolved models dir.
     - `test_save_metrics_writes_versioned_filename` — saves a
       synthetic metrics dict, asserts `metrics_v1.json` parses
       back identically.
     - `test_append_model_registry_writes_header_on_first_call` —
       empty CSV → header row written on first append.
     - `test_append_model_registry_is_idempotent_on_rerun` —
       appending the same `(model_name, version, git_commit)`
       twice produces exactly one row.
     - `test_model_registry_csv_columns_match_backend_schema` —
       exact column order match against Backend Schema
       §U-SCHEMA-13.
   - **Script** (`test_train_price_model_script.py`):
     - `test_train_price_model_script_runs_end_to_end_on_synthetic_parquet`
       — write a tiny synthetic `clean_listings.parquet` to
       `tmp_path` (with both Sale + Rent rows so both pipelines
       train), set the env var `HOUSINGIQ_PROCESSED_DIR` (or a
       CLI flag) to point the script at it, run via
       `subprocess.run`, assert both model `.pkl` files +
       `metrics_v1.json` + an appended `feature_selection_report.md`
       + a 2-row `model_registry.csv` land.
     - `test_train_price_model_script_is_idempotent_on_rerun` —
       re-running the script with no input change leaves
       `model_registry.csv` at exactly 2 rows (no duplicates).
     - `test_train_price_model_script_skips_rent_when_too_small`
       — synthetic parquet with 0 Rent rows → Rent block in
       `metrics_v1.json` has `skipped: true` with a `reason`.
     - `test_training_script_does_not_log_contact_fields` —
       grep the captured stdout for any column name matching
       `(contact|dealer|phone|email|photo|url|spid)` — must be
       absent.
2. `python -m pytest -m "not realdata"` from repo root still
   passes (no real-data dependency introduced by this spec).
3. `ruff check ml/training/ scripts/train_price_model.py
   tests/test_candidates.py tests/test_evaluation.py
   tests/test_selection.py tests/test_training_persistence.py
   tests/test_train_price_model_script.py` reports zero issues.
4. `python -c "from ml.training import CANDIDATE_MODELS,
   evaluate_model, select_winner, save_price_model; from
   ml.training.candidates import make_estimator;
   print(len(CANDIDATE_MODELS))"` from repo root prints `6`
   without error — public API imports cleanly.
5. `python scripts/train_price_model.py` from repo root exits 0
   and prints the summary INFO line (chosen model per transact
   type + test R²/MAE/RMSE + artifacts path) — manual smoke test
   of the script entry point.
6. After running step 5, `models/metrics_v1.json` parses as
   valid JSON with both `sale` and `rent` blocks (or `rent.skipped`
   if Rent < threshold), contains all 6 candidates' metrics, and
   has a `git_commit` field matching `git rev-parse HEAD`.
7. `data/processed/feature_selection_report.md` after running
   step 5 contains Step 12's Round 1 content (preserved) **plus**
   appended Round 2 + Round 3 sections + a final "kept feature
   list & rationale" section.
8. `data/model_registry.csv` after running step 5 has exactly
   one row per trained `transact_type` (1 if Rent was skipped,
   2 if both ran) with all Backend Schema §U-SCHEMA-13 columns
   populated.
9. `git status` after committing shows only the new files listed
   above, the modified `requirements.txt`, the modified
   `scripts/run_pipeline.py`, and the modified `ml/__init__.py`.
   No accidental additions to `app/`, `api/`,
   `data/processed/clean_listings.parquet`, `data/raw/`, or
   `notebooks/`.
10. `CLAUDE.md`'s "Implemented vs stub routes" table is unchanged
    — this spec adds **no routes**. The `POST /predict` FastAPI
    route stays a Stub until a follow-on spec wires the trained
    model.
11. `07-TRACKER.md` is updated via `/update-tracker` to mark Days
    22 (Baseline models + CV metrics) and Days 26 (Pipeline wrap
    + serialize `price_model_v1.pkl`) as **Done** with the actual
    date and the metrics achieved (R², MAE on the test set). Days
    23–25 (RF + GB tuning, XGBoost, final model selection + SHAP
    validation, FastAPI `/predict` route) remain Not Started
    because they belong to the follow-on specs (improvement
    levers + FastAPI wiring).
