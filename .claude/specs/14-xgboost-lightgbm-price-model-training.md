# Spec: XGBoost / LightGBM Price Model Training (v2)

## Overview
Train a **v2 boosted-tree price regression model** that improves on the
v1 baseline (Spec 13) by replacing the single global XGBoost candidate
with a **tuned + per-city / per-segment ensemble of XGBoost and
LightGBM**, evaluated against the same fixed 70/15/15 protocol
(`random_state=42`) and the same four headline metrics (R², MAE, RMSE,
MAPE on the original ₹ scale). This is the **first iteration of the
"30–35% MAE/RMSE reduction" tracking series** started by Spec 13:
every metric here is the reduction against the v1 baseline in
`models/metrics_v1.json`, not against an un-tuned single-model
reference. The winner is wrapped in an `sklearn.Pipeline(preprocessor
→ estimator)`, serialized via `joblib` to
`models/price_model_sale_v2.pkl` / `models/price_model_rent_v2.pkl`,
and paired with `models/metrics_v2.json` (same schema as v1 + a
`vs_v1` block quantifying the actual reduction). Module:
**price-prediction**.

This spec is the **first of three improvement-lever passes** the
literature doc (`09-LITERATURE-REVIEW`) calls out (TRD §U-TRD-2).
Levers 1–4 (stacking, Optuna tuning, geospatial features, smoothed
target encoding) all run inside this spec's training loop; levers
5–7 (SHAP-guided refinement, text-derived signal, outlier-robust
loss) land in the follow-on spec.

## Depends on
- **Step 13** — `13-baseline-regression-model-training` — produces
  `models/feature_pipeline_v1.pkl`, `models/feature_list_v1.json`,
  `models/price_model_sale_v1.pkl` / `price_model_rent_v1.pkl`
  (or `skipped: true` if Rent was too small), and
  `models/metrics_v1.json`. This spec consumes all four and
  quantifies improvement against them.
- **Step 12** — `12-feature-engineering-price-model` — provides the
  `LocalityAggregator` + fitted `ColumnTransformer` tuple used by
  the training script. The preprocessor is loaded, not refit
  (Rules §2.4). If a v2 preprocessor is needed, that's a separate
  spec — this one consumes v1 unchanged.
- **Step 11** — `11-price-prediction-input-schema-v3` — locks the
  16 input fields; the v2 model card documents the same field set.
- **Step 07** — `07-clean-listings-parquet-pipeline` — produces
  the canonical `data/processed/clean_listings.parquet`. The v2
  training script reads the same artifact.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` —
  provides `is_outlier` so training filters to
  `is_outlier == False`.
- **`ml/features/split.py`** — the 70/15/15 split helper with
  `random_state=42` (Step 12). Reused unchanged; v1 and v2 train
  on identical split boundaries, so metric deltas are attributable
  to model + levers, not to data drift.
- **`02-TRD.md` §10 + §U-TRD-2** — improvement-lever list,
  evaluation protocol, productionization checklist.
- **`05-BACKEND-SCHEMA.md` §6 + §U-SCHEMA-13** — model artifact
  filenames + `model_registry` row schema.
- **`08-RULES.md` §2.1–§2.5** — fixed evaluation protocol,
  train-only feature aggregation, versioned artifacts, paired
  metrics JSON.
- **`08-RULES.md` §5.4** — `random_state=42` everywhere.
- **`08-RULES.md` §13** — no MLOps tooling; this spec stays
  within the "script + versioned `.pkl` + JSON" pattern.
- **`09-LITERATURE-REVIEW-AND-IMPROVEMENT-PLAN.md`** — the 4
  base papers (B1–B4) + 18 supporting papers (S1–S18) cited as
  the justification for each lever. Every lever's Decision Log
  entry references the paper ID.
- **`12-feature-engineering-price-model.md` §"Rules for
  implementation"** — split helper, leakage rules,
  `transact_type`-as-routing-key conventions this spec inherits.

## Routes / Endpoints
No new routes/endpoints. This spec is offline model training +
serialization only. FastAPI wiring of `price_model_sale_v2.pkl` /
`price_model_rent_v2.pkl` into `POST /predict` is a separate spec
("FastAPI `/predict` route + smoke test") that consumes whichever
version is current at that time.

## Data / Schema changes
- **Read** `data/processed/clean_listings.parquet` (Step 07).
- **Read** `models/feature_pipeline_v1.pkl` (Step 12 — fitted
  preprocessor + locality aggregator tuple).
- **Read** `models/feature_list_v1.json` (Step 12 — final
  ordered feature list).
- **Read** `models/metrics_v1.json` (Step 13 — baseline metrics
  + chosen model name + per-city test metrics, used as the
  reduction baseline).
- **Write** `models/price_model_sale_v2.pkl` — `joblib.dump` of
  the full `sklearn.Pipeline` for Sale, preprocessor (re-loaded
  from v1 artifact, not refit) + the v2 winner estimator. If
  the v2 winner is a **stacked ensemble**, the `.pkl` holds the
  full `sklearn.ensemble.StackingRegressor` (or equivalent
  `Pipeline` wrapping it) — v1's `.pkl` is **not** overwritten.
- **Write** `models/price_model_rent_v2.pkl` — same, for Rent.
  Same skip rule as v1: if Rent rows < `RENT_MIN_ROWS` (500)
  after outlier filtering, write `metrics_v2.json.rent.skipped
  = true` and skip the artifact + registry row.
- **Write** `models/metrics_v2.json` — same schema as v1 plus
  a top-level `vs_v1` block quantifying the actual reduction
  (or shortfall) against the v1 chosen model:
  ```json
  {
    "version": "v2",
    "created_at": "<ISO timestamp>",
    "dataset_version": "clean_listings.parquet",
    "git_commit": "<sha>",
    "split": {"train": 0.70, "val": 0.15, "test": 0.15, "random_state": 42},
    "sale": { <same shape as v1 — candidates, chosen_model, chosen_metrics, per_city_test> },
    "rent": { <same shape, or {"skipped": true, "reason": "..."}> },
    "vs_v1": {
      "sale": {
        "v1_chosen_model": "<name from metrics_v1.json>",
        "v2_chosen_model": "<name>",
        "test_mae_pct_change": -12.4,
        "test_rmse_pct_change": -14.1,
        "test_r2_delta": 0.03,
        "target_pct_reduction": "30-35",
        "actual_pct_reduction_mae": 12.4,
        "actual_pct_reduction_rmse": 14.1,
        "target_met": false
      },
      "rent": { <same shape, or {"skipped": true}> }
    }
  }
  ```
  `pct_change = (v2 - v1) / v1 * 100` so a negative number means
  improvement (MAE/RMSE went down). `target_met` is `true` only
  if both MAE and RMSE reductions are ≥ 30% (the literature
  target from `09-LITERATURE-REVIEW` §5). Honest logging of
  shortfalls is a Rules §9.2 requirement.
- **Append** `data/processed/feature_selection_report.md` — adds
  one section "v2 — Improvement-Lever Pass 1" listing, per lever
  applied, (a) the lever name, (b) the literature paper ID
  (B1–B4 / S1–S18), (c) a one-line effect on validation MAE.
  The Round 1/2/3 sections from Steps 12/13 are preserved
  above; this spec only appends.
- **Append** `data/model_registry.csv` — one row per trained
  model per `transact_type`, matching the §U-SCHEMA-13 schema.
  `append_model_registry` is reused from Spec 13 and is
  idempotent on `(model_name, version, git_commit)` so re-runs
  don't duplicate rows.
- **No writes to `data/raw/`** (Rules §1.2).
- **No writes to `data/processed/clean_listings.parquet`**.
- **No writes to `data/processed/analytics_cache/`** — Week 6.
- **No application DB changes.**

## Templates / UI
None. Offline training only — no Flask templates, no static
assets, no HTML. The "30–35% improvement" figure lands in
`metrics_v2.json` and the Decision Log, not in the UI. The
recommender/insights UIs (later specs) read the same metrics
file for their own copy.

## Files to change / Files to create

**Create:**
- `ml/training/levers/` — new sub-package, one file per
  improvement lever, so each lever is independently testable +
  composable. Re-exports from `ml/training/levers/__init__.py`.
  - `__init__.py` — empty; re-exports the four lever
    constructors.
  - `optuna_search.py` — Lever 2 (Optuna Bayesian hyperparameter
    search for boosting models). Public API:
    - `OPTUNA_N_TRIALS: int = 40` — pinned constant. The
      literature (S8) recommends 30–50 trials for boosted trees;
      40 is a safe middle.
    - `OPTUNA_TIMEOUT_SEC: int | None = 600` — pinned upper
      bound; 10 minutes per candidate prevents runaway CI
      runs. If a trial hits the timeout, the best-so-far is
      returned and the run continues.
    - `optuna_search_xgb(X_train, y_train, X_val, y_val,
      n_trials: int = OPTUNA_N_TRIALS,
      timeout_sec: int | None = OPTUNA_TIMEOUT_SEC,
      random_state: int = 42) -> dict` — returns the best
      hyperparameters found + the best validation score.
      Search space (loose bounds, per literature):
      `max_depth ∈ [3, 10]`, `learning_rate ∈ [0.01, 0.3]`
      (log-uniform), `n_estimators ∈ [100, 1000]`,
      `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6,
      1.0]`, `min_child_weight ∈ [1, 10]`,
      `reg_alpha ∈ [1e-8, 1.0]` (log-uniform),
      `reg_lambda ∈ [1e-8, 1.0]` (log-uniform). Objective
      pinned to `reg:squarederror` with `tree_method="hist"`
      and `n_jobs=-1`. The Optuna study uses
      `sampler=TPESampler(seed=42)` for determinism. Logs
      each trial's value at DEBUG; logs the best value at
      INFO.
    - `optuna_search_lgbm(X_train, y_train, X_val, y_val,
      n_trials, timeout_sec, random_state) -> dict` — same
      shape, LightGBM-flavored search space:
      `num_leaves ∈ [15, 255]`, `max_depth ∈ [3, 10]`,
      `learning_rate ∈ [0.01, 0.3]`, `n_estimators ∈ [100,
      1000]`, `subsample ∈ [0.6, 1.0]`,
      `colsample_bytree ∈ [0.6, 1.0]`, `min_child_samples ∈
      [5, 50]`, `reg_alpha ∈ [1e-8, 1.0]`,
      `reg_lambda ∈ [1e-8, 1.0]`. Objective pinned to
      `regression` with `metric="rmse"` and
      `n_jobs=-1`. Same TPE sampler.
    - Both functions are **pure** (return a dict, do not
      side-effect the global Optuna logger beyond study-level
      logs). Pinned by tests that check the returned dict's
      keys and the fitted study's `best_value` against a
      hand-crafted 3-trial fixture.
  - `stacking.py` — Lever 1 (stacking ensemble). Public API:
    - `make_stacking_regressor(random_state: int = 42) ->
      StackingRegressor` — base learners per literature
      (B3, B4): `Ridge(alpha=1.0)`,
      `RandomForestRegressor(n_estimators=200, max_depth=None,
      random_state=42)`, `GradientBoostingRegressor(n_estimators=200,
      max_depth=4, learning_rate=0.05, random_state=42)`,
      `XGBRegressor(n_estimators=300, max_depth=6,
      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
      tree_method="hist", n_jobs=-1, random_state=42)` (v1
      defaults), `LGBMRegressor(...)` (v2 defaults). Meta-
      learner: `Ridge(alpha=1.0)` on out-of-fold base
      predictions. `cv=5`, `n_jobs=-1` on the stacking
      wrapper. The function takes the v1 + v2 estimator
      dicts (passed in) so the stacking layer composes
      whatever survived the candidate filter.
    - The function is factory-only (returns the unfitted
      `StackingRegressor`); fitting is the caller's job, so
      the same factory is reusable for both Sale and Rent.
  - `geospatial.py` — Lever 3 (distance-to-metro / distance-
    to-CBD features). Public API:
    - `METRO_STATIONS: dict[str, list[tuple[float,
      float]]]` — small per-city lookup table of (lat, lon)
      tuples for major metro stations per city. Pinned
      literal; values are documented in the module
      docstring with sources. *(ponytail: a 4-city × 5–10
      station lookup is ~30 lines; over-engineering this
      into a real spatial join or a GeoPandas pipeline is
      YAGNI at v1 — the lever is the *feature*, not the
      engineering.)*
    - `CITY_CENTERS: dict[str, tuple[float, float]]` —
      (lat, lon) per city center. Same provenance.
    - `haversine_km(lat1, lon1, lat2, lon2) -> float` —
      great-circle distance in km, pure stdlib math (no
      external geo dep).
    - `add_distance_features(df: pd.DataFrame) ->
      pd.DataFrame` — adds `distance_to_cbd_km` (min over
      all metro stations) + `distance_to_nearest_metro_km`
      per row. Uses `df["latitude"]` / `df["longitude"]`;
      rows with missing coords get NaN for both new
      columns (filled by the existing imputation step, not
      here). Pure function; no I/O.
    - The new features are added to `NUMERIC_FEATURES` via
      a documented constant `GEO_NUMERIC_FEATURES:
      tuple[str, ...] = ("distance_to_cbd_km",
      "distance_to_nearest_metro_km")` that the training
      script appends to the preprocessor's numeric block
      (via a `FeatureUnion`-style fit step — details in
      the script section below).
  - `target_encoding.py` — Lever 4 (smoothed target encoding
    for `sector` — already partially implemented in
    `LocalityAggregator` via `locality_smoothed_price`; this
    module extends with a **sector-level** smoother). Public
    API:
    - `SECTOR_SMOOTHING_PRIOR_WEIGHT: float = 20.0` — same
      as `SMOOTHING_PRIOR_WEIGHT` in Step 12. Pinned.
    - `class SectorTargetEncoder:` — sklearn-style fit /
      transform. **Filters to `is_outlier == False` on
      fit** (Rules §2.3). Computes per-`(city, sector)`
      Bayesian-smoothed mean of `price_per_sqft` toward the
      city mean. Stores the learned frame internally;
      `transform` is a left-join (never refits — Rules
      §8.2). Same LOO semantics as `LocalityAggregator`,
      but keyed on `(city, sector)` instead of `(city,
      locality)`, so it captures coarser-grained
      neighborhood price signal that complements the
      locality-level smoother. The new column
      `sector_smoothed_price` joins `NUMERIC_FEATURES`
      via the same `GEO_NUMERIC_FEATURES` pattern.

- `ml/training/evaluation.py` (MODIFY) — extend with v2
  metrics:
  - Add `vs_v1_metrics(v1_metrics: dict, v2_metrics: dict)
    -> dict` — returns the `vs_v1` block shape from
    `metrics_v2.json`. Computes `pct_change` against the
    v1 chosen model's test MAE/RMSE/R². Pure function.
  - Add `improvement_target_met(mae_pct_change: float,
    rmse_pct_change: float, target_pct: float = 30.0) ->
    bool` — `True` iff both reductions are ≥ target. Pinned
    default 30 matches the literature floor (Rules §9.2:
    "log the real number even if it comes in below").

- `ml/training/persistence.py` (MODIFY) — extend for v2:
  - Add `PRICE_MODEL_VERSION_V2: str = "v2"` — pinned
    constant.
  - `save_price_model(..., version: str = "v2")` — already
    accepts a version arg from v1; this spec passes
    `"v2"`. No code change needed; the v1 constant is
    the default for backward compatibility, but the v2
    script explicitly passes `"v2"`.
  - Add `load_v1_metrics(version: str = "v1") -> dict` —
    loads `models/metrics_{version}.json` and returns the
    parsed dict. Used by the v2 script to compute `vs_v1`.
    Raises `FileNotFoundError` with the expected path if
    the v1 metrics file is missing (so a fresh clone that
    skipped v1 surfaces the missing-dependency cleanly).

- `ml/training/report.py` (MODIFY) — extend for v2:
  - Add `write_v2_lever_section(lever_results: list[dict],
    out_path: Path) -> None` — appends the "v2 —
    Improvement-Lever Pass 1" section to
    `feature_selection_report.md`. Each entry in
    `lever_results` is a dict `{lever_name, paper_id,
    val_mae_before, val_mae_after, delta}`; the section
    formats them as a markdown table. Pinned by a test
    that checks the section header + a sample row.
  - The existing `write_round2_section` /
    `write_round3_section` from v1 are NOT re-called — v2
    appends only, never overwrites prior rounds.

- `scripts/train_price_model_v2.py` — the one entry point.
  Invoked as `python scripts/train_price_model_v2.py` from
  repo root. Idempotent (re-running with same git commit +
  same data appends nothing to `model_registry.csv`).
  Steps:
  1. Load `data/processed/clean_listings.parquet`.
  2. Filter to `is_outlier == False`.
  3. Split into Sale / Rent subsets per `transact_type`
     (Rent skip rule identical to v1: skip if n < 500
     after outlier filtering).
  4. For each subset with n >= 500:
     a. Call `split_train_val_test(df, target="price")`
        from `ml.features.split` to get the 70/15/15 split
        (same helper, same seed → identical boundaries
        to v1).
     b. Build the feature frame via
        `ml.features.feature_frame.build_feature_frame(df)`.
     c. Apply the fitted locality aggregator +
        preprocessor loaded from
        `models/feature_pipeline_v1.pkl` via
        `ml.features.persistence.load_feature_artifacts("v1")`.
        **The preprocessor is loaded, not refit** (Rules
        §2.4).
     d. **Lever 3** (geospatial): call
        `geospatial.add_distance_features(df)` to add
        `distance_to_cbd_km` and
        `distance_to_nearest_metro_km`. Add them to the
        numeric block of the preprocessor via a
        `FeatureUnion` (or, more simply, by re-fitting a
        sibling `StandardScaler` over the new columns
        and `hstack`-ing the result onto the preprocessor
        output). *(ponytail: simplest correct thing —
        avoid full `FeatureUnion` machinery; a single
        `StandardScaler` fit on the new columns, then
        `np.hstack` at transform time, is one screen of
        code and avoids a custom transformer class.)*
     e. **Lever 4** (sector target encoding): fit
        `SectorTargetEncoder` on the **train subset only**
        (Rules §2.3); transform the full subset. Add
        `sector_smoothed_price` to the numeric block the
        same way as the geo features.
     f. Build the v2 candidate list:
        - `"xgb_v1_defaults"` — same hyperparameters as
          v1's `XGBRegressor` (for direct comparability).
        - `"xgb_optuna"` — hyperparameters from
          `optuna_search_xgb(X_train, y_train, X_val,
          y_val)`.
        - `"lgbm_v1_defaults"` — `LGBMRegressor(
          n_estimators=500, max_depth=8, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8,
          min_child_samples=20, random_state=42,
          n_jobs=-1, verbose=-1)`.
        - `"lgbm_optuna"` — hyperparameters from
          `optuna_search_lgbm(X_train, y_train, X_val,
          y_val)`.
        - `"stacking"` — `make_stacking_regressor()` with
          the 4 surviving tree/linear base learners from
          the v1 candidate dict + the v2 XGB + LGBM
          defaults. Meta-learner is `Ridge(alpha=1.0)`.
     g. For each candidate name:
        - Build `Pipeline([("preproc", preproc), ("est",
          make_estimator(name))])`. For the stacking
          candidate, the inner estimator is the
          `StackingRegressor` from step f (which already
          contains the base learners + meta-learner).
        - Call `evaluate_model(...)` to get train/val/
          test metrics + `per_city_metrics(...)` for the
          test slice.
        - Append to `candidate_results[name]`.
     h. Run `select_winner(candidate_results)` to pick
        the name. Same selection rule as v1: lowest
        `val_rmse`, tie-break `val_mae`.
     i. Call `save_price_model(winner_pipeline,
        transact_type, version="v2")`.
     j. Accumulate into the v2 `metrics.json` payload:
        `candidate_results`, `chosen_model`,
        `chosen_metrics`, `per_city_test`.
  5. Load v1 metrics via `load_v1_metrics("v1")`. Compute
     `vs_v1` via `vs_v1_metrics(v1_test, v2_test)`. Log
     the actual reduction at INFO. If `target_met ==
     false`, log a WARNING with the actual % and the
     target % (Rules §9.2).
  6. Build the `metrics_v2.json` payload (with
     `git_commit`, `dataset_version`, `created_at`,
     `split`, `sale`, `rent`, `vs_v1`).
  7. Call `save_metrics(payload, version="v2")`.
  8. **SHAP for the v2 winner**: same TreeExplainer
     logic as v1 — append Round 3 to
     `feature_selection_report.md` (the spec appends
     a new "v2 — SHAP top-N" subsection inside Round 3
     with the v2 model's `mean |SHAP value|` ranking;
     the v1 SHAP table is preserved above).
  9. Compute Round 2 (tree + permutation importance) for
     the v2 candidate set; append to the report.
     *(ponytail: this is mechanical — same code as
     v1, re-run on the v2 models — pin to a small
     helper that takes a model + an X_val, not a
     bespoke v2 function.)*
  10. Write the v2 lever section via
      `write_v2_lever_section(lever_results, ...)`. Each
      row's `paper_id` comes from the literature doc:
      - Lever 1 (stacking) → B3, B4.
      - Lever 2 (Optuna) → S8.
      - Lever 3 (geospatial) → B2.
      - Lever 4 (target encoding) → S12.
  11. Append `data/model_registry.csv` with one row per
      `transact_type` (or skip the row if the subset was
      too small). Idempotent on `(model_name, version,
      git_commit)`.
  12. Log a summary line at INFO: chosen model per
      transact type + test R²/MAE/RMSE + actual % MAE
      reduction vs v1 + target-met boolean + artifacts
      at `<path>`.

- `tests/test_optuna_search.py` — pytest tests for the
  Optuna search wrappers. Uses a tiny synthetic
  regression fixture (sklearn `make_regression`),
  `n_trials=3` to keep the test fast, asserts the
  returned dict has the expected keys
  (`best_params`, `best_value`) and that the fitted
  study's `best_value` is finite. Pinned by
  `test_optuna_search_returns_best_params_dict`,
  `test_optuna_search_respects_random_state`,
  `test_optuna_search_xgb_and_lgbm_have_separate_search_spaces`.
- `tests/test_stacking.py` — pytest tests for
  `make_stacking_regressor`. Asserts the returned
  object is a `StackingRegressor` with the expected
  number of base estimators (5) and a `Ridge`
  meta-learner. Fast (tiny synthetic fixture, no
  Optuna).
- `tests/test_geospatial.py` — pytest tests for
  `haversine_km`, `add_distance_features`, and the
  pinned `METRO_STATIONS` / `CITY_CENTERS` tables.
  Asserts Delhi–Mumbai distance is within a known
  tolerance (~1400 km, the literature-accepted
  great-circle distance) — pins the haversine
  implementation.
- `tests/test_target_encoding.py` — pytest tests for
  `SectorTargetEncoder`. Mirrors the v1
  `LocalityAggregator` tests:
  `test_sector_target_encoder_fit_computes_group_means`,
  `test_sector_target_encoder_excludes_outliers_from_fit`,
  `test_sector_target_encoder_leave_one_out_semantics`,
  `test_sector_target_encoder_transform_does_not_refit`.
- `tests/test_v2_evaluation.py` — pytest tests for
  `vs_v1_metrics` + `improvement_target_met`:
  `test_vs_v1_metrics_computes_pct_change_correctly`,
  `test_improvement_target_met_true_at_30pct`,
  `test_improvement_target_met_false_below_30pct`.
- `tests/test_v2_report.py` — pytest tests for
  `write_v2_lever_section`: asserts the appended
  section header + a sample row's `paper_id` field
  appear in the output markdown.
- `tests/test_train_price_model_v2_script.py` — pytest
  test that runs `scripts/train_price_model_v2.py`
  against a tiny synthetic `clean_listings.parquet`
  (built in `tmp_path` by the test, with both Sale
  and Rent rows so both pipelines train), asserts:
  - both `price_model_sale_v2.pkl` and
    `price_model_rent_v2.pkl` exist after the run,
  - `metrics_v2.json` parses and contains both `sale`
    and `rent` blocks with the v2 candidate keys
    (xgb_v1_defaults, xgb_optuna, lgbm_v1_defaults,
    lgbm_optuna, stacking) + a `vs_v1` block,
  - `feature_selection_report.md` contains a
    "v2 — Improvement-Lever Pass 1" section header,
  - `model_registry.csv` has exactly 2 new rows
    appended for v2 (one per transact type) and no
    duplicates on re-run.

**Modify:**
- `scripts/run_pipeline.py` — append one line:
  `subprocess.run([sys.executable,
  "scripts/train_price_model_v2.py"], check=True)`
  after the Step 13 v1-training line, so v2 is
  reproducible end-to-end via `make pipeline`
  (TRD §13). No logic change; just sequence
  registration.
- `requirements.txt` — verify `lightgbm` is already
  pinned (Step 01 baseline). If absent, add with a
  pinned minor version. Flag the addition explicitly
  per CLAUDE.md "no new packages without checking
  first." `optuna` is also required for Lever 2 —
  add with a pinned minor version. Verify both via
  `pip freeze | grep -i <pkg>` before adding.
- `ml/training/__init__.py` — re-export
  `make_stacking_regressor`, `optuna_search_xgb`,
  `optuna_search_lgbm`, `add_distance_features`,
  `SectorTargetEncoder`, `vs_v1_metrics` so they're
  importable consistently with v1.
- `ml/__init__.py` — add `from ml import training  #
  noqa: F401` (already done by v1; no change needed,
  but the new `ml.training.levers` submodule is
  re-exported from `ml/training/__init__.py`).

**No changes** to:
- `app/`, `api/`, `data/raw/`,
  `data/processed/clean_listings.parquet`,
  `data/processed/feature_selection_report.md`'s
  Round 1/2/3 content from Steps 12/13 (v2 appends
  only).
- `notebooks/`, `migrations/`,
  `tests/conftest.py` (existing fixtures remain; new
  fixtures live in the new test files).
- `CLAUDE.md`'s "Implemented vs stub routes" table —
  this spec adds **no routes**.

## New dependencies
- `lightgbm` — for the `LGBMRegressor` candidate +
  Optuna search. Already in Step 01's stack per
  `02-TRD.md` §1 ("scikit-learn, XGBoost / LightGBM,
  SHAP"). Verify `pip freeze | grep -i lightgbm`
  returns a pinned version; add with a pinned minor
  version if absent.
- `optuna` — for the Bayesian hyperparameter search
  in Lever 2. **Not** in Step 01's stack — add
  explicitly. Pin to a recent minor (3.x at time of
  writing); `pip install optuna==3.6.1` (or current
  stable) with a one-line comment in
  `requirements.txt` citing literature S8.
- **No** new npm packages, **no** Flask/FastAPI
  route additions, **no** new DB drivers.
- **No** new scikit-learn or XGBoost versions — the
  v1 versions are reused.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The
  `model_registry.csv` is a flat-file append.
- **No dealer/contact/media-URL fields ever reach
  the UI or an export.** The v2 training script is
  offline-only; it does not log any column whose
  name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`.
  Pinned by
  `test_v2_training_script_does_not_log_contact_fields`.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no
  templates.
- **Model changes must reference the fixed
  evaluation protocol (Rules §2.1).** All v2
  candidates are trained on the same 70/15/15 split
  from `ml.features.split.split_train_val_test` (the
  same helper v1 uses, with the same `random_state
  =42`). v1 and v2 train on **identical** split
  boundaries, so the `vs_v1` metric deltas are
  attributable to the model + levers, not to data
  drift. Every estimator's constructor carries
  `random_state=42`. Optuna's sampler is seeded
  (`TPESampler(seed=42)`).
- **Log-transform the target (TRD §6.4, Step 12
  contract).** Same as v1: models train on
  `y = log1p(price)`. Metrics on the original ₹
  scale via `expm1`. The `Pipeline` does **not**
  include the inverse-transform; the FastAPI route
  layer (later spec) does it.
- **Outliers excluded from training (Rules §1.4).**
  Same as v1: filter to `is_outlier == False`
  before splitting. Outlier rows are present in the
  input parquet and remain in the analytics store.
- **`transact_type` is a routing key (TRD
  §U-TRD-4, Rules §10.3).** Same as v1: two
  separate pipelines (Sale + Rent), each
  independently trained. Rent is skipped with
  `skipped: true` if n < 500 after outlier
  filtering.
- **The preprocessor is loaded, not refit (Rules
  §2.4).** `feature_pipeline_v1.pkl` is loaded
  unchanged. The new geo + sector-encoded columns
  are added via a sibling `StandardScaler` +
  `np.hstack`, **not** by refitting the v1
  preprocessor. The fitted `SectorTargetEncoder`
  is **not** saved as part of the v1 preprocessor
  artifact — it's saved alongside, in a new
  `models/sector_target_encoder_v2.pkl`, and
  loaded by the v2 training script + future
  serving paths.
- **Leakage rule is hard (Rules §2.3, §8.4).**
  `SectorTargetEncoder.fit()` runs on
  `train_df[is_outlier == False]` only. `transform`
  is a pure join; it never refits. The
  leave-one-out semantics (per row, exclude own
  contribution from the group mean) are inherited
  from `LocalityAggregator` and pinned by
  `test_sector_target_encoder_leave_one_out_semantics`.
- **All randomness is seeded (Rules §5.4).**
  `random_state=42` on every estimator, on
  Optuna's `TPESampler`, on
  `permutation_importance`'s `n_repeats=10`.
- **Config values live in module constants.**
  `OPTUNA_N_TRIALS = 40`, `OPTUNA_TIMEOUT_SEC =
  600`, `RENT_MIN_ROWS = 500`,
  `SECTOR_SMOOTHING_PRIOR_WEIGHT = 20.0`,
  `IMPROVEMENT_TARGET_PCT = 30.0`,
  `PRICE_MODEL_VERSION_V2 = "v2"` are all module
  constants. Future tuning edits the constant and
  logs the change in the Decision Log.
- **Versioned artifacts, never overwritten in
  place (Rules §2.5).** `price_model_sale_v2.pkl`
  and `price_model_rent_v2.pkl` are new files; v1
  artifacts are **not** touched. Same for
  `metrics_v2.json` (a new file alongside v1).
- **`model_registry.csv` append is idempotent on
  re-run.** `append_model_registry` from v1 is
  reused; v2 rows have `version="v2"` and the same
  `(model_name, version, git_commit)` idempotency
  check skips duplicates.
- **Honest logging of shortfalls (Rules §9.2).**
  If `target_met == false` (i.e. v2's MAE/RMSE
  reduction is < 30%), the script logs a WARNING
  with the actual % and the target %, and
  `metrics_v2.json.vs_v1.target_met` is `false`.
  The Decision Log entry MUST cite the actual
  measured number, not the target. No "we
  probably would have hit it with more compute"
  rationalization.
- **Lever results are paper-cited (Rules §9.3).**
  Every row in the v2 lever section of
  `feature_selection_report.md` includes a
  `paper_id` from the literature doc (B1–B4,
  S1–S18). The Decision Log entry for "chose
  stacking as the v2 candidate" cites B3, B4;
  the entry for "chose Optuna for hyperparameter
  search" cites S8; etc.
- **No FastAPI imports in `ml/training/`.** This
  spec is offline-only. Wiring v2 (or whichever
  version is current) into the FastAPI route is a
  follow-on spec.
- **Logging uses stdlib `logging` only.** Same
  pattern as v1: one module-level logger per file
  (`logger = logging.getLogger(__name__)`);
  INFO for stage boundaries; WARNING for
  expected-but-noteworthy conditions (Rent
  skipped, target not met, city with n < 30 test
  rows); ERROR for hard failures.
- **Optuna's study-level logging is silenced.**
  Optuna is chatty by default; set
  `optuna.logging.set_verbosity(optuna.logging.WARNING)`
  once in the search wrappers so trial-by-trial
  output doesn't drown the stdlib logs. The
  `best_value` is logged at INFO once per search.
- **No notebook-only steps (Rules §5.3).**
  Everything in this spec is reproducible via
  `python scripts/train_price_model_v2.py`. No
  Jupyter cell fits an estimator, runs an Optuna
  study, or computes a metric that the script
  can't reproduce.

## Definition of done

1. `python -m pytest tests/test_optuna_search.py
   tests/test_stacking.py tests/test_geospatial.py
   tests/test_target_encoding.py
   tests/test_v2_evaluation.py tests/test_v2_report.py
   tests/test_train_price_model_v2_script.py -v`
   from repo root runs and passes. Tests required
   (exact names):
   - **Optuna** (`test_optuna_search.py`):
     - `test_optuna_search_returns_best_params_dict`
       — returned dict has `best_params` + `best_value`.
     - `test_optuna_search_respects_random_state` —
       two calls with the same seed return the same
       `best_value` (within a small tolerance).
     - `test_optuna_search_xgb_and_lgbm_have_separate_search_spaces`
       — XGB search includes `max_depth`; LGBM
       search includes `num_leaves`.
   - **Stacking** (`test_stacking.py`):
     - `test_make_stacking_regressor_returns_stacking_regressor`
       — type check.
     - `test_stacking_has_five_base_learners` — pin
       the base-learner count.
     - `test_stacking_meta_learner_is_ridge` — type
       check.
   - **Geospatial** (`test_geospatial.py`):
     - `test_haversine_delhi_mumbai_is_approximately_1400km`
       — pins the implementation against the
       literature-accepted great-circle distance.
     - `test_add_distance_features_adds_two_columns` —
       output has both
       `distance_to_cbd_km` and
       `distance_to_nearest_metro_km`.
     - `test_add_distance_features_handles_missing_coords`
       — NaN latitude/longitude produces NaN for
       both new columns, not an exception.
     - `test_metro_stations_constant_has_all_four_cities`
       — every city in the dataset has ≥ 1 entry.
   - **Target encoding** (`test_target_encoding.py`):
     - `test_sector_target_encoder_fit_computes_group_means`
       — synthetic 3-city, 5-sector fixture.
     - `test_sector_target_encoder_excludes_outliers_from_fit`
       — outlier rows' prices do not appear in the
       group means.
     - `test_sector_target_encoder_leave_one_out_semantics`
       — pin the LOO semantic (own contribution
       excluded from the group mean).
     - `test_sector_target_encoder_transform_does_not_refit`
       — fit on frame A; transform on frame B with
       a new (city, sector) inserted; assert the
       inserted row's column equals the city mean.
   - **v2 evaluation** (`test_v2_evaluation.py`):
     - `test_vs_v1_metrics_computes_pct_change_correctly`
       — synthetic v1 + v2 metrics; assert
       `mae_pct_change == (v2_mae - v1_mae) / v1_mae
       * 100`.
     - `test_improvement_target_met_true_at_30pct` —
       both MAE and RMSE reductions ≥ 30% returns
       `True`.
     - `test_improvement_target_met_false_below_30pct`
       — any single reduction < 30% returns
       `False`.
   - **v2 report** (`test_v2_report.py`):
     - `test_write_v2_lever_section_appends_header`
       — output markdown contains the "v2 —
       Improvement-Lever Pass 1" header.
     - `test_write_v2_lever_section_includes_paper_id`
       — a sample row's `paper_id` (e.g. "B3, B4")
       appears in the output.
   - **v2 script** (`test_train_price_model_v2_script.py`):
     - `test_train_price_model_v2_script_runs_end_to_end_on_synthetic_parquet`
       — write a tiny synthetic
       `clean_listings.parquet` to `tmp_path`
       (with both Sale + Rent rows so both
       pipelines train), set the env var
       `HOUSINGIQ_PROCESSED_DIR` (or a CLI flag)
       to point the script at it, run via
       `subprocess.run`, assert both v2 model
       `.pkl` files + `metrics_v2.json` + an
       appended `feature_selection_report.md` + a
       2-row `model_registry.csv` land.
     - `test_train_price_model_v2_script_is_idempotent_on_rerun`
       — re-running the script with no input
       change leaves `model_registry.csv` with no
       new v2 rows (idempotency check).
     - `test_train_price_model_v2_script_skips_rent_when_too_small`
       — synthetic parquet with 0 Rent rows →
       Rent block in `metrics_v2.json` has
       `skipped: true` with a `reason`.
     - `test_v2_training_script_does_not_log_contact_fields`
       — grep the captured stdout for any column
       name matching
       `(contact|dealer|phone|email|photo|url|spid)`
       — must be absent.
     - `test_v2_metrics_vs_v1_block_is_populated` —
       the v2 `metrics_v2.json` contains a
       `vs_v1.sale` block with
       `v1_chosen_model`, `v2_chosen_model`,
       `test_mae_pct_change`, `target_met`.
2. `python -m pytest -m "not realdata"` from repo
   root still passes (no real-data dependency
   introduced by this spec).
3. `ruff check ml/training/ scripts/
   tests/test_optuna_search.py tests/test_stacking.py
   tests/test_geospatial.py tests/test_target_encoding.py
   tests/test_v2_evaluation.py tests/test_v2_report.py
   tests/test_train_price_model_v2_script.py` reports
   zero issues.
4. `python -c "from ml.training import
   make_stacking_regressor, optuna_search_xgb,
   optuna_search_lgbm, add_distance_features,
   SectorTargetEncoder, vs_v1_metrics; print('ok')"`
   from repo root prints `ok` without error —
   public API imports cleanly.
5. `python scripts/train_price_model_v2.py` from
   repo root exits 0 and prints the summary INFO
   line (chosen model per transact type + test
   R²/MAE/RMSE + actual % MAE reduction vs v1 +
   target-met boolean + artifacts path) — manual
   smoke test of the script entry point.
6. After running step 5, `models/metrics_v2.json`
   parses as valid JSON with both `sale` and
   `rent` blocks (or `rent.skipped` if Rent <
   threshold), contains all 5 v2 candidates'
   metrics (xgb_v1_defaults, xgb_optuna,
   lgbm_v1_defaults, lgbm_optuna, stacking), has
   a `vs_v1` block, and has a `git_commit` field
   matching `git rev-parse HEAD`.
7. `data/processed/feature_selection_report.md`
   after running step 5 contains Steps 12/13's
   Round 1/2/3 content (preserved) **plus** an
   appended "v2 — Improvement-Lever Pass 1"
   section with one row per lever applied,
   including the `paper_id` citation.
8. `data/model_registry.csv` after running step 5
   has exactly one row per trained
   `transact_type` for v2 (1 if Rent was skipped,
   2 if both ran) with all Backend Schema
   §U-SCHEMA-13 columns populated. Re-running
   the script does not append duplicate rows.
9. `git status` after committing shows only the
   new files listed above, the modified
   `requirements.txt`, the modified
   `scripts/run_pipeline.py`, and the modified
   `ml/training/__init__.py` +
   `ml/training/evaluation.py` +
   `ml/training/persistence.py` +
   `ml/training/report.py`. No accidental
   additions to `app/`, `api/`,
   `data/processed/clean_listings.parquet`,
   `data/raw/`, or `notebooks/`.
10. `CLAUDE.md`'s "Implemented vs stub routes"
    table is unchanged — this spec adds **no
    routes**. The `POST /predict` FastAPI route
    stays a Stub until a follow-on spec wires
    whichever model version is current.
11. `07-TRACKER.md` is updated via
    `/update-tracker` to mark Day 56 (Improvement
    levers 1–4 on the regression model) as
    **Done** with the actual date and the
    measured MAE/RMSE reduction vs v1 (target
    30–35%, actual number logged honestly even
    if below target per Rules §9.2). Day 57
    (Levers 5–7 + final metrics_v3.json) remains
    Not Started.
