# Plan: Spec 14 — XGBoost / LightGBM Price Model Training (v2)

## Context
Spec 13 trained a **v1 baseline** (6 candidate models: Linear, Ridge, Lasso, RF,
GB, XGB) and shipped `models/price_model_{sale,rent}_v1.pkl` +
`models/metrics_v1.json` + a 2-row `model_registry.csv`. The literature doc
(`09-LITERATURE-REVIEW`) calls for a **30–35% MAE/RMSE reduction** over an
un-tuned baseline, achieved across three improvement-lever passes. Spec 14
delivers the **first pass** — levers 1–4 (stacking, Optuna tuning, geospatial
features, sector target encoding) — and writes **v2 artifacts alongside v1
without touching them** (Rules §2.5: versioned, never overwritten in place).
The `vs_v1` block in `metrics_v2.json` quantifies the actual reduction
honestly, even if it comes in below target (Rules §9.2).

The v1 codebase is the foundation: every module in `ml/training/` is
reusable, and v2 only **adds** (no breaking changes). The plan is structured
so each lever ships as an independently testable unit before the script
wires them together.

## Reusable v1 code (do not rewrite)
- `ml/features/split.py` — `split_train_val_test` with `random_state=42` →
  same split boundaries as v1, so deltas are attributable to model/levers.
- `ml/features/feature_frame.py::build_feature_frame` — 16-field feature
  frame builder.
- `ml/features/persistence.py::load_feature_artifacts("v1")` — fitted
  preprocessor + locality aggregator (load, never refit, Rules §2.4).
- `ml/training/candidates.py::make_estimator` + `CANDIDATE_MODELS` — v1's
  XGB defaults (reused as v2's `xgb_v1_defaults` candidate for direct
  comparability).
- `ml/training/evaluation.py` — `regression_metrics`, `evaluate_subset`,
  `per_city_metrics` (log→₹ inverse, MAPE epsilon guard). No change.
- `ml/training/selection.py::select_winner` — same rule (lowest val_rmse,
  tie-break val_mae).
- `ml/training/persistence.py` — `save_price_model`, `save_metrics`,
  `append_model_registry` already accept a `version` arg; v2 just passes
  `"v2"`.
- `ml/training/report.py::append_round_2_3` — used as-is for the
  Round 2/3 re-emission on v2 candidates.

## Step 0 — Branch + dependencies
1. **Verify branch** — already on
   `feature/xgboost-lightgbm-price-model-training`. If not, `git checkout
   feature/xgboost-lightgbm-price-model-training`.
2. **Add deps to `requirements.txt`** — verify `lightgbm==4.5.0` (already
   pinned per Step 01) and add `optuna==3.6.1` (or current stable, 3.x)
   with a one-line comment citing literature S8. Spec §"New dependencies".
   No other additions.

## Step 1 — Lever modules (sub-package `ml/training/levers/`)
Each lever ships in its own file under a new sub-package. The script in
Step 4 composes them.

### 1a. `ml/training/levers/__init__.py`
- Re-exports the four lever symbols so callers can write
  `from ml.training.levers import optuna_search_xgb, ...`.
- Spec §"Files to change / Files to create" → Create.

### 1b. `ml/training/levers/optuna_search.py` (Lever 2)
- Pinned constants: `OPTUNA_N_TRIALS = 40`, `OPTUNA_TIMEOUT_SEC = 600`.
- `optuna_search_xgb(X_train, y_train, X_val, y_val, ...) -> dict` —
  returns `{"best_params": {...}, "best_value": float}`. Search space per
  spec: `max_depth ∈ [3, 10]`, `learning_rate ∈ loguniform[0.01, 0.3]`,
  `n_estimators ∈ [100, 1000]`, `subsample ∈ [0.6, 1.0]`,
  `colsample_bytree ∈ [0.6, 1.0]`, `min_child_weight ∈ [1, 10]`,
  `reg_alpha`/`reg_lambda ∈ loguniform[1e-8, 1.0]`.
  `TPESampler(seed=42)`, objective `reg:squarederror`, `tree_method="hist"`,
  `n_jobs=-1`. Silences Optuna via
  `optuna.logging.set_verbosity(optuna.logging.WARNING)` at import.
- `optuna_search_lgbm(...)` — LGBM-flavored: `num_leaves ∈ [15, 255]`,
  `min_child_samples ∈ [5, 50]`, rest same shape. Objective `regression`,
  `metric="rmse"`, `n_jobs=-1`.
- Both functions are **pure** (return a dict; no global state mutation
  beyond the silenced logger).
- **Test** `tests/test_optuna_search.py` (3 tests per spec):
  `test_optuna_search_returns_best_params_dict`,
  `test_optuna_search_respects_random_state` (n_trials=3 to keep fast),
  `test_optuna_search_xgb_and_lgbm_have_separate_search_spaces`.

### 1c. `ml/training/levers/stacking.py` (Lever 1)
- `make_stacking_regressor(random_state: int = 42) -> StackingRegressor`
  with 5 base learners per spec: `Ridge(alpha=1.0)`,
  `RandomForestRegressor(n_estimators=200, random_state=42)`,
  `GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
  random_state=42)`, `XGBRegressor(v1 defaults, random_state=42)`,
  `LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
  random_state=42, n_jobs=-1, verbose=-1)`. Meta-learner:
  `Ridge(alpha=1.0)`, `cv=5`, `n_jobs=-1`. Factory only — caller fits.
- **Test** `tests/test_stacking.py` (3 tests): type check, 5 base learners,
  Ridge meta-learner.

### 1d. `ml/training/levers/geospatial.py` (Lever 3)
- Pinned literals (with documented sources in module docstring):
  - `METRO_STATIONS: dict[str, list[tuple[float, float]]]` — 5–10
    stations per city for all 4 cities.
  - `CITY_CENTERS: dict[str, tuple[float, float]]` — (lat, lon) per city.
- `haversine_km(lat1, lon1, lat2, lon2) -> float` — pure stdlib math
  (no GeoPandas / shapely dep).
- `add_distance_features(df: pd.DataFrame) -> pd.DataFrame` — adds
  `distance_to_cbd_km` (min over all metro stations) and
  `distance_to_nearest_metro_km`. NaN coords → NaN in both new columns.
- `GEO_NUMERIC_FEATURES: tuple[str, ...] = ("distance_to_cbd_km",
  "distance_to_nearest_metro_km")` — pinned constant the script uses to
  extend the preprocessor's numeric block.
- **Test** `tests/test_geospatial.py` (4 tests): haversine pins Delhi↔Mumbai
  ≈ 1400 km, two new columns added, NaN-coord handling, METRO_STATIONS
  covers all 4 cities.

### 1e. `ml/training/levers/target_encoding.py` (Lever 4)
- Pinned constant: `SECTOR_SMOOTHING_PRIOR_WEIGHT: float = 20.0`
  (same as `SMOOTHING_PRIOR_WEIGHT` in Step 12).
- `class SectorTargetEncoder:` — sklearn-style fit/transform. **Filters
  to `is_outlier == False` on `fit`** (Rules §2.3). Computes per-
  `(city, sector)` Bayesian-smoothed mean of `price_per_sqft` toward the
  city mean, using **leave-one-out semantics** per row (own contribution
  excluded from the group sum) — same LOO pattern as Step 12's
  `LocalityAggregator`. `transform` is a pure left-join (Rules §8.2);
  unseen `(city, sector)` falls back to the city mean. Exposes
  `fitted_aggregates_` + `city_priors_` attributes for tests.
- Save the fitted encoder as `models/sector_target_encoder_v2.pkl` (new
  artifact, alongside v2 models) via `joblib.dump`. The script loads it
  in Step 4.
- **Test** `tests/test_target_encoding.py` (4 tests, mirroring Step 12's
  `LocalityAggregator` tests): group means, outlier exclusion, LOO
  semantics, transform-doesn't-refit.

## Step 2 — Extend v1 modules for v2 metrics
Each modification is small and additive — v1 callers are unaffected.

### 2a. `ml/training/evaluation.py` (MODIFY)
- Add `vs_v1_metrics(v1_metrics: dict, v2_metrics: dict) -> dict` —
  returns the `vs_v1` block per spec. Pure function, no I/O.
- Add `IMPROVEMENT_TARGET_PCT: float = 30.0` constant + helper
  `improvement_target_met(mae_pct_change: float, rmse_pct_change: float,
  target_pct: float = IMPROVEMENT_TARGET_PCT) -> bool` — `True` iff both
  reductions ≥ target.
- **Test** `tests/test_v2_evaluation.py` (3 tests per spec): pct_change
  arithmetic, target-met true at 30%, target-met false below 30%.

### 2b. `ml/training/persistence.py` (MODIFY)
- Add `PRICE_MODEL_VERSION_V2: str = "v2"` constant.
- Add `load_metrics(version: str = "v1") -> dict` — loads
  `models/metrics_{version}.json`. Raises `FileNotFoundError` with the
  expected path on missing file. v1's `save_metrics` and
  `save_price_model` already accept a `version` kwarg; no change there.
- (No new test file — `load_metrics` is exercised by the script test
  in Step 4; a tiny in-line test in `test_training_persistence.py` is
  added if practical.)

### 2c. `ml/training/report.py` (MODIFY)
- Add `write_v2_lever_section(lever_results: list[dict], out_path: Path) ->
  None` — atomic append to `feature_selection_report.md` (same
  tempfile + `Path.replace` pattern as `append_round_2_3`). Each
  `lever_results` entry is
  `{"lever_name", "paper_id", "val_mae_before", "val_mae_after", "delta"}`.
  Renders a markdown table.
- **Test** `tests/test_v2_report.py` (2 tests): header appears, paper_id
  appears in output.

### 2d. `ml/training/__init__.py` (MODIFY)
- Re-export the new symbols: `make_stacking_regressor`,
  `optuna_search_xgb`, `optuna_search_lgbm`, `add_distance_features`,
  `SectorTargetEncoder`, `vs_v1_metrics`, `improvement_target_met`,
  `load_metrics`, `write_v2_lever_section`, `PRICE_MODEL_VERSION_V2`.
- Spec §"Modify" → `ml/training/__init__.py`.

## Step 3 — Candidates module extension
### 3a. `ml/training/candidates.py` (MODIFY)
- Add v2 candidate factory: `V2_CANDIDATE_MODELS: dict[str, Callable]`
  where values are zero-arg factories returning fresh estimators. Names:
  `xgb_v1_defaults` (clone of v1's XGB), `xgb_optuna` (placeholder
  fitted by script after Optuna call), `lgbm_v1_defaults`
  (`LGBMRegressor(...)` per spec), `lgbm_optuna` (placeholder), `stacking`
  (`make_stacking_regressor()`). The `make_v2_estimator(name, **overrides)`
  factory: for the two `_optuna` names, takes a `params` dict and
  instantiates the corresponding regressor with those params. For
  `stacking`, just returns `make_stacking_regressor()`.
- This module stays pure (no Optuna calls) — the script does the search
  then asks the factory for a fitted-ready estimator.
- Spec §"Files to change" → v2 candidate factory.

## Step 4 — v2 training script
### 4a. `scripts/train_price_model_v2.py` (CREATE)
- Mirrors `scripts/train_price_model.py` structure but:
  1. Loads `clean_listings.parquet` + filters `is_outlier == False`
     (same as v1).
  2. For each `transact_type` subset:
     a. `split_train_val_test` (Step 12 helper, identical seed).
     b. `load_feature_artifacts("v1")` (Rules §2.4).
     c. **`add_distance_features(df)`** (Lever 3) — adds 2 columns.
     d. **`SectorTargetEncoder().fit(train_df[is_outlier==False])` then
        `transform(df)`** (Lever 4) — adds `sector_smoothed_price`.
     e. **Build a sibling `StandardScaler` over the 3 new columns**;
        `np.hstack` the scaled new columns onto the v1 preprocessor
        output at transform time. Wrap as a small `GeoSectorScaler`
        helper inside the script (ponytail: avoid a custom sklearn
        transformer class — one helper function, 8 lines).
     f. **Optuna search** for XGB and LGBM (Lever 2) on `X_train`/
        `y_train`/`X_val`/`y_val` — pure call, returns `best_params`.
     g. For each of the 5 v2 candidates: build
        `Pipeline([("preproc", v1_preproc), ("geo", GeoSectorScaler()),
        ("est", make_v2_estimator(name, params=...))])`,
        call `evaluate_subset` (v1, unchanged), append to
        `candidate_results[name]`.
     h. `select_winner(candidate_results)` (v1, unchanged).
     i. `save_price_model(winner_pipe, transact_type, version="v2")`
        (v1 helper, new version).
     j. Save `sector_target_encoder_v2.pkl` next to the model
        (the encoder is needed by future serving paths, not by the
        script itself once training is done).
  3. Load v1 metrics via `load_metrics("v1")`; compute
     `vs_v1_metrics(...)` per transact type; log the actual % reduction
     at INFO + WARNING if `target_met == false` (Rules §9.2).
  4. Build `metrics_v2.json` payload (schema per spec) + `save_metrics(
     payload, version="v2")`.
  5. **Round 2/3**: re-use `append_round_2_3` from v1 with the v2
     candidate models' importances + SHAP for the v2 winner.
  6. Append the v2 lever section via `write_v2_lever_section` with
     paper IDs: Lever 1 → B3, B4; Lever 2 → S8; Lever 3 → B2; Lever 4 →
     S12.
  7. `append_model_registry` for v2 rows (idempotency check by
     `(model_name, version, git_commit)` skips duplicates).
  8. Summary INFO log: chosen model + test metrics + actual % MAE
     reduction vs v1 + `target_met` + artifacts path.
- CLI mirrors v1: `--parquet`, `--artifact-dir`, `--report-path`,
  `--registry-csv`. Reuses `HOUSINGIQ_*` env vars where applicable.
- **Test** `tests/test_train_price_model_v2_script.py` (5 tests per
  spec): end-to-end on synthetic parquet, idempotent re-run, rent
  skip-when-too-small, no-PII-logged (regex assertion on captured
  stdout), `vs_v1.sale` block populated in `metrics_v2.json`.

### 4b. `scripts/run_pipeline.py` (MODIFY)
- Append one line after the v1 training call:
  `subprocess.run([sys.executable, "scripts/train_price_model_v2.py"],
  check=True)`. Idempotent.

## Step 5 — Run + verify
1. `pip install -r requirements.txt` (pulls in `optuna`).
2. `python -m pytest tests/test_optuna_search.py tests/test_stacking.py
   tests/test_geospatial.py tests/test_target_encoding.py
   tests/test_v2_evaluation.py tests/test_v2_report.py
   tests/test_train_price_model_v2_script.py -v` — all green.
3. `python -m pytest -m "not realdata"` — no realdata tests broken.
4. `ruff check ml/training/ scripts/ tests/test_optuna_search.py
   tests/test_stacking.py tests/test_geospatial.py
   tests/test_target_encoding.py tests/test_v2_evaluation.py
   tests/test_v2_report.py tests/test_train_price_model_v2_script.py`
   — zero issues.
5. `python -c "from ml.training import make_stacking_regressor,
   optuna_search_xgb, optuna_search_lgbm, add_distance_features,
   SectorTargetEncoder, vs_v1_metrics; print('ok')"` — imports clean.
6. `python scripts/train_price_model_v2.py` — exits 0, summary INFO
   line printed.
7. Inspect `models/metrics_v2.json` — has `sale` + `rent` blocks (or
   `skipped`), 5 v2 candidate keys, `vs_v1` block, `git_commit` matches
   `git rev-parse HEAD`.
8. Inspect `data/processed/feature_selection_report.md` — Step 12/13
   content preserved, "v2 — Improvement-Lever Pass 1" appended with
   paper IDs.
9. Inspect `data/model_registry.csv` — 1 row (Rent skipped) or 2 rows
   (both ran) for v2. Re-run: no duplicates.
10. `git status` — only the expected new + modified files (no
    accidental writes to `app/`, `api/`, `data/raw/`,
    `data/processed/clean_listings.parquet`).
11. Update `07-TRACKER.md` via `/update-tracker` — mark Day 56 Done with
    actual % MAE reduction (honest, even if < 30% target, per Rules
    §9.2). Day 57 remains Not Started.

## Risks / open questions
1. **Synthetic-parquet test for Optuna** — Optuna is slow; the test
   fixture uses `n_trials=3` + `make_regression(n_samples=200)`. If CI
   times out, lower to `n_trials=2` + `n_samples=100` and pin in the
   test. Flag to user only if real-data run later shows the search
   needs more budget.
2. **METRO_STATIONS / CITY_CENTERS provenance** — the spec asks for
   sources in the module docstring but the v1 codebase has no precedent
   for source-cited literals. Plan: cite the city-center coordinates
   as "Google Maps geocoding, 2026" (generic) and metro station lists
   as "operator-published station coordinates, 2026." If the user has
   a specific source preference, they should flag before merge.
3. **Stacking `cv=5` overhead** — the stacking candidate's
   `StackingRegressor(cv=5, n_jobs=-1)` does 5-fold OOF prediction for
   each of the 5 base learners. On the full ~38k-row Sale subset
   (post-outlier), this is ~2–5 minutes. Acceptable for a one-shot
   training script; if it becomes a CI bottleneck, drop to `cv=3` and
   log the change in the Decision Log.
4. **Optuna timeout behavior** — `OPTUNA_TIMEOUT_SEC = 600` per spec.
   If a search times out mid-trial, Optuna returns the best-so-far
   (no error). The script logs the actual `best_value` and continues.
   Not a hard failure — just a smaller budget than `n_trials=40`.
5. **`save_metrics` re-run overwrite** — v1's `save_metrics` overwrites
   in place for the same version (the v1 comment in the script
   acknowledges this). v2 follows the same pattern: re-running with
   the same git commit + same data overwrites `metrics_v2.json`. This
   is consistent with v1's behavior and Rules §2.5 (the v1 file is
   not touched). Flag only if the user wants v2 to use a content-hash
   version string instead — would require touching persistence layer.
6. **v2 candidate list size** — 5 candidates × 2 splits (Sale, Rent) =
   10 pipeline fits + 2 Optuna studies (40 trials each). Total
   end-to-end training time is estimated 10–20 minutes on a single
   workstation. Acceptable for a one-shot v2 training; if it exceeds
   the user's tolerance, lower `OPTUNA_N_TRIALS` to 20 in a follow-on
   commit (and log in Decision Log).
