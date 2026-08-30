# Spec: Classification Model Training

## Overview
Train the HousingIQ Classification module's two classifiers using the leakage-safe feature frame (Spec 22) and the target labels (Spec 21). Per the v4 reframing in `01-PRD.md` §U4.1–§U4.2 and `08-RULES.md` §12, the module's **primary** deliverable is the `good_deal_verdict` 3-class classifier (`Good Deal` / `Fair Price` / `Overpriced`), optimized for **recall on the "Good Deal" class** — a missed good deal costs the user real money. The `price_tier` 4-class classifier (`Budget` / `Mid-Range` / `Premium` / `Luxury`) is **secondary**, powering the Recommender's "Fits my budget" filter and the Affordability chip. Both classifiers reuse the price model's feature set minus price-derived columns (Rules §8.1 + §12.4), evaluated with the fixed 70/15/15 protocol (`random_state=42`), and serialized as versioned `.pkl` artifacts under `models/`.

Module: **classification**.

## Depends on
- **Step 21** — `21-classification-target-definition` — supplies the target label files:
  - `data/processed/classifier/price_tier_labels.parquet` (with `price_tier` column)
  - `data/processed/classifier/good_deal_labels.parquet` (with `good_deal_verdict` column)
  - `data/processed/classifier/good_deal_thresholds.json` (per-city residual thresholds)
- **Step 22** — `22-classification-feature-reuse-from-price-schema` — supplies the classifier feature frame:
  - `data/processed/classifier/feature_frame.parquet` (with `split` column: `train`/`val`/`test`, `is_outlier`, 16 input-contract fields, engineered features, `locality_listing_count`, `has_<amenity>` flags; **no** `price`, `price_per_sqft`, `locality_avg_price_sqft`, `locality_smoothed_price`)
- **Step 12** — `12-feature-engineering-price-model` — supplies `ml.features.preprocessor.make_preprocessor` (the `ColumnTransformer` factory). The classifier training reuses this factory **without modification**; the leakage ban is enforced upstream by Spec 22 dropping price-derived columns before the preprocessor runs.
- **Step 15** — `15-price-model-evaluation-protocol` — supplies the fixed evaluation protocol (70/15/15, `random_state=42`, metrics on original scale, macro-F1 + per-class precision/recall/F1 + confusion matrix for classification).
- **`02-TRD.md` §U-TRD-1, §U-TRD-6, §U-TRD-8** — technical spec for classifier training: same input schema as regression, out-of-fold price predictions for `good_deal_verdict`, evaluation priority on Good Deal recall.
- **`05-BACKEND-SCHEMA.md` §U-SCHEMA-8, §U-SCHEMA-9** — final target field names (`good_deal_verdict`, `price_tier`) and model artifact filenames (`good_deal_classifier_v{n}.pkl`, `tier_classifier_v{n}.pkl`).
- **`08-RULES.md` §8, §12, §2.1, §2.3, §2.5** — classification rules (leakage ban, train-set-only boundaries, per-city evaluation, versioned artifacts), fixed split, leakage-safe aggregation, no overwrite.
- **`08-RULES.md` §13 + §15** — no MLflow/DVC/CI-CD; lightweight `model_registry` table (Backend Schema §U-SCHEMA-13) is the traceability mechanism.
- **`model-training-classification` skill** — key conventions: reuse price features minus price-derived, same split discipline, report per-class metrics + confusion matrix.

## Routes / Endpoints
No new routes/endpoints in this spec. The trained classifiers are serialized artifacts only. The FastAPI `/classify` route (which loads these artifacts) is a follow-on spec (Day 54 in the Tracker).

## Data / Schema changes

**Read:**
- `data/processed/classifier/feature_frame.parquet` (Spec 22) — the feature DataFrame with `listing_id`, `split`, `is_outlier`, and all classifier features.
- `data/processed/classifier/price_tier_labels.parquet` (Spec 21) — `listing_id` + `price_tier` (4-class).
- `data/processed/classifier/good_deal_labels.parquet` (Spec 21) — `listing_id` + `good_deal_verdict` (3-class, with `reason` for nulls).
- `data/processed/classifier/good_deal_thresholds.json` (Spec 21) — per-city residual thresholds (for reference/audit).

**Write (new directory `models/` — already exists):**
- `models/good_deal_classifier_v{n}.pkl` — full `sklearn.Pipeline` (preprocessing + final estimator) for the 3-class `good_deal_verdict` classifier. The preprocessing step is the **same `ColumnTransformer`** from `ml.features.preprocessor.make_preprocessor` (fit on the classifier feature frame's training subset). The estimator is the winning model from the candidate comparison (Logistic Regression baseline, Random Forest, XGBoost — selected by macro-F1 + Good Deal recall).
- `models/good_deal_classifier_metrics_v{n}.json` — evaluation metrics for the winning `good_deal_verdict` model:
  ```json
  {
    "model_version": "v1",
    "target": "good_deal_verdict",
    "train": {"accuracy": 0.72, "macro_f1": 0.68, "per_class": {"Good Deal": {"precision": 0.65, "recall": 0.82, "f1": 0.73}, "Fair Price": {...}, "Overpriced": {...}}, "confusion_matrix": [[...], [...], [...]]},
    "val":   {...},
    "test":  {...},
    "roc_auc_ovr": {"Good Deal": 0.87, "Fair Price": 0.79, "Overpriced": 0.84},
    "good_deal_recall": 0.82,
    "selected_estimator": "XGBoostClassifier",
    "hyperparameters": {...},
    "feature_count": 42,
    "n_train": 12345,
    "n_val": 2645,
    "n_test": 2645,
    "dataset_version": "clean_listings_2026-08-28.parquet",
    "git_commit": "abc1234"
  }
  ```
- `models/tier_classifier_v{n}.pkl` — full `sklearn.Pipeline` for the 4-class `price_tier` classifier (secondary). Same preprocessing, estimator selected by macro-F1 + multi-class ROC-AUC.
- `models/tier_classifier_metrics_v{n}.json` — evaluation metrics for the `price_tier` model (same schema as above, with 4 classes).
- `models/classifier_training_report.md` — append-only, one "Run YYYY-MM-DD HH:MM:SS" section per CLI run. Mirrors the style of `label_construction_report.md` / `feature_frame_report.md`:
  - Per-city `good_deal_verdict` class balance on train (so the reviewer sees if "Good Deal" is rare in Kolkata).
  - Per-city `price_tier` class balance on train.
  - Winning estimator name + hyperparameters for each classifier.
  - Full train/val/test metrics (accuracy, macro-F1, per-class P/R/F1, confusion matrix, ROC-AUC OvR).
  - **Good Deal recall** highlighted for the primary classifier.
  - A "Leakage audit" line: "Classifier feature frame verified to contain zero columns from {price, price_per_sqft, locality_avg_price_sqft, locality_smoothed_price}. Preprocessor fit on `is_outlier == False AND split == 'train'` only."
  - `dataset_version` and `git_commit`.

**No writes to:**
- `data/raw/`, `data/processed/clean_listings.parquet`, `data/processed/feature_selection_report.md`, `data/processed/analytics_cache/`, `data/app.db`, `data/model_registry.csv`.
- The input feature frame and label files (Spec 21/22 own them; this spec reads only).
- `notebooks/`.

## Templates / UI
None. This spec is offline model training — no Flask templates, no static assets, no HTML. The UI that surfaces classification results (`VerdictBadge`, `AffordabilityChip`, `/classify` page, Recommender tier filter, Analytics tile 14) is Tracker Day 55 — out of scope here.

## Files to change / Files to create

**Create:**
- `ml/classification/training.py` — the classifier training logic. Public API:
  - `CANDIDATE_ESTIMATORS: dict[str, tuple[type, dict]]` — pinned dictionary of candidate estimators and their default hyperparameter grids (mirrors the price model's `ml/training/candidates_v2.py` pattern but for classification):
    ```python
    {
        "LogisticRegression": (LogisticRegression, {"C": [0.1, 1.0, 10.0], "max_iter": [1000], "class_weight": ["balanced"]}),
        "RandomForestClassifier": (RandomForestClassifier, {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "class_weight": ["balanced", "balanced_subsample"]}),
        "XGBClassifier": (XGBClassifier, {"n_estimators": [200, 400], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}),
    }
    ```
    Order is the evaluation order; Logistic Regression is the baseline.
  - `PRIMARY_TARGET: str = "good_deal_verdict"` — the primary target column name.
  - `SECONDARY_TARGET: str = "price_tier"` — the secondary target column name.
  - `PRIMARY_LABEL_ORDER: tuple[str, ...] = ("Good Deal", "Fair Price", "Overpriced")` — pinned; matches `VERDICT_LABELS` from Spec 21.
  - `SECONDARY_LABEL_ORDER: tuple[str, ...] = ("Budget", "Mid-Range", "Premium", "Luxury")` — pinned; matches `TIER_LABELS` from Spec 21.
  - `SELECTION_METRIC_PRIMARY: str = "good_deal_recall"` — the metric that breaks ties for the primary classifier (recall on "Good Deal" class). Secondary tie-breaker: `macro_f1`.
  - `SELECTION_METRIC_SECONDARY: str = "macro_f1"` — the metric for the secondary classifier (macro-F1). Secondary tie-breaker: `roc_auc_ovr` (mean).
  - `MIN_GOOD_DEAL_RECALL: float = 0.70` — minimum acceptable recall on the "Good Deal" class for the primary classifier; if no candidate meets this, the CLI logs a WARNING and selects the best available (Rule §12.1: recall prioritized over accuracy).
  - `build_classifier_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline` — wraps the preprocessor + estimator into a single `sklearn.Pipeline` (mirrors the price model's pattern per Rules §2.4: "exact same Pipeline object used during evaluation").
  - `evaluate_classifier(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, label_order: tuple[str, ...], target_name: str) -> dict` — computes the full metric dict for one split (train/val/test): accuracy, macro-F1, per-class P/R/F1, confusion matrix (as list of lists, ordered by `label_order`), ROC-AUC OvR (per class), and for the primary target: `good_deal_recall` (recall of the "Good Deal" class). Returns a flat dict with keys prefixed by the split name (e.g., `train_accuracy`, `val_macro_f1`, `test_good_deal_recall`).
  - `train_one_target(features: pd.DataFrame, labels: pd.DataFrame, target_col: str, label_order: tuple[str, ...], preprocessor_factory: Callable, candidate_estimators: dict, selection_metric: str, min_recall: float | None, target_name: str, random_state: int = 42) -> tuple[Pipeline, dict, str]` — the core training loop for one target. Steps:
    1. Merge `features` + `labels` on `listing_id` (left join; features is the left frame so every feature row is preserved).
    2. Filter to `is_outlier == False` (training subset only; Rules §1.4 + §8.2).
    3. Split by the `split` column (already present from Spec 22) into train/val/test DataFrames.
    4. For each split, separate `X` (all columns except `listing_id`, `split`, `is_outlier`, `target_col`, `reason` if present) and `y` (the `target_col`).
    5. Drop rows where `y` is null (missing labels — e.g., `good_deal_verdict` null due to missing OOF predictions).
    6. Instantiate the preprocessor via `preprocessor_factory()` and **fit it on the training X only** (leakage-safe; Rules §2.3).
    7. Transform train/val/test X through the fitted preprocessor.
    8. For each candidate estimator: instantiate with default params, fit on transformed train, evaluate on transformed val via `evaluate_classifier`, record metrics.
    9. Select the winning estimator: highest `selection_metric` on val; if `min_recall` is set and no candidate meets it, log WARNING and pick the best `selection_metric` anyway.
    10. Refit the winning estimator on **train + val combined** (standard practice for final model; the test set remains untouched).
    11. Evaluate the refit pipeline on the held-out test set via `evaluate_classifier`.
    12. Return `(final_pipeline, full_metrics_dict, winning_estimator_name)`.
  - `train_both_classifiers(features_path: Path, price_tier_labels_path: Path, good_deal_labels_path: Path, preprocessor_factory: Callable, candidate_estimators: dict, output_dir: Path, model_version: str) -> tuple[dict, dict]` — the public entry point. Steps:
    1. Load `feature_frame.parquet`, `price_tier_labels.parquet`, `good_deal_labels.parquet`.
    2. Call `train_one_target` for `good_deal_verdict` (primary) with `SELECTION_METRIC_PRIMARY`, `MIN_GOOD_DEAL_RECALL`, `PRIMARY_LABEL_ORDER`.
    3. Call `train_one_target` for `price_tier` (secondary) with `SELECTION_METRIC_SECONDARY`, `min_recall=None`, `SECONDARY_LABEL_ORDER`.
    4. Serialize both pipelines to `output_dir / f"good_deal_classifier_{model_version}.pkl"` and `output_dir / f"tier_classifier_{model_version}.pkl"` via `joblib.dump`.
    5. Serialize both metrics dicts to `output_dir / f"good_deal_classifier_metrics_{model_version}.json"` and `output_dir / f"tier_classifier_metrics_{model_version}.json"`.
    6. Return the two metrics dicts.
    The function is **pure** with respect to its inputs; persistence is the caller's job (the CLI below).
- `ml/classification/training_report.py` — the report writer. Public API:
  - `write_classifier_training_report(*, output_dir: Path, good_deal_metrics: dict, tier_metrics: dict, good_deal_pipeline: Pipeline, tier_pipeline: Pipeline, feature_frame: pd.DataFrame, price_tier_labels: pd.DataFrame, good_deal_labels: pd.DataFrame, dataset_version: str, git_commit: str) -> Path` — appends one "Run YYYY-MM-DD HH:MM:SS" section to `models/classifier_training_report.md` (or writes the header on first call). Includes:
    - Per-city class balance for both targets on the training subset (`is_outlier == False AND split == 'train'`).
    - Winning estimator + hyperparameters for each classifier.
    - Full train/val/test metrics tables (markdown).
    - **Good Deal recall** called out for the primary classifier.
    - "Leakage audit" line (see Data / Schema changes).
    - `dataset_version` and `git_commit`.
  - `write_classifier_artifacts(good_deal_pipeline: Pipeline, tier_pipeline: Pipeline, good_deal_metrics: dict, tier_metrics: dict, output_dir: Path, model_version: str) -> tuple[Path, Path, Path, Path]` — writes the four `.pkl`/`.json` artifact files.
- `scripts/train_classifiers.py` — the CLI entry point. Invoked as:
  `python scripts/train_classifiers.py [--features-path data/processed/classifier/feature_frame.parquet] [--price-tier-labels-path data/processed/classifier/price_tier_labels.parquet] [--good-deal-labels-path data/processed/classifier/good_deal_labels.parquet] [--output-dir models] [--model-version v1] [--min-good-deal-recall 0.70]`. Steps:
  1. Parse args (one argparse group; reuses the same arg style as previous classifier CLIs).
  2. Import `ml.features.preprocessor.make_preprocessor` as the `preprocessor_factory`.
  3. Call `train_both_classifiers(...)` to get `(good_deal_metrics, tier_metrics)` and the two pipelines.
  4. Load the feature frame and label frames again for the report's per-city class balance.
  5. Call `write_classifier_artifacts(...)` to write the four model files.
  6. Call `write_classifier_training_report(...)` to append the run section.
  7. Print a one-line stdout summary: `"[OK] good_deal_classifier: {estimator} (test macro-F1={macro_f1:.3f}, Good Deal recall={recall:.3f}); tier_classifier: {estimator} (test macro-F1={macro_f1:.3f}); written to <output_dir>"`.
  8. Exit 0. Idempotent — a re-run with the same `model_version` overwrites the same four artifact files; a re-run with a different `model_version` writes a new set with the version in the filename (so multiple versions can coexist for comparison).

- `tests/test_classifier_training.py` — pytest tests for the training logic. Required tests (exact names):
  - `test_candidate_estimators_is_pinned` — `CANDIDATE_ESTIMATORS` has exactly the three keys above with the expected estimator classes.
  - `test_primary_target_is_pinned` — `PRIMARY_TARGET == "good_deal_verdict"`.
  - `test_secondary_target_is_pinned` — `SECONDARY_TARGET == "price_tier"`.
  - `test_primary_label_order_is_pinned` — `PRIMARY_LABEL_ORDER == ("Good Deal", "Fair Price", "Overpriced")`.
  - `test_secondary_label_order_is_pinned` — `SECONDARY_LABEL_ORDER == ("Budget", "Mid-Range", "Premium", "Luxury")`.
  - `test_selection_metric_primary_is_pinned` — `SELECTION_METRIC_PRIMARY == "good_deal_recall"`.
  - `test_selection_metric_secondary_is_pinned` — `SELECTION_METRIC_SECONDARY == "macro_f1"`.
  - `test_min_good_deal_recall_is_pinned` — `MIN_GOOD_DEAL_RECALL == 0.70`.
  - `test_build_classifier_pipeline_wraps_preprocessor_and_estimator` — synthetic preprocessor + estimator; assert the returned object is a `Pipeline` with two steps named `preprocessor` and `estimator`.
  - `test_evaluate_classifier_returns_all_metrics` — synthetic pipeline + X/y; assert the returned dict has keys for accuracy, macro_f1, per_class (with P/R/F1 for each label in `label_order`), confusion_matrix (list of lists, correct shape), roc_auc_ovr (per label), and for primary target: `good_deal_recall`.
  - `test_evaluate_classifier_confusion_matrix_order_matches_label_order` — synthetic 3-class problem; assert the confusion matrix rows/cols correspond to `label_order`.
  - `test_train_one_target_filters_outliers_and_null_labels` — synthetic frame with outliers and null `y`; assert those rows are excluded from train/val/test splits before fitting.
  - `test_train_one_target_fits_preprocessor_on_train_only` — pass a preprocessor factory that records which split it was fit on; assert it was called on the training X only.
  - `test_train_one_target_selects_by_selection_metric` — synthetic candidates where one has higher `selection_metric` on val; assert that candidate wins.
  - `test_train_one_target_respects_min_recall` — synthetic candidates where the best `selection_metric` has recall < `min_recall`; assert a WARNING is logged and the best `selection_metric` candidate is still selected (per Rules §12.1).
  - `test_train_one_target_refits_on_train_plus_val` — synthetic candidate; assert the final pipeline's estimator was fit on the combined train+val indices (check via a spy estimator that records fit indices).
  - `test_train_one_target_does_not_refit_on_test` — synthetic candidate; assert the test indices are never passed to `fit`.

- `tests/test_classifier_training_report.py` — pytest tests for the report writer. Required tests (exact names):
  - `test_write_classifier_training_report_writes_header_on_first_run` — empty `output_dir`; assert the report file exists with a header.
  - `test_write_classifier_training_report_appends_runs_not_overwrites` — call twice with different `git_commit`; assert both "Run" sections present.
  - `test_write_classifier_training_report_includes_leakage_audit_line` — assert the written "Run" section contains "Leakage audit" and lists the four dropped columns.
  - `test_write_classifier_training_report_includes_per_city_class_balance` — synthetic 4-city frame; assert each city appears with class counts for both targets.
  - `test_write_classifier_training_report_highlights_good_deal_recall` — assert the primary classifier's "Good Deal recall" value appears in the report.

- `tests/test_train_classifiers_cli.py` — pytest test for the CLI. Required tests (exact names):
  - `test_cli_runs_end_to_end_on_synthetic_artifacts` — build tiny synthetic `feature_frame.parquet` + `price_tier_labels.parquet` + `good_deal_labels.parquet` in `tmp_path`, invoke the CLI via `subprocess.run`, assert exit code 0 + the four artifact files exist in `tmp_path/output/`.
  - `test_cli_writes_deterministic_artifacts_on_rerun` — run the CLI twice with the same inputs; assert the `.pkl` files are byte-identical (joblib serialization is deterministic for the same object; no timestamp embedded).
  - `test_cli_does_not_log_contact_fields` — grep the captured stdout for any column name matching the regex `(contact|dealer|phone|email|photo|url|spid)` — must be absent.
  - `test_cli_output_pkl_contains_no_price_derived_columns` — load the produced `good_deal_classifier_v1.pkl`, inspect its `preprocessor` step's `feature_names_in_` (or the ColumnTransformer's `transformers_`); assert no column name is in `DROPPED_FOR_LEAKAGE` from Spec 22. Pins the leakage ban at the artifact level.

**Modify:**
- `ml/classification/__init__.py` — add `from ml.classification import training  # noqa: F401` and re-export the new public symbols: `train_both_classifiers`, `CANDIDATE_ESTIMATORS`, `PRIMARY_TARGET`, `SECONDARY_TARGET`, `PRIMARY_LABEL_ORDER`, `SECONDARY_LABEL_ORDER`, `SELECTION_METRIC_PRIMARY`, `SELECTION_METRIC_SECONDARY`, `MIN_GOOD_DEAL_RECALL`.
- `scripts/run_pipeline.py` — append one CLI line after the current classifier-feature-builder line (Spec 22's `scripts/build_classifier_features.py` invocation):
  ```python
  subprocess.run(
      [sys.executable, "scripts/train_classifiers.py", "--model-version", "v1"],
      check=False,
  )
  ```
  Same non-fatal pattern as Specs 21/22: pipeline prints the CLI's stdout summary regardless and proceeds to the next stage. Exit code surfaced in the pipeline's final summary.
- `requirements.txt` — no new packages; this spec only uses scikit-learn / XGBoost / joblib / pandas / numpy (already pinned) + stdlib. Verify with `pip freeze | grep -E "(scikit-learn|xgboost|joblib|numpy|pandas)"` and flag explicitly per CLAUDE.md "no new packages without checking first."

**No changes to:**
- `app/`, `api/`, `migrations/` (no Flask/FastAPI code in this spec; `/classify` route lands in follow-on spec).
- `data/raw/`, `data/processed/clean_listings.parquet`, `data/processed/feature_selection_report.md`, `data/processed/analytics_cache/`, `data/app.db`, `data/model_registry.csv`.
- `notebooks/`, `tests/conftest.py` (existing fixtures remain; new fixtures live in the new test files).
- `ml/features/*`, `ml/training/*` (read-only — this spec consumes `make_preprocessor` without modification).
- `CLAUDE.md`'s "Implemented vs stub routes" table — this spec adds **no routes**. The `POST /classify` FastAPI route stays a Stub until a follow-on spec wires the classifier trained here.

## New dependencies
None. The spec uses scikit-learn / XGBoost / joblib / pandas / numpy (already pinned) + stdlib (`argparse`, `json`, `logging`, `pathlib`, `subprocess`, `datetime`). **No** new pip/npm packages, **no** Flask/FastAPI route additions, **no** new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The spec reads Parquet + JSON, writes `.pkl` + `.json` + Markdown.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** The training code does not log any column whose name matches the regex `(contact|dealer|phone|email|photo|url|spid)`. Pinned by a test that greps the CLI's captured stdout for the regex — must be absent.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model evaluation must reference the fixed evaluation protocol (Rules §2.1).** The train/val/test split is the same `split` column from Spec 22 (which reused `split_train_val_test(target="price", random_state=42)`). Metrics are computed on the original scale (for classification, this means the label space — no inverse transform needed). Per-class P/R/F1 + confusion matrix + macro-F1 + ROC-AUC OvR are reported, not just accuracy (Rules §2.1 + `model-training-classification` skill).
- **Leakage ban (Rules §8.1 + §12.4).** The classifier feature frame (Spec 22) already drops all price-derived columns. This spec **verifies** the ban at the artifact level: the test `test_cli_output_pkl_contains_no_price_derived_columns` loads the trained pipeline and asserts its preprocessor's feature names contain none of `DROPPED_FOR_LEAKAGE`. The report's "Leakage audit" line documents the same.
- **`locality_listing_count` is kept (Spec 22 decision).** Count is not price-derived; it describes locality popularity. The Decision Log entry for Spec 22 documents this.
- **`luxury_category` is kept (Spec 22 decision).** Rules §10.2's "server-derived" rule applies to the API surface; offline consumption of the resolved value is allowed. The Decision Log entry for Spec 22 documents this.
- **`transact_type` is a routing key (Rules §10.3).** The classifier feature frame includes `transact_type` as a column. This spec trains **one global classifier per target** (not per-transact_type) — the feature frame already has `transact_type` as a one-hot encoded feature (via the shared preprocessor). If per-transact_type models prove better in a future iteration, that's a Spec 24+ concern. For v1, one global classifier per target is the simpler, auditable choice.
- **Reuse `make_preprocessor`, do not modify.** The function accepts the preprocessor factory; it instantiates and fits on the classifier training subset (`is_outlier == False AND split == 'train'`). This mirrors the price model's behavior and keeps the leakage-safe "fit on train only" rule.
- **`split` column reused, not recomputed.** The feature frame from Spec 22 already has the `split` column (train/val/test from `split_train_val_test(..., random_state=42)`). This spec reads it and splits by it — no re-splitting. A reviewer auditing "did the classifier see the same training subset as the regression?" gets the same answer either side.
- **Versioned artifacts, never overwritten in place (Rules §2.5).** The four artifact files include `model_version` in the filename (`good_deal_classifier_v1.pkl`, etc.). A re-run with the same `model_version` overwrites; a re-run with a different version writes a new file. The `classifier_training_report.md` is append-only.
- **No MLflow/DVC/etc. (Rules §13 + §15).** The `classifier_training_report.md` + `model_registry` table (Backend Schema §U-SCHEMA-13) are the audit trail.
- **Outlier rows are not deleted (Rules §1.4).** The training filters to `is_outlier == False`; the feature frame and label frames preserve all rows. The report's per-city class balance is computed on the training subset only.
- **Honest logging of shortfalls (Rules §9.2).** The report flags any city where "Good Deal" class has < 10 train rows, any per-class recall below the `MIN_GOOD_DEAL_RECALL` threshold, and any candidate that failed to meet the minimum recall. No "we probably have enough data" rationalization.
- **No notebook-only steps (Rules §5.3).** Everything in this spec is reproducible via `python scripts/train_classifiers.py --model-version v1`. No Jupyter cell computes a model the script can't reproduce.
- **No FastAPI imports in `ml/classification/`.** The training code is pure-Python; no FastAPI / httpx client imports. Keeps the trainer runnable in offline CI without a live inference service.
- **Logging uses stdlib `logging` only.** One module-level logger per file (`logger = logging.getLogger(__name__)`). INFO for stage boundaries; WARNING for expected-but-noteworthy conditions (per-city class count < 10, min-recall not met); ERROR for hard failures (missing input parquet, schema mismatch).
- **CLI never aborts the pipeline on non-zero exit.** The pipeline calls the CLI with `check=False`. The CLI's stdout summary is printed regardless; the pipeline proceeds. The CLI exits 0 on success and 1 only on a hard failure — missing input parquet, schema mismatch.
- **Single source of truth for candidate estimators.** `CANDIDATE_ESTIMATORS` is the only place the candidate list lives. The CLI, the training loop, and the tests all reference it. If a future spec adds a candidate, add it here once and it propagates.
- **`random_state=42` everywhere (Rules §5.4).** The candidate estimators that accept `random_state` (RandomForest, XGBoost) are instantiated with `random_state=42` in the training loop. The split is already pinned to 42 via Spec 22.

## Definition of done

1. `python -m pytest tests/test_classifier_training.py tests/test_classifier_training_report.py tests/test_train_classifiers_cli.py -v` from repo root runs and passes. Tests required (exact names):
   - **Training logic** (`test_classifier_training.py`):
     - `test_candidate_estimators_is_pinned`
     - `test_primary_target_is_pinned`
     - `test_secondary_target_is_pinned`
     - `test_primary_label_order_is_pinned`
     - `test_secondary_label_order_is_pinned`
     - `test_selection_metric_primary_is_pinned`
     - `test_selection_metric_secondary_is_pinned`
     - `test_min_good_deal_recall_is_pinned`
     - `test_build_classifier_pipeline_wraps_preprocessor_and_estimator`
     - `test_evaluate_classifier_returns_all_metrics`
     - `test_evaluate_classifier_confusion_matrix_order_matches_label_order`
     - `test_train_one_target_filters_outliers_and_null_labels`
     - `test_train_one_target_fits_preprocessor_on_train_only`
     - `test_train_one_target_selects_by_selection_metric`
     - `test_train_one_target_respects_min_recall`
     - `test_train_one_target_refits_on_train_plus_val`
     - `test_train_one_target_does_not_refit_on_test`
   - **Report writer** (`test_classifier_training_report.py`):
     - `test_write_classifier_training_report_writes_header_on_first_run`
     - `test_write_classifier_training_report_appends_runs_not_overwrites`
     - `test_write_classifier_training_report_includes_leakage_audit_line`
     - `test_write_classifier_training_report_includes_per_city_class_balance`
     - `test_write_classifier_training_report_highlights_good_deal_recall`
   - **CLI** (`test_train_classifiers_cli.py`):
     - `test_cli_runs_end_to_end_on_synthetic_artifacts`
     - `test_cli_writes_deterministic_artifacts_on_rerun`
     - `test_cli_does_not_log_contact_fields`
     - `test_cli_output_pkl_contains_no_price_derived_columns`
2. `python -m pytest -m "not realdata"` from repo root still passes (no real-data dependency introduced by this spec).
3. `ruff check ml/classification/training.py ml/classification/training_report.py ml/classification/__init__.py scripts/train_classifiers.py tests/test_classifier_training.py tests/test_classifier_training_report.py tests/test_train_classifiers_cli.py` reports zero issues.
4. `python -c "from ml.classification import train_both_classifiers, CANDIDATE_ESTIMATORS, PRIMARY_TARGET, SECONDARY_TARGET, PRIMARY_LABEL_ORDER, SECONDARY_LABEL_ORDER, SELECTION_METRIC_PRIMARY, SELECTION_METRIC_SECONDARY, MIN_GOOD_DEAL_RECALL; print(PRIMARY_TARGET, SECONDARY_TARGET, PRIMARY_LABEL_ORDER, SECONDARY_LABEL_ORDER)"` from repo root prints the expected pinned values without error — public API imports cleanly.
5. `python scripts/train_classifiers.py --model-version v1` from repo root exits 0 and prints the `[OK]` summary line — manual smoke test of the CLI.
6. After running step 5, the four artifact files exist at the canonical paths:
   - `models/good_deal_classifier_v1.pkl`
   - `models/good_deal_classifier_metrics_v1.json`
   - `models/tier_classifier_v1.pkl`
   - `models/tier_classifier_metrics_v1.json`
   And `models/classifier_training_report.md` contains a "Run YYYY-MM-DD HH:MM:SS" section with:
   - Per-city class balance for both targets on the training subset.
   - Winning estimator + hyperparameters for each classifier.
   - Full train/val/test metrics tables.
   - **Good Deal recall** highlighted for the primary classifier.
   - The "Leakage audit" line.
   - `dataset_version` and `git_commit`.
7. The artifact `.pkl` files **do not** contain any feature name in `DROPPED_FOR_LEAKAGE` — spot-check by loading the pipeline and inspecting `pipeline.named_steps['preprocessor'].feature_names_in_` (or the ColumnTransformer's transformers).
8. The `git_commit` recorded in the run's report section matches `git rev-parse HEAD` at the time the CLI was invoked.
9. `git status` after committing shows only the new files listed above, the modified `scripts/run_pipeline.py`, and the modified `ml/classification/__init__.py`. No accidental additions to `app/`, `api/`, `data/processed/clean_listings.parquet`, `data/processed/feature_selection_report.md`, `data/raw/`, `notebooks/`, `data/processed/analytics_cache/`, or `data/processed/classifier/{price_tier_labels,good_deal_labels,feature_frame,price_tier_quantile_boundaries,good_deal_thresholds}.parquet/json`.
10. `CLAUDE.md`'s "Implemented vs stub routes" table is unchanged — this spec adds **no routes**. The `POST /classify` FastAPI route stays a Stub until a follow-on spec (Day 54: serialize model + FastAPI /classify route + smoke test) wires the classifier trained here.
11. `07-TRACKER.md` is updated via `/update-tracker` to mark Day 52 (Train good_deal_verdict 3-class classifier) as **Done** with the actual measured Good Deal recall and macro-F1 from this spec's report. Day 53 (SHAP explanations + per-city confusion matrices) is moved from "Not Started" to "Not Started" with a `← needs Spec 23 trained models` note so the next session picks up from a clean handoff.
12. The Decision Log in `07-TRACKER.md` gains one new entry dated today with the rationale for: (a) the candidate estimator set (Logistic Regression baseline, Random Forest, XGBoost — no LightGBM in v1 to limit scope), (b) the selection metric priority (Good Deal recall for primary, macro-F1 for secondary), and (c) the single-global-classifier-per-target choice (vs per-transact_type) for v1. All three decisions are auditable from the report file in step 6 without rerunning any code.