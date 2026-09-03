# Implementation Plan: Spec 24 - Classification Model Evaluation

**Status:** Ready for execution
**Module:** Classification (Affordability & Investment-Tier Filter)
**Depends on:** Spec 23 (Classification Model Training), Spec 15 (Price Model Evaluation Protocol)
**Branch:** `feature/classification-model-evaluation`

---

## Overview

This plan implements the evaluation gate for the two Spec 23 classifiers by creating a new `ml/evaluation/classification/` package that mirrors Spec 15's `ml/evaluation/` structure exactly. The implementation follows the **ponytail** philosophy: reuse existing patterns, no new abstractions, minimum code that works.

---

## Phase 1: Core Package Structure (Foundation)

### 1.1 Create directory and `__init__.py`
- **File:** `ml/evaluation/classification/__init__.py`
- **Pattern:** Mirror `ml/evaluation/__init__.py` — re-export all public symbols
- **Exports:** protocol constants, `ClassificationEvaluationResult`, `evaluate()`, `format_summary()`, report functions, scoring functions, `protocol_split`, `__version__`
- **Version:** `"1.0.0"` (evaluator version, surfaced in every result)

### 1.2 Protocol Constants (`protocol.py`)
- **File:** `ml/evaluation/classification/protocol.py`
- **Pattern:** Mirror `ml/evaluation/protocol.py` — pinned constants with `Final` type hints
- **Constants:**
  - `SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}`
  - `RANDOM_STATE = 42`
  - `METRIC_NAMES = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "good_deal_recall", "price_tier_macro_f1")`
  - `CLASSIFIER_THRESHOLDS` dict with `good_deal` and `price_tier` sub-dicts
  - `PROTOCOL_VERSION = "1.0.0"`
  - `PROTOCOL_DOC_PATH = "docs/02-TRD.md"`
- **Import:** `RENT_MIN_ROWS` from `ml.training.candidates` (reused from price protocol)

### 1.3 Split Enforcement (`splits.py`)
- **File:** `ml/evaluation/classification/splits.py`
- **Pattern:** Mirror `ml/evaluation/splits.py` — single `protocol_split()` function
- **Implementation:** Two `train_test_split` calls (70/30 then 50/50 on 30%) with `stratify=df[target]`
- **Validation:** Raise `ValueError` if any class has < 2 members in test split
- **Returns:** `(train_df, val_df, test_df)` with reset indices

---

## Phase 2: Scoring & Gate Logic

### 2.1 Scoring Functions (`scoring.py`)
- **File:** `ml/evaluation/classification/scoring.py`
- **Pattern:** Mirror `ml/evaluation/scoring.py` — pure functions using sklearn metrics
- **Functions:**
  - `score_classifier(y_true, y_pred, target_name, labels)` → dict in `METRIC_NAMES` order
    - Computes: accuracy, precision_macro, recall_macro, f1_macro
    - `good_deal_recall` = recall for class 0 when `target_name=="good_deal"`
    - `price_tier_macro_f1` = alias for `f1_macro` when `target_name=="price_tier"`
  - `per_class_metrics(y_true, y_pred, labels)` → `{class_int: {precision, recall, f1, support}}`
  - `confusion_matrix_dict(y_true, y_pred, labels)` → `{"matrix": [...], "labels": [...]}`
- **Imports:** `from ml.evaluation.protocol import METRIC_NAMES` (shared constant)

### 2.2 Gate Evaluation (`gate.py`)
- **File:** `ml/evaluation/classification/gate.py`
- **Pattern:** Mirror `ml/evaluation/gate.py` — pure `evaluate()` function + frozen dataclass
- **Dataclass:** `ClassificationEvaluationResult` (all fields from spec Section 6)
- **Function:** `evaluate(model_path, version, target, processed_dir, models_dir, feature_frame_path, labels_dir)`
- **Steps (verbatim from spec):**
  1. Load `feature_frame.parquet` + appropriate labels parquet (`good_deal_labels.parquet` or `price_tier_labels.parquet`)
  2. Merge on `listing_id`, drop `is_outlier` rows
  3. `protocol_split` on target column (`good_deal_verdict` or `price_tier`)
  4. Load `classifier_preprocessor_v{version}.pkl` from `models_dir`
  5. Load model artifact (`good_deal_classifier_v{version}.pkl` or `price_tier_classifier_v{version}.pkl`)
  6. Transform X_train/X_val/X_test with preprocessor (note: pipeline already includes preprocessor)
  7. Predict on all three splits
  8. `score_classifier` on test split → metrics
  9. `per_class_metrics` + `confusion_matrix_dict` on test split
  10. Check thresholds from `CLASSIFIER_THRESHOLDS[target]`
  11. Return `ClassificationEvaluationResult`
- **Leakage Guard:** Assert no price-derived columns in feature columns passed to preprocessor:
  - `price_inr`, `price_per_sqft`, `locality_avg_price_sqft`, `locality_smoothed_price`
  - Raise `ValueError("leakage_detected: ...")` if found
- **Dataset Fingerprint:** SHA1 of first 1MB of feature_frame.parquet (reuse `_dataset_fingerprint` pattern)
- **Git Commit:** 12-char SHA via subprocess (reuse `_git_commit` pattern)
- **Evaluated At:** ISO8601 UTC

### 2.3 Report Persistence (`report.py`)
- **File:** `ml/evaluation/classification/report.py`
- **Pattern:** Mirror `ml/evaluation/report.py` — two functions
- **`write_evaluation_report(result, models_dir)`**
  - Writes `models/evaluation_reports/classification_eval_{target}_v{version}.json`
  - Uses `dataclasses.asdict()` for serialization
  - Creates directory if needed
  - Returns written `Path`
- **`append_protocol_section(result, report_path)`**
  - Appends `## Protocol Certification — {target} v{version}` section to `models/classification_training_report.md`
  - **Section content (Markdown):**
    - Protocol version + dataset fingerprint + git commit
    - Split sizes table
    - Test metrics table (METRIC_NAMES order)
    - Per-class metrics table
    - Confusion matrix (markdown table, reuse `_format_confusion_matrix` pattern from training_report.py)
    - Threshold checklist (✅/❌ per threshold)
    - Overall: **CERTIFIED** or **NOT CERTIFIED**
  - **Idempotent:** Replace existing section for same `target+version` (search for header, replace block)

---

## Phase 3: CLI Script

### 3.1 CLI Entry Point (`scripts/evaluate_classifiers.py`)
- **File:** `scripts/evaluate_classifiers.py`
- **Pattern:** Mirror `scripts/evaluate_price_model.py` exactly
- **Arguments:**
  - `--version` (required, e.g., "v1")
  - `--target` (repeatable, choices: "good_deal", "price_tier", required)
  - `--processed-dir` (default: env `HOUSINGIQ_PROCESSED_DIR` or `data/processed`)
  - `--models-dir` (default: env `HOUSINGIQ_ARTIFACT_DIR` or `models`)
  - `--feature-frame` (default: `data/processed/classifier/feature_frame.parquet`)
  - `--labels-dir` (default: `data/processed/classifier`)
  - `--report-path` (default: `models/classification_training_report.md`)
- **UTF-8 stdout/stderr reconfigure** (Windows cp1252 fix — copy from price script)
- **Git commit helper** (copy `_git_commit` from gate.py)
- **Main loop:** For each target:
  1. Build model path: `models_dir / f"{target}_classifier_{version}.pkl"`
  2. Check existence → error if missing
  3. Call `evaluate()` with all args
  4. Call `write_evaluation_report()` + `append_protocol_section()`
  5. Print one-line summary: `[PASS|FAIL] {target}_v{version} f1_macro={:.4f} good_deal_recall={:.4f} accuracy={:.4f}`
  6. Track overall RC (0 iff all pass)
- **Exit code:** 0 if all pass, 1 otherwise
- **Logging:** BasicConfig INFO level, same format as price script

---

## Phase 4: Test Suite (6 files)

### 4.1 `tests/test_classification_protocol.py`
- Verify all constants match spec values
- `PROTOCOL_VERSION` is semver string
- `CLASSIFIER_THRESHOLDS` has both keys with correct sub-keys

### 4.2 `tests/test_classification_splits.py`
- Fixture: 1000-row synthetic DataFrame with `good_deal_verdict` (3 classes) and `price_tier` (4 classes)
- Test `protocol_split` returns correct sizes (700/150/150 ±1)
- Test stratification preserves class proportions
- Test `random_state=42` yields identical splits
- Test `ValueError` when class has < 2 test members

### 4.3 `tests/test_classification_scoring.py`
- Fixture: synthetic y_true, y_pred arrays
- Test `score_classifier` returns all `METRIC_NAMES` keys in order
- Test `good_deal_recall` == recall for class 0 (target="good_deal")
- Test `price_tier_macro_f1` == `f1_macro` (target="price_tier")
- Test `per_class_metrics` returns correct structure
- Test `confusion_matrix_dict` returns JSON-serializable dict with correct shape

### 4.4 `tests/test_classification_gate.py`
- Fixture: synthetic feature_frame + labels + fitted preprocessor + trained model artifacts (use temp dirs)
- Test `evaluate()` returns `ClassificationEvaluationResult` with all fields populated
- Test leakage guard raises `ValueError` when price-derived columns present
- Test thresholds checked correctly: `overall_passed` = all thresholds pass
- Test `evaluated_at` valid ISO8601 UTC; `git_commit` 12-char or "unknown"
- Test dataset fingerprint changes when feature_frame content changes

### 4.5 `tests/test_classification_report.py`
- Fixture: temp dir, sample `ClassificationEvaluationResult`
- Test `write_evaluation_report` creates versioned JSON with correct structure
- Test `append_protocol_section` appends markdown with all required tables
- Test idempotency: running twice replaces (not duplicates) section

### 4.6 `tests/test_evaluate_classifiers_cli.py`
- **Pattern:** Mirror `tests/test_train_classifiers_cli.py` exactly
- Fixtures: `temp_dirs`, `synthetic_data` (same structure as training CLI tests)
- Tests:
  - CLI creates both evaluation report JSON files
  - CLI appends protocol sections to training report
  - CLI prints `[PASS]` or `[FAIL]` summary per target
  - Exit code 0 when both pass, 1 when any fails
  - Exits 1 with clear error when model artifact missing
  - Exits 1 with clear error when feature_frame or labels missing
  - Version arg reflected in output filenames

---

## Phase 5: Verification & Integration

### 5.1 Run test suite
```bash
pytest tests/test_classification_protocol.py -v
pytest tests/test_classification_splits.py -v
pytest tests/test_classification_scoring.py -v
pytest tests/test_classification_gate.py -v
pytest tests/test_classification_report.py -v
pytest tests/test_evaluate_classifiers_cli.py -v
```

### 5.2 Full integration test
```bash
# First ensure Spec 23 artifacts exist (run training if needed)
python scripts/train_classifiers.py --version 1 --features data/processed/classifier/feature_frame.parquet --labels data/processed/classifier --models-dir models --report models/classification_training_report.md

# Then run evaluation gate
python scripts/evaluate_classifiers.py --version v1 --target good_deal --target price_tier
```

### 5.3 Verify artifacts
- `models/evaluation_reports/classification_eval_good_deal_v1.json` exists
- `models/evaluation_reports/classification_eval_price_tier_v1.json` exists
- `models/classification_training_report.md` has two "## Protocol Certification" sections

### 5.4 Quality checks
```bash
# No new dependencies
pip check

# Lint
ruff check ml/evaluation/classification/ scripts/evaluate_classifiers.py tests/test_classification_*.py
```

---

## File Creation Order (Dependency Sequence)

| Order | File | Dependencies |
|-------|------|--------------|
| 1 | `ml/evaluation/classification/__init__.py` | — |
| 2 | `ml/evaluation/classification/protocol.py` | — |
| 3 | `ml/evaluation/classification/splits.py` | protocol.py |
| 4 | `ml/evaluation/classification/scoring.py` | protocol.py |
| 5 | `ml/evaluation/classification/gate.py` | protocol.py, splits.py, scoring.py, ml.features.persistence |
| 6 | `ml/evaluation/classification/report.py` | gate.py (for dataclass) |
| 7 | `scripts/evaluate_classifiers.py` | gate.py, report.py, protocol.py |
| 8 | `tests/test_classification_protocol.py` | protocol.py |
| 9 | `tests/test_classification_splits.py` | splits.py |
| 10 | `tests/test_classification_scoring.py` | scoring.py |
| 11 | `tests/test_classification_gate.py` | gate.py, protocol.py, splits.py, scoring.py |
| 12 | `tests/test_classification_report.py` | report.py, gate.py |
| 13 | `tests/test_evaluate_classifiers_cli.py` | CLI script, all above |

---

## Key Patterns to Reuse (No Reinvention)

| Pattern | Source | Reuse In |
|---------|--------|----------|
| `protocol_split` (70/15/15 stratified) | `ml/evaluation/splits.py` | `splits.py` |
| `_dataset_fingerprint` (SHA1 of parquet) | `ml/evaluation/gate.py` | `gate.py` |
| `_git_commit` (12-char SHA) | `ml/evaluation/gate.py` | `gate.py`, CLI |
| UTF-8 stdout reconfigure | `scripts/evaluate_price_model.py` | CLI |
| `write_evaluation_report` (JSON) | `ml/evaluation/report.py` | `report.py` |
| `append_protocol_section` (markdown) | `ml/evaluation/report.py` | `report.py` |
| `_format_confusion_matrix` | `ml/classification/training_report.py` | `report.py` |
| CLI structure (argparse, loop, RC) | `scripts/evaluate_price_model.py` | CLI |
| Test fixtures (temp dirs, synthetic data) | `tests/test_train_classifiers_cli.py` | All test files |
| `_run_cli` helper | `tests/test_train_classifiers_cli.py` | CLI test |

---

## Acceptance Checklist (from Spec §12)

- [ ] All 6 test files pass
- [ ] CLI certifies both classifiers against pinned thresholds
- [ ] Split is deterministic 70/15/15 stratified
- [ ] Scoring matches sklearn definitions exactly
- [ ] Leakage guard catches price-derived columns
- [ ] Reports written to `models/evaluation_reports/`
- [ ] Training report gets protocol section appended
- [ ] CLI exit code reflects overall pass/fail
- [ ] No new pip dependencies (`pip check` passes)

---

## Out of Scope (Do Not Implement)

- Per-city / per-locality breakdowns
- SHAP explainers
- FastAPI `/classify` integration
- Drift detection / A/B gate
- Any artifact types beyond what Spec 23 produces