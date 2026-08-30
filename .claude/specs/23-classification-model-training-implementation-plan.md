# Implementation Plan: Spec 23 — Classification Model Training

**Spec:** `.claude/specs/23-classification-model-training.md`  
**Branch:** `feature/classification-model-training`  
**Target:** Train two classifiers (`good_deal_verdict` primary, `price_tier` secondary) reusing leakage-safe features from Spec 22 and labels from Spec 21.

---

## Phase Overview

| Phase | Description | Files Created/Modified |
|-------|-------------|------------------------|
| 1 | Core training logic (`ml/classification/training.py`) | 1 new file |
| 2 | Report writer (`ml/classification/training_report.py`) | 1 new file |
| 3 | CLI entry point (`scripts/train_classifiers.py`) | 1 new file |
| 4 | Test coverage (3 test files) | 3 new files |
| 5 | Package exports & pipeline integration | 2 modified files |

---

## Phase 1 — Core Training Logic: `ml/classification/training.py`

**Purpose:** Pure training functions for both classifiers. No I/O, no CLI, no global state.

### Required Exports

```python
# Constants (pinned, matching Spec 21 labels)
GOOD_DEAL_LABELS: tuple[str, ...] = ("Good Deal", "Fair Deal", "Overpriced")
PRICE_TIER_LABELS: tuple[str, ...] = ("Budget", "Mid-Range", "Premium", "Luxury")

# Public API
def train_good_deal_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_weight: dict[int, float] | str = "balanced",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the primary 3-class good_deal_verdict classifier.
    
    Returns (fitted_pipeline, metrics_dict) where metrics_dict contains:
    - accuracy, macro_f1, per_class_precision, per_class_recall, per_class_f1
    - confusion_matrix (list[list[int]])
    - good_deal_recall (float) — the primary selection metric
    """
    ...

def train_price_tier_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_weight: dict[int, float] | str = "balanced",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the secondary 4-class price_tier classifier.
    
    Returns (fitted_pipeline, metrics_dict) with same structure as above.
    """
    ...

def evaluate_classifier(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Shared evaluation logic — returns the full metrics dict."""
    ...
```

### Implementation Details

1. **Model Selection** (per spec):
   - Baseline: `LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state)`
   - Candidate 1: `RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=random_state, n_jobs=-1)`
   - Candidate 2: `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1, eval_metric="mlogloss")`

2. **Selection Criterion**:
   - `good_deal_verdict`: **maximize recall on class "Good Deal" (index 0)** — this is the primary metric per TRD §U-TRD-8
   - `price_tier`: **maximize macro-F1** (balanced across 4 classes)

3. **Pipeline Structure**:
   - Each classifier is a `sklearn.pipeline.Pipeline` with steps: `[("preprocessor", ColumnTransformer), ("classifier", Estimator)]`
   - The preprocessor is the **same ColumnTransformer** from `ml.features.preprocessor` (fit on training data via `fit_preprocessor`)
   - Do NOT recreate the preprocessor — reuse the fitted one from the price model training flow

4. **Input Assumptions**:
   - `X_train`, `X_val` are **raw feature frames** (output of `build_classifier_feature_frame` from Spec 22) — NOT pre-transformed
   - `y_train`, `y_val` are integer-encoded labels (0, 1, 2 for good_deal; 0, 1, 2, 3 for price_tier)
   - The function fits the preprocessor internally on `X_train` (via `fit_preprocessor`)

5. **Class Weights**:
   - Default `"balanced"` for both
   - Allow override via `class_weight` param for experimentation

6. **Reproducibility**:
   - All random states pinned to `42`
   - XGBoost `eval_metric="mlogloss"` for multi-class

---

## Phase 2 — Report Writer: `ml/classification/training_report.py`

**Purpose:** Append-only Markdown report writer. Pure function, no I/O side effects in the core logic.

### Required Exports

```python
from pathlib import Path

def write_training_report(
    report_path: Path,
    model_version: int,
    good_deal_metrics: dict[str, Any],
    price_tier_metrics: dict[str, Any],
    good_deal_best_model: str,
    price_tier_best_model: str,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    feature_count: int,
    class_balance_good_deal: dict[str, int],
    class_balance_price_tier: dict[str, int],
    training_date: str,  # ISO format
) -> None:
    """Append a training run entry to the report file.
    
    Creates the file with a header if it doesn't exist.
    Each entry is a ## Training Run v{model_version} section.
    """
    ...
```

### Report Format (append-only, matching Spec 21/22 style)

```markdown
# Classification Model Training Report

## Training Run v1
**Date:** 2026-08-29  
**Train/Val/Test Split:** 127,400 / 27,300 / 27,300 (70/15/15)  
**Features:** 38 (after preprocessing)  
**Class Balance (good_deal_verdict):** Good Deal: 8,200 | Fair Deal: 15,400 | Overpriced: 3,700  
**Class Balance (price_tier):** Budget: 6,800 | Mid-Range: 10,200 | Premium: 7,100 | Luxury: 3,200  

### Good Deal Verdict (Primary)
**Best Model:** XGBClassifier  
**Selection Metric:** Good Deal Recall = 0.847  
| Metric | Value |
|--------|-------|
| Accuracy | 0.782 |
| Macro F1 | 0.756 |
| Good Deal Precision | 0.791 |
| **Good Deal Recall** | **0.847** |
| Good Deal F1 | 0.818 |
| Fair Deal Precision | 0.762 |
| Fair Deal Recall | 0.798 |
| Fair Deal F1 | 0.780 |
| Overpriced Precision | 0.714 |
| Overpriced Recall | 0.642 |
| Overpriced F1 | 0.676 |

**Confusion Matrix (rows=true, cols=pred):**
```
[[6945  112   43]
 [ 289 12289  22]
 [  98  189 2373]]
```

### Price Tier (Secondary)
**Best Model:** RandomForestClassifier  
**Selection Metric:** Macro F1 = 0.821  
| Metric | Value |
|--------|-------|
| Accuracy | 0.834 |
| Macro F1 | 0.821 |
| Budget Precision | 0.856 |
| Budget Recall | 0.872 |
| Budget F1 | 0.864 |
| Mid-Range Precision | 0.812 |
| Mid-Range Recall | 0.834 |
| Mid-Range F1 | 0.823 |
| Premium Precision | 0.834 |
| Premium Recall | 0.812 |
| Premium F1 | 0.823 |
| Luxury Precision | 0.789 |
| Luxury Recall | 0.765 |
| Luxury F1 | 0.777 |

**Confusion Matrix (rows=true, cols=pred):**
```
[[5928  712   89   71]
 [ 543 8508  987  162]
 [  67  892 5768  373]
 [  41  123  487 2449]]
```

---

## Training Run v2
...
```

---

## Phase 3 — CLI Entry Point: `scripts/train_classifiers.py`

**Purpose:** Reproducible training script with `argparse`, matching the pattern of `scripts/build_classification_labels.py` and `scripts/build_classifier_features.py`.

### CLI Interface

```bash
python scripts/train_classifiers.py \
    --features data/processed/classifier_features.parquet \
    --labels data/processed/classification_labels.parquet \
    --models-dir models \
    --report models/classification_training_report.md \
    --version 1 \
    --random-state 42
```

### Required Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--features` | Path | required | Path to `classifier_features.parquet` (Spec 22 output) |
| `--labels` | Path | required | Path to `classification_labels.parquet` (Spec 21 output) |
| `--models-dir` | Path | `models` | Output directory for `.pkl` artifacts |
| `--report` | Path | `models/classification_training_report.md` | Append-only report path |
| `--version` | int | required | Model version number (e.g., 1 → `good_deal_classifier_v1.pkl`) |
| `--random-state` | int | 42 | Random seed for reproducibility |

### CLI Flow

1. **Load data**: Read features + labels parquet files
2. **Validate alignment**: Same row count, same index (or join on `PROP_ID` if present)
3. **Split**: 70/15/15 with `random_state=42`, **stratified on `good_deal_verdict`** (primary target)
   - Use `train_test_split` twice: first 70/30, then 15/15 on the 30%
   - Stratify on `good_deal_verdict` for both splits
4. **Train both classifiers** via `training.py` functions
5. **Select best model** per selection criterion (Good Deal recall for primary, macro-F1 for secondary)
6. **Evaluate on test set** (held-out, never seen during training/selection)
7. **Serialize artifacts**:
   - `models/good_deal_classifier_v{version}.pkl` — full Pipeline (preprocessor + classifier)
   - `models/price_tier_classifier_v{version}.pkl` — full Pipeline
   - `models/classifier_preprocessor_v{version}.pkl` — fitted ColumnTransformer (for serving)
8. **Write report** via `training_report.py`
9. **Print summary** to stdout (model versions, key metrics, artifact paths)

### Error Handling

- Exit code 1 on any failure with clear error message
- Validate file existence before loading
- Validate label columns exist: `good_deal_verdict`, `price_tier`
- Validate no NaN in labels

---

## Phase 4 — Test Coverage (3 Files)

### 4.1 `tests/test_classification_training.py`

**Scope:** Unit tests for `ml.classification.training` functions.

| Test | Description |
|------|-------------|
| `test_good_deal_labels_constant` | `GOOD_DEAL_LABELS == ("Good Deal", "Fair Deal", "Overpriced")` |
| `test_price_tier_labels_constant` | `PRICE_TIER_LABELS == ("Budget", "Mid-Range", "Premium", "Luxury")` |
| `test_train_good_deal_returns_pipeline_and_metrics` | Returns `(Pipeline, dict)` with all required metric keys |
| `test_train_price_tier_returns_pipeline_and_metrics` | Same for price_tier |
| `test_good_deal_selection_prioritizes_recall` | Best model chosen by Good Deal recall, not accuracy |
| `test_price_tier_selection_prioritizes_macro_f1` | Best model chosen by macro-F1 |
| `test_evaluate_classifier_returns_full_metrics` | Confusion matrix, per-class P/R/F1 all present |
| `test_class_weight_balanced_default` | Default class_weight="balanced" used |
| `test_random_state_reproducibility` | Same random_state → identical model params |

**Fixtures:** Synthetic feature frame (38 cols) + labels matching Spec 21/22 schema.

### 4.2 `tests/test_classification_training_report.py`

**Scope:** Unit tests for `ml.classification.training_report`.

| Test | Description |
|------|-------------|
| `test_write_report_creates_file_with_header` | New file gets `# Classification Model Training Report` header |
| `test_write_report_appends_entry` | Second call appends `## Training Run v2` |
| `test_report_contains_all_sections` | Both classifiers, confusion matrices, class balance tables |
| `test_report_format_matches_spec` | Markdown tables, code fences for confusion matrices |

### 4.3 `tests/test_train_classifiers_cli.py`

**Scope:** Integration test for the CLI script (uses `tmp_path`, real parquet files).

| Test | Description |
|------|-------------|
| `test_cli_creates_both_model_artifacts` | `.pkl` files exist in models-dir after run |
| `test_cli_creates_preprocessor_artifact` | `classifier_preprocessor_v{n}.pkl` exists |
| `test_cli_writes_report` | Report file exists with correct version header |
| `test_cli_stratified_split` | Class proportions preserved in train/val/test |
| `test_cli_exits_nonzero_on_missing_files` | Missing features/labels → exit code 1 |
| `test_cli_version_in_artifact_names` | Version arg reflected in output filenames |

---

## Phase 5 — Package Exports & Pipeline Integration

### 5.1 `ml/classification/__init__.py`

**Add exports:**

```python
from .training import (
    GOOD_DEAL_LABELS,
    PRICE_TIER_LABELS,
    train_good_deal_classifier,
    train_price_tier_classifier,
    evaluate_classifier,
)
from .training_report import write_training_report

__all__ = [
    "GOOD_DEAL_LABELS",
    "PRICE_TIER_LABELS",
    "train_good_deal_classifier",
    "train_price_tier_classifier",
    "evaluate_classifier",
    "write_training_report",
]
```

### 5.2 `scripts/run_pipeline.py`

**Add a `train_classifiers` step** (if this script exists and orchestrates the full pipeline):

```python
def step_train_classifiers(args: argparse.Namespace) -> None:
    """Run classification model training (Spec 23)."""
    subprocess.run([
        sys.executable, "scripts/train_classifiers.py",
        "--features", "data/processed/classifier_features.parquet",
        "--labels", "data/processed/classification_labels.parquet",
        "--models-dir", "models",
        "--report", "models/classification_training_report.md",
        "--version", str(args.model_version),
        "--random-state", "42",
    ], check=True)
```

Add `--model-version` argument to the main parser and wire the step.

---

## File Tree After Implementation

```
ml/classification/
├── __init__.py              # ← modified (add exports)
├── tiers.py                 # Spec 21 (existing)
├── verdicts.py              # Spec 21 (existing)
├── features.py              # Spec 22 (existing)
├── training.py              # ← NEW (Phase 1)
├── training_report.py       # ← NEW (Phase 2)
scripts/
├── build_classification_labels.py      # Spec 21 (existing)
├── build_classifier_features.py        # Spec 22 (existing)
├── train_classifiers.py                # ← NEW (Phase 3)
├── run_pipeline.py                     # ← modified (Phase 5.2)
tests/
├── test_classification_training.py     # ← NEW (Phase 4.1)
├── test_classification_training_report.py  # ← NEW (Phase 4.2)
├── test_train_classifiers_cli.py       # ← NEW (Phase 4.3)
models/
├── good_deal_classifier_v1.pkl         # ← generated artifact
├── price_tier_classifier_v1.pkl        # ← generated artifact
├── classifier_preprocessor_v1.pkl      # ← generated artifact
├── classification_training_report.md   # ← generated report
```

---

## Dependencies (No New Packages)

All required packages already in `requirements.txt`:
- `pandas`, `numpy`, `scikit-learn` — core ML
- `xgboost` — XGBClassifier
- `joblib` — serialization (via `sklearn.externals.joblib` or direct)
- `pyyaml` — not needed, but already present

---

## Execution Order

```bash
# 1. Create the 3 implementation files
# 2. Create the 3 test files
# 3. Modify __init__.py and run_pipeline.py
# 4. Run tests
pytest tests/test_classification_training.py tests/test_classification_training_report.py tests/test_train_classifiers_cli.py -v

# 5. Run the CLI (requires Spec 21/22 outputs to exist)
python scripts/train_classifiers.py \
    --features data/processed/classifier_features.parquet \
    --labels data/processed/classification_labels.parquet \
    --models-dir models \
    --report models/classification_training_report.md \
    --version 1

# 6. Validate via housingiq-ml-evaluator (separate step)
```

---

## Acceptance Criteria (from Spec 23)

- [ ] Both classifiers trained with 70/15/15 stratified split (`random_state=42`)
- [ ] Good Deal classifier selected by **Good Deal recall** (primary metric)
- [ ] Price Tier classifier selected by **macro-F1**
- [ ] Full per-class precision/recall/F1 + confusion matrices in report
- [ ] Versioned `.pkl` artifacts in `models/` (never overwritten)
- [ ] Append-only `classification_training_report.md` with all required sections
- [ ] CLI reproduces exact same artifacts given same inputs + version
- [ ] All 3 test files pass
- [ ] No leakage: `price` / `price_per_sqft` never in classifier features (validated by reusing Spec 22 feature frame)