# Spec 24: Classification Model Evaluation

**Status:** Draft
**Module:** Classification (Affordability & Investment-Tier Filter)
**Depends on:** Spec 23 (Classification Model Training), Spec 15 (Price Model Evaluation Protocol)
**Author:** HousingIQ Team
**Date:** 2026-08-30

---

## 1. Purpose

This spec defines the **evaluation gate** for the two classification models trained in Spec 23:

| Model | Target | Classes | Primary Metric |
|-------|--------|---------|----------------|
| `good_deal_classifier` | `good_deal_verdict` | 3 (Good Deal / Fair Deal / Overpriced) | **Good Deal recall** ≥ 0.80 |
| `price_tier_classifier` | `price_tier` | 4 (Budget / Mid-Range / Premium / Luxury) | **macro-F1** ≥ 0.75 |

The gate mirrors **Spec 15's price model evaluation protocol** — same split, same determinism, same artifact discipline — adapted for classification metrics. A model is **certified** iff it clears every threshold for its respective target.

---

## 2. Scope

### In Scope
- New package `ml/evaluation/classification/` with protocol constants, split enforcement, scoring, gate evaluation, report writing, and CLI.
- CLI script `scripts/evaluate_classifiers.py` — the certification entry point.
- Versioned JSON report artifacts under `models/evaluation_reports/`.
- Protocol certification section appended to the classification training report (`models/classification_training_report.md`).
- Comprehensive test coverage following Spec 23's test patterns.

### Out of Scope
- Retraining or hyperparameter tuning (Spec 23).
- FastAPI `/classify` endpoint integration (future spec).
- SHAP explainers for classifiers (future spec — gate only certifies metrics).
- Per-city / per-locality breakdowns (v1 gate is global; per-segment added if PRD demands it).

---

## 3. Protocol Constants (Pinned)

File: `ml/evaluation/classification/protocol.py`

```python
# 70/15/15 split — identical to price protocol
SPLIT_RATIOS: Final[dict[str, float]] = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_STATE: Final[int] = 42

# Headline metrics the gate reports (order is significant for report tables)
METRIC_NAMES: Final[tuple[str, ...]] = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "good_deal_recall",        # good_deal_classifier only
    "price_tier_macro_f1",     # price_tier_classifier only
)

# Pass/fail thresholds — PRD targets
# good_deal_classifier: Good Deal recall ≥ 0.80 (primary), macro-F1 ≥ 0.70 (floor)
# price_tier_classifier: macro-F1 ≥ 0.75 (primary), accuracy ≥ 0.70 (floor)
CLASSIFIER_THRESHOLDS: Final[dict[str, dict[str, float]]] = {
    "good_deal": {
        "good_deal_recall_min": 0.80,
        "f1_macro_min": 0.70,
        "accuracy_min": 0.65,
    },
    "price_tier": {
        "f1_macro_min": 0.75,
        "accuracy_min": 0.70,
        "recall_macro_min": 0.65,
    },
}

PROTOCOL_VERSION: Final[str] = "1.0.0"
PROTOCOL_DOC_PATH: Final[str] = "docs/02-TRD.md"
```

**Any drift in these values is a deliberate protocol revision** — bump `PROTOCOL_VERSION` and update `docs/02-TRD.md` simultaneously.

---

## 4. Split Enforcement

File: `ml/evaluation/classification/splits.py`

```python
def protocol_split(
    df: pd.DataFrame,
    target: str,                    # "good_deal_verdict" or "price_tier"
    *,
    ratios: dict[str, float] = SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split on `target` with pinned random_state.
    Returns (train_df, val_df, test_df) with reset indices.
    Raises ValueError if any class has < 2 members in the test split.
    """
```

- Uses `sklearn.model_selection.train_test_split` twice (70/30 then 50/50 on the 30%) with `stratify=df[target]`.
- Identical random_state ensures **bit-for-bit reproducibility** across runs.
- Test validates exact split sizes on a 1000-row synthetic frame (700/150/150 ± stratification variance).

---

## 5. Scoring Functions

File: `ml/evaluation/classification/scoring.py`

```python
def score_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_name: str,               # "good_deal" or "price_tier"
    labels: list[int],              # [0,1,2] or [0,1,2,3]
) -> dict[str, float]:
    """
    Returns dict in METRIC_NAMES order with:
      - accuracy
      - precision_macro
      - recall_macro
      - f1_macro
      - good_deal_recall (target_name=="good_deal" only; recall for class 0)
      - price_tier_macro_f1 (target_name=="price_tier" only; alias for f1_macro)
    Pure function; no side effects.
    """
```

```python
def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[int, dict[str, float]]:
    """
    Returns {class_int: {precision, recall, f1, support}} for confusion matrix
    and per-class reporting. Pure function.
    """
```

```python
def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, list[list[int]]]:
    """
    Returns {"matrix": [[...]], "labels": [...]} — JSON-serializable.
    """
```

---

## 6. Gate Evaluation Entry Point

File: `ml/evaluation/classification/gate.py`

```python
@dataclasses.dataclass(frozen=True)
class ClassificationEvaluationResult:
    version: str                          # e.g. "v1"
    target: str                           # "good_deal" or "price_tier"
    protocol_version: str                 # "1.0.0"
    dataset_version: str                  # fingerprint of feature_frame.parquet
    git_commit: str                       # 12-char SHA
    split_sizes: dict[str, int]           # {"train": 700, "val": 150, "test": 150}
    metrics: dict[str, float]             # test-set headline metrics
    per_class_test: dict[int, dict]       # per-class precision/recall/f1/support
    confusion_matrix: dict                # {"matrix": [...], "labels": [...]}
    thresholds_passed: dict[str, bool]    # one entry per threshold in CLASSIFIER_THRESHOLDS[target]
    overall_passed: bool                  # all(thresholds_passed.values())
    evaluated_at: str                     # ISO8601 UTC
    evaluator_version: str                # "1.0.0"
```

```python
def evaluate(
    model_path: Path | str,
    version: str,
    target: str,                          # "good_deal" or "price_tier"
    processed_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    feature_frame_path: Path | str | None = None,
    labels_dir: Path | str | None = None,
) -> ClassificationEvaluationResult:
    """
    Pure function — no file writes. Steps:
      1. Load feature_frame.parquet + labels (good_deal_labels.parquet or price_tier_labels.parquet).
      2. Merge on listing_id, drop is_outlier rows.
      3. protocol_split on target column.
      4. Load classifier_preprocessor_v{version}.pkl from models_dir.
      5. Load model artifact (good_deal_classifier_v{version}.pkl or price_tier_classifier_v{version}.pkl).
      6. Transform X_train/X_val/X_test with preprocessor.
      7. Predict on all three splits.
      8. score_classifier on test split → metrics.
      9. per_class_metrics + confusion_matrix_dict on test split.
      10. Check thresholds from CLASSIFIER_THRESHOLDS[target].
      11. Return ClassificationEvaluationResult.
    """
```

**Leakage guard:** The gate asserts that no price-derived column (`price_inr`, `price_per_sqft`, `locality_avg_price_sqft`, `locality_smoothed_price`) appears in the feature columns passed to the preprocessor. If any are present, `evaluate()` raises `ValueError("leakage_detected: ...")`.

---

## 7. Report Persistence

File: `ml/evaluation/classification/report.py`

```python
def write_evaluation_report(
    result: ClassificationEvaluationResult,
    models_dir: Path | str,
) -> Path:
    """
    Writes models/evaluation_reports/classification_eval_{target}_v{version}.json
    with the full ClassificationEvaluationResult (dataclass → dict via asdict).
    Returns the written Path.
    """
```

```python
def append_protocol_section(
    result: ClassificationEvaluationResult,
    report_path: Path | str,
) -> None:
    """
    Appends a "## Protocol Certification — {target} v{version}" section to
    the classification training report (models/classification_training_report.md).
    Section includes:
      - Protocol version + dataset fingerprint + git commit
      - Split sizes table
      - Test metrics table (METRIC_NAMES order)
      - Per-class metrics table
      - Confusion matrix (markdown table)
      - Threshold checklist (✅/❌ per threshold)
      - Overall: **CERTIFIED** or **NOT CERTIFIED**
    Idempotent — replaces existing section for same target+version if present.
    """
```

---

## 8. CLI Script

File: `scripts/evaluate_classifiers.py`

```bash
python scripts/evaluate_classifiers.py \
    --version v1 \
    --target good_deal \
    --target price_tier \
    [--processed-dir data/processed] \
    [--models-dir models] \
    [--feature-frame data/processed/classifier/feature_frame.parquet] \
    [--labels-dir data/processed/classifier] \
    [--report-path models/classification_training_report.md]
```

- `--target` is repeatable; typical invocation certifies both classifiers.
- For each target:
  1. Calls `ml.evaluation.classification.gate.evaluate(...)`.
  2. Calls `ml.evaluation.classification.report.write_evaluation_report`.
  3. Calls `ml.evaluation.classification.report.append_protocol_section`.
  4. Prints one-line summary: `[PASS|FAIL] {target}_v{version} f1_macro={:.4f} good_deal_recall={:.4f} accuracy={:.4f}`.
- Exit code: `0` iff **every** target's `overall_passed == True`, else `1`.
- No new pip packages — stdlib `argparse` + the `ml.evaluation.classification` package.

---

## 9. Files to Create

```
ml/evaluation/classification/
├── __init__.py
├── protocol.py          # pinned constants (Section 3)
├── splits.py            # protocol_split (Section 4)
├── scoring.py           # score_classifier, per_class_metrics, confusion_matrix_dict (Section 5)
├── gate.py              # ClassificationEvaluationResult + evaluate() (Section 6)
└── report.py            # write_evaluation_report + append_protocol_section (Section 7)

scripts/
└── evaluate_classifiers.py    # CLI gate (Section 8)

tests/
├── test_classification_protocol.py
├── test_classification_splits.py
├── test_classification_scoring.py
├── test_classification_gate.py
├── test_classification_report.py
└── test_evaluate_classifiers_cli.py
```

---

## 10. Test Requirements

### `test_classification_protocol.py`
- Constants match the pinned values in this spec.
- `PROTOCOL_VERSION` is a semver string.
- `CLASSIFIER_THRESHOLDS` has both `"good_deal"` and `"price_tier"` keys with correct sub-keys.

### `test_classification_splits.py`
- `protocol_split` returns 3 DataFrames with correct sizes on 1000-row frame (700/150/150 ±1).
- Stratification preserves class proportions (each class count in test ≈ 15% of total).
- `random_state=42` yields identical splits across calls.
- Raises `ValueError` if any class would have < 2 test members.

### `test_classification_scoring.py`
- `score_classifier` returns all METRIC_NAMES keys in correct order.
- `good_deal_recall` equals recall for class 0 (Good Deal) when target="good_deal".
- `price_tier_macro_f1` equals `f1_macro` when target="price_tier".
- `per_class_metrics` returns dict with precision/recall/f1/support for each class.
- `confusion_matrix_dict` returns JSON-serializable structure with correct shape.

### `test_classification_gate.py`
- `evaluate()` returns `ClassificationEvaluationResult` with all fields populated.
- Leakage guard raises `ValueError` if price-derived columns present in features.
- Thresholds checked correctly: `overall_passed` is `True` iff all thresholds pass.
- `evaluated_at` is valid ISO8601 UTC; `git_commit` is 12-char or "unknown".
- Dataset fingerprint changes when feature_frame.parquet content changes.

### `test_classification_report.py`
- `write_evaluation_report` creates versioned JSON file with correct structure.
- `append_protocol_section` appends markdown section with all required tables.
- Idempotent: running twice for same target+version replaces (not duplicates) the section.

### `test_evaluate_classifiers_cli.py`
- CLI creates both evaluation report JSON files.
- CLI appends protocol sections to training report.
- CLI prints `[PASS]` or `[FAIL]` summary line per target.
- Exit code 0 when both pass, 1 when any fails.
- Exits 1 with clear error when model artifact missing.
- Exits 1 with clear error when feature_frame or labels missing.
- Version arg reflected in output filenames.

---

## 11. Integration Notes

- The gate **reads the same artifacts** Spec 23 produces:
  - `models/good_deal_classifier_v{n}.pkl`
  - `models/price_tier_classifier_v{n}.pkl`
  - `models/classifier_preprocessor_v{n}.pkl`
  - `data/processed/classifier/feature_frame.parquet`
  - `data/processed/classifier/good_deal_labels.parquet`
  - `data/processed/classifier/price_tier_labels.parquet`
- No new artifact types — the gate only *consumes* what training produces.
- The classification training report (`models/classification_training_report.md`) becomes the **single certification document** — training metrics + protocol certification in one file.
- FastAPI `/classify` endpoint (future) will load the certified model artifacts; the gate does not interact with the API.

---

## 12. Acceptance Criteria

| Criterion | Verification |
|-----------|--------------|
| Gate certifies both classifiers against pinned thresholds | `pytest tests/test_evaluate_classifiers_cli.py -v` passes |
| Split is deterministic 70/15/15 stratified | `test_classification_splits.py` passes |
| Scoring matches sklearn definitions exactly | `test_classification_scoring.py` passes |
| Leakage guard catches price-derived columns | `test_classification_gate.py::test_leakage_guard` passes |
| Reports written to `models/evaluation_reports/` | `test_classification_report.py` passes |
| Training report gets protocol section appended | `test_classification_report.py::test_append_protocol_section` passes |
| CLI exit code reflects overall pass/fail | `test_evaluate_classifiers_cli.py` passes |
| No new pip dependencies | `pip check` passes; `requirements.txt` unchanged |

---

## 13. Future Extensions (Not in v1)

- Per-city / per-locality metric breakdowns in the gate.
- SHAP explainer generation for certified classifiers.
- Drift detection: compare current test metrics vs. certification metrics.
- A/B gate: certify challenger model against champion's certified metrics.