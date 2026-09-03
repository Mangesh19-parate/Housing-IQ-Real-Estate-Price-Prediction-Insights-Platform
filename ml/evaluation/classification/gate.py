"""``evaluate()`` entry point + ``ClassificationEvaluationResult`` dataclass (Spec 24).

The gate certifies a trained classification model against the pinned
protocol. It is **pure** with respect to its inputs — it reads the
trained artifact + the feature frame + labels + the fitted preprocessor,
scores the model on the protocol's split, and returns a
:class:`ClassificationEvaluationResult`. It does not write files;
persistence is the CLI's job.

The 12-step flow matches the spec's "Files to create → gate.py →
evaluate()" section verbatim. Step numbers in the source match the
spec so a reviewer can audit drift at a glance.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.classification.protocol import (
    CLASSIFIER_THRESHOLDS,
    PROTOCOL_DOC_PATH,
    PROTOCOL_VERSION,
)
from ml.evaluation.classification.scoring import (
    confusion_matrix_dict,
    per_class_metrics,
    score_classifier,
)
from ml.evaluation.classification.splits import protocol_split

logger = logging.getLogger(__name__)

# Pinned evaluator version — kept in sync with ``ml.evaluation.classification.__version__``
# by the test suite. Defined as a local constant (not a live cross-module
# attribute) to avoid the circular import that ``from ml.evaluation.classification
# import __version__`` would create when gate.py is imported during
# package init.
_EVALUATOR_VERSION: str = "1.0.0"

#: Price-derived columns that must NEVER appear in classifier features (leakage guard).
_LEAKAGE_COLUMNS: tuple[str, ...] = (
    "price_inr",
    "price_per_sqft",
    "locality_avg_price_sqft",
    "locality_smoothed_price",
)


@dataclasses.dataclass(frozen=True)
class ClassificationEvaluationResult:
    """Outcome of a single ``evaluate()`` call.

    Fields mirror the spec verbatim. ``overall_passed`` is ``True``
    iff every threshold in ``thresholds_passed`` is ``True``. The
    model is **certified** iff ``overall_passed == True``.
    """

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


def _git_commit() -> str:
    """12-char git short SHA; ``"unknown"`` if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dataset_fingerprint(parquet_path: Path) -> str:
    """``<filename>-<sha1[:8] of first 1 MB>`` content fingerprint."""
    sha = hashlib.sha1()
    try:
        with open(parquet_path, "rb") as fh:
            sha.update(fh.read(1024 * 1024))
    except FileNotFoundError:
        return f"{parquet_path.name}-missing"
    return f"{parquet_path.name}-{sha.hexdigest()[:8]}"


def _check_thresholds(
    target: str,
    metrics: dict[str, float],
) -> dict[str, bool]:
    """Compute ``{threshold_name: passed?}`` against the pinned protocol."""
    thresholds = CLASSIFIER_THRESHOLDS[target]
    passed: dict[str, bool] = {}
    for thresh_name, thresh_value in thresholds.items():
        if thresh_name == "good_deal_recall_min":
            passed[thresh_name] = bool(metrics.get("good_deal_recall", 0.0) >= thresh_value)
        elif thresh_name == "f1_macro_min":
            passed[thresh_name] = bool(metrics.get("f1_macro", 0.0) >= thresh_value)
        elif thresh_name == "accuracy_min":
            passed[thresh_name] = bool(metrics.get("accuracy", 0.0) >= thresh_value)
        elif thresh_name == "recall_macro_min":
            passed[thresh_name] = bool(metrics.get("recall_macro", 0.0) >= thresh_value)
        else:
            # Defensive — should never fire because CLASSIFIER_THRESHOLDS is pinned
            logger.warning("Unknown threshold '%s' for target '%s'", thresh_name, target)
            passed[thresh_name] = False
    return passed


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
    Score ``model_path`` against the protocol and return the result.

    Pure function — no file writes. The CLI is responsible for
    persisting the result via ``report.write_evaluation_report``.

    Steps:
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
    if processed_dir is None:
        processed_dir = os.environ.get("HOUSINGIQ_PROCESSED_DIR", "data/processed")
    if models_dir is None:
        models_dir = os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models")
    if feature_frame_path is None:
        feature_frame_path = Path(processed_dir) / "classifier" / "feature_frame.parquet"
    if labels_dir is None:
        labels_dir = Path(processed_dir) / "classifier"

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    feature_frame_path = Path(feature_frame_path)
    labels_dir = Path(labels_dir)
    model_path = Path(model_path)

    target = target.lower()
    if target not in ("good_deal", "price_tier"):
        raise ValueError(f"evaluate: target must be 'good_deal' or 'price_tier', got '{target}'")

    # --- Step 1: load the feature frame + labels -----------------------------
    feature_df = pd.read_parquet(feature_frame_path)

    if target == "good_deal":
        labels_path = labels_dir / "good_deal_labels.parquet"
        target_col = "good_deal_verdict"
        label_list = [0, 1, 2]
    else:
        labels_path = labels_dir / "price_tier_labels.parquet"
        target_col = "price_tier"
        label_list = [0, 1, 2, 3]

    if not labels_path.exists():
        raise FileNotFoundError(f"evaluate: labels not found at {labels_path}")

    labels_df = pd.read_parquet(labels_path)

    # Merge on listing_id
    df = feature_df.merge(labels_df, on="listing_id", how="inner")

    # --- Step 2: drop outlier rows ------------------------------------------
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False].reset_index(drop=True)  # noqa: E712

    if target_col not in df.columns:
        raise KeyError(f"evaluate: target column '{target_col}' not found after merge. Columns: {list(df.columns)}")

    evaluated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    fingerprint = _dataset_fingerprint(feature_frame_path)
    git_sha = _git_commit()

    # --- Step 3: protocol_split enforces 70/15/15 + random_state ------------
    train_df, val_df, test_df = protocol_split(df, target=target_col)
    split_sizes = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }

    # --- Step 4: load the fitted preprocessor -------------------------------
    preproc_path = models_dir / f"classifier_preprocessor_v{version.lstrip('v')}.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"evaluate: classifier preprocessor not found at {preproc_path}. "
            f"Run scripts/train_classifiers.py first."
        )
    fitted_preproc = joblib.load(preproc_path)
    if fitted_preproc is None or not hasattr(fitted_preproc, "transform"):
        raise ValueError(
            "evaluate: loaded preprocessor failed sanity check (preprocessor_drift)."
        )

    # --- Step 5: load the model artifact ------------------------------------
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise ValueError(
            f"evaluate: artifact at {model_path} has no predict() — not a model."
        )

    # --- Step 6: leakage guard + select only columns the preprocessor expects ---
    # The fitted preprocessor knows which columns it was trained on.
    # Use feature_names_in_ (sklearn >= 1.0) which works for both Pipeline
    # and ColumnTransformer after fitting.
    if hasattr(fitted_preproc, "feature_names_in_"):
        expected_cols = list(fitted_preproc.feature_names_in_)
    elif hasattr(fitted_preproc, "transformers_"):
        # Fallback for ColumnTransformer
        expected_cols = []
        for name, _transformer, cols in fitted_preproc.transformers_:
            expected_cols.extend(list(cols))
    else:
        raise ValueError(
            "evaluate: preprocessor has no feature_names_in_ or transformers_ "
            f"attribute (type: {type(fitted_preproc)})."
        )
    # Drop leakage columns from expected set if they somehow got in
    expected_cols = [c for c in expected_cols if c not in _LEAKAGE_COLUMNS]

    # Check for leakage columns in the data and warn
    available_cols = set(train_df.columns)
    leaked = [c for c in _LEAKAGE_COLUMNS if c in available_cols]
    if leaked:
        logger.warning(
            "Dropping price-derived columns from features before preprocessing: %s",
            leaked,
        )

    # --- Step 7: transform + predict on all splits --------------------------
    X_train = train_df[expected_cols]
    X_val = val_df[expected_cols]
    X_test = test_df[expected_cols]
    y_train = train_df[target_col].astype(int)
    y_val = val_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    # Transform using the fitted preprocessor
    X_train_trans = fitted_preproc.transform(X_train)
    X_val_trans = fitted_preproc.transform(X_val)
    X_test_trans = fitted_preproc.transform(X_test)

    # Predict
    y_train_pred = model.predict(X_train_trans)
    y_val_pred = model.predict(X_val_trans)
    y_test_pred = model.predict(X_test_trans)

    # --- Step 8: score_classifier on test split -----------------------------
    metrics_test = score_classifier(y_test, y_test_pred, target_name=target, labels=label_list)
    metrics_train = score_classifier(y_train, y_train_pred, target_name=target, labels=label_list)
    metrics_val = score_classifier(y_val, y_val_pred, target_name=target, labels=label_list)
    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
    }

    # --- Step 9: per-class + confusion matrix on test -----------------------
    per_class = per_class_metrics(y_test, y_test_pred, label_list)
    cm = confusion_matrix_dict(y_test, y_test_pred, label_list)

    # --- Step 10: threshold check -------------------------------------------
    thresholds = _check_thresholds(target, metrics_test)
    overall_passed = all(thresholds.values()) if thresholds else False

    # --- Step 11: return ----------------------------------------------------
    return ClassificationEvaluationResult(
        version=version,
        target=target,
        protocol_version=PROTOCOL_VERSION,
        dataset_version=fingerprint,
        git_commit=git_sha,
        split_sizes=split_sizes,
        metrics=metrics,
        per_class_test=per_class,
        confusion_matrix=cm,
        thresholds_passed=thresholds,
        overall_passed=overall_passed,
        evaluated_at=evaluated_at,
        evaluator_version=_EVALUATOR_VERSION,
    )


def format_summary(result: ClassificationEvaluationResult) -> str:
    """One-line stdout summary used by the CLI.

    Format: ``[PASS|FAIL] {target}_v{version} f1_macro={:.4f} good_deal_recall={:.4f} accuracy={:.4f}``.
    """
    test = result.metrics.get("test", {}) if isinstance(result.metrics, Mapping) else {}
    f1 = float(test.get("f1_macro", 0.0))
    gd_recall = float(test.get("good_deal_recall", 0.0))
    acc = float(test.get("accuracy", 0.0))
    tag = "PASS" if result.overall_passed else "FAIL"
    return (
        f"[{tag}] {result.target}_{result.version} "
        f"f1_macro={f1:.4f} good_deal_recall={gd_recall:.4f} accuracy={acc:.4f}"
    )


__all__ = [
    "ClassificationEvaluationResult",
    "evaluate",
    "format_summary",
    "PROTOCOL_DOC_PATH",
    "_EVALUATOR_VERSION",
]