"""Metric-scoring layer for classification evaluation — Spec 24.

Pure functions for the headline classification metrics (accuracy,
macro-precision, macro-recall, macro-F1) plus target-specific metrics
(Good Deal recall, price_tier macro-F1 alias), per-class breakdowns,
and confusion matrices. All functions are pure; no side effects.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.evaluation.classification.protocol import METRIC_NAMES

#: Ordered list of class labels for good_deal_verdict (index-aligned with integer encoding)
GOOD_DEAL_LABELS: Final[tuple[str, ...]] = ("Good Deal", "Fair Deal", "Overpriced")

#: Ordered list of class labels for price_tier (index-aligned with integer encoding)
PRICE_TIER_LABELS: Final[tuple[str, ...]] = ("Budget", "Mid-Range", "Premium", "Luxury")


def score_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_name: str,               # "good_deal" or "price_tier"
    labels: list[int],              # [0,1,2] or [0,1,2,3]
) -> dict[str, float]:
    """
    Return dict in METRIC_NAMES order with:
      - accuracy
      - precision_macro
      - recall_macro
      - f1_macro
      - good_deal_recall (target_name=="good_deal" only; recall for class 0)
      - price_tier_macro_f1 (target_name=="price_tier" only; alias for f1_macro)

    Pure function; no side effects.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    # Core metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=labels))
    recall_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=labels))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=labels))

    # Target-specific metrics
    if target_name == "good_deal":
        # Recall for class 0 ("Good Deal")
        per_class_recall = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels)
        good_deal_recall = float(per_class_recall[0]) if len(per_class_recall) > 0 else 0.0
        price_tier_macro_f1 = f1_macro  # alias for consistency, not used for this target
    elif target_name == "price_tier":
        good_deal_recall = 0.0  # not used for this target
        price_tier_macro_f1 = f1_macro
    else:
        raise ValueError(f"score_classifier: unknown target_name '{target_name}'")

    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name == "accuracy":
            out[name] = acc
        elif name == "precision_macro":
            out[name] = precision_macro
        elif name == "recall_macro":
            out[name] = recall_macro
        elif name == "f1_macro":
            out[name] = f1_macro
        elif name == "good_deal_recall":
            out[name] = good_deal_recall
        elif name == "price_tier_macro_f1":
            out[name] = price_tier_macro_f1
        else:
            # Defensive — should never fire because METRIC_NAMES is the pinned source of truth
            raise ValueError(f"score_classifier: unknown metric '{name}'")
    return out


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[int, dict[str, float]]:
    """
    Return ``{class_int: {precision, recall, f1, support}}`` for confusion matrix
    and per-class reporting. Pure function.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    precision = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels)
    recall = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels)
    f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels)
    support = np.bincount(y_true_arr, minlength=max(labels) + 1) if len(y_true_arr) > 0 else np.zeros(len(labels), dtype=int)

    out: dict[int, dict[str, float]] = {}
    for i, cls in enumerate(labels):
        out[cls] = {
            "precision": float(precision[i]) if i < len(precision) else 0.0,
            "recall": float(recall[i]) if i < len(recall) else 0.0,
            "f1": float(f1[i]) if i < len(f1) else 0.0,
            "support": int(support[cls]) if cls < len(support) else 0,
        }
    return out


def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, list[list[int]]]:
    """
    Return ``{"matrix": [[...]], "labels": [...]}`` — JSON-serializable.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    return {
        "matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
    }


__all__ = [
    "GOOD_DEAL_LABELS",
    "PRICE_TIER_LABELS",
    "METRIC_NAMES",
    "confusion_matrix_dict",
    "per_class_metrics",
    "score_classifier",
]