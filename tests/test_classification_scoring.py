"""Tests for classification scoring functions (Spec 24)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.classification.scoring import (
    confusion_matrix_dict,
    per_class_metrics,
    score_classifier,
)


@pytest.fixture
def synthetic_predictions():
    """Synthetic y_true, y_pred for 3-class (good_deal) and 4-class (price_tier)."""
    np.random.seed(42)
    n = 300

    # 3-class: Good Deal (0), Fair Deal (1), Overpriced (2)
    y_true_3 = np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2])
    y_pred_3 = np.random.choice([0, 1, 2], n, p=[0.25, 0.55, 0.2])

    # 4-class: Budget (0), Mid-Range (1), Premium (2), Luxury (3)
    y_true_4 = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1])
    y_pred_4 = np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15])

    return {
        "y_true_3": y_true_3,
        "y_pred_3": y_pred_3,
        "y_true_4": y_true_4,
        "y_pred_4": y_pred_4,
        "labels_3": [0, 1, 2],
        "labels_4": [0, 1, 2, 3],
    }


def test_score_classifier_returns_all_metric_names(synthetic_predictions):
    """score_classifier returns all METRIC_NAMES keys in correct order."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = score_classifier(y_true, y_pred, target_name="good_deal", labels=[0, 1, 2])

    from ml.evaluation.classification.protocol import METRIC_NAMES
    assert list(result.keys()) == list(METRIC_NAMES)


def test_score_classifier_good_deal_recall_is_class_0_recall(synthetic_predictions):
    """good_deal_recall equals recall for class 0 when target='good_deal'."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = score_classifier(y_true, y_pred, target_name="good_deal", labels=[0, 1, 2])

    # Compute recall for class 0 manually
    from sklearn.metrics import recall_score
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    expected = float(per_class_recall[0])

    assert result["good_deal_recall"] == expected


def test_score_classifier_price_tier_macro_f1_is_f1_macro(synthetic_predictions):
    """price_tier_macro_f1 equals f1_macro when target='price_tier'."""
    y_true = synthetic_predictions["y_true_4"]
    y_pred = synthetic_predictions["y_pred_4"]
    result = score_classifier(y_true, y_pred, target_name="price_tier", labels=[0, 1, 2, 3])

    assert result["price_tier_macro_f1"] == result["f1_macro"]


def test_score_classifier_values_in_range(synthetic_predictions):
    """All metrics are in valid range [0, 1]."""
    for target, y_true_key, y_pred_key, labels in [
        ("good_deal", "y_true_3", "y_pred_3", [0, 1, 2]),
        ("price_tier", "y_true_4", "y_pred_4", [0, 1, 2, 3]),
    ]:
        y_true = synthetic_predictions[y_true_key]
        y_pred = synthetic_predictions[y_pred_key]
        result = score_classifier(y_true, y_pred, target_name=target, labels=labels)
        for name, value in result.items():
            assert 0.0 <= value <= 1.0, f"{target}.{name}={value} out of range"


def test_per_class_metrics_structure(synthetic_predictions):
    """per_class_metrics returns dict with precision/recall/f1/support for each class."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = per_class_metrics(y_true, y_pred, labels=[0, 1, 2])

    assert set(result.keys()) == {0, 1, 2}
    for cls in [0, 1, 2]:
        assert set(result[cls].keys()) == {"precision", "recall", "f1", "support"}
        assert all(0.0 <= v <= 1.0 for k, v in result[cls].items() if k != "support")
        assert isinstance(result[cls]["support"], int)


def test_per_class_metrics_support_matches_counts(synthetic_predictions):
    """Support in per_class_metrics matches actual class counts in y_true."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = per_class_metrics(y_true, y_pred, labels=[0, 1, 2])

    for cls in [0, 1, 2]:
        expected = int(np.sum(y_true == cls))
        assert result[cls]["support"] == expected


def test_confusion_matrix_dict_shape(synthetic_predictions):
    """confusion_matrix_dict returns correct shape for 3-class."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = confusion_matrix_dict(y_true, y_pred, labels=[0, 1, 2])

    assert "matrix" in result
    assert "labels" in result
    assert len(result["matrix"]) == 3
    assert all(len(row) == 3 for row in result["matrix"])
    assert result["labels"] == ["0", "1", "2"]


def test_confusion_matrix_dict_json_serializable(synthetic_predictions):
    """confusion_matrix_dict returns JSON-serializable structure."""
    import json
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = confusion_matrix_dict(y_true, y_pred, labels=[0, 1, 2])

    # Should not raise
    json.dumps(result)


def test_confusion_matrix_dict_sum_equals_n(synthetic_predictions):
    """Confusion matrix sum equals number of samples."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    result = confusion_matrix_dict(y_true, y_pred, labels=[0, 1, 2])

    total = sum(sum(row) for row in result["matrix"])
    assert total == len(y_true)


def test_score_classifier_raises_on_unknown_target(synthetic_predictions):
    """score_classifier raises ValueError for unknown target_name."""
    y_true = synthetic_predictions["y_true_3"]
    y_pred = synthetic_predictions["y_pred_3"]
    with pytest.raises(ValueError, match="unknown target_name"):
        score_classifier(y_true, y_pred, target_name="unknown", labels=[0, 1, 2])