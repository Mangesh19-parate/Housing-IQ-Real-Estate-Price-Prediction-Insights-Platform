"""Tests for classification evaluation protocol constants (Spec 24)."""

from __future__ import annotations

from ml.evaluation.classification.protocol import (
    CLASSIFIER_THRESHOLDS,
    METRIC_NAMES,
    PROTOCOL_DOC_PATH,
    PROTOCOL_VERSION,
    RANDOM_STATE,
    SPLIT_RATIOS,
)


def test_protocol_version_is_semver():
    """PROTOCOL_VERSION is a valid semver string."""
    parts = PROTOCOL_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_split_ratios_sum_to_one():
    """SPLIT_RATIOS sum to 1.0."""
    total = sum(SPLIT_RATIOS.values())
    assert abs(total - 1.0) < 1e-9


def test_split_ratios_values():
    """SPLIT_RATIOS has correct 70/15/15 values."""
    assert SPLIT_RATIOS == {"train": 0.70, "val": 0.15, "test": 0.15}


def test_random_state_pinned():
    """RANDOM_STATE is pinned to 42."""
    assert RANDOM_STATE == 42


def test_metric_names_order():
    """METRIC_NAMES has correct order and values."""
    expected = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "good_deal_recall",
        "price_tier_macro_f1",
    )
    assert METRIC_NAMES == expected


def test_classifier_thresholds_structure():
    """CLASSIFIER_THRESHOLDS has both targets with correct sub-keys."""
    assert "good_deal" in CLASSIFIER_THRESHOLDS
    assert "price_tier" in CLASSIFIER_THRESHOLDS

    good_deal = CLASSIFIER_THRESHOLDS["good_deal"]
    assert "good_deal_recall_min" in good_deal
    assert "f1_macro_min" in good_deal
    assert "accuracy_min" in good_deal

    price_tier = CLASSIFIER_THRESHOLDS["price_tier"]
    assert "f1_macro_min" in price_tier
    assert "accuracy_min" in price_tier
    assert "recall_macro_min" in price_tier


def test_classifier_thresholds_values():
    """CLASSIFIER_THRESHOLDS values match PRD targets."""
    gd = CLASSIFIER_THRESHOLDS["good_deal"]
    assert gd["good_deal_recall_min"] == 0.80
    assert gd["f1_macro_min"] == 0.70
    assert gd["accuracy_min"] == 0.65

    pt = CLASSIFIER_THRESHOLDS["price_tier"]
    assert pt["f1_macro_min"] == 0.75
    assert pt["accuracy_min"] == 0.70
    assert pt["recall_macro_min"] == 0.65


def test_protocol_doc_path():
    """PROTOCOL_DOC_PATH points to TRD."""
    assert PROTOCOL_DOC_PATH == "docs/02-TRD.md"