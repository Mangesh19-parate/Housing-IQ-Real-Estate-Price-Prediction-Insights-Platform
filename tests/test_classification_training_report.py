"""Tests for ml.classification.training_report (Spec 23)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ml.classification.training_report import write_training_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metrics_good_deal() -> dict:
    """Sample metrics for good_deal_verdict (3-class)."""
    return {
        "accuracy": 0.782,
        "macro_f1": 0.756,
        "per_class_precision": {
            "Good Deal": 0.791,
            "Fair Deal": 0.762,
            "Overpriced": 0.714,
        },
        "per_class_recall": {
            "Good Deal": 0.847,
            "Fair Deal": 0.798,
            "Overpriced": 0.642,
        },
        "per_class_f1": {
            "Good Deal": 0.818,
            "Fair Deal": 0.780,
            "Overpriced": 0.676,
        },
        "confusion_matrix": [
            [6945, 112, 43],
            [289, 12289, 22],
            [98, 189, 2373],
        ],
        "primary_class_recall": 0.847,
    }


@pytest.fixture
def sample_metrics_price_tier() -> dict:
    """Sample metrics for price_tier (4-class)."""
    return {
        "accuracy": 0.834,
        "macro_f1": 0.821,
        "per_class_precision": {
            "Budget": 0.856,
            "Mid-Range": 0.812,
            "Premium": 0.834,
            "Luxury": 0.789,
        },
        "per_class_recall": {
            "Budget": 0.872,
            "Mid-Range": 0.834,
            "Premium": 0.812,
            "Luxury": 0.765,
        },
        "per_class_f1": {
            "Budget": 0.864,
            "Mid-Range": 0.823,
            "Premium": 0.823,
            "Luxury": 0.777,
        },
        "confusion_matrix": [
            [5928, 712, 89, 71],
            [543, 8508, 987, 162],
            [67, 892, 5768, 373],
            [41, 123, 487, 2449],
        ],
        "primary_class_recall": 0.872,
    }


@pytest.fixture
def temp_report_path() -> Path:
    """Create a temporary report file path."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        path = Path(f.name)
    # Clean up after test
    yield path
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_report_creates_file_with_header(
    temp_report_path,
    sample_metrics_good_deal,
    sample_metrics_price_tier,
):
    """New file gets '# Classification Model Training Report' header."""
    write_training_report(
        report_path=temp_report_path,
        model_version=1,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="XGBClassifier",
        price_tier_best_model="RandomForestClassifier",
        train_rows=127400,
        val_rows=27300,
        test_rows=27300,
        feature_count=38,
        class_balance_good_deal={"Good Deal": 8200, "Fair Deal": 15400, "Overpriced": 3700},
        class_balance_price_tier={"Budget": 6800, "Mid-Range": 10200, "Premium": 7100, "Luxury": 3200},
        training_date="2026-08-29",
    )

    content = temp_report_path.read_text(encoding="utf-8")
    assert content.startswith("# Classification Model Training Report")
    assert "## Training Run v1" in content


def test_write_report_appends_entry(
    temp_report_path,
    sample_metrics_good_deal,
    sample_metrics_price_tier,
):
    """Second call appends '## Training Run v2' section."""
    # First write
    write_training_report(
        report_path=temp_report_path,
        model_version=1,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="XGBClassifier",
        price_tier_best_model="RandomForestClassifier",
        train_rows=127400,
        val_rows=27300,
        test_rows=27300,
        feature_count=38,
        class_balance_good_deal={"Good Deal": 8200, "Fair Deal": 15400, "Overpriced": 3700},
        class_balance_price_tier={"Budget": 6800, "Mid-Range": 10200, "Premium": 7100, "Luxury": 3200},
        training_date="2026-08-29",
    )

    # Second write (different version)
    write_training_report(
        report_path=temp_report_path,
        model_version=2,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="RandomForestClassifier",
        price_tier_best_model="XGBClassifier",
        train_rows=130000,
        val_rows=27800,
        test_rows=27800,
        feature_count=38,
        class_balance_good_deal={"Good Deal": 8500, "Fair Deal": 15800, "Overpriced": 3700},
        class_balance_price_tier={"Budget": 7000, "Mid-Range": 10500, "Premium": 7200, "Luxury": 3100},
        training_date="2026-08-30",
    )

    content = temp_report_path.read_text(encoding="utf-8")
    # Should have both sections
    assert content.count("## Training Run v") == 2
    assert "## Training Run v1" in content
    assert "## Training Run v2" in content
    # v2 should appear after v1
    assert content.index("## Training Run v2") > content.index("## Training Run v1")


def test_report_contains_all_sections(
    temp_report_path,
    sample_metrics_good_deal,
    sample_metrics_price_tier,
):
    """Report contains both classifiers, confusion matrices, class balance tables."""
    write_training_report(
        report_path=temp_report_path,
        model_version=1,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="XGBClassifier",
        price_tier_best_model="RandomForestClassifier",
        train_rows=127400,
        val_rows=27300,
        test_rows=27300,
        feature_count=38,
        class_balance_good_deal={"Good Deal": 8200, "Fair Deal": 15400, "Overpriced": 3700},
        class_balance_price_tier={"Budget": 6800, "Mid-Range": 10200, "Premium": 7100, "Luxury": 3200},
        training_date="2026-08-29",
    )

    content = temp_report_path.read_text(encoding="utf-8")

    # Header info
    assert "**Date:** 2026-08-29" in content
    assert "**Train/Val/Test Split:** 127,400 / 27,300 / 27,300 (70/15/15)" in content
    assert "**Features (post-preprocessing):** 38" in content

    # Class balance tables
    assert "**Class Balance (good_deal_verdict):**" in content
    assert "Good Deal: 8,200" in content
    assert "Fair Deal: 15,400" in content
    assert "Overpriced: 3,700" in content
    assert "**Class Balance (price_tier):**" in content
    assert "Budget: 6,800" in content
    assert "Mid-Range: 10,200" in content
    assert "Premium: 7,100" in content
    assert "Luxury: 3,200" in content

    # Good Deal section
    assert "### Good Deal Verdict (Primary)" in content
    assert "**Best Model:** XGBClassifier" in content
    assert "**Selection Metric:** Good Deal Recall = 0.8470" in content

    # Price Tier section
    assert "### Price Tier (Secondary)" in content
    assert "**Best Model:** RandomForestClassifier" in content
    assert "**Selection Metric:** Macro F1 = 0.8210" in content


def test_report_format_matches_spec(
    temp_report_path,
    sample_metrics_good_deal,
    sample_metrics_price_tier,
):
    """Markdown tables, code fences for confusion matrices."""
    write_training_report(
        report_path=temp_report_path,
        model_version=1,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="XGBClassifier",
        price_tier_best_model="RandomForestClassifier",
        train_rows=127400,
        val_rows=27300,
        test_rows=27300,
        feature_count=38,
        class_balance_good_deal={"Good Deal": 8200, "Fair Deal": 15400, "Overpriced": 3700},
        class_balance_price_tier={"Budget": 6800, "Mid-Range": 10200, "Premium": 7100, "Luxury": 3200},
        training_date="2026-08-29",
    )

    content = temp_report_path.read_text(encoding="utf-8")

    # Metric tables should have proper markdown format
    assert "| Metric | Value |" in content
    assert "|--------|-------|" in content
    assert "| Accuracy | 0.7820 |" in content
    assert "| Macro F1 | 0.7560 |" in content

    # Per-class tables
    assert "| Class | Precision | Recall | F1 |" in content
    assert "|-------|-----------|--------|----|" in content
    assert "| Good Deal | 0.7910 | 0.8470 | 0.8180 |" in content
    assert "| Budget | 0.8560 | 0.8720 | 0.8640 |" in content

    # Confusion matrices in code fences
    assert content.count("```") >= 2  # Two confusion matrices
    # Check confusion matrix content
    assert "6945" in content
    assert "5928" in content


def test_report_handles_existing_file_without_header(
    temp_report_path,
    sample_metrics_good_deal,
    sample_metrics_price_tier,
):
    """If file exists but without expected header, prepends header."""
    # Write garbage content first
    temp_report_path.write_text("Some existing content\n", encoding="utf-8")

    write_training_report(
        report_path=temp_report_path,
        model_version=1,
        good_deal_metrics=sample_metrics_good_deal,
        price_tier_metrics=sample_metrics_price_tier,
        good_deal_best_model="XGBClassifier",
        price_tier_best_model="RandomForestClassifier",
        train_rows=100,
        val_rows=20,
        test_rows=20,
        feature_count=10,
        class_balance_good_deal={"Good Deal": 10, "Fair Deal": 20, "Overpriced": 5},
        class_balance_price_tier={"Budget": 5, "Mid-Range": 10, "Premium": 3, "Luxury": 2},
        training_date="2026-08-29",
    )

    content = temp_report_path.read_text(encoding="utf-8")
    # Should have added the header
    assert content.startswith("# Classification Model Training Report")
    assert "Some existing content" in content  # Original preserved