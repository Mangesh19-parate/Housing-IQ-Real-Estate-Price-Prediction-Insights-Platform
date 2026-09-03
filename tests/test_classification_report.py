"""Tests for classification evaluation report persistence (Spec 24)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ml.evaluation.classification.gate import ClassificationEvaluationResult
from ml.evaluation.classification.report import (
    append_protocol_section,
    write_evaluation_report,
)


@pytest.fixture
def sample_result():
    """Create a sample ClassificationEvaluationResult for testing."""
    return ClassificationEvaluationResult(
        version="v1",
        target="good_deal",
        protocol_version="1.0.0",
        dataset_version="feature_frame.parquet-abc12345",
        git_commit="abcdef123456",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={
            "train": {"accuracy": 0.85, "f1_macro": 0.82, "good_deal_recall": 0.78},
            "val": {"accuracy": 0.83, "f1_macro": 0.80, "good_deal_recall": 0.75},
            "test": {"accuracy": 0.82, "f1_macro": 0.79, "good_deal_recall": 0.81,
                     "precision_macro": 0.78, "recall_macro": 0.77, "price_tier_macro_f1": 0.79},
        },
        per_class_test={
            0: {"precision": 0.85, "recall": 0.81, "f1": 0.83, "support": 30},
            1: {"precision": 0.75, "recall": 0.78, "f1": 0.76, "support": 90},
            2: {"precision": 0.72, "recall": 0.70, "f1": 0.71, "support": 30},
        },
        confusion_matrix={
            "matrix": [[25, 4, 1], [10, 70, 10], [3, 7, 20]],
            "labels": ["0", "1", "2"],
        },
        thresholds_passed={
            "good_deal_recall_min": True,
            "f1_macro_min": True,
            "accuracy_min": True,
        },
        overall_passed=True,
        evaluated_at="2026-08-30T12:00:00+00:00",
        evaluator_version="1.0.0",
    )


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        models = tmp / "models"
        models.mkdir(parents=True)
        yield {"models": models, "report": tmp / "classification_training_report.md"}


def test_write_evaluation_report_creates_json(temp_dirs, sample_result):
    """write_evaluation_report creates versioned JSON file with correct structure."""
    d = temp_dirs
    report_path = write_evaluation_report(sample_result, d["models"])

    expected_name = "classification_eval_good_deal_v1.json"
    assert report_path.name == expected_name
    assert report_path.exists()

    # Verify content
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "v1"
    assert data["target"] == "good_deal"
    assert data["protocol_version"] == "1.0.0"
    assert data["dataset_version"] == "feature_frame.parquet-abc12345"
    assert data["git_commit"] == "abcdef123456"
    assert data["split_sizes"] == {"train": 700, "val": 150, "test": 150}
    assert "test" in data["metrics"]
    assert data["overall_passed"] is True
    assert data["evaluated_at"] == "2026-08-30T12:00:00+00:00"
    assert data["evaluator_version"] == "1.0.0"


def test_write_evaluation_report_creates_dir(temp_dirs, sample_result):
    """write_evaluation_report creates evaluation_reports directory if needed."""
    d = temp_dirs
    # Don't create evaluation_reports dir - write_evaluation_report should create it
    report_path = write_evaluation_report(sample_result, d["models"])
    assert (d["models"] / "evaluation_reports").exists()


def test_write_evaluation_report_price_tier(temp_dirs):
    """write_evaluation_report works for price_tier target."""
    d = temp_dirs
    result = ClassificationEvaluationResult(
        version="v2",
        target="price_tier",
        protocol_version="1.0.0",
        dataset_version="feature_frame.parquet-def67890",
        git_commit="fedcba654321",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={
            "test": {"accuracy": 0.78, "f1_macro": 0.76, "price_tier_macro_f1": 0.76,
                     "precision_macro": 0.75, "recall_macro": 0.74, "good_deal_recall": 0.0},
        },
        per_class_test={
            0: {"precision": 0.80, "recall": 0.78, "f1": 0.79, "support": 40},
            1: {"precision": 0.72, "recall": 0.75, "f1": 0.73, "support": 50},
            2: {"precision": 0.70, "recall": 0.68, "f1": 0.69, "support": 40},
            3: {"precision": 0.65, "recall": 0.62, "f1": 0.63, "support": 20},
        },
        confusion_matrix={
            "matrix": [[35, 3, 2, 0], [5, 38, 5, 2], [3, 4, 30, 3], [1, 2, 3, 14]],
            "labels": ["0", "1", "2", "3"],
        },
        thresholds_passed={
            "f1_macro_min": True,
            "accuracy_min": True,
            "recall_macro_min": False,
        },
        overall_passed=False,
        evaluated_at="2026-08-30T12:00:00+00:00",
        evaluator_version="1.0.0",
    )

    report_path = write_evaluation_report(result, d["models"])
    assert report_path.name == "classification_eval_price_tier_v2.json"
    assert report_path.exists()


def test_append_protocol_section_creates_report(temp_dirs, sample_result):
    """append_protocol_section creates report with protocol section."""
    d = temp_dirs
    append_protocol_section(sample_result, d["report"])

    assert d["report"].exists()
    content = d["report"].read_text(encoding="utf-8")

    assert "# Classification Model Training Report" in content
    assert "## Protocol Certification — good_deal v1" in content
    assert "**Protocol Version:** 1.0.0" in content
    assert "**Dataset Fingerprint:** feature_frame.parquet-abc12345" in content
    assert "**Git Commit:** abcdef123456" in content
    assert "### Split Sizes" in content
    assert "| Train | 700 |" in content
    assert "| Validation | 150 |" in content
    assert "| Test | 150 |" in content
    assert "### Test Metrics" in content
    assert "| accuracy | 0.8200 |" in content
    assert "| f1_macro | 0.7900 |" in content
    assert "| good_deal_recall | 0.8100 |" in content
    assert "### Per-Class Metrics (Test)" in content
    assert "| 0 | 0.8500 | 0.8100 | 0.8300 | 30 |" in content
    assert "### Confusion Matrix (Test)" in content
    assert "### Threshold Checklist" in content
    assert "| good_deal_recall_min | ✅ PASS |" in content
    assert "| f1_macro_min | ✅ PASS |" in content
    assert "| accuracy_min | ✅ PASS |" in content
    assert "**Overall:** **CERTIFIED**" in content


def test_append_protocol_section_not_certified(temp_dirs):
    """append_protocol_section shows NOT CERTIFIED when overall_passed=False."""
    d = temp_dirs
    result = ClassificationEvaluationResult(
        version="v1",
        target="price_tier",
        protocol_version="1.0.0",
        dataset_version="feature_frame.parquet-abc12345",
        git_commit="abcdef123456",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={"test": {"accuracy": 0.65, "f1_macro": 0.60, "price_tier_macro_f1": 0.60,
                           "precision_macro": 0.58, "recall_macro": 0.55, "good_deal_recall": 0.0}},
        per_class_test={},
        confusion_matrix={"matrix": [], "labels": []},
        thresholds_passed={"f1_macro_min": False, "accuracy_min": False, "recall_macro_min": False},
        overall_passed=False,
        evaluated_at="2026-08-30T12:00:00+00:00",
        evaluator_version="1.0.0",
    )

    append_protocol_section(result, d["report"])
    content = d["report"].read_text(encoding="utf-8")

    assert "**Overall:** **NOT CERTIFIED**" in content
    assert "| f1_macro_min | ❌ FAIL |" in content


def test_append_protocol_section_idempotent(temp_dirs, sample_result):
    """append_protocol_section replaces (not duplicates) section for same target+version."""
    d = temp_dirs
    # First append
    append_protocol_section(sample_result, d["report"])
    content1 = d["report"].read_text(encoding="utf-8")
    count1 = content1.count("## Protocol Certification — good_deal v1")

    # Second append (same target+version)
    append_protocol_section(sample_result, d["report"])
    content2 = d["report"].read_text(encoding="utf-8")
    count2 = content2.count("## Protocol Certification — good_deal v1")

    assert count1 == 1
    assert count2 == 1  # Not duplicated


def test_append_protocol_section_multiple_targets(temp_dirs):
    """append_protocol_section handles multiple targets separately."""
    d = temp_dirs

    result1 = ClassificationEvaluationResult(
        version="v1", target="good_deal", protocol_version="1.0.0",
        dataset_version="feature_frame.parquet-abc12345", git_commit="abcdef123456",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={"test": {"accuracy": 0.82, "f1_macro": 0.79, "good_deal_recall": 0.81,
                           "precision_macro": 0.78, "recall_macro": 0.77, "price_tier_macro_f1": 0.79}},
        per_class_test={}, confusion_matrix={"matrix": [], "labels": []},
        thresholds_passed={"good_deal_recall_min": True, "f1_macro_min": True, "accuracy_min": True},
        overall_passed=True, evaluated_at="2026-08-30T12:00:00+00:00", evaluator_version="1.0.0",
    )

    result2 = ClassificationEvaluationResult(
        version="v1", target="price_tier", protocol_version="1.0.0",
        dataset_version="feature_frame.parquet-abc12345", git_commit="abcdef123456",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={"test": {"accuracy": 0.78, "f1_macro": 0.76, "price_tier_macro_f1": 0.76,
                           "precision_macro": 0.75, "recall_macro": 0.74, "good_deal_recall": 0.0}},
        per_class_test={}, confusion_matrix={"matrix": [], "labels": []},
        thresholds_passed={"f1_macro_min": True, "accuracy_min": True, "recall_macro_min": True},
        overall_passed=True, evaluated_at="2026-08-30T12:00:00+00:00", evaluator_version="1.0.0",
    )

    append_protocol_section(result1, d["report"])
    append_protocol_section(result2, d["report"])

    content = d["report"].read_text(encoding="utf-8")
    assert "## Protocol Certification — good_deal v1" in content
    assert "## Protocol Certification — price_tier v1" in content
    assert content.count("## Protocol Certification") == 2


def test_append_protocol_section_preserves_existing_content(temp_dirs, sample_result):
    """append_protocol_section preserves existing report content."""
    d = temp_dirs
    # Pre-populate with existing content
    existing = "# Classification Model Training Report\n\n## Training Run v1\n\nExisting content here.\n\n---\n"
    d["report"].write_text(existing, encoding="utf-8")

    append_protocol_section(sample_result, d["report"])
    content = d["report"].read_text(encoding="utf-8")

    assert "Existing content here." in content
    assert "## Protocol Certification — good_deal v1" in content