"""Tests for ``ml.evaluation.report`` — write_evaluation_report + append_protocol_section."""

from __future__ import annotations

import json
from pathlib import Path

from ml.evaluation.gate import EvaluationResult
from ml.evaluation.report import (
    append_protocol_section,
    write_evaluation_report,
)


def _make_result(evaluated_at: str = "2026-08-15T12:00:00+00:00") -> EvaluationResult:
    return EvaluationResult(
        version="v1",
        transact_type="sale",
        protocol_version="1.0.0",
        dataset_version="clean_listings.parquet-abcdef12",
        git_commit="abcdef123456",
        split_sizes={"train": 700, "val": 150, "test": 150},
        metrics={
            "train": {"r2": 0.90, "mae": 100_000.0, "rmse": 200_000.0, "mape": 5.0},
            "val": {"r2": 0.82, "mae": 250_000.0, "rmse": 400_000.0, "mape": 8.0},
            "test": {"r2": 0.81, "mae": 300_000.0, "rmse": 500_000.0, "mape": 9.0},
        },
        per_city_test={
            "Gurgaon": {"r2": 0.83, "mae": 280_000.0, "rmse": 450_000.0, "mape": 8.5},
            "Mumbai": {"r2": 0.79, "mae": 320_000.0, "rmse": 520_000.0, "mape": 9.5},
        },
        within_tol_15_pct=0.72,
        latency_p95_ms=120.5,
        thresholds_passed={
            "r2_min": True,
            "r2_stretch": False,
            "mae_pct_within_15_at_least": True,
            "rent_min_rows": True,
            "p95_latency_ms_max": True,
        },
        overall_passed=True,
        evaluated_at=evaluated_at,
        evaluator_version="1.0.0",
    )


def test_write_evaluation_report_writes_versioned_filename(
    tmp_path: Path,
) -> None:
    result = _make_result()
    path = write_evaluation_report(result, tmp_path)
    assert path.name == "evaluation_report_v1_sale.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["version"] == "v1"
    assert payload["transact_type"] == "sale"
    assert payload["protocol_version"] == "1.0.0"
    assert payload["git_commit"] == "abcdef123456"


def test_write_evaluation_report_rerun_uses_timestamp_suffix(
    tmp_path: Path,
) -> None:
    result_a = _make_result(evaluated_at="2026-08-15T12:00:00+00:00")
    result_b = _make_result(evaluated_at="2026-08-15T13:30:00+00:00")

    path_a = write_evaluation_report(result_a, tmp_path)
    path_b = write_evaluation_report(result_b, tmp_path)

    # First run lands at the canonical name; second lands at a
    # rerun_<timestamp>-suffixed sibling so the original is preserved
    # (Rules §2.5: never overwrite).
    assert path_a.name == "evaluation_report_v1_sale.json"
    assert "rerun_" in path_b.name
    assert path_b.name != path_a.name
    assert path_a.exists()
    assert path_b.exists()

    # Original content is preserved verbatim.
    original_payload = json.loads(path_a.read_text())
    assert original_payload["evaluated_at"] == "2026-08-15T12:00:00+00:00"


def test_write_evaluation_report_same_timestamp_is_idempotent(
    tmp_path: Path,
) -> None:
    result = _make_result(evaluated_at="2026-08-15T12:00:00+00:00")
    path_a = write_evaluation_report(result, tmp_path)
    path_b = write_evaluation_report(result, tmp_path)
    assert path_a == path_b
    # Only one file exists (no rerun-suffix sibling).
    siblings = list(tmp_path.glob("evaluation_report_v1_sale*.json"))
    assert len(siblings) == 1


def test_append_protocol_section_appends_not_overwrites(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "feature_selection_report.md"
    sentinel = "## Round 1 — correlation + Lasso (Step 12)\n\n[sentinel content]\n"
    report_path.write_text(sentinel, encoding="utf-8")

    result = _make_result()
    append_protocol_section(result, report_path)

    text = report_path.read_text(encoding="utf-8")
    assert "[sentinel content]" in text  # pre-existing content preserved
    assert "Protocol Certification" in text  # new section appended
    assert text.index("[sentinel content]") < text.index("Protocol Certification")


def test_append_protocol_section_creates_file_when_missing(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "new_report.md"
    assert not report_path.exists()

    result = _make_result()
    append_protocol_section(result, report_path)

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Protocol Certification" in text
    assert "sale_v1" in text
