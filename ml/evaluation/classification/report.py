"""Report persistence for classification evaluation (Spec 24).

Two artifacts:
    models/evaluation_reports/{target}_v{version}_evaluation.json  —
        full ClassificationEvaluationResult as JSON.
    models/classification_training_report.md  — append-only Markdown
        with one "## Protocol Certification — {target} v{version}" section
        per evaluation run. Idempotent: re-running with same target+version
        replaces the section rather than duplicating.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

from ml.evaluation.classification.gate import ClassificationEvaluationResult

logger = logging.getLogger(__name__)


def write_evaluation_report(
    result: ClassificationEvaluationResult,
    out_dir: Path | str | None = None,
) -> Path:
    """
    Write the evaluation result to ``models/evaluation_reports/classification_eval_{target}_v{version}.json``.

    The filename uses ``v{version}`` where ``version`` is the stripped version
    number (e.g., "v1" -> "v1", "1" -> "v1"). This matches the CLI's
    ``--version v1`` convention and the training script's artifact naming.

    Args:
        result: ClassificationEvaluationResult from evaluate().
        out_dir: Optional override for the output directory.
                 Defaults to ``models`` (function will create ``evaluation_reports`` subdir).

    Returns:
        Path to the written JSON file.
    """
    if out_dir is None:
        out_dir = Path("models")
    out_dir = Path(out_dir)
    eval_dir = out_dir / "evaluation_reports"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Normalize version: "v1" -> "v1", "1" -> "v1"
    version_suffix = result.version.lstrip("v")
    filename = f"classification_eval_{result.target}_v{version_suffix}.json"
    report_path = eval_dir / filename

    data = dataclasses.asdict(result)

    # Write atomically
    report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def append_protocol_section(
    result: ClassificationEvaluationResult,
    report_path: Path | str,
) -> None:
    """
    Append a "## Protocol Certification — {target} v{version}" section to
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
    report_path = Path(report_path)

    # Build the section content
    test_metrics = result.metrics.get("test", {}) if isinstance(result.metrics, dict) else {}

    version_suffix = result.version.lstrip("v")

    section_lines = [
        f"## Protocol Certification — {result.target} v{version_suffix}",
        f"**Protocol Version:** {result.protocol_version}  ",
        f"**Dataset Fingerprint:** {result.dataset_version}  ",
        f"**Git Commit:** {result.git_commit}  ",
        f"**Evaluated At:** {result.evaluated_at}  ",
        f"**Evaluator Version:** {result.evaluator_version}  ",
        "",
        "### Split Sizes",
        "",
        "| Split | Count |",
        "|-------|-------|",
        f"| Train | {result.split_sizes.get('train', 0):,} |",
        f"| Validation | {result.split_sizes.get('val', 0):,} |",
        f"| Test | {result.split_sizes.get('test', 0):,} |",
        "",
        "### Test Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    # METRIC_NAMES order from protocol
    from ml.evaluation.classification.protocol import METRIC_NAMES
    for metric_name in METRIC_NAMES:
        value = test_metrics.get(metric_name)
        if value is not None:
            section_lines.append(f"| {metric_name} | {value:.4f} |")

    section_lines.extend([
        "",
        "### Per-Class Metrics (Test)",
        "",
        _format_per_class_table(result.per_class_test),
        "",
        "### Confusion Matrix (Test)",
        "",
        "**Rows = True, Cols = Predicted**",
        "",
        _format_confusion_matrix(result.confusion_matrix.get("matrix", [])),
        "",
        "### Threshold Checklist",
        "",
        _format_thresholds_table(result.thresholds_passed),
        "",
        f"**Overall:** **{'CERTIFIED' if result.overall_passed else 'NOT CERTIFIED'}**",
        "",
        "---",
        "",
    ])

    new_section = "\n".join(section_lines)

    # Pattern to find existing section for same target+version
    pattern = (
        rf"## Protocol Certification — {re.escape(result.target)} v{re.escape(version_suffix)}"
        r".*?(?=\n## |\Z)"
    )

    # Read existing content or create new with header
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        if re.search(pattern, existing, re.DOTALL):
            # Replace existing section
            new_content = re.sub(pattern, new_section, existing, flags=re.DOTALL)
        else:
            # Append after the header (first line should be "# Classification Model Training Report")
            if existing.startswith("# Classification Model Training Report"):
                new_content = existing.rstrip() + "\n\n" + new_section
            else:
                new_content = "# Classification Model Training Report\n\n" + existing.rstrip() + "\n\n" + new_section
    else:
        new_content = "# Classification Model Training Report\n\n" + new_section

    # Write atomically
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(new_content, encoding="utf-8")


def _format_per_class_table(per_class: dict[int, dict]) -> str:
    """Format per-class metrics as a markdown table."""
    if not per_class:
        return "| Class | Precision | Recall | F1 | Support |\n|-------|-----------|--------|----|---------|\n| (no data) | — | — | — | — |"
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for cls, metrics in sorted(per_class.items()):
        lines.append(
            f"| {cls} | {metrics.get('precision', 0.0):.4f} | "
            f"{metrics.get('recall', 0.0):.4f} | "
            f"{metrics.get('f1', 0.0):.4f} | "
            f"{metrics.get('support', 0)} |"
        )
    return "\n".join(lines)


def _format_confusion_matrix(matrix: list[list[int]]) -> str:
    """Format confusion matrix as a markdown table."""
    if not matrix:
        return "(empty)"
    n = len(matrix)
    header = "| | " + " | ".join(f"Pred {i}" for i in range(n)) + " |"
    sep = "|---" * (n + 1) + "|"
    rows = []
    for i, row in enumerate(matrix):
        rows.append(f"| True {i} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def _format_thresholds_table(thresholds: dict[str, bool]) -> str:
    """Format threshold checklist as a markdown table."""
    lines = ["| Threshold | Status |", "|-----------|--------|"]
    for name, passed in thresholds.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"| {name} | {status} |")
    return "\n".join(lines)


__all__ = [
    "append_protocol_section",
    "write_evaluation_report",
]