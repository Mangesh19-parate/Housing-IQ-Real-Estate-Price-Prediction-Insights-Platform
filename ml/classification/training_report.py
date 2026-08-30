"""Append-only classification training report writer (Spec 23).

Authority:
    - .claude/specs/23-classification-model-training.md
    - Pattern from Spec 21/22: append-only .md reports with versioned headers

Produces a Markdown file with one `## Training Run v{n}` section per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_confusion_matrix(cm: list[list[int]]) -> str:
    """Format confusion matrix as a Markdown code fence."""
    lines = ["```"]
    for row in cm:
        lines.append(" ".join(f"{val:5d}" for val in row))
    lines.append("```")
    return "\n".join(lines)


def _format_class_balance_table(balance: dict[str, int]) -> str:
    """Format class balance as a Markdown inline list."""
    parts = [f"{label}: {count:,}" for label, count in balance.items()]
    return " | ".join(parts)


def _format_metrics_table(metrics: dict[str, Any], labels: tuple[str, ...]) -> str:
    """Format per-class metrics as a Markdown table."""
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        "",
        "| Class | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for label in labels:
        p = metrics["per_class_precision"].get(label, 0.0)
        r = metrics["per_class_recall"].get(label, 0.0)
        f1 = metrics["per_class_f1"].get(label, 0.0)
        lines.append(f"| {label} | {p:.4f} | {r:.4f} | {f1:.4f} |")
    return "\n".join(lines)


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
    training_date: str,  # ISO format (YYYY-MM-DD)
) -> None:
    """Append a training run entry to the report file.

    Creates the file with a header if it doesn't exist.
    Each entry is a `## Training Run v{model_version}` section.

    Args:
        report_path: Path to the Markdown report file.
        model_version: Integer version (e.g., 1 -> v1).
        good_deal_metrics: Full metrics dict from evaluate_classifier for good_deal_verdict.
        price_tier_metrics: Full metrics dict from evaluate_classifier for price_tier.
        good_deal_best_model: Name of selected model (e.g., "XGBClassifier").
        price_tier_best_model: Name of selected model.
        train_rows: Number of training samples.
        val_rows: Number of validation samples.
        test_rows: Number of test samples.
        feature_count: Number of features after preprocessing (output dimension).
        class_balance_good_deal: Dict of label -> count for good_deal_verdict.
        class_balance_price_tier: Dict of label -> count for price_tier.
        training_date: ISO date string (YYYY-MM-DD).
    """
    # Build the entry content
    entry_lines = [
        f"## Training Run v{model_version}",
        f"**Date:** {training_date}",
        f"**Train/Val/Test Split:** {train_rows:,} / {val_rows:,} / {test_rows:,} (70/15/15)",
        f"**Features (post-preprocessing):** {feature_count}",
        f"**Class Balance (good_deal_verdict):** {_format_class_balance_table(class_balance_good_deal)}",
        f"**Class Balance (price_tier):** {_format_class_balance_table(class_balance_price_tier)}",
        "",
        "### Good Deal Verdict (Primary)",
        f"**Best Model:** {good_deal_best_model}",
        f"**Selection Metric:** Good Deal Recall = {good_deal_metrics['primary_class_recall']:.4f}",
        "",
        _format_metrics_table(good_deal_metrics, ("Good Deal", "Fair Deal", "Overpriced")),
        "",
        "**Confusion Matrix (rows=true, cols=pred):**",
        _format_confusion_matrix(good_deal_metrics["confusion_matrix"]),
        "",
        "### Price Tier (Secondary)",
        f"**Best Model:** {price_tier_best_model}",
        f"**Selection Metric:** Macro F1 = {price_tier_metrics['macro_f1']:.4f}",
        "",
        _format_metrics_table(price_tier_metrics, ("Budget", "Mid-Range", "Premium", "Luxury")),
        "",
        "**Confusion Matrix (rows=true, cols=pred):**",
        _format_confusion_matrix(price_tier_metrics["confusion_matrix"]),
        "",
        "---",
        "",
    ]

    entry = "\n".join(entry_lines)

    # Read existing content or create new with header
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        # Append after the header (first line should be "# Classification Model Training Report")
        if existing.startswith("# Classification Model Training Report"):
            new_content = existing.rstrip() + "\n\n" + entry
        else:
            # Unexpected format - prepend header, preserve existing content, then append entry
            new_content = "# Classification Model Training Report\n\n" + existing.rstrip() + "\n\n" + entry
    else:
        new_content = "# Classification Model Training Report\n\n" + entry

    # Write atomically
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(new_content, encoding="utf-8")


__all__ = ["write_training_report"]