"""Feature-selection report helpers (Spec 13).

Reads ``data/processed/feature_selection_report.md`` (Step 12 wrote
Round 1), appends Round 2 (tree-based + permutation importance) +
Round 3 (SHAP ranking) + Final feature list sections, writes the
combined file atomically. The append is the only legitimate way to
modify the report — never overwrite Step 12's content.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

#: Default report path; ``HOUSINGIQ_REPORT_PATH`` overrides for tests.
DEFAULT_REPORT_PATH: Path = Path(
    os.environ.get(
        "HOUSINGIQ_REPORT_PATH", "data/processed/feature_selection_report.md"
    )
)


def feature_importance_table(
    estimator,
    feature_names: list[str],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Return top-N ``(feature_name, importance)`` pairs for tree models.

    Returns an empty list for non-tree estimators (Linear/Ridge/Lasso)
    — the caller decides whether to log a WARNING. Sorted by
    importance descending.
    """
    if not hasattr(estimator, "feature_importances_"):
        return []
    pairs = list(zip(list(feature_names), list(estimator.feature_importances_)))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs[:top_n]


def _render_importance_table(
    title: str, rows: Iterable[tuple[str, float]]
) -> str:
    lines = [f"### {title}", ""]
    rows = list(rows)
    if not rows:
        lines.append("_n/a — non-tree model or no feature_importances_.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| rank | feature | importance |")
    lines.append("|------|---------|------------|")
    for i, (name, val) in enumerate(rows, 1):
        lines.append(f"| {i} | `{name}` | {val:.6f} |")
    lines.append("")
    return "\n".join(lines)


def _render_perm_table(
    rows: Iterable[tuple[str, float]], top_n: int = 20
) -> str:
    lines = ["### Permutation importance (validation slice, n_repeats=10)", ""]
    rows = sorted(rows, key=lambda kv: kv[1], reverse=True)[:top_n]
    if not rows:
        lines.append("_n/a — permutation importance not computed._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| rank | feature | mean importance |")
    lines.append("|------|---------|-----------------|")
    for i, (name, val) in enumerate(rows, 1):
        lines.append(f"| {i} | `{name}` | {val:.6f} |")
    lines.append("")
    return "\n".join(lines)


def _render_shap_table(
    rows: Iterable[tuple[str, float]], top_n: int = 20
) -> str:
    lines = ["### SHAP ranking (mean |SHAP| over test slice)", ""]
    rows = sorted(rows, key=lambda kv: kv[1], reverse=True)[:top_n]
    if not rows:
        lines.append(
            "_n/a — non-tree model winner; SHAP ranking deferred to a "
            "tree-based winner._"
        )
        lines.append("")
        return "\n".join(lines)
    lines.append("| rank | feature | mean \\|SHAP\\| |")
    lines.append("|------|---------|----------------|")
    for i, (name, val) in enumerate(rows, 1):
        lines.append(f"| {i} | `{name}` | {val:.6f} |")
    lines.append("")
    return "\n".join(lines)


def append_round_2_3(
    report_path: Path | str,
    rf_importances: list[tuple[str, float]],
    gb_importances: list[tuple[str, float]],
    xgb_importances: list[tuple[str, float]],
    perm_importances: list[tuple[str, float]],
    shap_importances: list[tuple[str, float]],
    winner_name: str,
    chosen_metrics: dict | None = None,
) -> None:
    """Atomically append Round 2 + Round 3 + Final sections to the report.

    Existing content (Step 12's Round 1) is preserved verbatim. The
    atomic write pattern (``tempfile`` + ``Path.replace``) guarantees
    the file is never half-written on a crash.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else ""
    )

    blocks = [
        "\n---\n\n",
        "## Round 2 — Tree-based + permutation importance\n\n",
        "Computed by the Step 13 training script against the v1 "
        "baseline. Top-N features per tree model + permutation "
        "importance from the validation slice of the winner "
        f"(`{winner_name}`).\n\n",
        _render_importance_table(
            "Random Forest impurity importance", rf_importances
        ),
        _render_importance_table(
            "Gradient Boosting impurity importance", gb_importances
        ),
        _render_importance_table(
            "XGBoost impurity importance", xgb_importances
        ),
        _render_perm_table(perm_importances),
        "\n## Round 3 — SHAP ranking\n\n",
        _render_shap_table(shap_importances),
        "\n## Final feature list & rationale\n\n",
        "Decision rule (TRD §9): a feature is kept if it ranks in "
        "the top-N by at least 2 of the 4 model-based methods "
        "(RF importance, GB importance, Permutation, SHAP), or if "
        "it has a clear, non-zero Lasso coefficient. Top features "
        "in **every** ranking method in this run:\n\n",
    ]

    # Cross-method consensus: intersection of all four top-20 sets.
    consensus_sets: list[set[str]] = []
    for rows in (rf_importances, gb_importances, xgb_importances,
                 perm_importances):
        consensus_sets.append({name for name, _ in rows[:20]})
    consensus = set.intersection(*consensus_sets) if consensus_sets else set()

    if shap_importances:
        shap_top20 = {name for name, _ in shap_importances[:20]}
        consensus &= shap_top20

    if consensus:
        blocks.append("| feature | methods agreeing |")
        blocks.append("|---------|------------------|")
        for name in sorted(consensus):
            blocks.append(f"| `{name}` | 4 of 4 |")
        blocks.append("")
    else:
        blocks.append(
            "_No single feature appeared in all 4 method rankings — "
            "feature set is the union of Step 12's correlation "
            "filter + every tree-model top-20._\n\n"
        )

    if chosen_metrics is not None:
        blocks.append("**Winner test metrics** (original ₹ scale):\n\n")
        for split in ("train", "val", "test"):
            m = chosen_metrics.get(split, {})
            if not m:
                continue
            blocks.append(
                f"- {split}: R²={m.get('r2', float('nan')):.4f}, "
                f"MAE=₹{m.get('mae', float('nan')):,.0f}, "
                f"RMSE=₹{m.get('rmse', float('nan')):,.0f}, "
                f"MAPE={m.get('mape', float('nan')):.2f}%\n"
            )
        blocks.append("\n")

    blocks.append(
        f"_Generated by `scripts/train_price_model.py` against the "
        f"v1 baseline. Winner: `{winner_name}`._\n"
    )

    appended = "".join(blocks)

    # Atomic write: tempfile in the same dir, then replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".report_", suffix=".md.tmp", dir=str(report_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(existing)
            fh.write(appended)
        Path(tmp_name).replace(report_path)
    except Exception:
        # Clean up the temp file on any failure to avoid leaving litter.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    logger.info(
        "Appended Round 2 + Round 3 + Final sections to %s (winner=%s).",
        report_path,
        winner_name,
    )


__all__ = [
    "DEFAULT_REPORT_PATH",
    "append_round_2_3",
    "feature_importance_table",
]
