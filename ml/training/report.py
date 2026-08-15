"""Feature-selection report helpers (Spec 13).

Reads ``data/processed/feature_selection_report.md`` (Step 12 wrote
Round 1), appends Round 2 (tree-based + permutation importance) +
Round 3 (SHAP ranking) + Final feature list sections, writes the
combined file atomically. The append is the only legitimate way to
modify the report — never overwrite Step 12's content.
"""

from __future__ import annotations

import json
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


def write_v2_lever_section(
    lever_results: dict,
    winner_name: str,
    improvement_pct: dict[str, dict[str, float]] | None = None,
    target_met: dict[str, dict[str, bool]] | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Append a 'Spec 14 — improvement levers' section to the report.

    Distinct from ``append_round_2_3``: this is v2-only and covers the
    4 levers (stacking, Optuna, geo, target encoding) — not the v1
    feature-selection rounds. Same atomic-write discipline as the v1
    helper.

    ``lever_results`` shape (best-effort; missing keys render as
    ``_n/a — not run_``):

        {
            "stacking": {"val_rmse": ..., "test_rmse": ...},
            "optuna_xgb": {"best_value": -..., "best_params": {...}},
            "optuna_lgbm": {"best_value": -..., "best_params": {...}},
            "geo_features": ["distance_to_cbd_km", ...],
            "sector_encoding": {"smoothing_prior": 20.0, ...},
        }

    Returns the path that was written.
    """
    report_path = Path(out_path) if out_path else DEFAULT_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else ""
    )

    blocks: list[str] = [
        "\n---\n\n",
        "## Spec 14 — Improvement levers (v2 boosted-tree model)\n\n",
        f"v2 winner model: `{winner_name}`. The four levers (stacking, "
        "Optuna tuning, geospatial features, sector target encoding) "
        "were exercised independently; the model with the lowest "
        "validation RMSE was selected.\n\n",
    ]

    # Lever 1 — Stacking
    blocks.append("### Lever 1 — Stacking (5 base learners + Ridge meta)\n\n")
    st = lever_results.get("stacking") or {}
    if st:
        blocks.append("| split | RMSE (₹) | MAE (₹) | R² |\n")
        blocks.append("|-------|----------|---------|----|\n")
        for split in ("train", "val", "test"):
            m = st.get(f"{split}_metrics") or st.get(split)
            if not isinstance(m, dict):
                continue
            blocks.append(
                f"| {split} | {m.get('rmse', float('nan')):,.0f} | "
                f"{m.get('mae', float('nan')):,.0f} | "
                f"{m.get('r2', float('nan')):.4f} |\n"
            )
        blocks.append("\n")
    else:
        blocks.append("_n/a — stacking not run._\n\n")

    # Lever 2 — Optuna
    blocks.append("### Lever 2 — Optuna Bayesian search (XGB + LGBM)\n\n")
    for tag, label in (
        ("optuna_xgb", "XGBoost"),
        ("optuna_lgbm", "LightGBM"),
    ):
        o = lever_results.get(tag) or {}
        blocks.append(f"**{label}**\n\n")
        if o:
            val = o.get("best_value")
            if val is not None:
                # val is negative RMSE; report the positive value.
                blocks.append(f"- best neg-RMSE: `{val:.4f}` "
                              f"(≈ RMSE ₹{-val:,.0f})\n")
            params = o.get("best_params") or {}
            if params:
                # Trim verbosity for readability.
                keep = {k: v for k, v in params.items()
                        if k in {
                            "max_depth", "num_leaves", "learning_rate",
                            "n_estimators", "subsample", "colsample_bytree",
                            "min_child_weight", "min_child_samples",
                            "reg_alpha", "reg_lambda",
                        }}
                blocks.append(
                    "- best params: `" + json.dumps(keep, default=str) + "`\n"
                )
        else:
            blocks.append("_n/a — not run._\n")
        blocks.append("\n")

    # Lever 3 — Geospatial
    blocks.append("### Lever 3 — Geospatial features\n\n")
    geo_cols = lever_results.get("geo_features") or []
    if geo_cols:
        blocks.append("Added columns: `" + "`, `".join(geo_cols) + "`.\n\n")
    else:
        blocks.append("_n/a — geo features not added._\n\n")

    # Lever 4 — Sector target encoding
    blocks.append("### Lever 4 — Sector target encoding\n\n")
    sec = lever_results.get("sector_encoding") or {}
    if sec:
        blocks.append(
            f"- smoothing prior weight: `{sec.get('smoothing_prior_weight', 20.0)}`\n"
            f"- output column: `{sec.get('output_column', 'sector_smoothed_price')}`\n"
            f"- groups fitted: `{sec.get('n_groups', 'n/a')}`\n\n"
        )
    else:
        blocks.append("_n/a — sector encoding not run._\n\n")

    # Improvement table — honest shortfall per Rules §9.2.
    if improvement_pct:
        blocks.append("### v2 vs v1 — test-slice improvement (%)\n\n")
        blocks.append(
            "_Positive = v2 better (lower error / higher R²)._\n\n"
        )
        blocks.append("| metric | sale | rent |\n")
        blocks.append("|--------|------|------|\n")
        for metric in ("mae", "rmse", "mape", "r2"):
            cells = improvement_pct.get(metric, {})
            if not cells:
                continue
            blocks.append(
                f"| {metric.upper()} | "
                f"{cells.get('sale', float('nan')):+.2f}% | "
                f"{cells.get('rent', float('nan')):+.2f}% |\n"
            )
        blocks.append("\n")

        if target_met:
            blocks.append("**Spec target (≥32.5% MAE/RMSE reduction):**\n\n")
            for metric in ("mae", "rmse"):
                cells = target_met.get(metric, {})
                if not cells:
                    continue
                bits = ", ".join(
                    f"{ttype}: {'✓' if met else '✗'}"
                    for ttype, met in cells.items()
                )
                blocks.append(f"- {metric.upper()} — {bits}\n")
            blocks.append("\n")

    blocks.append(
        f"_Generated by `scripts/train_price_model_v2.py` against the "
        f"v2 boosted-tree stack. Winner: `{winner_name}`._\n"
    )

    appended = "".join(blocks)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".report_v2_", suffix=".md.tmp", dir=str(report_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(existing)
            fh.write(appended)
        Path(tmp_name).replace(report_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    logger.info(
        "Appended v2 lever section to %s (winner=%s).",
        report_path,
        winner_name,
    )
    return report_path


__all__ = [
    "DEFAULT_REPORT_PATH",
    "append_round_2_3",
    "feature_importance_table",
    "write_v2_lever_section",
]
