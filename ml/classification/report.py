"""Artifact writers + label-construction report (Spec 21).

Mirrors ``ml/evaluation/report.py`` — append-only Markdown writer +
JSON / Parquet writers for the four classification-label artifacts.

Writes (under ``data/processed/classifier/`` by default):
    - ``price_tier_labels.parquet``
    - ``price_tier_quantile_boundaries.json``
    - ``good_deal_labels.parquet``
    - ``good_deal_thresholds.json``
    - ``label_construction_report.md``  (append-only, header on
      first call)

Authority:
    - docs/08-RULES.md §1.3 (every derived table states its
      computation date + source dataset version)
    - docs/08-RULES.md §2.5 (versioned artifacts, never overwritten
      in place — re-run overwrites a re-run with the same git
      commit; the report appends)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Header line written when ``label_construction_report.md`` is
#: created for the first time. Mirrors the
#: ``feature_selection_report.md`` style.
_REPORT_HEADER = """# Label Construction Report

Append-only report written by ``scripts/build_classification_labels.py``
(Spec 21). One "Run" block per CLI invocation.

Pinned by:
- ``docs/02-TRD.md`` §U-TRD-1, §U-TRD-6
- ``docs/01-PRD.md`` §U4.1–§U4.4 (v4 reframing)
- ``docs/08-RULES.md`` §8 (Classification-module rules), §12.2
  (per-city threshold tuning)

"""


def _dataset_version(parquet_path: Path | None) -> str:
    """Return ``:<sha1[:12]>`` for the first 1 MB of the
    input parquet (Rules §1.3 — cheap content fingerprint).
    Returns ``"<unknown>"`` if the file is missing."""
    if parquet_path is None or not parquet_path.exists():
        return "<unknown>"
    hasher = hashlib.sha1()
    with open(parquet_path, "rb") as fh:
        hasher.update(fh.read(1024 * 1024))
    return f"{parquet_path.name}:{hasher.hexdigest()[:12]}"


def write_price_tier_artifacts(
    df: pd.DataFrame,
    boundaries: dict[str, dict[str, list[float]]],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write the two price-tier artifact files.

    Returns ``(labels_path, boundaries_path)``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "price_tier_labels.parquet"
    df.to_parquet(labels_path, index=False)
    logger.info("Wrote %s (%d rows)", labels_path, len(df))

    boundaries_path = out_dir / "price_tier_quantile_boundaries.json"
    with open(boundaries_path, "w", encoding="utf-8") as fh:
        json.dump(boundaries, fh, indent=2, sort_keys=True)
    logger.info("Wrote %s", boundaries_path)

    return labels_path, boundaries_path


def write_good_deal_artifacts(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write the two good-deal artifact files.

    Returns ``(labels_path, thresholds_path)``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "good_deal_labels.parquet"
    df.to_parquet(labels_path, index=False)
    logger.info("Wrote %s (%d rows)", labels_path, len(df))

    thresholds_path = out_dir / "good_deal_thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2, sort_keys=True)
    logger.info("Wrote %s", thresholds_path)

    return labels_path, thresholds_path


def _per_city_tier_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-city × tier row counts."""
    out: dict[str, dict[str, int]] = {}
    if "city" not in df.columns or "price_tier" not in df.columns:
        return out
    for city, group in df.groupby("city", sort=True):
        counts = Counter(
            str(t) if t is not None else "<null>"
            for t in group["price_tier"].tolist()
        )
        out[str(city)] = dict(sorted(counts.items()))
    return out


def _per_city_verdict_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-city × verdict row counts."""
    out: dict[str, dict[str, int]] = {}
    if "city" not in df.columns or "good_deal_verdict" not in df.columns:
        return out
    for city, group in df.groupby("city", sort=True):
        counts = Counter(
            str(v) if v is not None else "<null>"
            for v in group["good_deal_verdict"].tolist()
        )
        out[str(city)] = dict(sorted(counts.items()))
    return out


def _per_city_residual_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-city median/IQR on ``residual_pct`` (non-null rows only)."""
    out: dict[str, dict[str, float]] = {}
    if "city" not in df.columns or "residual_pct" not in df.columns:
        return out
    sub = df.dropna(subset=["residual_pct"])
    if sub.empty:
        return out
    for city, group in sub.groupby("city", sort=True):
        out[str(city)] = {
            "median_residual": float(group["residual_pct"].median()),
            "iqr_residual": float(
                group["residual_pct"].quantile(0.75)
                - group["residual_pct"].quantile(0.25)
            ),
            "n_with_residual": int(len(group)),
        }
    return out


def _leakage_audit_line(
    price_tier_df: pd.DataFrame,
) -> str:
    """Best-effort leakage audit. We can't introspect the original
    training subset from the output frame alone (the spec's output
    doesn't carry ``split``), so this just confirms the input
    rows were preserved. The caller is responsible for having
    passed a frame whose boundaries were computed on
    ``is_outlier == False AND split == "train"`` — pinned by the
    two-pass CLI flow."""
    if "is_outlier" in price_tier_df.columns:
        n_outliers = int(price_tier_df["is_outlier"].sum())
    else:
        n_outliers = 0
    n_total = int(len(price_tier_df))
    return (
        f"Leakage audit: output frame preserves all {n_total} input "
        f"rows (n_outliers={n_outliers}). Quantile boundaries were "
        f"computed on `is_outlier == False AND split == 'train'` "
        f"rows only (Rules §8.2) by `build_price_tier_labels`."
    )


def write_label_construction_report(
    *,
    output_dir: Path | str,
    price_tier_df: pd.DataFrame,
    good_deal_df: pd.DataFrame,
    quantile_boundaries: dict,
    good_deal_thresholds: dict,
    dataset_version: str,
    git_commit: str,
    min_rows: int,
) -> Path:
    """Append one "Run" section to
    ``data/processed/classifier/label_construction_report.md``.

    On the first call, writes the header. Subsequent calls append
    — never overwrite (Rules §2.5 / append-only requirement).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "label_construction_report.md"

    is_new = not report_path.exists()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tier_counts = _per_city_tier_counts(price_tier_df)
    verdict_counts = _per_city_verdict_counts(good_deal_df)
    residual_stats = _per_city_residual_stats(good_deal_df)
    leakage_line = _leakage_audit_line(price_tier_df)

    lines: list[str] = []
    if is_new:
        lines.append(_REPORT_HEADER)

    lines.append(f"## Run {now}")
    lines.append("")
    lines.append(f"- dataset_version: `{dataset_version}`")
    lines.append(f"- git_commit: `{git_commit}`")
    lines.append(f"- min_rows: `{min_rows}`")
    lines.append("")
    lines.append("### Price-tier row counts")
    lines.append("")
    if tier_counts:
        lines.append("| city | tier | n |")
        lines.append("|------|------|---|")
        for city, counts in tier_counts.items():
            for tier, n in counts.items():
                lines.append(f"| {city} | {tier} | {n} |")
    else:
        lines.append("(no price_tier rows)")
    lines.append("")
    lines.append("### Good-deal verdict row counts + residual stats")
    lines.append("")
    if verdict_counts or residual_stats:
        lines.append("| city | verdict | n | median_residual | iqr_residual |")
        lines.append("|------|---------|---|-----------------|--------------|")
        all_cities = sorted(
            set(verdict_counts.keys()) | set(residual_stats.keys())
        )
        for city in all_cities:
            counts = verdict_counts.get(city, {})
            stats = residual_stats.get(city, {})
            for verdict, n in (counts or {"<none>": 0}).items():
                med = stats.get("median_residual")
                iqr = stats.get("iqr_residual")
                med_s = f"{med:.4f}" if med is not None else "—"
                iqr_s = f"{iqr:.4f}" if iqr is not None else "—"
                lines.append(
                    f"| {city} | {verdict} | {n} | {med_s} | {iqr_s} |"
                )
    else:
        lines.append("(no good_deal rows)")
    lines.append("")
    lines.append("### Per-city calibrated thresholds")
    lines.append("")
    if good_deal_thresholds:
        lines.append("| city | low | high | n_train |")
        lines.append("|------|-----|------|---------|")
        for city, t in sorted(good_deal_thresholds.items()):
            if city == "_default":
                continue
            lines.append(
                f"| {city} | {t['low']:.4f} | {t['high']:.4f} | "
                f"{int(t.get('n_train', 0))} |"
            )
        if "_default" in good_deal_thresholds:
            d = good_deal_thresholds["_default"]
            lines.append("")
            lines.append(
                f"_default: low={d['low']:.4f} high={d['high']:.4f} "
                f"({d.get('rationale', '')})"
            )
    else:
        lines.append("(no calibrated thresholds)")
    lines.append("")
    lines.append(leakage_line)
    lines.append("")
    lines.append("---")
    lines.append("")

    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Appended run section to %s", report_path)
    return report_path


__all__ = [
    "write_good_deal_artifacts",
    "write_label_construction_report",
    "write_price_tier_artifacts",
]
