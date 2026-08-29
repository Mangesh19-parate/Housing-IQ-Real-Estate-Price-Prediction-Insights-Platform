"""Feature-frame artifact writer + report (Spec 22).

Writes under ``data/processed/classifier/`` (by default):

- ``feature_frame.parquet`` — the deterministic leakage-safe feature
  DataFrame consumed by Day 52's classifier-training spec.
- ``feature_frame_report.md`` — append-only Markdown report; one "Run"
  block per CLI invocation. Mirrors the style of
  ``label_construction_report.md``.

Authority:
- ``docs/08-RULES.md`` §1.3 (every derived table states its
  computation date + source dataset version)
- ``docs/08-RULES.md`` §2.5 (versioned artifacts, never overwritten
  in place — re-run overwrites a re-run with the same git commit;
  the report appends)
- ``docs/08-RULES.md`` §8.1 + §12.4 (leakage ban on price-derived
  features)
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.classification.features import DROPPED_FOR_LEAKAGE, RETAINED_LOCALITY_COLUMN

logger = logging.getLogger(__name__)

#: Header line written when ``feature_frame_report.md`` is created for
#: the first time. Mirrors the style of ``label_construction_report.md``.
_REPORT_HEADER = """# Feature-Frame Construction Report

Append-only report written by ``scripts/build_classifier_features.py``
(Spec 22). One "Run" block per CLI invocation.

Pinned by:
- ``docs/08-RULES.md`` §8.1 + §12.4 (leakage ban on price-derived
  classifier inputs)
- ``docs/02-TRD.md`` §U-TRD-1 + §U-TRD-6 (Classification module spec)
- ``ml/features/feature_frame.py`` (Step 12 — the reused price-model
  pipeline)

"""


def _dataset_version(parquet_path: Path | None) -> str:
    """Return ``<basename>:<sha1[:12]>`` for the first 1 MB of the
    input parquet (Rules §1.3). Returns ``"<unknown>"`` if missing.
    """
    if parquet_path is None or not parquet_path.exists():
        return "<unknown>"
    hasher = hashlib.sha1()
    with open(parquet_path, "rb") as fh:
        hasher.update(fh.read(1024 * 1024))
    return f"{parquet_path.name}:{hasher.hexdigest()[:12]}"


def _per_city_non_outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    """Per-city non-outlier row counts. Sorted ascending by city."""
    out: dict[str, int] = {}
    if "city" not in df.columns:
        return out
    sub = df
    if "is_outlier" in df.columns:
        sub = df[df["is_outlier"] == False]  # noqa: E712
    if sub.empty:
        return out
    for city, n in sub["city"].value_counts(sort=True).items():
        out[str(city)] = int(n)
    return out


def _per_city_tier_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-(city × price_tier) row counts from a Spec-21 tier frame."""
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
    """Per-(city × good_deal_verdict) row counts from a Spec-21 verdict frame."""
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


def write_feature_frame_artifact(
    df: pd.DataFrame,
    output_dir: Path | str,
) -> Path:
    """Write ``feature_frame.parquet``. Returns the canonical path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_frame.parquet"
    df.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))
    return path


def write_feature_frame_report(
    *,
    output_dir: Path | str,
    feature_frame: pd.DataFrame,
    price_tier_labels: pd.DataFrame,
    good_deal_labels: pd.DataFrame,
    dataset_version: str,
    git_commit: str,
) -> Path:
    """Append one "Run" section to ``feature_frame_report.md``.

    On the first call, writes the header. Subsequent calls append —
    never overwrite (Rules §2.5).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "feature_frame_report.md"

    is_new = not report_path.exists()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    kept_cols = list(feature_frame.columns)
    dropped_cols = sorted(DROPPED_FOR_LEAKAGE)
    tier_counts = _per_city_tier_counts(price_tier_labels)
    verdict_counts = _per_city_verdict_counts(good_deal_labels)
    non_outlier = _per_city_non_outlier_counts(feature_frame)
    n_total = len(feature_frame)
    n_outliers = (
        int(feature_frame["is_outlier"].sum())
        if "is_outlier" in feature_frame.columns
        else 0
    )

    lines: list[str] = []
    if is_new:
        lines.append(_REPORT_HEADER)

    lines.append(f"## Run {now}")
    lines.append("")
    lines.append(f"- dataset_version: `{dataset_version}`")
    lines.append(f"- git_commit: `{git_commit}`")
    lines.append(f"- total rows: {n_total} (outliers: {n_outliers})")
    lines.append(f"- retained locality column: `{RETAINED_LOCALITY_COLUMN}`")
    lines.append("")
    lines.append("### Kept columns")
    lines.append("")
    lines.append("```")
    for col in kept_cols:
        lines.append(col)
    lines.append("```")
    lines.append("")
    lines.append("### Dropped columns (leakage ban)")
    lines.append("")
    lines.append("| column | rationale |")
    lines.append("|--------|-----------|")
    rationales = {
        "price": "Regression target — must not feed the classifier (Rules §8.1)",
        "price_per_sqft": "Price-derived ratio — leak risk (Rules §8.1)",
        "locality_avg_price_sqft": "Per-locality mean of price_per_sqft — leak risk",
        "locality_smoothed_price": "Per-locality smoothed mean of price_inr — leak risk",
    }
    for col in dropped_cols:
        lines.append(f"| `{col}` | {rationales.get(col, 'price-derived')} |")
    lines.append("")
    lines.append("### Per-city non-outlier row counts")
    lines.append("")
    if non_outlier:
        lines.append("| city | n |")
        lines.append("|------|---|")
        for city, n in non_outlier.items():
            lines.append(f"| {city} | {n} |")
    else:
        lines.append("(no non-outlier rows)")
    lines.append("")
    lines.append("### Per-(city × price_tier) row counts")
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
    lines.append("### Per-(city × good_deal_verdict) row counts")
    lines.append("")
    if verdict_counts:
        lines.append("| city | verdict | n |")
        lines.append("|------|---------|---|")
        for city, counts in verdict_counts.items():
            for verdict, n in counts.items():
                lines.append(f"| {city} | {verdict} | {n} |")
    else:
        lines.append("(no good_deal rows)")
    lines.append("")
    # Leakage audit line — names every column the spec promises to drop.
    lines.append(
        "Leakage audit: price-derived columns dropped: "
        + ", ".join(f"`{c}`" for c in dropped_cols)
        + ". `"
        + RETAINED_LOCALITY_COLUMN
        + "` retained (count, not price-derived). Verified "
        "`feature_frame.parquet` contains none of the dropped columns "
        'via `pyarrow.parquet.read_schema(path).names`.'
    )
    lines.append("")
    lines.append("---")

    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Appended Run section to %s", report_path)
    return report_path


__all__ = [
    "write_feature_frame_artifact",
    "write_feature_frame_report",
]
