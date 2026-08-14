"""Build the Round 1 feature-selection report (Spec 12 Phase 5).

Writes ``data/processed/feature_selection_report.md`` with the six
required sections from the spec:

    1. Engineered column summary
    2. Numeric correlation matrix (Round 1 multicollinearity filter)
    3. Categorical cardinality table
    4. Locality-aggregate preview
    5. Top amenities selected
    6. Round 1 selection decisions

Rounds 2 + 3 (tree-based importance, permutation, SHAP) are produced
by the future training spec — a hand-off note is included.

No fitted model is required for this script.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.features.feature_frame import (  # noqa: E402
    ENGINEERED_COLUMNS,
    select_top_amenities,
)
from ml.features.locality_aggregator import LocalityAggregator  # noqa: E402
from ml.features.persistence import ENGINEERED_FEATURE_RECIPE  # noqa: E402

logger = logging.getLogger("build_feature_selection_report")


_NUMERIC_FOR_CORR: tuple[str, ...] = (
    "bedRoom",
    "bathroom",
    "built_up_area",
    "n_amenities",
    "n_features",
    "floor_ratio",
    "age_bucket_ord",
    "bath_bed_ratio",
    "area_per_bedroom",
    "locality_listing_count",
    "top_amenities_count",
    "price_per_sqft",
    "price_inr",
)

_CATEGORICAL_FOR_CARDINALITY: tuple[str, ...] = (
    "city",
    "property_type",
    "facing",
    "agePossession",
    "furnishing_type",
    "luxury_category",
    "floor_category",
    "balcony",
)


def _df_md(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                "" if pd.isna(v) else str(v)[:60] for v in row.tolist()
            )
            + " |"
        )
    return "\n".join(lines)


def _engineered_summary(df: pd.DataFrame) -> str:
    parts = ["## 1. Engineered column summary", ""]
    parts.append(
        "Per-column: definition (from `engineered_feature_recipe`), dtype, "
        "missingness %, sample statistics per city."
    )
    parts.append("")
    rows = []
    for col in ENGINEERED_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        rows.append(
            {
                "column": col,
                "definition": ENGINEERED_FEATURE_RECIPE.get(col, ""),
                "dtype": str(series.dtype),
                "missing_%": round(100.0 * series.isna().mean(), 2),
                "mean": round(float(series.mean(skipna=True) or 0), 3),
                "median": round(float(series.median(skipna=True) or 0), 3),
                "std": round(float(series.std(skipna=True) or 0), 3),
            }
        )
    parts.append(_df_md(pd.DataFrame(rows)))
    return "\n".join(parts)


def _numeric_correlation(df: pd.DataFrame) -> str:
    parts = ["## 2. Numeric correlation matrix", ""]
    cols = [c for c in _NUMERIC_FOR_CORR if c in df.columns]
    if not cols:
        parts.append("_(no numeric columns available)_")
        return "\n".join(parts)
    corr = df[cols].corr(numeric_only=True).round(3)
    parts.append("Pearson correlations on the numeric feature set. Round 1's "
                 "|corr| > 0.9 multicollinearity filter is the rule of thumb.")
    parts.append("")
    parts.append(_df_md(corr.reset_index().rename(columns={"index": "feature"})))
    # Flag pairs > 0.9.
    flagged = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) > 0.9:
                flagged.append((a, b, round(float(v), 3)))
    if flagged:
        parts.append("")
        parts.append("**Pairs flagged |corr| > 0.9:**")
        for a, b, v in flagged:
            parts.append(f"- `{a}` ↔ `{b}` = {v}")
    else:
        parts.append("")
        parts.append("_No pairs flagged._")
    return "\n".join(parts)


def _categorical_cardinality(df: pd.DataFrame) -> str:
    parts = ["## 3. Categorical cardinality table", ""]
    parts.append(
        "Number of unique values per categorical feature + top-5 most-"
        "frequent values (overall; per-city breakdown deferred to the "
        "training spec's Round 2 report)."
    )
    parts.append("")
    rows = []
    for col in _CATEGORICAL_FOR_CARDINALITY:
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts().head(5)
        top5 = ", ".join(f"{idx} ({cnt})" for idx, cnt in counts.items())
        rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_5": top5,
            }
        )
    parts.append(_df_md(pd.DataFrame(rows)))
    return "\n".join(parts)


def _locality_preview(df: pd.DataFrame, agg: LocalityAggregator) -> str:
    parts = ["## 4. Locality-aggregate preview", ""]
    fitted = agg.fitted_aggregates_
    if fitted is None or fitted.empty:
        parts.append("_(no locality aggregates fitted)_")
        return "\n".join(parts)
    # Top/bottom 5 by mean price_per_sqft per city.
    # We need the per-locality mean — reconstruct from the fitted sums.
    fitted = fitted.copy()
    fitted["mean_price_per_sqft"] = fitted["sum_price_per_sqft"] / fitted["count"].replace(
        0, pd.NA
    )
    rows_top = []
    rows_bot = []
    for city, sub in fitted.groupby("city"):
        nonzero = sub[sub["count"] > 0]
        if nonzero.empty:
            continue
        top = nonzero.nlargest(5, "mean_price_per_sqft")
        bot = nonzero.nsmallest(5, "mean_price_per_sqft")
        for _, r in top.iterrows():
            rows_top.append(
                {
                    "city": city,
                    "locality": r["locality"],
                    "mean_price_per_sqft": round(float(r["mean_price_per_sqft"]), 1),
                    "count": int(r["count"]),
                }
            )
        for _, r in bot.iterrows():
            rows_bot.append(
                {
                    "city": city,
                    "locality": r["locality"],
                    "mean_price_per_sqft": round(float(r["mean_price_per_sqft"]), 1),
                    "count": int(r["count"]),
                }
            )
    parts.append("**Top 5 highest-priced localities per city (by mean price_per_sqft):**")
    parts.append("")
    if rows_top:
        parts.append(_df_md(pd.DataFrame(rows_top)))
    else:
        parts.append("_(none)_")
    parts.append("")
    parts.append("**Top 5 lowest-priced localities per city:**")
    parts.append("")
    if rows_bot:
        parts.append(_df_md(pd.DataFrame(rows_bot)))
    else:
        parts.append("_(none)_")
    parts.append("")
    parts.append(
        "Smoothing prior weight = `SMOOTHING_PRIOR_WEIGHT` "
        f"({float(os.environ.get('SMOOTHING_PRIOR_WEIGHT', 20.0))}). "
        "At ~20 listings per locality the smoother leans ~50/50 toward "
        "the locality mean vs. the city mean."
    )
    return "\n".join(parts)


def _top_amenities(df: pd.DataFrame) -> str:
    parts = ["## 5. Top amenities selected", ""]
    top = select_top_amenities(df, k=10)
    if not top:
        parts.append("_(no amenities detected in the corpus)_")
        return "\n".join(parts)
    counts: dict[str, int] = {}
    flat = (
        df["amenities_list"].dropna().explode().dropna()
        if "amenities_list" in df.columns
        else pd.Series(dtype=object)
    )
    for label in top:
        counts[label] = int((flat == label).sum())
    rows = [
        {"amenity": k, "corpus_frequency": v}
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    parts.append("Top-10 amenities (K=10) the `has_<amenity>` flags are built for.")
    parts.append("")
    parts.append(_df_md(pd.DataFrame(rows)))
    return "\n".join(parts)


def _round1_decisions(df: pd.DataFrame) -> str:
    """Round 1 selection decisions — explicit keep/drop with rationale."""
    parts = ["## 6. Round 1 selection decisions", ""]
    parts.append("Decision rule (per TRD §9): keep if in the top-N by ≥2 of "
                 "the 4 model-based methods (RF importance, GB importance, "
                 "Permutation, SHAP), or has a non-zero Lasso coefficient. "
                 "This Round 1 report covers the correlation filter + base/"
                 "engineered column rationale only; Round 2 (tree importance)"
                 " + Round 3 (permutation + SHAP) require a fitted model and "
                 "are produced by `scripts/train_price_model.py` (future spec).")
    parts.append("")
    # Keep decisions.
    parts.append("### Kept")
    parts.append("")
    keep_rows = []
    for col in ENGINEERED_COLUMNS:
        keep_rows.append(
            {
                "column": col,
                "rationale": ENGINEERED_FEATURE_RECIPE.get(col, ""),
            }
        )
    parts.append(_df_md(pd.DataFrame(keep_rows)))
    parts.append("")
    # Drop decisions: only column-level candidates worth flagging in
    # Round 1 are built-up-area vs area_per_bedroom (multicollinearity
    # is expected), and any raw input columns the preprocessor
    # explicitly excludes (e.g. `sector`, `transact_type`).
    parts.append("### Excluded by design")
    parts.append("")
    parts.append(_df_md(
        pd.DataFrame(
            [
                {
                    "column": "sector",
                    "rationale": "Target-encoded via LocalityAggregator "
                                 "(smoothed) — one-hot would explode "
                                 "cardinality across 100+ localities × 4 "
                                 "cities (TRD §U-TRD-3).",
                },
                {
                    "column": "transact_type",
                    "rationale": "Routing key (Sale/Rent → separate "
                                 "pipelines), not a model feature "
                                 "(TRD §U-TRD-4, Rules §10.3).",
                },
                {
                    "column": "price_inr",
                    "rationale": "Training target only. Excluded from "
                                 "NUMERIC_FEATURES, ORDINAL_FEATURES, "
                                 "ONEHOT_FEATURES.",
                },
            ]
        )
    ))
    return "\n".join(parts)


def _handoff_note() -> str:
    return (
        "\n## Hand-off\n\n"
        "Round 2 (RF/GB tree-based importance + permutation) and "
        "Round 3 (SHAP mean |value| ranking) reports will be appended "
        "by `scripts/train_price_model.py` (future spec) using the "
        "saved fitted model.\n"
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the Round 1 feature-selection report."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_PROCESSED_DIR",
                str(_REPO_ROOT / "data" / "processed"),
            )
        )
        / "clean_listings.parquet",
        help="Path to the cleaned Parquet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "processed" / "feature_selection_report.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        logger.error("Parquet not found: %s", args.parquet)
        return 2

    df = pd.read_parquet(args.parquet)
    # Filter to non-outlier for the locality preview — same rule the
    # aggregator uses.
    clean = df[df["is_outlier"] == False]  # noqa: E712
    agg = LocalityAggregator().fit(clean)

    sections = [
        "# Feature Selection Report — Round 1",
        "",
        "_Source: `data/processed/clean_listings.parquet`. "
        f"Rows: {len(df)} total, {len(clean)} non-outlier._",
        "",
        _engineered_summary(clean),
        "",
        _numeric_correlation(clean),
        "",
        _categorical_cardinality(clean),
        "",
        _locality_preview(clean, agg),
        "",
        _top_amenities(clean),
        "",
        _round1_decisions(clean),
        _handoff_note(),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
