"""``ml.cleaning.outliers`` — Step 06 outlier flagging layer.

Implements ``02-TRD.md`` §6 verbatim:

  1. Per-city [1st, 99th] percentile bounds on numeric columns (TRD §6.1).
  2. Per-city Tukey fence (Q1 − 1.5×IQR, Q3 + 1.5×IQR) on numeric columns
     (TRD §6.2).
  3. Domain-rule caps on ``bedRoom`` / ``bathroom`` (>15), with villa /
     farmhouse / independent-house exemptions (TRD §6.3).

Adds two columns to the canonical frame:
  * ``is_outlier`` (bool) — or-combined flag across all three methods.
  * ``outlier_reasons`` (object, list[str]) — fixed-set reason codes drawn
    from ``{"percentile_<col>", "iqr_<col>", "domain_<col>"}``.

Flagged rows are RETAINED, not deleted — Rules §1.4 is binding. Training-time
exclusion is the modeling step's responsibility (Week 3–4), not this one.
The ``log1p(price)`` target transform from TRD §6.4 is also out of scope —
Day 13 of the implementation plan owns that.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

_LOG: logging.Logger = logging.getLogger("ml.cleaning.outliers")

# The three numeric columns TRD §6 names for IQR + percentile capping.
OUTLIER_NUMERIC_COLUMNS: Final[tuple[str, ...]] = ("price_inr", "area_sqft", "price_per_sqft")

# TRD §6.1 bounds.
PERCENTILE_LOWER: Final[float] = 0.01
PERCENTILE_UPPER: Final[float] = 0.99

# TRD §6.2 Tukey fence.
IQR_MULTIPLIER: Final[float] = 1.5

# TRD §6.3 domain-rule caps. ``property_type`` is the canonical column from
# Step 05's CANONICAL_COLUMNS (no ``_label`` suffix).
OUTLIER_DOMAIN_RULES: Final[dict[str, dict[str, object]]] = {
    "bedRoom": {"max": 15, "note": "unless property_type is villa/farmhouse/independent house"},
    "bathroom": {"max": 15, "note": "unless property_type is villa/farmhouse/independent house"},
}

OUTLIER_PROPERTY_TYPE_EXEMPTIONS: Final[frozenset[str]] = frozenset(
    {"villa", "farmhouse", "independent house"}
)

# Single source of truth for the reason-list column name.
OUTLIER_REASON_COLUMN: Final[str] = "outlier_reasons"

# All possible reason strings — kept tight so outlier_reasons stays JSON-safe.
_REASON_PREFIX_NUMERIC: Final[tuple[str, str]] = ("percentile_", "iqr_")
_REASON_PREFIX_DOMAIN: Final[str] = "domain_"


def flag_percentile_outliers(df: pd.DataFrame, column: str) -> pd.Series:
    """Per-city [1st, 99th]-percentile outlier flag for ``column``.

    Returns a bool Series aligned to ``df.index``. Rows with NaN in the
    column are not flagged (NaN comparison is undefined). Bounds are
    per-city so a Mumbai luxury flat and a Kolkata budget flat are
    evaluated against their own distribution.
    """
    if column not in df.columns or "city" not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")

    def _per_city_flag(s: pd.Series) -> pd.Series:
        if s.dropna().empty:
            return pd.Series(False, index=s.index, dtype="bool")
        lo = s.quantile(PERCENTILE_LOWER)
        hi = s.quantile(PERCENTILE_UPPER)
        return (s < lo) | (s > hi)

    return df.groupby("city")[column].transform(_per_city_flag).fillna(False).astype("bool")


def flag_iqr_outliers(df: pd.DataFrame, column: str) -> pd.Series:
    """Per-city Tukey-fence outlier flag for ``column`` (Q1 − 1.5×IQR / Q3 + 1.5×IQR)."""
    if column not in df.columns or "city" not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")

    def _per_city_flag(s: pd.Series) -> pd.Series:
        non_null = s.dropna()
        if non_null.empty:
            return pd.Series(False, index=s.index, dtype="bool")
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - IQR_MULTIPLIER * iqr
        hi = q3 + IQR_MULTIPLIER * iqr
        return (s < lo) | (s > hi)

    return df.groupby("city")[column].transform(_per_city_flag).fillna(False).astype("bool")


def flag_domain_rule_outliers(df: pd.DataFrame) -> pd.Series:
    """Apply ``OUTLIER_DOMAIN_RULES`` per column. ``bedRoom`` / ``bathroom``
    cap is overridden for rows whose ``property_type`` is in
    ``OUTLIER_PROPERTY_TYPE_EXEMPTIONS``.
    """
    if df.empty:
        return pd.Series(False, index=df.index, dtype="bool")

    mask = pd.Series(False, index=df.index, dtype="bool")
    for column, rule in OUTLIER_DOMAIN_RULES.items():
        if column not in df.columns:
            continue
        cap = rule.get("max")
        if cap is None:
            continue
        # Real-data canonical frame can emit bedRoom/bathroom as string dtype
        # (e.g. "3.0" rather than 3). Coerce to numeric; non-parseable → NaN
        # which compares False against the cap.
        col_values = pd.to_numeric(df[column], errors="coerce")
        col_mask = col_values > cap
        if column in ("bedRoom", "bathroom") and "property_type" in df.columns:
            exempt = df["property_type"].isin(OUTLIER_PROPERTY_TYPE_EXEMPTIONS)
            col_mask = col_mask & ~exempt
        mask = mask | col_mask.fillna(False).astype("bool")
    return mask


def _empty_reasons(length: int) -> pd.Series:
    """Empty-list Series of length ``length`` with object dtype."""
    return pd.Series([[] for _ in range(length)], index=range(length), dtype="object")


def _log_outlier_summary(df: pd.DataFrame) -> None:
    """Per-city outlier count log line: ``{city: {n_rows, n_outliers, pct}}``."""
    if df.empty or "city" not in df.columns:
        _LOG.info("outliers.summary empty")
        return
    parts: list[str] = []
    for city, group in df.groupby("city"):
        n_rows = len(group)
        n_out = int(group["is_outlier"].sum()) if "is_outlier" in group.columns else 0
        pct = round((n_out / n_rows * 100) if n_rows else 0.0, 2)
        parts.append(f"{city}:{{n_rows={n_rows},n_outliers={n_out},pct={pct}}}")
    _LOG.info("outliers.summary per_city=%s", " ".join(parts))


def flag_all_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level helper. Returns the input frame with two columns added/overwritten:
    ``is_outlier`` (bool) and ``outlier_reasons`` (object, list[str]).
    """
    out = df.copy()
    n = len(out)

    reasons: list[list[str]] = [[] for _ in range(n)]

    # Numeric: percentile + IQR per OUTLIER_NUMERIC_COLUMNS.
    for column in OUTLIER_NUMERIC_COLUMNS:
        if column not in out.columns:
            continue
        for prefix, flagger in (
            (_REASON_PREFIX_NUMERIC[0], flag_percentile_outliers),
            (_REASON_PREFIX_NUMERIC[1], flag_iqr_outliers),
        ):
            mask = flagger(out, column)
            for i, flagged in enumerate(mask.values):
                if flagged:
                    reasons[i].append(f"{prefix}{column}")

    # Domain rules.
    # We need per-column diagnostics to stamp the right reason code per row,
    # so we re-compute per-column masks below (flag_domain_rule_outliers only
    # returns an or-combined mask without per-column reasons).
    domain_reasons_per_row: list[list[str]] = [[] for _ in range(n)]
    for column, rule in OUTLIER_DOMAIN_RULES.items():
        if column not in out.columns:
            continue
        cap = rule.get("max")
        if cap is None:
            continue
        col_values = pd.to_numeric(out[column], errors="coerce")
        col_mask = col_values > cap
        if column in ("bedRoom", "bathroom") and "property_type" in out.columns:
            exempt = out["property_type"].isin(OUTLIER_PROPERTY_TYPE_EXEMPTIONS)
            col_mask = col_mask & ~exempt
        col_mask = col_mask.fillna(False)
        for i, flagged in enumerate(col_mask.values):
            if bool(flagged):
                domain_reasons_per_row[i].append(f"{_REASON_PREFIX_DOMAIN}{column}")
    for i, extra in enumerate(domain_reasons_per_row):
        reasons[i].extend(extra)

    out["is_outlier"] = pd.array([bool(r) for r in reasons], dtype="bool")
    out[OUTLIER_REASON_COLUMN] = pd.Series(reasons, dtype="object")
    _log_outlier_summary(out)
    return out
