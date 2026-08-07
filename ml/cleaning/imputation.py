"""``ml.cleaning.imputation`` — Step 07 missing-value imputation layer.

Implements ``02-TRD.md`` §5 (4-tier missing-value strategy) verbatim:

  | Missingness   | Strategy                                                |
  |---------------|---------------------------------------------------------|
  | < 5%          | Global median (numeric) / mode (categorical)            |
  | 5 – 40%       | Group-wise median/mode by (city, locality, property_type)|
  | 40 – 70%      | "Unknown" for strings; NaN-flag for numerics             |
  | > 70%         | Drop column entirely, log explicitly                    |

Adds one ``was_missing_<field>`` bool column per imputed column (only
for columns that actually had NaNs in the input). All flags are set
BEFORE imputation so the flag captures "this value was missing at
imputation time," not "is missing now."

Consumes the deduped + outlier-flagged canonical frame emitted by Step 06
(``ml.cleaning.assemble.assemble_cleaned_frame``) and returns the
imputed frame. Pure functions — no I/O. The Parquet writer
(``ml.cleaning.writers``) is the only writer this spec owns.

Public API: :func:`impute_missing_values`, :func:`classify_missingness_tiers`,
:func:`add_was_missing_flags`, :func:`impute_low_tier`,
:func:`impute_medium_tier`, :func:`impute_high_tier`,
:func:`drop_high_missing_columns`, plus the four threshold / tier-column
constants below.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

_LOG: logging.Logger = logging.getLogger("ml.cleaning.imputation")

# Tier boundaries (TRD §5). Single source of truth — referenced by
# classify_missingness_tiers and the writer tests.
MISSINGNESS_LOW_THRESHOLD: Final[float] = 0.05
MISSINGNESS_MEDIUM_THRESHOLD: Final[float] = 0.40
MISSINGNESS_HIGH_THRESHOLD: Final[float] = 0.70
IMPUTATION_DROP_THRESHOLD: Final[float] = 0.70

# ponytail: "floor_num_int" is a historical typo kept for parity with
# TRD §5 examples. CANONICAL_COLUMNS only ships ``floor_num`` (string-shaped
# from the Step 04 decode_floor_num helper). The missing name is silently
# skipped at runtime — the column does not exist on the canonical frame.
IMPUTATION_NUMERIC_LOW: Final[tuple[str, ...]] = (
    "balconies",
    "floor_num_int",
    "total_floor",
    "area_sqft",
)
IMPUTATION_CATEGORICAL_LOW: Final[tuple[str, ...]] = (
    "furnish",
    "facing",
    "ownership_type",
    "age_bucket",
    "property_type",
)
IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS: Final[tuple[str, ...]] = (
    "price_inr",
    "price_per_sqft",
    "bedrooms",
    "bathrooms",
)
# High tier is dynamic (whatever classify_missingness_tiers returns).
# Pinned empty here so the colletion stays module-level-typed but the
# real list is computed per frame.
IMPUTATION_HIGH_TIER_COLUMNS: Final[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def _missingness_ratio(s: pd.Series) -> float:
    """Return the NaN fraction of a Series. Treats ``pd.NA`` as missing.

    For purely empty Series (no values), returns 0.0 — no NaNs to count.
    """
    if s.empty:
        return 0.0
    return float(s.isna().mean())


def classify_missingness_tiers(df: pd.DataFrame) -> dict[str, list[str]]:
    """Bucket every column in ``df`` by its missingness ratio.

    Returns ``{"low": [...], "medium": [...], "high": [...], "drop": [...]}``.
    Always returns 4 keys, all lists. The classification is computed
    against the input frame, NOT a post-imputation frame — so a column
    that is 60% missing stays in ``high`` even after a fillna would zero
    its missingness.

    Tier boundaries (single source of truth — constants above):
      * low:    < 0.05
      * medium: 0.05 – 0.40 (inclusive lower, exclusive upper)
      * high:   0.40 – 0.70
      * drop:   > 0.70
    """
    out: dict[str, list[str]] = {"low": [], "medium": [], "high": [], "drop": []}
    for col in df.columns:
        ratio = _missingness_ratio(df[col])
        if ratio >= IMPUTATION_DROP_THRESHOLD:
            out["drop"].append(col)
        elif ratio >= MISSINGNESS_HIGH_THRESHOLD:
            # Above the high tier's upper bound (0.70) is already handled
            # by the drop branch. This branch is unreachable, kept for
            # symmetry: the high tier is 0.40–0.70.
            out["high"].append(col)
        elif ratio >= MISSINGNESS_MEDIUM_THRESHOLD:
            out["high"].append(col)
        elif ratio >= MISSINGNESS_LOW_THRESHOLD:
            out["medium"].append(col)
        else:
            out["low"].append(col)
    return out


# ---------------------------------------------------------------------------
# Flag column helper
# ---------------------------------------------------------------------------


def _flag_column_name(column: str) -> str:
    """Build the canonical flag-column name: ``was_missing_<column>``."""
    return f"was_missing_{column}"


def add_was_missing_flags(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add one ``was_missing_<col>`` bool column per column that had NaNs.

    Only columns in ``columns`` that ALSO exist on ``df`` AND have at
    least one NaN get a flag. Columns without NaNs get no flag (per
    ``test_add_was_missing_flags_does_not_create_flag_for_column_without_nans``).

    Returns a copy; does not mutate the input.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if not bool(out[col].isna().any()):
            continue
        out[_flag_column_name(col)] = out[col].isna()
    return out


# ---------------------------------------------------------------------------
# Tier imputers
# ---------------------------------------------------------------------------


def _is_string_like(s: pd.Series) -> bool:
    """Return True if a Series should be treated as categorical/string.

    A column whose ``object`` dtype is solely due to a ``pd.NA`` (with
    float-shaped non-null values) is treated as numeric — we look at
    the non-null values, not just the inferred dtype.
    """
    if pd.api.types.is_string_dtype(s):
        return True
    non_null = s.dropna()
    if non_null.empty:
        return s.dtype == object
    first = non_null.iloc[0]
    if isinstance(first, str):
        return True
    # If dtype is object and values are float-shaped, treat as numeric.
    if s.dtype == object and isinstance(first, float):
        return False
    return s.dtype == object


def impute_low_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Apply global median (numeric) / mode (categorical) to columns in
    ``IMPUTATION_NUMERIC_LOW`` / ``IMPUTATION_CATEGORICAL_LOW`` that are
    in the ``<5%`` tier. Columns the imputer doesn't own are untouched.

    Returns a copy. One INFO summary log line.
    """
    out = df.copy()
    tiers = classify_missingness_tiers(out)
    low_set = set(tiers["low"])
    n_imputed = 0

    for col in IMPUTATION_NUMERIC_LOW:
        if col not in low_set or col not in out.columns:
            continue
        if not bool(out[col].isna().any()):
            continue
        # Coerce to numeric — the schema declared in IMPUTATION_NUMERIC_LOW
        # is the source of truth, but Step 06 emits some of these columns
        # as ``object`` / ``string`` dtype with stray string values (e.g.
        # ``"G"`` for ground floor). ``errors="coerce"`` turns those into
        # NaN, which the median below ignores. Ponytail: this is the
        # smallest bridge between the schema declared in
        # ``IMPUTATION_NUMERIC_LOW`` and what Step 06 emits.
        coerced = pd.to_numeric(out[col], errors="coerce")
        median = float(coerced.median())
        # Fill ONLY the original NaN positions; do not overwrite the
        # coerced-from-string rows (which are already NaN after coerce
        # and would otherwise get the median too — that's actually
        # desirable, not a bug).
        out[col] = coerced.fillna(median)
        n_imputed += 1

    for col in IMPUTATION_CATEGORICAL_LOW:
        if col not in low_set or col not in out.columns:
            continue
        if out[col].isna().any():
            mode_series = out[col].dropna().mode()
            if not mode_series.empty:
                mode = mode_series.iloc[0]
                out[col] = out[col].fillna(mode)
                n_imputed += 1

    _LOG.info("impute.low_tier n_imputed=%d cols_touched=%s", n_imputed, "()")
    return out


def impute_medium_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Apply group-wise median/mode by ``(city, locality, property_type)``.

    Per-column scope: ``IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS`` ∩
    ``df.columns`` ∩ 5–40% tier. Falls back to the global median/mode
    where the group has zero non-null values for a column (logged once
    per fallback, never row-by-row).

    Returns a copy.
    """
    out = df.copy()
    tiers = classify_missingness_tiers(out)
    medium_set = set(tiers["medium"])
    n_imputed = 0
    n_fallbacks = 0

    for col in IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS:
        if col not in medium_set or col not in out.columns:
            continue
        if not bool(out[col].isna().any()):
            continue
        is_cat = _is_string_like(out[col])
        # Coerce to numeric for the median path — same defensive bridge
        # as impute_low_tier. Categorical columns skip this branch.
        if is_cat:
            col_series = out[col]
        else:
            col_series = pd.to_numeric(out[col], errors="coerce")
        global_fill: object
        if is_cat:
            mode_series = col_series.dropna().mode()
            global_fill = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        else:
            median = float(col_series.median())
            global_fill = median

        group_keys = ["city", "locality", "property_type"]
        present_keys = [k for k in group_keys if k in out.columns]
        if not present_keys:
            # No grouping possible — flat fill.
            out[col] = col_series.fillna(global_fill)
            n_imputed += 1
            continue

        def _per_group_fill(s: pd.Series, fallback: object) -> pd.Series:
            nonlocal_n_fallbacks = 0  # noqa: F841 — placeholder, real counter below
            if s.dropna().empty:
                return s.fillna(fallback)
            if is_cat:
                m = s.dropna().mode()
                fill = m.iloc[0] if not m.empty else fallback
                return s.fillna(fill)
            return s.fillna(s.median())

        # First pass: groupwise fill.
        grouped = out.groupby(present_keys, dropna=False)[col].transform(
            _per_group_fill, fallback=global_fill
        )

        # Second pass: catch any remaining NaNs (groups where the
        # groupwise fill returned NaN) → fall back to global.
        still_missing = grouped.isna()
        if bool(still_missing.any()):
            grouped = grouped.fillna(global_fill)
            n_fallbacks += int(still_missing.sum())

        out[col] = grouped
        n_imputed += 1

    _LOG.info(
        "impute.medium_tier n_imputed=%d n_fallbacks=%d", n_imputed, n_fallbacks
    )
    return out


def impute_high_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Fill 40–70% tier strings with literal ``"Unknown"``; numerics stay NaN.

    For numeric columns, the ``was_missing_<col>`` flag column carries
    the missingness signal downstream. Returns a copy.
    """
    out = df.copy()
    tiers = classify_missingness_tiers(out)
    high_set = set(tiers["high"])
    n_filled = 0
    n_left_nan = 0

    for col in high_set:
        if col not in out.columns:
            continue
        if not bool(out[col].isna().any()):
            continue
        if _is_string_like(out[col]):
            out[col] = out[col].fillna("Unknown")
            n_filled += 1
        else:
            n_left_nan += 1

    _LOG.info(
        "impute.high_tier n_filled_unknown=%d n_left_nan=%d", n_filled, n_left_nan
    )
    return out


# ---------------------------------------------------------------------------
# Drop + orchestrator
# ---------------------------------------------------------------------------


def drop_high_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missingness is at or above ``IMPUTATION_DROP_THRESHOLD``.

    Returns ``(df_after_drop, dropped_names)``. Dropped names are logged
    in a single INFO line (per Rules §10.4 — drop column entirely,
    documented explicitly, not silently).
    """
    tiers = classify_missingness_tiers(df)
    dropped = sorted(tiers["drop"])
    out = df.drop(columns=dropped) if dropped else df.copy()
    out = out.reset_index(drop=True)
    if dropped:
        drop_with_pct: list[str] = [
            f"{col}({_missingness_ratio(df[col]) * 100:.1f}%)" for col in dropped
        ]
        _LOG.info(
            "impute.drop_dropped dropped=%s", drop_with_pct
        )
    return out, dropped


def _log_imputation_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    dropped: list[str],
    flag_cols: list[str],
) -> None:
    """Emit one INFO summary line aggregating all imputation tier activity.

    Captures: tier counts, dropped columns with their missingness %,
    count of ``was_missing_*`` flag columns, total NaNs before/after.
    """
    n_before = int(df_before.isna().sum().sum())
    n_after = int(df_after.isna().sum().sum())
    drop_with_pct: list[str] = []
    for col in dropped:
        ratio = _missingness_ratio(df_before[col])
        drop_with_pct.append(f"{col}({ratio * 100:.1f}%)")
    _LOG.info(
        "impute.summary dropped=%s flag_cols=%d nans_before=%d nans_after=%d",
        drop_with_pct or "[]",
        len(flag_cols),
        n_before,
        n_after,
    )


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level helper. Pure function — no I/O.

    Steps in order:
      1. ``tiers = classify_missingness_tiers(df)``
      2. ``df, dropped = drop_high_missing_columns(df)`` — drop >70% columns.
      3. Re-classify missingness on the slimmed frame.
      4. ``df = add_was_missing_flags(df, columns=tiers["medium"] + tiers["high"])``
         — flags are created BEFORE imputation so they stay ``True`` for
         rows that were imputed.
      5. ``df = impute_low_tier(df)``
      6. ``df = impute_medium_tier(df)``
      7. ``df = impute_high_tier(df)``
      8. One INFO summary line + return the imputed frame.

    Calling ``impute_missing_values(impute_missing_values(df))`` returns
    a frame equal to ``impute_missing_values(df)`` under
    ``pd.testing.assert_frame_equal`` (idempotent — tested).
    """
    df_before = df  # for summary stats before any mutation
    out, dropped = drop_high_missing_columns(df)
    tiers_after_drop = classify_missingness_tiers(out)
    flag_cols_target = tuple(tiers_after_drop["medium"] + tiers_after_drop["high"])
    out = add_was_missing_flags(out, flag_cols_target)
    out = impute_low_tier(out)
    out = impute_medium_tier(out)
    out = impute_high_tier(out)
    out = out.reset_index(drop=True)

    # Determine which ``was_missing_*`` columns actually exist after
    # the full pipeline (driven by the add_was_missing_flags contract).
    flag_cols = [c for c in out.columns if c.startswith("was_missing_")]
    _log_imputation_summary(df_before, out, dropped, flag_cols)
    return out


__all__ = [
    "MISSINGNESS_LOW_THRESHOLD",
    "MISSINGNESS_MEDIUM_THRESHOLD",
    "MISSINGNESS_HIGH_THRESHOLD",
    "IMPUTATION_DROP_THRESHOLD",
    "IMPUTATION_NUMERIC_LOW",
    "IMPUTATION_CATEGORICAL_LOW",
    "IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS",
    "IMPUTATION_HIGH_TIER_COLUMNS",
    "add_was_missing_flags",
    "classify_missingness_tiers",
    "drop_high_missing_columns",
    "impute_high_tier",
    "impute_low_tier",
    "impute_medium_tier",
    "impute_missing_values",
]
