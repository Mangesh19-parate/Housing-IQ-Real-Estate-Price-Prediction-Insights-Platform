"""``ml.cleaning.dedup`` — Step 06 deduplication layer.

Implements ``02-TRD.md`` §4.9 (drop duplicate ``PROP_ID`` rows). All 4 city
frames emitted by Step 05 (``ml.cleaning.canonical_mapping``) carry a
canonical ``listing_id`` column populated from raw ``PROP_ID`` /
``PROPHEADID`` / ``PROPERTY_ID`` per city. This module is the single place
that drops duplicates.

Conflict tiebreaker policy (single source of truth — ``CONFLICT_TIEBREAKER_ORDER``):
  1. Most non-null canonical fields wins.
  2. Most-recent ``register_date`` wins. Step 05 leaves ``register_date`` as
     raw free-text (e.g. ``"29th Sep, 2023"``); lex sort is deterministic and
     acceptable for tiebreaking.
  3. First-seen input row wins (stable row_order tiebreaker).

Public API: :func:`deduplicate_listings`, :func:`compute_nonnull_field_count`,
:data:`DEDUP_KEY_COLUMN`, :data:`CONFLICT_TIEBREAKER_ORDER`.

Outliers are flagged, NOT deleted — see ``ml.cleaning.outliers``.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS

_LOG: logging.Logger = logging.getLogger("ml.cleaning.dedup")

# Single source of truth for which column drives dedup.
DEDUP_KEY_COLUMN: Final[str] = "listing_id"

# Order matters: first criterion is the strongest. See module docstring.
CONFLICT_TIEBREAKER_ORDER: Final[tuple[str, ...]] = (
    "nonnull_fields_count",  # computed per-row
    "register_date",         # raw text, lex sort is deterministic
    "row_order",             # input row position (stable)
)


def compute_nonnull_field_count(df: pd.DataFrame) -> pd.Series:
    """Vectorized per-row count of non-null values across ``CANONICAL_COLUMNS``.

    Returns an ``int`` Series aligned to ``df.index``. Columns not present in
    ``df`` are treated as all-null (so adding new canonical columns doesn't
    crash older frames).
    """
    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index, dtype="int64")
    return df[cols].notna().sum(axis=1).astype("int64")


def _log_dedup_drop(reason: str, count: int, total: int) -> None:
    """Emit a structured INFO log line for one dedup drop bucket.

    Mirrors the style of ``ml.cleaning.parsing._log_unparseable`` — single
    log line per reason, aggregated (per-row drops are not logged individually).
    """
    if total <= 0:
        pct = 0.0
    else:
        pct = round((count / total) * 100, 2)
    _LOG.info(
        "dedup.drop reason=%s count=%d total=%d pct=%.2f",
        reason, count, total, pct,
    )


def _coerce_listing_id_key(df: pd.DataFrame) -> pd.Series:
    """Return a Series with stripped, string-cast listing_id values.

    Whitespace-only strings collapse to empty string (treated as null by
    ``deduplicate_listings``). NaN passes through.
    """
    return df[DEDUP_KEY_COLUMN].astype("string").str.strip()


def deduplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop null/empty ``listing_id`` rows and dedup duplicates by tiebreaker.

    Steps:
      1. Drop rows where ``listing_id`` is null/empty/NaN — log count.
      2. Strip whitespace on ``listing_id``, cast to str.
      3. Sort by ``CONFLICT_TIEBREAKER_ORDER`` (descending nnull,
         descending register_date, ascending row_order) within each
         ``listing_id`` group.
      4. ``groupby("listing_id").first()`` — keep the winning row.
      5. Log summary: input rows, dropped (no listing_id),
         dropped (duplicate listing_id), output rows.

    Pure function. Calling it twice on the same input yields equal outputs.
    No filesystem side effects.
    """
    if df.empty:
        return df.reset_index(drop=True)

    rows_in = len(df)
    key = _coerce_listing_id_key(df)
    keep_mask = key.notna() & (key.astype(str) != "")
    dropped_no_key = int((~keep_mask).sum())
    if dropped_no_key:
        _log_dedup_drop("no_listing_id", dropped_no_key, rows_in)

    work = df.loc[keep_mask].copy()
    if work.empty:
        return work.reset_index(drop=True)

    # Strip the canonical column too so survivors don't carry leading/trailing
    # whitespace. groupby("listing_id") keys off the helper, but the returned
    # frame's listing_id column should be clean.
    work[DEDUP_KEY_COLUMN] = key.loc[keep_mask].values
    work["_dedup_listing_id"] = key.loc[keep_mask].values
    work["_nonnull"] = compute_nonnull_field_count(work).values
    work["_row_order"] = np.arange(len(work), dtype="int64")

    # Sort: descending nnull, descending register_date (NaN last), ascending row_order.
    sort_cols = ["_nonnull", "register_date", "_row_order"]
    ascending = [False, False, True]
    na_position = "last"  # NaT/NaN tiebreaker values sink to bottom
    sorted_work = work.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position=na_position,
        kind="mergesort",  # stable
    )

    deduped = sorted_work.groupby("_dedup_listing_id", as_index=False).first()
    dropped_duplicate = rows_in - dropped_no_key - len(deduped)
    if dropped_duplicate > 0:
        _log_dedup_drop("duplicate_listing_id", dropped_duplicate, rows_in)

    # Drop the helper columns and restore canonical column ordering.
    deduped = deduped.drop(columns=["_dedup_listing_id", "_nonnull", "_row_order"])
    canonical_cols = [c for c in CANONICAL_COLUMNS if c in deduped.columns]
    other_cols = [c for c in deduped.columns if c not in canonical_cols]
    deduped = deduped[canonical_cols + other_cols].reset_index(drop=True)

    _LOG.info(
        "dedup.summary rows_in=%d dropped_no_listing_id=%d dropped_duplicate=%d rows_out=%d",
        rows_in, dropped_no_key, dropped_duplicate, len(deduped),
    )
    return deduped
