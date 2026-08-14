"""Deterministic feature-DataFrame builder for the price model.

Pure-Python module; no I/O. Consumes the canonical cleaned DataFrame
emitted by Step 07 (``data/processed/clean_listings.parquet``) and emits
a deterministic feature frame in the locked column order defined by
``ENGINEERED_COLUMNS`` + the 16 input-contract fields.

The 16 input-contract field names are imported from
``api.schemas.predict_v3.INPUT_FIELDS_V3`` — that tuple is the
authoritative wire-format contract (Step 11), and the feature frame must
match it column-for-column so the FastAPI request body can be losslessly
turned into a feature row.

Leakage note: ``price_per_sqft`` is a within-row ratio
(``price_inr / built_up_area``), allowed for the price model. The
classifier spec removes it before training (Rules §8.1). The locality
aggregator columns are NOT computed here — they are produced by
``LocalityAggregator`` and added downstream.
"""

from __future__ import annotations

import re
from enum import Enum

import pandas as pd

from api.schemas.predict_v3 import INPUT_FIELDS_V3

# ---------------------------------------------------------------------------
# ENGINEERED_COLUMNS — the ordered tuple of columns build_feature_frame() adds
# beyond the 16 contract fields. Locked at 11 entries.
#
# The last three (locality_*) come from LocalityAggregator; the rest are
# pure row-level math applied by derive_row_features().
# ---------------------------------------------------------------------------
ENGINEERED_COLUMNS: tuple[str, ...] = (
    "price_per_sqft",
    "n_amenities",
    "n_features",
    "floor_ratio",
    "age_bucket_ord",
    "bath_bed_ratio",
    "area_per_bedroom",
    "locality_avg_price_sqft",
    "locality_listing_count",
    "locality_smoothed_price",
    "top_amenities_count",
)

# ---------------------------------------------------------------------------
# Enums — match Step 11's Pydantic enums verbatim (string values must agree
# so train-time labels and serve-time API requests share the same vocabulary).
# ---------------------------------------------------------------------------


class AgeBucket(str, Enum):
    NEW = "New Property"
    RELATIVELY_NEW = "Relatively New"
    MODERATELY_OLD = "Moderately Old"
    OLD = "Old Property"
    UNDER_CONSTRUCTION = "Under Construction"


class FloorCategory(str, Enum):
    LOW = "Low Floor"
    MID = "Mid Floor"
    HIGH = "High Floor"


class LuxuryCategory(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FurnishingType(str, Enum):
    UNFURNISHED = "Unfurnished"
    SEMIFURNISHED = "Semifurnished"
    FURNISHED = "Furnished"


class Balcony(str, Enum):
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    THREE_PLUS = "3+"


# ---------------------------------------------------------------------------
# Ordinal mappings — order is the resale desirability axis (newer is more
# valuable), not raw age. Pinned because OrdinalEncoder is order-sensitive
# and train/serve skew if this ever drifts.
# ---------------------------------------------------------------------------
AGE_BUCKET_ORDINAL: dict[str, int] = {
    "New Property": 0,
    "Under Construction": 1,
    "Relatively New": 2,
    "Moderately Old": 3,
    "Old Property": 4,
}

FLOOR_CATEGORY_ORDINAL: dict[str, int] = {
    "Low Floor": 0,
    "Mid Floor": 1,
    "High Floor": 2,
}

LUXURY_CATEGORY_ORDINAL: dict[str, int] = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

FURNISHING_TYPE_ORDINAL: dict[str, int] = {
    "Unfurnished": 0,
    "Semifurnished": 1,
    "Furnished": 2,
}

BALCONY_ORDINAL: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "3+": 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify_amenity(label: str) -> str:
    """Convert ``"Swimming Pool"`` -> ``"swimming_pool"``.

    Lowercase, snake_case, strips punctuation, collapses whitespace. Used
    to turn amenity labels into column names for the ``has_<amenity>``
    flags.
    """
    s = label.lower().strip()
    # Replace any non-alphanumeric run with a single underscore.
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def select_top_amenities(df: pd.DataFrame, k: int = 10) -> list[str]:
    """Return the top-K most-frequent amenity labels across the frame.

    Explodes ``amenities_list`` (a list-valued column; each entry is a
    list of decoded strings) and counts frequencies. Pure function.
    """
    if "amenities_list" not in df.columns or k <= 0:
        return []
    # Flat-explode: collect all amenities across all rows.
    flat = (
        df["amenities_list"]
        .dropna()
        .explode()
        .dropna()
    )
    if flat.empty:
        return []
    counts = flat.value_counts()
    return counts.head(k).index.tolist()


def derive_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the non-locality engineered columns to a copy of ``df``.

    Adds (in this order):
        - ``price_per_sqft``: ``price_inr / built_up_area`` (within-row,
          leakage-free)
        - ``n_amenities``, ``n_features``: list-length counts
        - ``floor_ratio``: ``floor_num / total_floor``
        - ``age_bucket_ord``: ordinal from ``AGE_BUCKET_ORDINAL``
        - ``bath_bed_ratio``: ``bathroom / bedRoom`` (NaN if bedRoom == 0)
        - ``area_per_bedroom``: ``built_up_area / bedRoom`` (NaN if 0)
        - ``has_<amenity_slug>``: one bool column per top-K amenity
        - ``top_amenities_count``: how many of the top-K amenities this
          row has (cheap density signal)

    Pure: returns a new frame, does not mutate ``df``. Missing input
    columns are tolerated (the corresponding engineered column is
    populated with NaN) — see ``build_feature_frame`` for the strict
    validation that gates feature-frame production.
    """
    out = df.copy()
    n_rows = len(out)

    # price_per_sqft — within-row ratio, leakage-free (both operands are
    # observed per row, not a group average).
    if "price_inr" in out.columns and "built_up_area" in out.columns:
        out["price_per_sqft"] = out["price_inr"] / out["built_up_area"].replace(
            0, pd.NA
        )
    else:
        out["price_per_sqft"] = pd.NA

    # n_amenities / n_features — list-length counts.
    if "amenities_list" in out.columns:
        out["n_amenities"] = (
            out["amenities_list"].apply(lambda v: len(v) if isinstance(v, list) else 0)
        )
    else:
        out["n_amenities"] = 0
    if "features_list" in out.columns:
        out["n_features"] = (
            out["features_list"].apply(lambda v: len(v) if isinstance(v, list) else 0)
        )
    else:
        out["n_features"] = 0

    # floor_ratio — divide by zero -> NaN.
    if "floor_num" in out.columns and "total_floor" in out.columns:
        denom = out["total_floor"].replace(0, pd.NA)
        out["floor_ratio"] = out["floor_num"] / denom
    else:
        out["floor_ratio"] = pd.NA

    # age_bucket_ord — ordinal via the pinned dict.
    if "agePossession" in out.columns:
        out["age_bucket_ord"] = out["agePossession"].map(AGE_BUCKET_ORDINAL)
    else:
        out["age_bucket_ord"] = pd.NA

    # bath_bed_ratio / area_per_bedroom — divide-by-zero guard.
    if "bedRoom" in out.columns:
        bed_safe = out["bedRoom"].replace(0, pd.NA)
        if "bathroom" in out.columns:
            out["bath_bed_ratio"] = out["bathroom"] / bed_safe
        else:
            out["bath_bed_ratio"] = pd.NA
        if "built_up_area" in out.columns:
            out["area_per_bedroom"] = out["built_up_area"] / bed_safe
        else:
            out["area_per_bedroom"] = pd.NA
    else:
        out["bath_bed_ratio"] = pd.NA
        out["area_per_bedroom"] = pd.NA

    # top-K amenity flags + density count.
    top = select_top_amenities(out, k=10)
    counts = pd.Series([0] * n_rows, index=out.index, dtype="int64")
    if "amenities_list" in out.columns and top:
        amen_sets = out["amenities_list"].apply(
            lambda v: set(v) if isinstance(v, list) else set()
        )
        for label in top:
            slug = slugify_amenity(label)
            col = f"has_{slug}"
            out[col] = amen_sets.apply(lambda s, _lbl=label: _lbl in s).astype("bool")
            counts = counts + out[col].astype("int64")
    else:
        # No amenities column or no top-K — emit has_* = False columns
        # so the column contract is stable across frames.
        for label in top:
            slug = slugify_amenity(label)
            out[f"has_{slug}"] = pd.Series([False] * n_rows, index=out.index)
    out["top_amenities_count"] = counts

    return out


# Columns that are diagnostic flags (not model inputs) and must be
# dropped before the preprocessor sees the frame.
_DIAGNOSTIC_PREFIXES: tuple[str, ...] = ("was_missing_",)
_DIAGNOSTIC_EXACT: tuple[str, ...] = ("is_outlier",)


# The 16-field contract uses "amenities" as the wire-format name, but
# the cleaning layer emits "amenities_list". Map between them so callers
# can pass either form.
_API_TO_CLEANING_NAME: dict[str, str] = {"amenities": "amenities_list"}


def _resolve_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` where the INPUT_FIELDS_V3 contract names are present.

    If the cleaning layer emitted ``amenities_list`` (its canonical
    spelling) but the API contract expects ``amenities``, alias it. The
    alias is internal-only — the final output column order matches the
    INPUT_FIELDS_V3 wire names.
    """
    work = df.copy()
    for api_name, clean_name in _API_TO_CLEANING_NAME.items():
        if api_name not in work.columns and clean_name in work.columns:
            work[api_name] = work[clean_name]
    return work


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level feature-DataFrame builder.

    Steps:
        1. Validate input has every column in ``INPUT_FIELDS_V3`` plus
           ``is_outlier`` (Step 06) and any ``was_missing_*`` flags
           (Step 07). Raises ``ValueError`` listing missing columns.
        2. Apply ``derive_row_features`` (row-level math + amenity
           flags).
        3. Drop ``is_outlier`` and ``was_missing_*`` (diagnostic, not
           model inputs). Outlier rows are still present — the training
           script applies the ``is_outlier == False`` filter at training
           time, not here.
        4. Reorder columns: 16 contract fields first (per
           ``INPUT_FIELDS_V3`` order), then ``ENGINEERED_COLUMNS``. Final
           order is deterministic and pinned by a test.

    Pure function: returns a new DataFrame, no I/O.
    """
    work = _resolve_input_columns(df)
    required = list(INPUT_FIELDS_V3) + ["is_outlier"]
    was_missing = [c for c in work.columns if c.startswith("was_missing_")]
    required += was_missing
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(
            f"build_feature_frame: input frame is missing required columns: {missing}"
        )

    work = derive_row_features(work)

    # Drop diagnostic flags. Outlier rows remain; is_outlier filter is
    # training-side only (LocalityAggregator also reads is_outlier
    # separately at fit time).
    drop_cols = list(_DIAGNOSTIC_EXACT) + [
        c for c in work.columns if c in _DIAGNOSTIC_PREFIXES or c.startswith("was_missing_")
    ]
    work = work.drop(columns=[c for c in drop_cols if c in work.columns])

    # Final deterministic column order: INPUT_FIELDS_V3 then ENGINEERED_COLUMNS.
    final_cols = list(INPUT_FIELDS_V3) + list(ENGINEERED_COLUMNS)
    # ENGINEERED_COLUMNS may include fields not present (e.g. locality_*
    # until LocalityAggregator has run). Surface them as NaN columns.
    for col in final_cols:
        if col not in work.columns:
            work[col] = pd.NA
    return work[final_cols]
