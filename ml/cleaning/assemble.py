"""``ml.cleaning.assemble`` — Step 06 orchestrator.

Single public function :func:`assemble_cleaned_frame` ties together:

  * Step 02 — :func:`assert_raw_readonly` (immutability gate) +
    :func:`load_raw_city_frames` (not used directly; per-city CSV reads go
    through Step 05 mappers).
  * Step 04 — :func:`load_facet_frames` (facet decode tables).
  * Step 05 — :func:`map_city` (per-city canonical mappers).
  * Step 06 — :func:`deduplicate_listings` + :func:`flag_all_outliers`.

Step 07 (missing-value imputation) imports :func:`assemble_cleaned_frame` as
its single entry point. No writes to ``data/processed/`` here — Step 07 owns
the first Parquet write.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from ml.cleaning.canonical_mapping import map_city
from ml.cleaning.dedup import deduplicate_listings
from ml.cleaning.facet_decoders import load_facet_frames
from ml.cleaning.ingest import assert_raw_readonly
from ml.cleaning.outliers import flag_all_outliers

_LOG: logging.Logger = logging.getLogger("ml.cleaning.assemble")

# Same filenames as Step 02's RAW_FILE_TO_CITY (kept here so the assembler
# has a self-contained city → filename map without depending on Step 02's
# private tuple ordering).
ASSEMBLE_CITY_FILES: Final[dict[str, str]] = {
    "Gurgaon": "gurgaon_10k.csv",
    "Hyderabad": "hyderabad.csv",
    "Kolkata": "kolkata.csv",
    "Mumbai": "mumbai.csv",
}

# What ``assemble_cleaned_frame`` logs at the end (single summary line).
ASSEMBLE_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "rows_in",
    "rows_dropped_no_listing_id",
    "rows_dropped_duplicate",
    "rows_in_after_dedup",
    "rows_flagged_outlier",
    "rows_in_after_outlier_flag",
    "per_city_breakdown",
)


def _derive_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``price_per_sqft = price_inr / area_sqft`` where both are non-null.

    Step 05 leaves ``price_per_sqft`` as ``pd.NA`` by design (price_per_sqft is
    a feature-engineering concern per Step 05's docstring). TRD §6 requires
    outlier flagging on this column, so the assembler derives it here and
    passes the result to ``flag_all_outliers``. Overwrites the column in
    place; rows where either input is NaN or ``area_sqft <= 0`` stay NaN.
    """
    out = df.copy()
    if "price_inr" not in out.columns or "area_sqft" not in out.columns:
        return out
    mask = (
        out["price_inr"].notna()
        & out["area_sqft"].notna()
        & (out["area_sqft"] > 0)
    )
    if "price_per_sqft" not in out.columns:
        out["price_per_sqft"] = pd.NA
    out.loc[mask, "price_per_sqft"] = (
        out.loc[mask, "price_inr"] / out.loc[mask, "area_sqft"]
    )
    return out


def assemble_cleaned_frame(raw_dir: Path, facet_dir: Path) -> pd.DataFrame:
    """Read raw + facets, run per-city mappers, concat, dedup, flag outliers.

    ``raw_dir`` is the folder containing the 4 city CSVs
    (``data/raw/``). ``facet_dir`` is the folder containing the 15 facet
    CSVs (``data/raw/facets/``).

    Returns the assembled DataFrame. Does NOT write to ``data/processed/``.
    Step 07 is responsible for the Parquet write.
    """
    # ``assert_raw_readonly`` expects the parent data dir, whose ``/raw``
    # subfolder holds the immutable CSVs. ``raw_dir`` is ``data/raw/`` →
    # pass ``raw_dir.parent``.
    assert_raw_readonly(raw_dir.parent)

    facets = load_facet_frames(facet_dir)
    rows_in_per_city: dict[str, int] = {}
    frames: list[pd.DataFrame] = []
    for city, filename in ASSEMBLE_CITY_FILES.items():
        city_df = map_city(city, raw_dir / filename, facets)
        rows_in_per_city[city] = len(city_df)
        frames.append(city_df)
    df = pd.concat(frames, ignore_index=True)
    rows_in = len(df)

    # Derive price_per_sqft BEFORE outlier flagging — TRD §6 needs it.
    df = _derive_price_per_sqft(df)

    # Dedup. Capture dropped counts from the log line indirectly via
    # DataFrame length diff (dedup.py also logs the exact bucket counts).
    rows_after_dedup = len(df)
    df = deduplicate_listings(df)
    rows_after_dedup = len(df)

    # Flag outliers.
    df = flag_all_outliers(df)
    rows_after_outliers = len(df)
    rows_flagged = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0

    # Per-city breakdown of the final frame.
    per_city: dict[str, dict[str, int]] = {}
    if "city" in df.columns:
        for city, group in df.groupby("city"):
            per_city[city] = {
                "n_rows": int(len(group)),
                "n_outliers": int(group["is_outlier"].sum())
                if "is_outlier" in group.columns else 0,
            }

    _LOG.info(
        "assemble.summary rows_in=%d rows_in_after_dedup=%d rows_flagged_outlier=%d "
        "rows_in_after_outlier_flag=%d per_city_breakdown=%s",
        rows_in, rows_after_dedup, rows_flagged, rows_after_outliers, per_city,
    )
    return df
