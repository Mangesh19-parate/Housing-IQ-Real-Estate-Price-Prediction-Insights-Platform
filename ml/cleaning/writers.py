"""``ml.cleaning.writers`` — Step 07 Parquet-writer layer.

Writes the canonical ``data/processed/clean_listings.parquet`` — the
single training-and-serving artifact every downstream consumer
(regression, classification, recommender, analytics, insights) reads.

Sidecar metadata file: alongside the Parquet, a ``<file>.parquet.meta.json``
is written with the dataset version, row/column counts, column list,
computation timestamp, and source-raw filenames. Satisfies Rules §1.5
("every derived table … states its computation date and source dataset
version").

Public API:
  :data:`CLEAN_LISTINGS_PARQUET_PATH`        -- default output path
  :data:`CLEAN_LISTINGS_DATASET_VERSION`     -- version tag in sidecar
  :data:`CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER` -- import-time order
  :func:`build_clean_listings_columns_order` -- per-frame order builder
  :func:`write_clean_listings_parquet`
  :func:`read_clean_listings_parquet`
  :func:`verify_clean_listings_parquet`

The writer is the **only** spec allowed to write
``data/processed/clean_listings.parquet``. Re-entry is gated by ``persist``
on the orchestrator (``ml.cleaning.pipeline.run_clean_listings_pipeline``).
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS

_LOG: logging.Logger = logging.getLogger("ml.cleaning.writers")

# Single source of truth for the output path (referenced by tests and
# scripts/run_pipeline.py).
CLEAN_LISTINGS_PARQUET_PATH: Final[Path] = Path("data/processed/clean_listings.parquet")

# Bumped whenever the schema changes incompatibly (column drop, dtype change).
CLEAN_LISTINGS_DATASET_VERSION: Final[str] = "v1"

# Import-time constant covers the columns discoverable without the frame.
# ``was_missing_*`` columns are dynamic (one per medium/high-tier column
# at runtime); the writer rebuilds the full ordered tuple via
# :func:`build_clean_listings_columns_order`. ``CANONICAL_COLUMNS`` already
# includes ``is_outlier`` as its last entry — appending it again would
# set-dedup to a no-op, so we only add ``outlier_reasons`` here.
#
# In the full per-frame order (see :func:`build_clean_listings_columns_order`),
# sorted ``was_missing_*`` columns are inserted between ``is_outlier`` and
# ``outlier_reasons``, so ``outlier_reasons`` is always the LAST column.
CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER: Final[tuple[str, ...]] = (
    *CANONICAL_COLUMNS,
    "outlier_reasons",
)

# Sidecar metadata filename. The ``with_suffix(path.suffix + ".meta.json")``
# pattern resolves to ``<file>.parquet.meta.json`` for our path — that
# two-dot extension is intentional (Rules §1.5 sidecar convention).
_META_SUFFIX: Final[str] = ".meta.json"

# Source-raw filenames, captured once for the sidecar metadata. Same
# filenames the Step 06 assembler uses; duplicated here so writers can
# be exercised in isolation without importing ``ml.cleaning.assemble``.
_SOURCE_RAW_FILES: Final[tuple[str, ...]] = (
    "data/raw/gurgaon_10k.csv",
    "data/raw/hyderabad.csv",
    "data/raw/kolkata.csv",
    "data/raw/mumbai.csv",
    "data/raw/facets/*.csv",
)


# ---------------------------------------------------------------------------
# Column-order builder
# ---------------------------------------------------------------------------


def build_clean_listings_columns_order(df: pd.DataFrame) -> tuple[str, ...]:
    """Build the per-frame deterministic column order for the Parquet.

    Order: ``CANONICAL_COLUMNS`` (already includes ``is_outlier``) →
    sorted ``was_missing_*`` columns → ``outlier_reasons`` (always last).
    Insertion order is preserved via ``dict.fromkeys`` semantics
    (Python 3.7+); the final ``outlier_reasons`` re-append places it at
    the end after the dynamically-named flags. Extra columns in the frame
    that are NOT in this set are dropped with a warning by the writer
    (defensive — should not happen in practice).
    """
    was_missing = sorted(c for c in df.columns if c.startswith("was_missing_"))
    ordered_keys = list(dict.fromkeys((*CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER, *was_missing)))
    # Re-append ``outlier_reasons`` so it sits at the END (after any
    # was_missing_* columns) regardless of where it appeared in the
    # import-time tuple. Idempotent if it's already last.
    if ordered_keys[-1] != "outlier_reasons":
        ordered_keys = [c for c in ordered_keys if c != "outlier_reasons"]
        ordered_keys.append("outlier_reasons")
    return tuple(ordered_keys)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _sidecar_path(parquet_path: Path) -> Path:
    """Build the sidecar metadata path: ``<file>.parquet.meta.json``."""
    return parquet_path.with_suffix(parquet_path.suffix + _META_SUFFIX)


def _write_sidecar_metadata(
    parquet_path: Path,
    df: pd.DataFrame,
    source_raw_files: tuple[str, ...],
) -> None:
    """Emit a JSON sidecar with the version, counts, columns, timestamp.

    Satisfies Rules §1.5 — every derived table states its computation
    date and source dataset version.
    """
    payload = {
        "dataset_version": CLEAN_LISTINGS_DATASET_VERSION,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw_files": list(source_raw_files),
    }
    sidecar = _sidecar_path(parquet_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_clean_listings_parquet(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Write ``df`` to ``output_path`` (default :data:`CLEAN_LISTINGS_PARQUET_PATH`).

    Steps:
      1. Resolve ``output_path`` (default the constant).
      2. ``mkdir(parents=True, exist_ok=True)``.
      3. Reorder columns to :func:`build_clean_listings_columns_order`;
         columns not in that order are dropped with a warning (defensive;
         no row loss).
      4. ``df.to_parquet(path, index=False, engine="pyarrow")``.
      5. Write the sidecar ``<file>.parquet.meta.json``.
      6. Return the path written.
    """
    target = output_path or CLEAN_LISTINGS_PARQUET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ordered = build_clean_listings_columns_order(df)
    extras = [c for c in df.columns if c not in ordered]
    if extras:
        warnings.warn(
            f"write_clean_listings_parquet: dropping unexpected columns "
            f"not in CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER: {extras}",
            stacklevel=2,
        )
        df = df.drop(columns=extras)

    df = df[list(ordered)]

    df.to_parquet(target, index=False, engine="pyarrow")
    _write_sidecar_metadata(target, df, _SOURCE_RAW_FILES)
    _LOG.info(
        "writers.write path=%s rows=%d cols=%d version=%s",
        target,
        len(df),
        len(df.columns),
        CLEAN_LISTINGS_DATASET_VERSION,
    )
    return target


# ---------------------------------------------------------------------------
# Reader + verifier
# ---------------------------------------------------------------------------


def read_clean_listings_parquet(path: Path | None = None) -> pd.DataFrame:
    """Read the Parquet (default :data:`CLEAN_LISTINGS_PARQUET_PATH`)."""
    target = path or CLEAN_LISTINGS_PARQUET_PATH
    return pd.read_parquet(target, engine="pyarrow")


def verify_clean_listings_parquet(path: Path | None = None) -> dict:
    """Sanity-check the Parquet and return a verification dict.

    Keys: ``exists``, ``row_count``, ``column_count``,
    ``columns_match_canonical_order``, ``listing_id_unique``,
    ``has_is_outlier``, ``has_was_missing_columns``, ``error``.

    On a missing file, ``exists=False`` and all other keys are
    absent (or False). On a read failure, ``exists=False, error=str(e)``.
    """
    target = path or CLEAN_LISTINGS_PARQUET_PATH
    result: dict = {"exists": target.exists()}
    if not result["exists"]:
        result["row_count"] = 0
        result["column_count"] = 0
        result["columns_match_canonical_order"] = False
        result["listing_id_unique"] = False
        result["has_is_outlier"] = False
        result["has_was_missing_columns"] = False
        return result

    try:
        df = read_clean_listings_parquet(target)
    except Exception as exc:  # pragma: no cover — defensive
        result["exists"] = False
        result["error"] = str(exc)
        return result

    cols = list(df.columns)
    result["row_count"] = int(len(df))
    result["column_count"] = int(len(cols))
    expected_prefix = list(build_clean_listings_columns_order(df))
    result["columns_match_canonical_order"] = cols == expected_prefix
    result["listing_id_unique"] = (
        "listing_id" in cols and bool(df["listing_id"].is_unique)
    )
    result["has_is_outlier"] = "is_outlier" in cols
    result["has_was_missing_columns"] = any(
        c.startswith("was_missing_") for c in cols
    )
    return result


__all__ = [
    "CLEAN_LISTINGS_PARQUET_PATH",
    "CLEAN_LISTINGS_DATASET_VERSION",
    "CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER",
    "build_clean_listings_columns_order",
    "read_clean_listings_parquet",
    "verify_clean_listings_parquet",
    "write_clean_listings_parquet",
]
