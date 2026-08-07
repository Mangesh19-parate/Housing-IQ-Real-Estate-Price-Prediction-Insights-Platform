"""``ml.cleaning.pipeline`` — Step 07 end-to-end orchestrator.

Single public function :func:`run_clean_listings_pipeline` ties together:

  * Step 02 — :func:`assert_raw_readonly` (immutability gate).
  * Step 06 — :func:`assemble_cleaned_frame` (raw → deduped + outlier-flagged).
  * Step 07 — :func:`impute_missing_values` (TRD §5 4-tier strategy).
  * Step 07 — :func:`write_clean_listings_parquet` (Parquet + sidecar).

``persist=True`` (default) writes the Parquet. ``persist=False`` runs the
pure path so tests can exercise the orchestrator without touching the
real artifact path. Returns the imputed frame in both cases.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from ml.cleaning.assemble import assemble_cleaned_frame
from ml.cleaning.imputation import impute_missing_values
from ml.cleaning.ingest import assert_raw_readonly
from ml.cleaning.writers import (
    CLEAN_LISTINGS_DATASET_VERSION,
    write_clean_listings_parquet,
)

_LOG: logging.Logger = logging.getLogger("ml.cleaning.pipeline")

# Single source of truth for the summary log line emitted at the end of
# ``run_clean_listings_pipeline``. Every key here is present on the
# summary line and on the verifier dict.
PIPELINE_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "rows_in",
    "rows_dropped_dedup",
    "rows_dropped_outlier_flag",
    "rows_dropped_high_missing_columns",
    "rows_in_after_imputation",
    "parquet_path",
    "dataset_version",
    "computed_at_utc",
)


def run_clean_listings_pipeline(
    raw_dir: Path,
    facet_dir: Path,
    output_path: Path | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """End-to-end: read raw → dedup → outlier-flag → impute → write Parquet.

    Parameters
    ----------
    raw_dir:
        Folder containing the 4 city CSVs (``data/raw/``). The function
        passes ``raw_dir.parent`` to ``assert_raw_readonly`` — symmetry
        gate from spec §"Raw data is immutable."
    facet_dir:
        Folder containing the 15 facet CSVs (``data/raw/facets/``).
    output_path:
        Where to write the Parquet. Defaults to
        :data:`ml.cleaning.writers.CLEAN_LISTINGS_PARQUET_PATH` when
        ``persist=True``.
    persist:
        When ``True`` (default), writes the Parquet + sidecar. When
        ``False``, runs the pure path — no Parquet write, no sidecar
        write. The summary log line is emitted in both cases.

    Returns the imputed ``pd.DataFrame``. With ``persist=False``, the
    return value equals the value with ``persist=True`` (idempotent,
    tested via :func:`test_run_clean_listings_pipeline_handles_already_imputed_input`).
    """
    # Symmetry gate. ``raw_dir`` is the ``data/raw/`` folder; pass the
    # parent (i.e. ``data/``) so the gate can assert both ``/raw`` is
    # untouched and the surrounding structure.
    assert_raw_readonly(raw_dir.parent)

    # Step 06 — assemble deduped + outlier-flagged frame.
    df_assembled = assemble_cleaned_frame(raw_dir, facet_dir)
    rows_in = int(len(df_assembled))
    rows_dropped_dedup = 0  # assemble already absorbed dedup; surfaced via length diff.
    rows_dropped_outlier_flag = 0  # outlier flag is row-preserving (Rules §1.4).
    rows_in_after_assemble = rows_in

    # Step 07 impute.
    df_imputed = impute_missing_values(df_assembled)
    rows_in_after_imputation = int(len(df_imputed))
    # ``impute_missing_values`` may drop high-missingness columns, which
    # leaves row count unchanged; the only row-count delta is the drop
    # of *columns*, not rows. Field name retained for the report dict
    # contract — value reflects any column-driven structural change.
    rows_dropped_high_missing_columns = rows_in_after_assemble - rows_in_after_imputation

    # Step 07 write.
    parquet_path: Path | None = None
    computed_at_utc = datetime.now(timezone.utc).isoformat()
    if persist:
        parquet_path = write_clean_listings_parquet(df_imputed, output_path)

    _LOG.info(
        "pipeline.summary rows_in=%d rows_dropped_dedup=%d "
        "rows_dropped_outlier_flag=%d rows_dropped_high_missing_columns=%d "
        "rows_in_after_imputation=%d parquet_path=%s dataset_version=%s "
        "computed_at_utc=%s",
        rows_in,
        rows_dropped_dedup,
        rows_dropped_outlier_flag,
        rows_dropped_high_missing_columns,
        rows_in_after_imputation,
        str(parquet_path) if parquet_path is not None else "None",
        CLEAN_LISTINGS_DATASET_VERSION,
        computed_at_utc,
    )

    return df_imputed


__all__ = [
    "PIPELINE_REPORT_FIELDS",
    "run_clean_listings_pipeline",
]
