#!/usr/bin/env python
"""Build all 13 analytics cache JSON files from clean_listings.parquet.

This script generates precomputed chart data for the Analytics dashboard,
served via FastAPI GET /api/analytics/<chart_id>?city=<city>.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.analytics import build_all_cache_files
from ml.cleaning.writers import CLEAN_LISTINGS_PARQUET_PATH

_LOG = logging.getLogger("scripts.build_analytics_cache")

# Output directory for cache files
ANALYTICS_CACHE_DIR = Path("data/processed/analytics_cache")


def main() -> None:
    """Load clean_listings.parquet and build all 13 cache files for 5 city slices."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _LOG.info("Loading clean_listings.parquet...")
    df = pd.read_parquet(CLEAN_LISTINGS_PARQUET_PATH)
    _LOG.info("Loaded %d rows", len(df))

    _LOG.info("Building analytics cache files in %s", ANALYTICS_CACHE_DIR)
    written = build_all_cache_files(df, ANALYTICS_CACHE_DIR)

    _LOG.info("Successfully built %d cache files", len(written))
    for f in written:
        _LOG.info("  - %s", f)


if __name__ == "__main__":
    main()