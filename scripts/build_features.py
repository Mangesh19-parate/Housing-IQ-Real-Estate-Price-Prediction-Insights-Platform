"""Build the price-model feature artifacts (Spec 12 Phase 5).

Reads the canonical cleaned Parquet emitted by Step 07, fits the
leakage-safe locality aggregator + the ColumnTransformer on the
training subset only (the 70% train split with ``is_outlier == False``
rows), and serializes the artifacts under ``models/``.

Invoked as ``python scripts/build_features.py`` from repo root.
Idempotent — re-running overwrites the same artifact files.

Logs (stdlib logging INFO):
    1. rows in
    2. rows after outlier filter
    3. aggregator row count
    4. preprocessor fit time + output dim
    5. artifacts written
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Repo root on sys.path so ``from ml.features import ...`` works whether
# this is run as a script or imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.features.feature_frame import (  # noqa: E402
    build_feature_frame,
    derive_row_features,
)
from ml.features.locality_aggregator import LocalityAggregator  # noqa: E402
from ml.features.persistence import (  # noqa: E402
    ARTIFACT_DIR,
    save_feature_artifacts,
)
from ml.features.preprocessor import (  # noqa: E402
    fit_preprocessor,
    transform_with_preprocessor,
)
from ml.features.split import split_train_val_test  # noqa: E402

logger = logging.getLogger("build_features")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build price-model feature pipeline artifacts."
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
        help="Path to the cleaned Parquet (default: "
        "$HOUSINGIQ_PROCESSED_DIR/clean_listings.parquet or "
        "data/processed/clean_listings.parquet).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Where to write feature_pipeline_v1.pkl + feature_list_v1.json.",
    )
    args = parser.parse_args()

    parquet_path: Path = args.parquet
    artifact_dir: Path = args.artifact_dir

    if not parquet_path.exists():
        logger.error("Parquet not found: %s", parquet_path)
        return 2

    logger.info("Reading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("rows in: %d", len(df))

    # Split first — train/val/test on the cleaned frame.
    train_df, _val_df, _test_df = split_train_val_test(df)
    logger.info(
        "split sizes: train=%d, val=%d, test=%d",
        len(train_df),
        len(_val_df),
        len(_test_df),
    )

    # Filter outliers from the training subset only (Rules §2.3, §8.4).
    clean_train = train_df[train_df["is_outlier"] == False]  # noqa: E712
    logger.info("rows after outlier filter: %d", len(clean_train))

    # Build the deterministic feature frame (16 contract fields + 11
    # engineered columns). Locality cols arrive as NaN here.
    feature_frame = build_feature_frame(clean_train)
    logger.info(
        "feature frame columns: %d (16 contract + 11 engineered)",
        len(feature_frame.columns),
    )

    # Fit locality aggregator on the training subset (needs locality +
    # city + price columns; build_feature_frame strips locality because
    # it's not a regression input, so we re-use clean_train here).
    agg = LocalityAggregator().fit(clean_train)
    logger.info(
        "aggregator row count: %d (city, locality) groups",
        agg.n_groups_,
    )

    # The preprocessor expects the engineered columns (age_bucket_ord,
    # bath_bed_ratio, area_per_bedroom, top_amenities_count, has_*). Apply
    # ``derive_row_features`` to clean_train (which still has locality) so
    # the aggregator can layer on the locality_* columns in fit_preprocessor.
    train_with_features = derive_row_features(clean_train)

    # Fit the preprocessor on the feature frame (which has the 11
    # engineered columns including the locality_* columns the
    # aggregator will produce).
    t0 = time.perf_counter()
    prep = fit_preprocessor(train_with_features, agg)
    fit_seconds = time.perf_counter() - t0

    # Materialize the post-transform feature names by transforming the
    # same training frame (the feature names come from the fitted
    # transformer's `get_feature_names_out`).
    post_train = transform_with_preprocessor(
        prep, agg.transform(train_with_features)
    )
    feature_names = list(post_train.columns)
    logger.info(
        "preprocessor fit in %.2fs; post-transform dim = %d",
        fit_seconds,
        len(feature_names),
    )

    paths = save_feature_artifacts(
        prep,
        agg,
        feature_names,
        artifact_dir=artifact_dir,
    )
    logger.info("artifacts written: %s", paths)

    return 0


if __name__ == "__main__":
    sys.exit(main())
