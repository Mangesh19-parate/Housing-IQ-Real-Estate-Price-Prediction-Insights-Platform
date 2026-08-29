"""CLI: build the HousingIQ classifier feature frame (Spec 22).

Reads ``data/processed/clean_listings.parquet`` (and the Spec-21
``price_tier_labels.parquet`` + ``good_deal_labels.parquet`` for the
report's per-cell counts); writes:

    data/processed/classifier/feature_frame.parquet
    data/processed/classifier/feature_frame_report.md   (append)

Plus a one-line ``[OK]`` summary on stdout.

Authority:
- ``docs/08-RULES.md`` §8.1 + §12.4 (leakage ban on price-derived
  classifier inputs)
- ``ml/features/feature_frame.py`` + ``ml/features/locality_aggregator.py``
  (Step 12 — the reused price-model pipeline)

Exit code:
    0 on success.
    1 on hard failure (missing input parquet, schema mismatch).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force UTF-8 stdout/stderr on Windows (cp1252 default).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml.classification import (  # noqa: E402
    DROPPED_FOR_LEAKAGE,
    RETAINED_LOCALITY_COLUMN,
    build_classifier_feature_frame,
)
from ml.classification.feature_report import (  # noqa: E402
    write_feature_frame_artifact,
    write_feature_frame_report,
)

logger = logging.getLogger("build_classifier_features")

#: Pinned regex for contact/PII/URL column names. Mirrors
#: ``scripts/build_classification_labels.py:66``.
CONTACT_FIELD_REGEX: str = r"(contact|dealer|phone|email|photo|url|spid)"


def _git_commit() -> str:
    """12-char git short SHA; ``"unknown"`` if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dataset_version(parquet_path: Path) -> str:
    """``<basename>:<sha1[:12]>`` for the first 1 MB of the parquet."""
    import hashlib

    if not parquet_path.exists():
        return "<missing>"
    hasher = hashlib.sha1()
    with open(parquet_path, "rb") as fh:
        hasher.update(fh.read(1024 * 1024))
    return f"{parquet_path.name}:{hasher.hexdigest()[:12]}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the HousingIQ classifier feature frame "
            "(Spec 22). Reuses the price-model feature pipeline and "
            "drops every price-derived column. Writes "
            "`feature_frame.parquet` and appends a Run section to "
            "`feature_frame_report.md`. Exit 0 on success; 1 on "
            "hard failure."
        )
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_PROCESSED_DIR",
                str(_REPO_ROOT / "data" / "processed"),
            )
        ),
        help="Directory containing clean_listings.parquet.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_CLASSIFIER_OUTPUT_DIR",
                str(_REPO_ROOT / "data" / "processed" / "classifier"),
            )
        ),
        help="Output directory for the classifier feature artifacts.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    parquet_path = args.processed_dir / "clean_listings.parquet"

    if not parquet_path.exists():
        logger.error(
            "clean_listings.parquet not found at %s — run the cleaning "
            "pipeline (Step 07) first.",
            parquet_path,
        )
        print(f"[FAIL] clean_listings_missing at {parquet_path}")
        return 1

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        logger.error("pandas is required but not installed.")
        return 1

    logger.info("Reading %s", parquet_path)
    clean_listings = pd.read_parquet(parquet_path)

    # ---- schema check ----
    # Note: the real parquet has `price_inr`, NOT `price` (Spec 21's CLI
    # uses "price" — a latent bug on its side). Spec 22 checks the
    # correct column name.
    required_cols = {
        "listing_id",
        "city",
        "locality",
        "transact_type",
        "price_inr",
        "price_per_sqft",
        "is_outlier",
    }
    missing = required_cols - set(clean_listings.columns)
    if missing:
        logger.error(
            "clean_listings.parquet is missing required columns: %s",
            sorted(missing),
        )
        print(f"[FAIL] schema_mismatch missing={sorted(missing)}")
        return 1

    # ---- build feature frame ----
    logger.info("Building classifier feature frame...")
    try:
        feature_frame = build_classifier_feature_frame(clean_listings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build feature frame: %s", exc)
        print(f"[FAIL] build_raised: {exc}")
        return 1

    n_outliers = (
        int(feature_frame["is_outlier"].sum())
        if "is_outlier" in feature_frame.columns
        else 0
    )
    logger.info("feature_frame: %d rows (%d outliers)", len(feature_frame), n_outliers)

    # ---- load Spec 21's labels for the report's per-cell counts ----
    tier_path = args.output_dir / "price_tier_labels.parquet"
    good_path = args.output_dir / "good_deal_labels.parquet"
    tier_df = pd.read_parquet(tier_path) if tier_path.exists() else pd.DataFrame()
    good_df = pd.read_parquet(good_path) if good_path.exists() else pd.DataFrame()
    if tier_df.empty:
        logger.warning(
            "%s missing — per-(city × price_tier) section will be empty.",
            tier_path,
        )
    if good_df.empty:
        logger.warning(
            "%s missing — per-(city × good_deal_verdict) section will be empty.",
            good_path,
        )

    # ---- persist artifacts ----
    try:
        write_feature_frame_artifact(feature_frame, args.output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write artifact: %s", exc)
        print(f"[FAIL] write_artifact_raised: {exc}")
        return 1

    # ---- append run section to the report ----
    try:
        write_feature_frame_report(
            output_dir=args.output_dir,
            feature_frame=feature_frame,
            price_tier_labels=tier_df,
            good_deal_labels=good_df,
            dataset_version=_dataset_version(parquet_path),
            git_commit=_git_commit(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write report: %s", exc)
        # Non-fatal — artifact is already on disk.

    # ---- one-line [OK] summary ----
    print(
        f"[OK] feature_frame rows={len(feature_frame)} (outliers={n_outliers}); "
        f"dropped={sorted(DROPPED_FOR_LEAKAGE)}; "
        f"retained={RETAINED_LOCALITY_COLUMN}; "
        f"written to {args.output_dir}"
    )
    logger.info(
        "Classifier features built (git=%s, output_dir=%s)",
        _git_commit(),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
