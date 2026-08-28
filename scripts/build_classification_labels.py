"""CLI: build the HousingIQ Classification target labels (Spec 21).

Reads ``data/processed/clean_listings.parquet`` + (optionally)
``data/processed/oof_predictions_{model_version}.parquet``; writes:

    data/processed/classifier/price_tier_labels.parquet
    data/processed/classifier/price_tier_quantile_boundaries.json
    data/processed/classifier/good_deal_labels.parquet
    data/processed/classifier/good_deal_thresholds.json
    data/processed/classifier/label_construction_report.md   (append)

Plus a one-line ``[OK]`` summary on stdout.

Two-pass ``build_good_deal_labels`` flow:
    1. First pass with ``thresholds=None`` — every verdict is null,
       ``residual_pct`` populated.
    2. ``calibrate_good_deal_thresholds(...)`` reads the
       ``residual_pct`` + ``is_outlier`` + train-mask to derive
       per-city thresholds.
    3. Second pass with the calibrated thresholds — verdicts
       populated from the per-city band.

Exit code:
    0 on success.
    1 on hard failure (missing input parquet, schema mismatch).
    A null-OOF scenario is a WARNING, not a failure — verdicts
    come back null with ``reason="missing_oof_prediction"``.
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

# Force UTF-8 stdout/stderr on Windows (cp1252 default) — the
# ``≥`` and ``±`` glyphs in the summary would otherwise crash with
# ``UnicodeEncodeError``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml.classification import (  # noqa: E402
    build_good_deal_labels,
    build_price_tier_labels,
    calibrate_good_deal_thresholds,
    write_good_deal_artifacts,
    write_label_construction_report,
    write_price_tier_artifacts,
)
from ml.features.split import split_train_val_test  # noqa: E402

logger = logging.getLogger("build_classification_labels")

#: Pinned regex for contact/PII/URL column names. Mirrors
#: ``scripts/evaluate_price_model.py:68``. The CLI's stdout must
#: never log any column matching this regex (Rules §1.1).
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
    """``:<sha1[:12]>`` for the first 1 MB of the parquet.
    Cheap content fingerprint per Rules §1.3.
    """
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
            "Build the HousingIQ Classification target labels "
            "(price_tier + good_deal_verdict). Writes four artifacts "
            "under data/processed/classifier/ and appends a Run "
            "section to label_construction_report.md. "
            "Exit 0 on success; 1 on hard failure."
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
        help="Output directory for the four classifier artifacts.",
    )
    p.add_argument(
        "--model-version",
        default="v2",
        help=(
            "Price model version to consume OOF predictions from "
            "(reads oof_predictions_{model_version}.parquet). "
            "Default 'v2' (current best price model)."
        ),
    )
    p.add_argument(
        "--min-city-rows",
        type=int,
        default=100,
        help=(
            "Minimum non-outlier training rows per "
            "(city, transact_type) group for per-city calibration. "
            "Below this, the city uses default ±10% thresholds. "
            "Default 100."
        ),
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    parquet_path = args.processed_dir / "clean_listings.parquet"
    oof_path = args.processed_dir / f"oof_predictions_{args.model_version}.parquet"

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

    # Validate required columns. Missing → hard failure.
    required_cols = {
        "listing_id",
        "city",
        "locality",
        "transact_type",
        "price",
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

    # ---- price_tier builder ----
    logger.info("Building price_tier labels (pass 1)...")
    tier_df, boundaries = build_price_tier_labels(
        clean_listings,
        min_rows=args.min_city_rows,
    )
    n_tier_with_label = int(tier_df["price_tier"].notna().sum())
    logger.info("price_tier: %d rows with a label", n_tier_with_label)

    # Derive the train mask from the same fixed 70/15/15 split the
    # tier builder used. The tier builder already applied
    # split_train_val_test internally; we replicate the call here
    # so calibrate_good_deal_thresholds sees the same train subset.
    train_df, _val_df, _test_df = split_train_val_test(
        clean_listings, target="price"
    )
    is_train_mask = clean_listings.index.isin(train_df.index)

    # ---- good_deal_verdict builder (first pass) ----
    logger.info("Building good_deal_labels (pass 1: residuals)...")
    oof = None
    if oof_path.exists():
        logger.info("Loading OOF predictions from %s", oof_path)
        oof = pd.read_parquet(oof_path)
    else:
        logger.warning(
            "OOF predictions not found at %s — every verdict will be "
            "null with reason='missing_oof_prediction'.",
            oof_path,
        )

    good_deal_df_pass1, _ = build_good_deal_labels(
        clean_listings,
        oof_predictions=oof,
        thresholds=None,
        model_version=args.model_version,
    )

    # ---- calibrate per-city thresholds ----
    logger.info("Calibrating per-city verdict thresholds...")
    thresholds = calibrate_good_deal_thresholds(
        good_deal_df_pass1,
        is_train_mask=is_train_mask,
    )

    # ---- good_deal_verdict builder (second pass with calibrated thresholds) ----
    logger.info("Building good_deal_labels (pass 2: verdicts)...")
    good_deal_df, _ = build_good_deal_labels(
        clean_listings,
        oof_predictions=oof,
        thresholds=thresholds,
        model_version=args.model_version,
    )
    n_verdict = int(good_deal_df["good_deal_verdict"].notna().sum())
    n_no_oof = int(good_deal_df["reason"].notna().sum())
    logger.info("good_deal_verdict: %d rows with verdict", n_verdict)
    logger.info("good_deal_verdict: %d rows without OOF", n_no_oof)

    # ---- persist artifacts ----
    try:
        tier_paths = write_price_tier_artifacts(
            tier_df, boundaries, args.output_dir
        )
        verdict_paths = write_good_deal_artifacts(
            good_deal_df, thresholds, args.output_dir
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write artifacts: %s", exc)
        print(f"[FAIL] write_artifacts_raised: {exc}")
        return 1

    # ---- append run section to the report ----
    try:
        write_label_construction_report(
            output_dir=args.output_dir,
            price_tier_df=tier_df,
            good_deal_df=good_deal_df,
            quantile_boundaries=boundaries,
            good_deal_thresholds=thresholds,
            dataset_version=_dataset_version(parquet_path),
            git_commit=_git_commit(),
            min_rows=args.min_city_rows,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write report: %s", exc)
        # Non-fatal — artifacts are already on disk.

    # ---- one-line [OK] summary ----
    threshold_summary = {
        city: (
            round(t["low"], 4),
            round(t["high"], 4),
        )
        for city, t in thresholds.items()
        if city != "_default"
    }
    print(
        f"[OK] price_tier rows={n_tier_with_label} "
        f"good_deal rows={n_verdict} (with verdict), "
        f"{n_no_oof} rows without OOF; "
        f"per-city thresholds: {threshold_summary}; "
        f"written to {args.output_dir}"
    )
    logger.info(
        "Classification labels built (git=%s, tier_paths=%s, "
        "verdict_paths=%s)",
        _git_commit(),
        tier_paths,
        verdict_paths,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
