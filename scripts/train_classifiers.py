"""CLI: train the HousingIQ Classification models (Spec 23).

Reads:
    data/processed/classifier/feature_frame.parquet          (Spec 22)
    data/processed/classifier/price_tier_labels.parquet      (Spec 21)
    data/processed/classifier/good_deal_labels.parquet       (Spec 21)

Writes:
    models/good_deal_classifier_v{n}.pkl
    models/price_tier_classifier_v{n}.pkl
    models/classifier_preprocessor_v{n}.pkl
    models/classification_training_report.md    (append)

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

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from ml.classification import (  # noqa: E402
    GOOD_DEAL_LABELS,
    PRICE_TIER_LABELS,
    evaluate_classifier,
    train_good_deal_classifier,
    train_price_tier_classifier,
    write_training_report,
)
from ml.features.locality_aggregator import LocalityAggregator  # noqa: E402
from ml.features.split import split_train_val_test  # noqa: E402

logger = logging.getLogger("train_classifiers")


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
            "Train the HousingIQ Classification models "
            "(good_deal_verdict + price_tier). Reads Spec 21/22 "
            "artifacts, trains both classifiers with 70/15/15 "
            "stratified split, serializes versioned .pkl artifacts, "
            "and appends a run section to classification_training_report.md. "
            "Exit 0 on success; 1 on hard failure."
        )
    )
    p.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to classifier feature_frame.parquet (Spec 22 output).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "Path to directory containing price_tier_labels.parquet "
            "and good_deal_labels.parquet (Spec 21 output)."
        ),
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_MODELS_DIR",
                str(_REPO_ROOT / "models"),
            )
        ),
        help="Output directory for .pkl model artifacts.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_CLASSIFICATION_REPORT",
                str(_REPO_ROOT / "models" / "classification_training_report.md"),
            )
        ),
        help="Append-only Markdown report path.",
    )
    p.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version number (e.g., 1 -> good_deal_classifier_v1.pkl).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42).",
    )
    return p.parse_args()


def _load_and_join(
    features_path: Path,
    labels_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features and labels; join on listing_id; return (X, y_good_deal, y_price_tier)."""
    logger.info("Loading features from %s", features_path)
    features = pd.read_parquet(features_path)

    tier_path = labels_dir / "price_tier_labels.parquet"
    good_path = labels_dir / "good_deal_labels.parquet"

    if not tier_path.exists():
        raise FileNotFoundError(f"price_tier_labels.parquet not found at {tier_path}")
    if not good_path.exists():
        raise FileNotFoundError(f"good_deal_labels.parquet not found at {good_path}")

    logger.info("Loading labels from %s and %s", tier_path, good_path)
    tier_df = pd.read_parquet(tier_path)
    good_df = pd.read_parquet(good_path)

    # Both label frames should have listing_id + the target column
    required_tier = {"listing_id", "price_tier"}
    required_good = {"listing_id", "good_deal_verdict"}
    missing_tier = required_tier - set(tier_df.columns)
    missing_good = required_good - set(good_df.columns)
    if missing_tier:
        raise ValueError(f"price_tier_labels missing columns: {missing_tier}")
    if missing_good:
        raise ValueError(f"good_deal_labels missing columns: {missing_good}")

    # Join labels on listing_id
    labels = tier_df[["listing_id", "price_tier"]].merge(
        good_df[["listing_id", "good_deal_verdict"]], on="listing_id", how="inner"
    )

    # Join features + labels on listing_id
    if "listing_id" not in features.columns:
        raise ValueError("feature_frame missing 'listing_id' column")

    merged = features.merge(labels, on="listing_id", how="inner")
    logger.info("Joined features + labels: %d rows", len(merged))

    # Split X and y
    X = merged.drop(columns=["price_tier", "good_deal_verdict"])
    y_tier = merged["price_tier"].astype(int)
    y_good = merged["good_deal_verdict"].astype(int)

    # Validate no NaN in labels
    if y_tier.isna().any():
        raise ValueError("price_tier labels contain NaN")
    if y_good.isna().any():
        raise ValueError("good_deal_verdict labels contain NaN")

    # Validate label ranges
    invalid_tier = y_tier[~y_tier.isin(range(len(PRICE_TIER_LABELS)))]
    if len(invalid_tier) > 0:
        raise ValueError(f"price_tier contains invalid values: {invalid_tier.unique()}")
    invalid_good = y_good[~y_good.isin(range(len(GOOD_DEAL_LABELS)))]
    if len(invalid_good) > 0:
        raise ValueError(f"good_deal_verdict contains invalid values: {invalid_good.unique()}")

    logger.info(
        "Class balance (price_tier): %s",
        y_tier.value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Class balance (good_deal_verdict): %s",
        y_good.value_counts().sort_index().to_dict(),
    )

    return X, y_good, y_tier


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    # Validate input paths
    if not args.features.exists():
        logger.error("Features file not found: %s", args.features)
        print(f"[FAIL] features_missing at {args.features}")
        return 1
    if not args.labels.exists():
        logger.error("Labels directory not found: %s", args.labels)
        print(f"[FAIL] labels_dir_missing at {args.labels}")
        return 1

    try:
        # ---- Load and join ----
        X, y_good_deal, y_price_tier = _load_and_join(args.features, args.labels)

        # ---- Split: 70/15/15 stratified on good_deal_verdict (primary target) ----
        # Use the same fixed split helper as the price model
        # We need to create a temporary DataFrame with the target for stratification
        temp_df = X.copy()
        temp_df["good_deal_verdict"] = y_good_deal
        temp_df["price_tier"] = y_price_tier

        logger.info("Applying 70/15/15 stratified split (random_state=%d)...", args.random_state)
        train_df, val_df, test_df = split_train_val_test(temp_df, target="good_deal_verdict")

        # Extract X and y for each split
        X_train = train_df.drop(columns=["good_deal_verdict", "price_tier"])
        y_good_train = train_df["good_deal_verdict"]
        y_tier_train = train_df["price_tier"]

        X_val = val_df.drop(columns=["good_deal_verdict", "price_tier"])
        y_good_val = val_df["good_deal_verdict"]
        y_tier_val = val_df["price_tier"]

        X_test = test_df.drop(columns=["good_deal_verdict", "price_tier"])
        y_good_test = test_df["good_deal_verdict"]
        y_tier_test = test_df["price_tier"]

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # ---- Fit LocalityAggregator on training data (for target encoding) ----
        logger.info("Fitting LocalityAggregator on training data...")
        locality_agg = LocalityAggregator()
        locality_agg.fit(train_df)

        # ---- Train Good Deal Verdict Classifier (Primary) ----
        logger.info("=" * 60)
        logger.info("Training GOOD_DEAL_VERDICT classifier (primary)...")
        logger.info("=" * 60)
        good_deal_pipe, good_deal_val_metrics = train_good_deal_classifier(
            X_train=X_train,
            y_train=y_good_train,
            X_val=X_val,
            y_val=y_good_val,
            locality_aggregator=locality_agg,
            class_weight="balanced",
            random_state=args.random_state,
        )
        good_deal_best_model = getattr(good_deal_pipe, "best_model_name_", "Unknown")

        # ---- Train Price Tier Classifier (Secondary) ----
        logger.info("=" * 60)
        logger.info("Training PRICE_TIER classifier (secondary)...")
        logger.info("=" * 60)
        price_tier_pipe, price_tier_val_metrics = train_price_tier_classifier(
            X_train=X_train,
            y_train=y_tier_train,
            X_val=X_val,
            y_val=y_tier_val,
            locality_aggregator=locality_agg,
            class_weight="balanced",
            random_state=args.random_state,
        )
        price_tier_best_model = getattr(price_tier_pipe, "best_model_name_", "Unknown")

        # ---- Evaluate on Test Set (held-out) ----
        logger.info("=" * 60)
        logger.info("Evaluating on TEST SET (held-out)...")
        logger.info("=" * 60)

        good_deal_test_metrics = evaluate_classifier(
            good_deal_pipe, X_test, y_good_test, GOOD_DEAL_LABELS, primary_class_index=0
        )
        price_tier_test_metrics = evaluate_classifier(
            price_tier_pipe, X_test, y_tier_test, PRICE_TIER_LABELS, primary_class_index=0
        )

        logger.info(
            "Good Deal Test: accuracy=%.4f, macro_f1=%.4f, Good Deal recall=%.4f",
            good_deal_test_metrics["accuracy"],
            good_deal_test_metrics["macro_f1"],
            good_deal_test_metrics["primary_class_recall"],
        )
        logger.info(
            "Price Tier Test: accuracy=%.4f, macro_f1=%.4f",
            price_tier_test_metrics["accuracy"],
            price_tier_test_metrics["macro_f1"],
        )

        # ---- Serialize Artifacts ----
        args.models_dir.mkdir(parents=True, exist_ok=True)

        good_deal_path = args.models_dir / f"good_deal_classifier_v{args.version}.pkl"
        price_tier_path = args.models_dir / f"price_tier_classifier_v{args.version}.pkl"
        preprocessor_path = args.models_dir / f"classifier_preprocessor_v{args.version}.pkl"

        logger.info("Saving artifacts...")
        joblib.dump(good_deal_pipe, good_deal_path)
        joblib.dump(price_tier_pipe, price_tier_path)

        # Extract the fitted preprocessor from the pipeline (both share the same preprocessor)
        # The preprocessor is the first step in the pipeline
        fitted_preprocessor = good_deal_pipe.named_steps["preprocessor"]
        joblib.dump(fitted_preprocessor, preprocessor_path)

        logger.info("Saved: %s", good_deal_path)
        logger.info("Saved: %s", price_tier_path)
        logger.info("Saved: %s", preprocessor_path)

        # ---- Write Report ----
        from datetime import date

        training_date = date.today().isoformat()

        # Compute class balances for the report
        class_balance_good = {
            GOOD_DEAL_LABELS[i]: int((y_good_deal == i).sum())
            for i in range(len(GOOD_DEAL_LABELS))
        }
        class_balance_tier = {
            PRICE_TIER_LABELS[i]: int((y_price_tier == i).sum())
            for i in range(len(PRICE_TIER_LABELS))
        }

        # Get feature count from fitted preprocessor output
        feature_count = len(fitted_preprocessor.get_feature_names_out())

        write_training_report(
            report_path=args.report,
            model_version=args.version,
            good_deal_metrics=good_deal_test_metrics,
            price_tier_metrics=price_tier_test_metrics,
            good_deal_best_model=good_deal_best_model,
            price_tier_best_model=price_tier_best_model,
            train_rows=len(X_train),
            val_rows=len(X_val),
            test_rows=len(X_test),
            feature_count=feature_count,
            class_balance_good_deal=class_balance_good,
            class_balance_price_tier=class_balance_tier,
            training_date=training_date,
        )
        logger.info("Report appended to %s", args.report)

        # ---- One-line [OK] summary ----
        print(
            f"[OK] good_deal={good_deal_best_model} "
            f"(test_acc={good_deal_test_metrics['accuracy']:.4f}, "
            f"GoodDeal_recall={good_deal_test_metrics['primary_class_recall']:.4f}); "
            f"price_tier={price_tier_best_model} "
            f"(test_acc={price_tier_test_metrics['accuracy']:.4f}, "
            f"macro_f1={price_tier_test_metrics['macro_f1']:.4f}); "
            f"v{args.version} artifacts in {args.models_dir}; "
            f"report at {args.report}"
        )
        logger.info(
            "Training complete (git=%s, version=%d)",
            _git_commit(),
            args.version,
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        print(f"[FAIL] training_raised: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())