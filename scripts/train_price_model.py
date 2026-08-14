"""Train the v1 baseline price regression (Spec 13).

Consumes the Step 12 feature artifacts + Step 07 cleaned Parquet and
trains 6 candidate estimators (Linear, Ridge, Lasso, RF, GB, XGB) per
``transact_type`` (Sale + Rent). The winner per split is wrapped in a
full ``Pipeline(preprocessor → estimator)`` and serialized via joblib;
metrics land in ``models/metrics_v1.json``, Round 2/3 sections in
``data/processed/feature_selection_report.md``, and one row per
trained model in ``data/model_registry.csv``.

Invoked as ``python scripts/train_price_model.py`` from repo root.
Idempotent: re-running with identical data + git commit produces no
new CSV rows + no new pkl files (existing artifacts are overwritten in
place for the same version, which is intentional for v1 — a v2 future
spec writes ``v2`` artifacts and leaves v1 untouched).

Prerequisites:
    1. ``scripts/build_features.py`` (Step 12) must have been run at
       least once — produces ``feature_pipeline_v1.pkl`` +
       ``feature_list_v1.json``.
    2. ``data/processed/clean_listings.parquet`` (Step 07 output).

Logging uses stdlib ``logging`` only (Rules §5.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Repo root on sys.path so ``from ml.training import ...`` works whether
# this is run as a script or imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.features.feature_frame import build_feature_frame  # noqa: E402
from ml.features.persistence import load_feature_artifacts  # noqa: E402
from ml.features.split import FIXED_RANDOM_STATE, split_train_val_test  # noqa: E402
from ml.training.candidates import (  # noqa: E402
    CANDIDATE_MODELS,
    PRICE_MODEL_VERSION,
    RENT_MIN_ROWS,
    candidate_hyperparameters,
    make_estimator,
)
from ml.training.evaluation import evaluate_subset  # noqa: E402
from ml.training.persistence import (  # noqa: E402
    ARTIFACT_DIR,
    REGISTRY_CSV_PATH,
    append_model_registry,
    save_metrics,
    save_price_model,
)
from ml.training.report import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    append_round_2_3,
    feature_importance_table,
)
from ml.training.selection import select_winner  # noqa: E402

logger = logging.getLogger("train_price_model")

# Regex pattern used by the test suite to assert no PII-adjacent
# columns leak into logs (Rules §1.1). Pinned here so the test
# references the same constant.
_CONTACT_FIELD_REGEX = r"(contact|dealer|phone|email|photo|url|spid)"


def _git_commit(repo_root: Path) -> str:
    """Return the current git HEAD sha (12 chars), or 'unknown' on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_feature_frame_for_training(
    clean_df: pd.DataFrame, fitted_aggregator, fitted_preprocessor
) -> pd.DataFrame:
    """Build the full feature frame (16 fields + engineered + locality cols).

    Order matters: ``LocalityAggregator.transform`` must run BEFORE
    ``build_feature_frame`` so the 3 ``locality_*`` columns are
    present in the frame that ``build_feature_frame`` reorders.
    """
    with_locality = fitted_aggregator.transform(clean_df)
    feat = build_feature_frame(with_locality)
    return feat


def _train_one_transact_type(
    sub: pd.DataFrame,
    feat: pd.DataFrame,
    fitted_preprocessor,
    city_column: str = "city",
) -> dict | None:
    """Train all 6 candidates on a single ``transact_type`` subset.

    Returns the per-transact-type metrics dict (or ``None`` if the
    subset has fewer than ``RENT_MIN_ROWS`` rows after outlier
    filtering).
    """
    if len(sub) < RENT_MIN_ROWS:
        return {
            "skipped": True,
            "reason": f"n={len(sub)} < {RENT_MIN_ROWS}",
        }

    train_df, val_df, test_df = split_train_val_test(sub, target="price")

    # Align feature rows with split rows by integer position (sklearn
    # preserves order in train_test_split when shuffle=True +
    # random_state is set).
    train_idx = sub.index.get_indexer(train_df.index)
    val_idx = sub.index.get_indexer(val_df.index)
    test_idx = sub.index.get_indexer(test_df.index)

    X_train, X_val, X_test = (
        feat.iloc[train_idx].reset_index(drop=True),
        feat.iloc[val_idx].reset_index(drop=True),
        feat.iloc[test_idx].reset_index(drop=True),
    )
    y_train = np.log1p(train_df["price_inr"].to_numpy(dtype=float))
    y_val = np.log1p(val_df["price_inr"].to_numpy(dtype=float))
    y_test = np.log1p(test_df["price_inr"].to_numpy(dtype=float))
    city_test = X_test[city_column].reset_index(drop=True)

    candidate_results: dict[str, dict] = {}
    trained_pipes: dict[str, Pipeline] = {}

    for name in CANDIDATE_MODELS:
        pipe = Pipeline(
            [("preproc", fitted_preprocessor), ("est", make_estimator(name))]
        )
        res = evaluate_subset(
            pipe,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            city_test=city_test,
        )
        candidate_results[name] = res
        trained_pipes[name] = pipe
        logger.info(
            "    [%s] val_rmse=%.0f  val_r2=%.4f  val_mae=%.0f",
            name,
            res["val"]["rmse"],
            res["val"]["r2"],
            res["val"]["mae"],
        )

    winner_name = select_winner(candidate_results)
    winner_pipe = trained_pipes[winner_name]
    save_price_model(winner_pipe, transact_type=_detect_transact_type(sub))

    return {
        "candidates": candidate_results,
        "chosen_model": winner_name,
        "chosen_metrics": candidate_results[winner_name],
        "per_city_test": candidate_results[winner_name]["per_city_test"],
        "_trained_pipes": trained_pipes,  # used by main() for SHAP/perm report
    }


def _detect_transact_type(sub: pd.DataFrame) -> str:
    """Return the single ``transact_type`` value in ``sub`` (assumed unique)."""
    if "transact_type" not in sub.columns:
        return "Sale"  # safe default; tests always pass a transact_type column
    vals = sub["transact_type"].unique()
    if len(vals) != 1:
        raise ValueError(
            f"_detect_transact_type: subset has multiple values {vals!r}"
        )
    return str(vals[0])


def _compute_report_inputs(
    trained_pipes: dict[str, Pipeline],
    feature_names: list[str],
    X_val,
    y_val,
) -> dict:
    """Compute RF/GB/XGB impurity + permutation importance + SHAP ranking."""
    rf_imp = feature_importance_table(
        trained_pipes["random_forest"].named_steps["est"], feature_names
    )
    gb_imp = feature_importance_table(
        trained_pipes["gradient_boosting"].named_steps["est"], feature_names
    )
    xgb_imp = feature_importance_table(
        trained_pipes["xgboost"].named_steps["est"], feature_names
    )

    # Permutation importance on the validation slice (cheaper than full
    # train; n_repeats=10 per TRD §9).
    perm = permutation_importance(
        trained_pipes[
            "random_forest"  # any tree works; use RF as the canonical
        ],
        X_val,
        y_val,
        n_repeats=10,
        random_state=FIXED_RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    perm_pairs = list(zip(feature_names, perm.importances_mean))

    # SHAP for the chosen model — tree-based only (Rules §9).
    return {
        "rf_imp": rf_imp,
        "gb_imp": gb_imp,
        "xgb_imp": xgb_imp,
        "perm_pairs": perm_pairs,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the v1 baseline price regression."
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
        help="Path to the cleaned Parquet.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory holding feature_pipeline_v1.pkl + feature_list_v1.json "
        "and where price_model_*.pkl + metrics_v1.json are written.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Feature-selection report path to append Round 2/3 to.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=REGISTRY_CSV_PATH,
        help="Model registry CSV path.",
    )
    args = parser.parse_args()

    git_sha = _git_commit(_REPO_ROOT)
    logger.info("git_commit=%s", git_sha)
    logger.info("Loading %s", args.parquet)
    clean_df = pd.read_parquet(args.parquet)
    logger.info("Loaded %d rows.", len(clean_df))

    # Outlier filter (Rules §1.4): exclude from training, keep in analytics store.
    non_outlier = clean_df[clean_df["is_outlier"] == False]  # noqa: E712
    logger.info(
        "After outlier filter: %d rows (%.1f%% of full set).",
        len(non_outlier),
        100 * len(non_outlier) / max(len(clean_df), 1),
    )

    # Load Step 12 artifacts (Rules §2.4: never refit).
    preproc, agg, feature_names = load_feature_artifacts(
        version=PRICE_MODEL_VERSION, artifact_dir=args.artifact_dir
    )
    logger.info(
        "Loaded feature artifacts v%s (%d features).",
        PRICE_MODEL_VERSION,
        len(feature_names),
    )

    # Build feature frame for the non-outlier subset (LOO-safe because
    # the aggregator was fit on the Step 12 train rows, and we're
    # applying it here in transform mode — no refit).
    feat = _build_feature_frame_for_training(non_outlier, agg, preproc)
    logger.info("Built feature frame: %d rows × %d columns.", *feat.shape)

    payload = {
        "version": PRICE_MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "clean_listings.parquet",
        "git_commit": git_sha,
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "random_state": FIXED_RANDOM_STATE,
        },
        "sale": {},
        "rent": {},
    }

    for ttype in ("Sale", "Rent"):
        sub = non_outlier[non_outlier["transact_type"] == ttype]
        logger.info("=== %s subset: %d rows ===", ttype, len(sub))
        result = _train_one_transact_type(sub, feat, preproc)
        if result is None:
            payload[ttype.lower()] = {
                "skipped": True,
                "reason": "train_one returned None",
            }
            continue
        if result.get("skipped"):
            payload[ttype.lower()] = result
            logger.info(
                "Skipped %s pipeline: %s", ttype, result["reason"]
            )
            continue

        # Compute Round 2/3 report inputs once per transact type.
        winner_name = result["chosen_model"]
        trained = result.pop("_trained_pipes")

        # Need val slice for permutation importance. Rebuild from
        # train_df/val_df via split helper.
        train_df, val_df, _test_df = split_train_val_test(
            sub, target="price"
        )
        val_idx = sub.index.get_indexer(val_df.index)
        X_val = feat.iloc[val_idx].reset_index(drop=True)
        y_val = np.log1p(val_df["price_inr"].to_numpy(dtype=float))

        rep = _compute_report_inputs(
            trained, feature_names, X_val, y_val
        )

        # SHAP only for tree-model winners.
        winner_est = trained[winner_name].named_steps["est"]
        shap_pairs: list[tuple[str, float]] = []
        if hasattr(winner_est, "feature_importances_") and winner_name in {
            "random_forest",
            "gradient_boosting",
            "xgboost",
        }:
            try:
                import shap

                # Explain on the *preprocessed* matrix — TreeExplainer
                # expects the model's input, which is what the
                # preprocessor emits.
                preproc_matrix = trained[winner_name].named_steps[
                    "preproc"
                ].transform(X_val)
                explainer = shap.TreeExplainer(winner_est)
                shap_values = explainer.shap_values(preproc_matrix)
                mean_abs = np.abs(shap_values).mean(axis=0)
                shap_pairs = list(zip(feature_names, mean_abs))
                shap_pairs.sort(key=lambda kv: kv[1], reverse=True)
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "SHAP computation failed (%s); skipping.", e
                )
        else:
            logger.warning(
                "Winner %s is non-tree; SHAP ranking skipped.",
                winner_name,
            )

        append_round_2_3(
            report_path=args.report_path,
            rf_importances=rep["rf_imp"],
            gb_importances=rep["gb_imp"],
            xgb_importances=rep["xgb_imp"],
            perm_importances=rep["perm_pairs"],
            shap_importances=shap_pairs,
            winner_name=winner_name,
            chosen_metrics=result["chosen_metrics"],
        )

        # Registry row — one per trained transact type.
        test_metrics = result["chosen_metrics"]["test"]
        append_model_registry(
            {
                "model_name": f"price_model_{ttype.lower()}",
                "version": PRICE_MODEL_VERSION,
                "training_dataset_version": "clean_listings.parquet",
                "git_commit": git_sha,
                "training_date": datetime.now(timezone.utc).isoformat(),
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
                "hyperparameters": json.dumps(
                    candidate_hyperparameters(winner_name), default=str
                ),
                "feature_hash": hashlib.sha1(
                    "".join(feature_names).encode()
                ).hexdigest()[:16],
            },
            csv_path=args.registry_csv,
        )

        # Drop _trained_pipes from payload (not serializable, large).
        payload[ttype.lower()] = {
            "candidates": result["candidates"],
            "chosen_model": winner_name,
            "chosen_metrics": result["chosen_metrics"],
            "per_city_test": result["per_city_test"],
        }

    save_metrics(payload, artifact_dir=args.artifact_dir)

    # Summary log line per spec DoD §5.
    summary = {
        t: (
            payload[t].get("chosen_model")
            or payload[t].get("skipped")
            or "missing"
        )
        for t in ("sale", "rent")
    }
    logger.info("DONE. %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
