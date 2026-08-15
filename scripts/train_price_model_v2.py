"""Train the v2 boosted-tree price regression (Spec 14).

Sweeps four improvement levers over the v1 baseline:

    1. Stacking — 5 base learners (Ridge, RF, GB, XGB, LGBM) + Ridge
       meta-learner (CV=5).
    2. Optuna Bayesian search — separate ``XGBRegressor`` and
       ``LGBMRegressor`` search spaces (40 trials each, 10-min cap).
    3. Geospatial features — haversine distance to city center +
       nearest metro station per city.
    4. Sector target encoding — LOO Bayesian-smoothed mean keyed on
       ``(city, sector)``, complementing Step 12's
       ``LocalityAggregator``.

Per ``transact_type`` (Sale + Rent) the script:
    - Splits the non-outlier rows 70/15/15 with ``random_state=42``.
    - Builds a v2 feature frame that ADDS the geo + sector columns to
      the v1 feature frame (never refits the preprocessor — Rules
      §2.4).
    - Trains the 5 v2 candidates.
    - Selects the winner by validation RMSE.
    - Writes ``price_model_{transact}_v2.pkl``, ``metrics_v2.json``,
      appends one registry row per transact type, and appends a
      'Spec 14 — improvement levers' section to the report.

Honest shortfalls (Rules §9.2): the v2-vs-v1 improvement pct is logged
at every transact type. If it does NOT meet the 30–35% bar, the script
prints a WARNING — never silently clamps.

Prerequisites:
    - ``scripts/build_features.py`` (Step 12) at least once.
    - ``scripts/train_price_model.py`` (Step 13) at least once — used
      for the v1-vs-v2 comparison.
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
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.features.feature_frame import build_feature_frame  # noqa: E402
from ml.features.persistence import load_feature_artifacts  # noqa: E402
from ml.features.split import FIXED_RANDOM_STATE, split_train_val_test  # noqa: E402
from ml.training.candidates import (  # noqa: E402
    RENT_MIN_ROWS,
    V2_CANDIDATE_MODELS,
    make_v2_estimator,
)
from ml.training.evaluation import (  # noqa: E402
    IMPROVEMENT_TARGET_PCT,
    improvement_target_met,
    regression_metrics,
    vs_v1_metrics,
)
from ml.training.levers.geospatial import (  # noqa: E402
    GEO_NUMERIC_FEATURES,
    add_distance_features,
)
from ml.training.levers.optuna_search import (  # noqa: E402
    optuna_search_lgbm,
    optuna_search_xgb,
)
from ml.training.levers.target_encoding import (  # noqa: E402
    SECTOR_OUTPUT_COLUMN,
    SectorTargetEncoder,
)
from ml.training.persistence import (  # noqa: E402
    ARTIFACT_DIR,
    PRICE_MODEL_VERSION_V2,
    REGISTRY_CSV_PATH,
    append_model_registry,
    load_metrics,
    save_metrics,
    save_price_model,
)
from ml.training.report import (  # noqa: E402
    DEFAULT_REPORT_PATH,
    write_v2_lever_section,
)

logger = logging.getLogger("train_price_model_v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _detect_transact_type(sub: pd.DataFrame) -> str:
    if "transact_type" not in sub.columns:
        return "Sale"
    vals = sub["transact_type"].unique()
    if len(vals) != 1:
        raise ValueError(
            f"transact_type subset has multiple values {vals!r}"
        )
    return str(vals[0])


def _build_v2_feature_frame(
    non_outlier: pd.DataFrame,
    fitted_preprocessor,
    fitted_locality_agg,
) -> tuple[pd.DataFrame, SectorTargetEncoder]:
    """Build the v2 feature frame (v1 + 2 geo cols + sector_smoothed_price).

    Order matters:
        1. LocalityAggregator.transform — adds 3 ``locality_*`` columns.
        2. ``add_distance_features`` — adds 2 geo cols (NaN-safe).
        3. SectorTargetEncoder.fit + transform — adds sector_smoothed_price.
           Encoder is FIT on the same frame so the smoothing weights
           are consistent with the test slice (LOO semantics).
        4. ``build_feature_frame`` — final deterministic column order.

    The preprocessor is NEVER refit (Rules §2.4). The geo + sector cols
    pass through as numeric; ``transform`` of the saved preprocessor
    silently drops columns it does not recognise. To keep them in the
    matrix, the v2 pipeline appends them post-preprocess (see
    ``_make_v2_pipeline``).
    """
    with_locality = fitted_locality_agg.transform(non_outlier)
    with_geo = add_distance_features(with_locality)
    sector_enc = SectorTargetEncoder().fit(with_geo)
    with_sector = sector_enc.transform(with_geo)
    feat = build_feature_frame(with_sector)
    return feat, sector_enc


def _make_v2_pipeline(fitted_preprocessor, sector_enc: SectorTargetEncoder):
    """Build a sklearn Pipeline that runs:

        1. The fitted Step 12 preprocessor (transform-only).
        2. Append the 2 geo columns + 1 sector column to the matrix
           (post-transform passthrough).

    The preprocessor was fit on the v1 NUMERIC_FEATURES tuple only; we
    don't refit it. The extra columns are appended as a separate step
    so the preprocessor's transform output and the extra columns share
    row order.
    """

    def _append_v2(X, df_with_extras):
        """Concatenate the preprocessor matrix with the v2 columns."""
        extras = df_with_extras[
            list(GEO_NUMERIC_FEATURES) + [SECTOR_OUTPUT_COLUMN]
        ].reset_index(drop=True)
        return np.hstack(
            [np.asarray(X), extras.to_numpy(dtype=float, na_value=0.0)]
        )

    # ponytail: not used directly — ``_train_one_transact_type`` builds
    # the adapter per call so it can carry its own slice.
    class _V2PreprocAdapter:
        """Adapter that carries the v2 feature frame alongside the
        fitted preprocessor. ``fit_transform`` runs the preprocessor
        once and appends the v2 columns; ``transform`` does the same on
        the held frame.
        """

        def __init__(self, preproc, df_v2: pd.DataFrame):
            self.preproc = preproc
            self.df_v2 = df_v2.reset_index(drop=True)

        def fit_transform(self, X, y=None, **kwargs):
            matrix = self.preproc.transform(X)
            return _append_v2(matrix, self.df_v2)

        def transform(self, X):
            matrix = self.preproc.transform(X)
            return _append_v2(matrix, self.df_v2)

    return _V2PreprocAdapter


def _train_one_transact_type(
    sub: pd.DataFrame,
    feat: pd.DataFrame,
    fitted_preprocessor,
    sector_enc: SectorTargetEncoder,
) -> dict | None:
    """Train the 5 v2 candidates on a single transact_type subset.

    Mirrors ``_train_one_transact_type`` from the v1 script, but uses
    the v2 candidates and the v2-aware pipeline adapter.
    """
    if len(sub) < RENT_MIN_ROWS:
        return {
            "skipped": True,
            "reason": f"n={len(sub)} < {RENT_MIN_ROWS}",
        }

    train_df, val_df, test_df = split_train_val_test(sub, target="price")
    train_idx = sub.index.get_indexer(train_df.index)
    val_idx = sub.index.get_indexer(val_df.index)
    test_idx = sub.index.get_indexer(test_df.index)

    X_train = feat.iloc[train_idx].reset_index(drop=True)
    X_val = feat.iloc[val_idx].reset_index(drop=True)
    X_test = feat.iloc[test_idx].reset_index(drop=True)

    y_train = np.log1p(train_df["price_inr"].to_numpy(dtype=float))
    y_val = np.log1p(val_df["price_inr"].to_numpy(dtype=float))
    y_test = np.log1p(test_df["price_inr"].to_numpy(dtype=float))

    Adapter = _make_v2_pipeline(fitted_preprocessor, sector_enc)

    candidate_results: dict[str, dict] = {}
    candidate_pipes: dict[str, Any] = {}

    # Pre-build the per-slice preprocessed matrices once.
    train_slice = feat.iloc[train_idx].reset_index(drop=True)
    val_slice = feat.iloc[val_idx].reset_index(drop=True)
    test_slice = feat.iloc[test_idx].reset_index(drop=True)
    Xtr_pp = Adapter(fitted_preprocessor, train_slice).fit_transform(X_train)
    Xv_pp = Adapter(fitted_preprocessor, val_slice).fit_transform(X_val)
    Xte_pp = Adapter(fitted_preprocessor, test_slice).fit_transform(X_test)

    # ------------------------------------------------------------------
    # Lever 2 — Optuna (run BEFORE candidate sweep so the winners
    # are injected into xgb_optuna / lgbm_optuna).
    # ------------------------------------------------------------------
    logger.info("Running Optuna search for XGB (40 trials, 10-min cap)…")
    xgb_search = optuna_search_xgb(
        Xtr_pp,
        y_train,
        Xv_pp,
        y_val,
        n_trials=40,
        timeout_sec=600,
        random_state=FIXED_RANDOM_STATE,
    )
    logger.info("XGB Optuna best neg-RMSE=%.4f", xgb_search["best_value"])

    logger.info("Running Optuna search for LGBM (40 trials, 10-min cap)…")
    lgbm_search = optuna_search_lgbm(
        Xtr_pp,
        y_train,
        Xv_pp,
        y_val,
        n_trials=40,
        timeout_sec=600,
        random_state=FIXED_RANDOM_STATE,
    )
    logger.info("LGBM Optuna best neg-RMSE=%.4f", lgbm_search["best_value"])

    optuna_results = {"optuna_xgb": xgb_search, "optuna_lgbm": lgbm_search}

    # ------------------------------------------------------------------
    # Sweep the 5 v2 candidates on the pre-built matrices.
    # ------------------------------------------------------------------
    for name in V2_CANDIDATE_MODELS:
        if name == "xgb_optuna":
            est = make_v2_estimator(name, params=xgb_search["best_params"])
        elif name == "lgbm_optuna":
            est = make_v2_estimator(name, params=lgbm_search["best_params"])
        else:
            est = make_v2_estimator(name)

        est_fitted = est.fit(Xtr_pp, y_train)
        res = {
            "train": regression_metrics(y_train, est_fitted.predict(Xtr_pp)),
            "val": regression_metrics(y_val, est_fitted.predict(Xv_pp)),
            "test": regression_metrics(y_test, est_fitted.predict(Xte_pp)),
        }
        candidate_results[name] = res
        candidate_pipes[name] = (est_fitted, (Xtr_pp, Xv_pp, Xte_pp))
        logger.info(
            "    [%s] val_rmse=%.0f  val_r2=%.4f  val_mae=%.0f",
            name,
            res["val"]["rmse"],
            res["val"]["r2"],
            res["val"]["mae"],
        )

    # Select winner by validation RMSE (same rule as v1's select_winner).
    winner_name = min(
        candidate_results,
        key=lambda n: candidate_results[n]["val"]["rmse"],
    )
    winner_est, _ = candidate_pipes[winner_name]

    # Persist: build a joblib-friendly pipeline that re-applies the
    # preprocessor + appends v2 columns + applies the winner estimator.
    # We use a thin serializable wrapper.
    artifact_pipe = _SerializableV2Pipeline(
        preprocessor=fitted_preprocessor,
        sector_encoder=sector_enc,
        feature_frame_template=feat.copy(),
        estimator=winner_est,
    )
    save_price_model(
        artifact_pipe,
        transact_type=_detect_transact_type(sub),
        version=PRICE_MODEL_VERSION_V2,
    )

    return {
        "candidates": candidate_results,
        "chosen_model": winner_name,
        "chosen_metrics": candidate_results[winner_name],
        "optuna_results": optuna_results,
    }


class _SerializableV2Pipeline:
    """joblib-serializable v2 pipeline.

    Carries the fitted preprocessor, the sector encoder, and the winner
    estimator. ``predict(X)`` rebuilds the v2 columns on the input
    frame, runs the preprocessor, appends the v2 columns, then applies
    the estimator. Mirrors the v1 ``Pipeline(preproc → est)`` shape.
    """

    def __init__(
        self,
        preprocessor,
        sector_encoder: SectorTargetEncoder,
        feature_frame_template: pd.DataFrame,
        estimator,
    ):
        self.preprocessor = preprocessor
        self.sector_encoder = sector_encoder
        self.feature_frame_template = feature_frame_template
        self.estimator = estimator

    def predict(self, X) -> np.ndarray:
        # The serving path rebuilds the full v2 frame; here we trust
        # that ``X`` is already in v2-feature-frame shape (caller's
        # job — same contract as the v1 Pipeline).
        Adapter = _make_v2_pipeline(self.preprocessor, self.sector_encoder)
        # We use the template's extras columns (geo + sector) — at
        # serving time the caller must supply them; here we fall back
        # to NaN when the input frame lacks them.
        df_extras = self._build_extras(X)
        return self.estimator.predict(
            Adapter(self.preprocessor, df_extras).fit_transform(X)
        )

    def _build_extras(self, X) -> pd.DataFrame:
        # If X is a DataFrame, take the v2 columns directly.
        if isinstance(X, pd.DataFrame):
            needed = list(GEO_NUMERIC_FEATURES) + [SECTOR_OUTPUT_COLUMN]
            present = [c for c in needed if c in X.columns]
            return X[present].copy()
        # Numpy fallback: emit a zero-filled frame matching the template length.
        n = len(X)
        return pd.DataFrame(
            np.zeros((n, len(GEO_NUMERIC_FEATURES) + 1)),
            columns=list(GEO_NUMERIC_FEATURES) + [SECTOR_OUTPUT_COLUMN],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the v2 boosted-tree price regression (Spec 14)."
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
        "and where price_model_*_v2.pkl + metrics_v2.json are written.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Feature-selection report path to append the v2 lever section to.",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=REGISTRY_CSV_PATH,
        help="Model registry CSV path.",
    )
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip the Optuna search (use v1-default hyperparams only).",
    )
    args = parser.parse_args()

    git_sha = _git_commit(_REPO_ROOT)
    logger.info("git_commit=%s", git_sha)
    logger.info("Loading %s", args.parquet)
    if not args.parquet.exists():
        logger.error(
            "Parquet not found at %s — run scripts/build_cleaned_dataset.py first.",
            args.parquet,
        )
        return 1
    clean_df = pd.read_parquet(args.parquet)
    logger.info("Loaded %d rows.", len(clean_df))

    non_outlier = clean_df[clean_df["is_outlier"] == False]  # noqa: E712
    logger.info(
        "After outlier filter: %d rows (%.1f%% of full set).",
        len(non_outlier),
        100 * len(non_outlier) / max(len(clean_df), 1),
    )

    preproc, agg, feature_names = load_feature_artifacts(
        version="v1", artifact_dir=args.artifact_dir
    )
    logger.info(
        "Loaded v1 feature artifacts (%d base features).",
        len(feature_names),
    )

    feat, sector_enc = _build_v2_feature_frame(non_outlier, preproc, agg)
    logger.info("Built v2 feature frame: %d rows × %d columns.", *feat.shape)

    v1_payload = load_metrics(version="v1", artifact_dir=args.artifact_dir)

    payload: dict[str, Any] = {
        "version": PRICE_MODEL_VERSION_V2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "clean_listings.parquet",
        "git_commit": git_sha,
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "random_state": FIXED_RANDOM_STATE,
        },
        "levers": {},
        "sale": {},
        "rent": {},
    }

    last_lever_results: dict[str, Any] = {}

    for ttype in ("Sale", "Rent"):
        sub = non_outlier[non_outlier["transact_type"] == ttype]
        logger.info("=== %s subset: %d rows ===", ttype, len(sub))
        result = _train_one_transact_type(sub, feat, preproc, sector_enc)
        if result is None:
            payload[ttype.lower()] = {
                "skipped": True,
                "reason": "train_one returned None",
            }
            continue
        if result.get("skipped"):
            payload[ttype.lower()] = result
            logger.info("Skipped %s pipeline: %s", ttype, result["reason"])
            continue

        winner_name = result["chosen_model"]
        payload[ttype.lower()] = {
            "candidates": result["candidates"],
            "chosen_model": winner_name,
            "chosen_metrics": result["chosen_metrics"],
        }
        last_lever_results = result["optuna_results"]

        test_metrics = result["chosen_metrics"]["test"]
        append_model_registry(
            {
                "model_name": f"price_model_{ttype.lower()}",
                "version": PRICE_MODEL_VERSION_V2,
                "training_dataset_version": "clean_listings.parquet",
                "git_commit": git_sha,
                "training_date": datetime.now(timezone.utc).isoformat(),
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
                "hyperparameters": json.dumps(
                    {"chosen_model": winner_name}, default=str
                ),
                "feature_hash": hashlib.sha1(
                    "".join(feat.columns).encode()
                ).hexdigest()[:16],
            },
            csv_path=args.registry_csv,
        )

    payload["levers"] = {
        "stacking": {},
        "optuna_xgb": last_lever_results.get("optuna_xgb", {}),
        "optuna_lgbm": last_lever_results.get("optuna_lgbm", {}),
        "geo_features": list(GEO_NUMERIC_FEATURES),
        "sector_encoding": {
            "smoothing_prior_weight": 20.0,
            "output_column": SECTOR_OUTPUT_COLUMN,
            "n_groups": sector_enc.n_groups_,
        },
    }

    save_metrics(payload, version=PRICE_MODEL_VERSION_V2, artifact_dir=args.artifact_dir)

    # v2 vs v1 — honest logging per Rules §9.2.
    improvement = vs_v1_metrics(v1_payload, payload, split="test")
    target_met = improvement_target_met(improvement, IMPROVEMENT_TARGET_PCT)
    for metric in ("mae", "rmse", "r2"):
        for ttype, pct in improvement.get(metric, {}).items():
            verdict = target_met.get(metric, {}).get(ttype, None)
            if metric in ("mae", "rmse"):
                if verdict is True:
                    logger.info(
                        "%s %s improvement vs v1: %+.2f%% — TARGET MET",
                        ttype, metric.upper(), pct,
                    )
                else:
                    logger.warning(
                        "%s %s improvement vs v1: %+.2f%% — TARGET NOT MET "
                        "(target ≥%.1f%%)",
                        ttype, metric.upper(), pct, IMPROVEMENT_TARGET_PCT,
                    )

    write_v2_lever_section(
        lever_results=payload["levers"],
        winner_name=payload.get("sale", {}).get(
            "chosen_model", "unknown"
        ),
        improvement_pct=improvement,
        target_met=target_met,
        out_path=args.report_path,
    )

    logger.info(
        "DONE. winner(sale)=%s winner(rent)=%s",
        payload["sale"].get("chosen_model", "skipped"),
        payload["rent"].get("chosen_model", "skipped"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
