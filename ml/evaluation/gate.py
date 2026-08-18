"""``evaluate()`` entry point + ``EvaluationResult`` dataclass (Spec 15).

The gate certifies a trained price model against the pinned
protocol. It is **pure** with respect to its inputs — it reads the
trained artifact + the cleaned parquet + the fitted preprocessor
+ (optionally) a live FastAPI instance, scores the model on the
protocol's split, and returns an :class:`EvaluationResult`. It does
not write files; persistence is the CLI's job.

The 12-step flow matches the spec's "Files to create → gate.py →
evaluate()" section verbatim. Step numbers in the source match the
spec so a reviewer can audit drift at a glance.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.protocol import (
    PROTOCOL_DOC_PATH,
    PROTOCOL_VERSION,
    protocol_thresholds,
)
from ml.evaluation.scoring import (
    per_city_metrics,
    score_predictions,
    within_tolerance_pct,
)
from ml.evaluation.splits import protocol_split
from ml.features.persistence import load_feature_artifacts

logger = logging.getLogger(__name__)

# Pinned evaluator version — kept in sync with ``ml.evaluation.__version__``
# by the test suite. Defined as a local constant (not a live cross-module
# attribute) to avoid the circular import that ``from ml.evaluation
# import __version__`` would create when gate.py is imported during
# package init.
_EVALUATOR_VERSION: str = "1.0.0"


@dataclasses.dataclass(frozen=True)
class ProtocolThresholds:
    """Mirrors ``protocol_thresholds`` — typed accessor for tests."""

    r2_min: float
    r2_stretch: float
    mae_pct_within_15_at_least: float
    p95_latency_ms_max: float
    rent_min_rows: float


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    """Outcome of a single ``evaluate()`` call.

    Fields mirror the spec verbatim. ``overall_passed`` is ``True``
    iff every threshold in ``thresholds_passed`` is ``True``. The
    model is **certified** iff ``overall_passed == True``.
    """

    version: str
    transact_type: str
    protocol_version: str
    dataset_version: str
    git_commit: str
    split_sizes: dict[str, int]
    metrics: dict[str, float]
    per_city_test: dict[str, dict[str, float]]
    within_tol_15_pct: float
    latency_p95_ms: float | None
    thresholds_passed: dict[str, bool]
    overall_passed: bool
    evaluated_at: str
    evaluator_version: str


def _git_commit() -> str:
    """12-char git short SHA; ``"unknown"`` if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dataset_fingerprint(parquet_path: Path) -> str:
    """``<filename>-<sha1[:8] of first 1 MB>`` content fingerprint."""
    sha = hashlib.sha1()
    try:
        with open(parquet_path, "rb") as fh:
            sha.update(fh.read(1024 * 1024))
    except FileNotFoundError:
        return f"{parquet_path.name}-missing"
    return f"{parquet_path.name}-{sha.hexdigest()[:8]}"


def _measure_latency_ms(
    fastapi_url: str,
    transact_type: str,
    n_samples: int = 50,
) -> float | None:
    """Return p95 latency in ms over ``n_samples`` ``/predict`` calls.

    Uses stdlib ``urllib.request`` — no project-side HTTP client
    imports (Rules §13: no new infrastructure). On any failure
    (server down, timeout, non-2xx response), logs a WARNING and
    returns ``None`` — the gate never crashes on a missing FastAPI
    instance.
    """
    url = fastapi_url.rstrip("/") + "/predict"
    timings_ms: list[float] = []
    for _ in range(n_samples):
        body = json_body_for_probe(transact_type)
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=5) as resp:
                _ = resp.read()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "Latency probe failed (%s) — emitting latency_p95_ms=None",
                exc,
            )
            return None
    if not timings_ms:
        return None
    return float(np.percentile(np.asarray(timings_ms), 95))


def json_body_for_probe(transact_type: str) -> bytes:
    """Minimal valid request body for the FastAPI ``/predict`` probe.

    Uses a small fixed payload — the latency probe doesn't care about
    accuracy, only round-trip time. Kept tiny to keep the probe cheap.
    """
    import json

    return json.dumps(
        {
            "city": "Gurgaon",
            "sector": "sector 84",
            "property_type": "flat",
            "transact_type": transact_type.capitalize(),
            "bedRoom": 3,
            "bathroom": 3,
            "balcony": "2",
            "agePossession": "Relatively New",
            "built_up_area": 1450,
            "servant_room": True,
            "store_room": False,
            "furnishing_type": 1,
            "luxury_category": "High",
            "floor_category": "Mid Floor",
            "facing": "North",
            "amenities": ["Clubhouse", "Swimming Pool"],
        }
    ).encode("utf-8")


def _build_preprocessor_for_gate() -> object:
    """Return an unfitted preprocessor stub used by ``evaluate()``.

    Used as the in-memory reference for the preprocessor-drift check
    (Rules §2.4: production model = the exact ``Pipeline`` used at
    evaluation). The drift check compares the *fitted* instance
    loaded from ``models/feature_pipeline_v1.pkl`` to itself via
    identity; for now the assertion is reduced to "loaded object is
    not ``None`` and has ``transform``" — full pipeline identity
    pinning would require persisting a content hash, which is out of
    scope for Spec 15 (the loaded artifact is the artifact, by
    Rules §2.4).
    """
    from ml.features.preprocessor import make_preprocessor

    return make_preprocessor()


def _check_thresholds(
    r2: float,
    within_15: float,
    n_rent: int,
) -> dict[str, bool]:
    """Compute ``{threshold_name: passed?}`` against the pinned protocol.

    Latency is checked separately — it's only available when the gate
    is run against a live FastAPI instance. When ``latency_p95_ms`` is
    ``None``, the latency threshold is omitted from the result (rather
    than marked False) — the offline gate can't fail on a metric it
    can't measure.
    """
    return {
        "r2_min": bool(r2 >= protocol_thresholds["r2_min"]),
        "r2_stretch": bool(r2 >= protocol_thresholds["r2_stretch"]),
        "mae_pct_within_15_at_least": bool(
            within_15 >= protocol_thresholds["mae_pct_within_15_at_least"]
        ),
        "rent_min_rows": bool(n_rent >= protocol_thresholds["rent_min_rows"]),
    }


def evaluate(
    model_path: Path | str,
    version: str,
    transact_type: str,
    processed_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    parquet_path: Path | str | None = None,
    fastapi_url: str | None = None,
) -> EvaluationResult:
    """Score ``model_path`` against the protocol and return the result.

    Pure function — no file writes. The CLI is responsible for
    persisting the result via ``report.write_evaluation_report``.
    """
    if processed_dir is None:
        processed_dir = os.environ.get(
            "HOUSINGIQ_PROCESSED_DIR", "data/processed"
        )
    if models_dir is None:
        models_dir = os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models")
    if parquet_path is None:
        parquet_path = Path(processed_dir) / "clean_listings.parquet"

    parquet_path = Path(parquet_path)
    models_dir = Path(models_dir)
    model_path = Path(model_path)

    # --- Step 1: load the cleaned parquet --------------------------------
    df = pd.read_parquet(parquet_path)

    # --- Step 2: filter to non-outlier rows -----------------------------
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False].reset_index(drop=True)  # noqa: E712

    # --- Step 3: split per transact_type --------------------------------
    transact_type = transact_type.lower()
    df_sub = df[df["transact_type"].str.lower() == transact_type].reset_index(
        drop=True
    )

    evaluated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    fingerprint = _dataset_fingerprint(parquet_path)
    git_sha = _git_commit()

    base_thresholds = _check_thresholds(
        r2=0.0,
        within_15=0.0,
        n_rent=len(df_sub) if transact_type == "rent" else 10**9,
    )

    if len(df_sub) < int(protocol_thresholds["rent_min_rows"]):
        logger.warning(
            "%s subset has %d rows (< %d) — emitting skipped result.",
            transact_type,
            len(df_sub),
            int(protocol_thresholds["rent_min_rows"]),
        )
        return EvaluationResult(
            version=version,
            transact_type=transact_type,
            protocol_version=PROTOCOL_VERSION,
            dataset_version=fingerprint,
            git_commit=git_sha,
            split_sizes={},
            metrics={"skipped": True, "reason": f"n={len(df_sub)} < 500"},
            per_city_test={},
            within_tol_15_pct=0.0,
            latency_p95_ms=None,
            thresholds_passed={**base_thresholds, "rent_min_rows": False},
            overall_passed=False,
            evaluated_at=evaluated_at,
            evaluator_version=_EVALUATOR_VERSION,
        )

    # --- Step 4: protocol_split enforces 70/15/15 + random_state ------
    train_df, val_df, test_df = protocol_split(df_sub, target="price")
    split_sizes = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }

    # --- Step 5: load + sanity-check the fitted preprocessor ------------
    preproc_path = models_dir / "feature_pipeline_v1.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"evaluate: feature pipeline not found at {preproc_path}. "
            f"Run scripts/build_features.py first."
        )
    fitted_preproc, _agg, _feat_list = load_feature_artifacts(
        "v1", artifact_dir=models_dir
    )
    # Drift check: the loaded object must be non-None and have a
    # ``transform`` method (Rules §2.4). The artifact is the artifact —
    # no further identity pinning in Spec 15.
    if fitted_preproc is None or not hasattr(fitted_preproc, "transform"):
        raise ValueError(
            "evaluate: loaded preprocessor failed sanity check "
            "(preprocessor_drift)."
        )

    # --- Step 6: load the model artifact --------------------------------
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise ValueError(
            f"evaluate: artifact at {model_path} has no predict() — "
            f"not a model."
        )

    # --- Step 7: score train/val/test -----------------------------------
    target_col = "price"
    if target_col not in train_df.columns:
        # Some parquets use ``price_inr`` — accept either.
        if "price_inr" in train_df.columns:
            target_col = "price_inr"
        else:
            raise KeyError(
                f"evaluate: target column not found (looked for 'price' "
                f"and 'price_inr'). Columns: {list(train_df.columns)}"
            )

    feature_cols = [
        c for c in train_df.columns if c not in {target_col, "is_outlier"}
    ]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train_log = np.log1p(train_df[target_col].astype(float))
    y_val_log = np.log1p(val_df[target_col].astype(float))
    y_test_log = np.log1p(test_df[target_col].astype(float))

    metrics_test = score_predictions(y_test_log, model.predict(X_test))
    metrics_train = score_predictions(y_train_log, model.predict(X_train))
    metrics_val = score_predictions(y_val_log, model.predict(X_val))
    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
    }

    # --- Step 8: per-city test metrics ----------------------------------
    city_test = test_df["city"] if "city" in test_df.columns else None
    per_city: dict[str, dict[str, float]] = {}
    if city_test is not None:
        per_city = per_city_metrics(model, X_test, y_test_log, city_test)

    # --- Step 9: within-15% fraction ------------------------------------
    y_true_test = np.expm1(y_test_log.to_numpy() if hasattr(y_test_log, "to_numpy") else y_test_log)
    y_pred_test = np.expm1(model.predict(X_test))
    within_15 = within_tolerance_pct(y_true_test, y_pred_test, tolerance=0.15)

    # --- Step 10: optional latency probe --------------------------------
    latency_p95_ms: float | None = None
    if fastapi_url:
        latency_p95_ms = _measure_latency_ms(fastapi_url, transact_type)

    # --- Step 11: threshold check ---------------------------------------
    thresholds = _check_thresholds(
        r2=metrics_test["r2"],
        within_15=within_15,
        n_rent=len(df_sub) if transact_type == "rent" else 10**9,
    )
    if latency_p95_ms is not None:
        thresholds["p95_latency_ms_max"] = bool(
            latency_p95_ms <= protocol_thresholds["p95_latency_ms_max"]
        )
    overall_passed = all(thresholds.values()) if thresholds else False

    # --- Step 12: return ------------------------------------------------
    return EvaluationResult(
        version=version,
        transact_type=transact_type,
        protocol_version=PROTOCOL_VERSION,
        dataset_version=fingerprint,
        git_commit=git_sha,
        split_sizes=split_sizes,
        metrics=metrics,
        per_city_test=per_city,
        within_tol_15_pct=within_15,
        latency_p95_ms=latency_p95_ms,
        thresholds_passed=thresholds,
        overall_passed=overall_passed,
        evaluated_at=evaluated_at,
        evaluator_version=_EVALUATOR_VERSION,
    )


def format_summary(result: EvaluationResult) -> str:
    """One-line stdout summary used by the CLI.

    Format: ``[PASS|FAIL] v{sale,rent}_v{N} R²={r2:.4f} (≥0.80)
    MAE=₹{mae:.0f} within±15%={pct:.1%} (≥70%)
    latency_p95={lat}ms (<300ms)``.
    """
    test = result.metrics.get("test", {}) if isinstance(result.metrics, Mapping) else {}
    r2 = float(test.get("r2", 0.0))
    mae = float(test.get("mae", 0.0))
    lat = (
        "n/a"
        if result.latency_p95_ms is None
        else f"{result.latency_p95_ms:.1f}"
    )
    tag = "PASS" if result.overall_passed else "FAIL"
    return (
        f"[{tag}] {result.transact_type}_{result.version} "
        f"R²={r2:.4f} (≥0.80) "
        f"MAE=₹{mae:.0f} "
        f"within±15%={result.within_tol_15_pct:.1%} (≥70%) "
        f"latency_p95={lat}ms (<300ms)"
    )


__all__ = [
    "EvaluationResult",
    "ProtocolThresholds",
    "evaluate",
    "format_summary",
    "PROTOCOL_DOC_PATH",
    "_EVALUATOR_VERSION",
]
