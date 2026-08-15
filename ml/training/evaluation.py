"""Metric computation + per-model evaluation (Spec 13).

Pure-Python module. ``regression_metrics`` accepts log-space targets
(``np.log1p(price)``) and inverse-transforms via ``np.expm1`` before
scoring — this is the single source of truth for "train on log, report
on original scale" (TRD §6.4, Rules §2.1).
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

#: Small-sample threshold for per-city WARNINGs (Rules §8.5 — log but
#: still emit the row).
SMALL_CITY_TEST_ROWS: Final[int] = 30

#: MAPE epsilon guard. Avoids divide-by-zero on near-zero prices; no
#: expected post-cleaning (Step 07 drops ₹0 rows) but pinned anyway.
_MAPE_EPS: Final[float] = 1.0

#: Target improvement for the v2 boosted-tree model (Spec 14).
#: 30–35% MAE/RMSE reduction vs the v1 baseline — pinned at the
#: midpoint (32.5%). Used by ``improvement_target_met`` to flag a
#: shortfall honestly (Rules §9.2).
IMPROVEMENT_TARGET_PCT: Final[float] = 32.5


def regression_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """Return ``{r2, mae, rmse, mape}`` on the ORIGINAL ₹ scale.

    Inputs are expected in log-space (``np.log1p(price_inr)``). The
    function applies ``np.expm1`` to both true + predicted before
    computing metrics, so a model that learns ``log(price)`` is
    reported in ₹. ``MAPE`` is ``mean(|y_true - y_pred| /
    max(y_true, eps)) * 100`` — the ``eps`` guard avoids divide-by-zero.

    Pure function; no side effects.
    """
    y_true = np.expm1(np.asarray(y_true_log, dtype=float))
    y_pred = np.expm1(np.asarray(y_pred_log, dtype=float))

    denom = np.maximum(y_true, _MAPE_EPS)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
    }


def evaluate_subset(
    pipeline,
    X_train,
    y_train_log,
    X_val,
    y_val_log,
    X_test,
    y_test_log,
    city_test: pd.Series | None = None,
) -> dict:
    """Fit ``pipeline`` on train, score train/val/test, optionally per-city.

    Returns ``{"train": {...}, "val": {...}, "test": {...},
    "per_city_test": {...}}`` (the last key only if ``city_test`` is
    provided). Each metrics dict has the four ``regression_metrics``
    keys on the original ₹ scale.

    Pure with respect to ``X_*`` / ``y_*`` (the ``pipeline.fit`` call is
    the only mutation; caller should not reuse the pipeline afterwards).
    """
    pipeline.fit(X_train, y_train_log)
    out: dict = {
        "train": regression_metrics(y_train_log, pipeline.predict(X_train)),
        "val": regression_metrics(y_val_log, pipeline.predict(X_val)),
        "test": regression_metrics(y_test_log, pipeline.predict(X_test)),
    }
    if city_test is not None:
        out["per_city_test"] = per_city_metrics(
            pipeline, X_test, y_test_log, city_test
        )
    return out


def per_city_metrics(
    pipeline,
    X_test,
    y_test_log,
    city_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Slice test-set metrics by ``city_test`` (a ``pd.Series`` aligned
    with ``X_test``). Logs a WARNING for any city with fewer than
    ``SMALL_CITY_TEST_ROWS`` rows but still emits the row.
    """
    if not isinstance(city_test, pd.Series):
        raise TypeError(
            f"city_test must be a pd.Series, got {type(city_test).__name__}"
        )
    out: dict[str, dict[str, float]] = {}
    city_arr = city_test.reset_index(drop=True)
    X_reset = X_test.reset_index(drop=True)
    y_arr = np.asarray(y_test_log)

    for city in sorted(city_arr.unique()):
        mask = (city_arr == city).values
        n = int(mask.sum())
        if n == 0:
            continue
        if n < SMALL_CITY_TEST_ROWS:
            logger.warning(
                "City %s has only %d test rows (below %d) — "
                "per-city metrics may be noisy.",
                city,
                n,
                SMALL_CITY_TEST_ROWS,
            )
        out[str(city)] = regression_metrics(
            y_arr[mask], pipeline.predict(X_reset[mask])
        )
    return out


def vs_v1_metrics(
    v1_payload: dict, v2_metrics: dict, split: str = "test"
) -> dict[str, dict[str, float]]:
    """Return per-metric pct-improvement of v2 over v1 on a given split.

    ``v1_payload`` is the full v1 ``metrics_v1.json`` payload (the
    ``{"sale": {...}, "rent": {...}}`` shape). ``v2_metrics`` is a
    per-transact-type chosen-metrics dict shaped like
    ``{"train": {...}, "val": {...}, "test": {...}}`` (or a single
    flat metrics dict for an older API).

    Returns a nested dict ``{metric_name: {sale: pct, rent: pct}}``.
    A positive pct means v2 is **better** (lower error is good, higher
    R² is good). When v1 has no row for a transact type, that
    sub-key is omitted (the caller decides whether to WARN).

    Honest shortfalls (Rules §9.2): a NEGATIVE pct is left as-is —
    never silently clamped to zero.
    """
    out: dict[str, dict[str, float]] = {
        "r2": {},
        "mae": {},
        "rmse": {},
        "mape": {},
    }
    # Metric semantics: lower-is-better (negative pct = improvement).
    _LOWER_BETTER = {"mae", "rmse", "mape"}
    for ttype in ("sale", "rent"):
        block = v1_payload.get(ttype, {})
        chosen = block.get("chosen_metrics") if isinstance(block, dict) else None
        if not chosen:
            logger.warning(
                "vs_v1_metrics: v1 payload missing chosen_metrics for %s",
                ttype,
            )
            continue
        v1_split = chosen.get(split, {})
        if not v1_split:
            logger.warning(
                "vs_v1_metrics: v1 payload missing %s metrics for %s",
                split,
                ttype,
            )
            continue
        v2_split = (
            v2_metrics.get(split, v2_metrics)
            if isinstance(v2_metrics, dict)
            else {}
        )
        for metric in out:
            v1_v = v1_split.get(metric)
            v2_v = v2_split.get(metric)
            if v1_v is None or v2_v is None:
                continue
            if v1_v == 0:
                # Pinned: no v1 baseline; skip rather than divide by zero.
                logger.warning(
                    "vs_v1_metrics: v1 %s is zero for %s — skipping.",
                    metric,
                    ttype,
                )
                continue
            delta_pct = (v1_v - v2_v) / abs(v1_v) * 100.0
            out[metric][ttype] = float(delta_pct)
    return out


def improvement_target_met(
    improvement_pct: dict[str, dict[str, float]],
    target_pct: float = IMPROVEMENT_TARGET_PCT,
) -> dict[str, dict[str, bool]]:
    """Decide which (metric, transact_type) cells meet the v2 target.

    Improvement pct is computed by ``vs_v1_metrics`` (positive = better).
    The Spec 14 success criterion is 30–35% MAE/RMSE reduction — a
    metric cell is ``True`` only when the improvement **strictly
    exceeds** ``target_pct`` (i.e. positive delta, the metric was
    lower-is-better, and the magnitude clears the bar).
    """
    out: dict[str, dict[str, bool]] = {}
    for metric, cells in improvement_pct.items():
        out[metric] = {}
        for ttype, pct in cells.items():
            out[metric][ttype] = bool(pct >= target_pct)
    return out


__all__ = [
    "IMPROVEMENT_TARGET_PCT",
    "SMALL_CITY_TEST_ROWS",
    "evaluate_subset",
    "improvement_target_met",
    "per_city_metrics",
    "regression_metrics",
    "vs_v1_metrics",
]
