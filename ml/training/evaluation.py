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


__all__ = [
    "SMALL_CITY_TEST_ROWS",
    "evaluate_subset",
    "per_city_metrics",
    "regression_metrics",
]
