"""Metric-scoring layer — Spec 15.

Pure functions for the four headline metrics (R², MAE, RMSE, MAPE)
on the original ₹ scale, plus the within-tolerance fraction that
backs the PRD "MAE within ±15% for 70% of listings" threshold.

Re-imports ``per_city_metrics`` from ``ml.training.evaluation`` so
the gate and the training scripts agree on per-city math (no
second copy).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ml.evaluation.protocol import METRIC_NAMES
from ml.training.evaluation import per_city_metrics  # noqa: F401  re-export

#: Small epsilon for the within-tolerance fraction — guards against
#: divide-by-zero on near-zero ``y_true`` rows. No such rows are
#: expected post-cleaning, but pinned for safety.
_WITHIN_TOL_EPS: Final[float] = 1.0


def score_predictions(
    y_true: np.ndarray,
    y_pred_log: np.ndarray,
    invert_log: bool = True,
) -> dict[str, float]:
    """Return ``{r2, mae, rmse, mape}`` in ``METRIC_NAMES`` order.

    When ``invert_log=True``, both inputs are expected in log space
    (``np.log1p(price)``) and the function applies ``np.expm1`` to
    both true + predicted before scoring — this enforces the
    single-source-of-truth rule from ``08-RULES.md`` §2.1: "train on
    log, report on original scale."

    ``MAPE`` is computed with an epsilon guard (``max(y_true, 1.0)``)
    to avoid divide-by-zero on near-zero prices. Pure function; no
    side effects.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred_log, dtype=float)

    if invert_log:
        y_true_arr = np.expm1(y_true_arr)
        y_pred_arr = np.expm1(y_pred_arr)

    denom = np.maximum(y_true_arr, 1.0)
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name == "r2":
            out[name] = float(r2_score(y_true_arr, y_pred_arr))
        elif name == "mae":
            out[name] = float(mean_absolute_error(y_true_arr, y_pred_arr))
        elif name == "rmse":
            out[name] = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        elif name == "mape":
            out[name] = mape
        else:
            # Defensive — should never fire because METRIC_NAMES is
            # the pinned source of truth.
            raise ValueError(f"Unknown metric: {name!r}")
    return out


def within_tolerance_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 0.15,
) -> float:
    """Fraction of rows where ``|y_pred - y_true| / y_true <= tolerance``.

    Used to score the PRD "MAE within ±15% for 70% of test listings"
    threshold. Both inputs are expected on the **original ₹ scale**
    (caller has already ``expm1``'d if the model trained on log).

    Returns a value in ``[0.0, 1.0]``. The epsilon guard
    (``max(y_true, 1.0)``) prevents divide-by-zero on near-zero rows.
    Pure function; no side effects.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.size == 0:
        return 0.0

    denom = np.maximum(y_true_arr, _WITHIN_TOL_EPS)
    abs_rel_err = np.abs(y_pred_arr - y_true_arr) / denom
    return float(np.mean(abs_rel_err <= tolerance))


__all__ = [
    "METRIC_NAMES",
    "per_city_metrics",
    "score_predictions",
    "within_tolerance_pct",
]
