"""Per-city residual-threshold calibrator for ``good_deal_verdict`` (Spec 21).

Given a frame of ``residual_pct`` values (typically the output of
``build_good_deal_labels`` after the first pass — verdicts null),
compute the per-city residual band that defines ``Good Deal`` /
``Fair Price`` / ``Overpriced`` for each city.

Per-city formula::

    low  = median_residual - 0.5 * iqr_residual
    high = median_residual + 0.5 * iqr_residual

then clamped to ``[default_low - 0.10, default_high + 0.10]`` so a
pathological city can't produce a band wider than ±20%.

Cities with fewer than ``MIN_CITY_TRAIN_ROWS`` non-outlier training
rows fall back to the default ±10% with a logged WARNING.

Authority:
    - docs/02-TRD.md §U-TRD-6 (tunable per city)
    - docs/08-RULES.md §12.2 (re-validate thresholds per city)
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from ml.classification.verdicts import DEFAULT_THRESHOLD_HIGH, DEFAULT_THRESHOLD_LOW

logger = logging.getLogger(__name__)

#: Minimum non-outlier training rows for per-city calibration. Below
#: this, the city uses the default ±10% thresholds (Rules §12.2).
MIN_CITY_TRAIN_ROWS: Final[int] = 100

#: Per-city band = ``median ± BAND_HALF_WIDTH × IQR``. Half the IQR
#: is the within-city "noise floor" half-band — anything beating
#: half a band is meaningfully under- or over-priced.
BAND_HALF_WIDTH: Final[float] = 0.5

#: Maximum allowed band extension beyond the default. A pathological
#: city with very high IQR can't push the band past ±20% even if
#: the raw formula would suggest it.
CLAMP_MARGIN: Final[float] = 0.10


def _iqr(s: pd.Series) -> float:
    """Interquartile range (75th − 25th percentile) of ``s``."""
    q75 = float(s.quantile(0.75))
    q25 = float(s.quantile(0.25))
    return q75 - q25


def calibrate_good_deal_thresholds(
    good_deal_labels: pd.DataFrame,
    is_train_mask: pd.Series,
    default_low: float = DEFAULT_THRESHOLD_LOW,
    default_high: float = DEFAULT_THRESHOLD_HIGH,
) -> dict[str, dict[str, float]]:
    """Return per-city calibrated residual thresholds.

    Filters to ``is_outlier == False`` AND ``is_train_mask == True``
    AND ``residual_pct`` not null. Per city:
        - If ``n < MIN_CITY_TRAIN_ROWS``: fall back to
          ``(default_low, default_high)`` and log a WARNING.
        - Else: emit ``(low=median - 0.5*iqr,
          high=median + 0.5*iqr)`` clamped to
          ``[default_low - 0.10, default_high + 0.10]``.

    Returns a dict containing:
        - one entry per city: ``{city: {"low", "high", "n_train",
          "median_residual", "iqr_residual"}}``
        - a ``"_default"`` entry carrying ``{"low", "high",
          "rationale"}`` so the caller can persist a single file
          that round-trips.

    Pure function — does not write to disk.
    """
    if "city" not in good_deal_labels.columns:
        raise KeyError("calibrate_good_deal_thresholds: missing 'city' column")
    if "residual_pct" not in good_deal_labels.columns:
        raise KeyError(
            "calibrate_good_deal_thresholds: missing 'residual_pct' column"
        )
    if len(is_train_mask) != len(good_deal_labels):
        raise ValueError(
            "calibrate_good_deal_thresholds: is_train_mask length "
            f"({len(is_train_mask)}) does not match frame length "
            f"({len(good_deal_labels)})"
        )

    is_outlier_series = good_deal_labels.get(
        "is_outlier", pd.Series(False, index=good_deal_labels.index)
    )
    df = good_deal_labels[
        (~is_outlier_series.astype(bool))
        & is_train_mask.astype(bool)
        & good_deal_labels["residual_pct"].notna()
    ]

    low_floor = default_low - CLAMP_MARGIN
    high_ceiling = default_high + CLAMP_MARGIN

    out: dict[str, dict[str, float]] = {}

    for city, group in df.groupby("city", sort=True):
        n = len(group)
        if n < MIN_CITY_TRAIN_ROWS:
            logger.warning(
                "calibrate_good_deal_thresholds: city=%s has n=%d "
                "training rows < MIN_CITY_TRAIN_ROWS=%d; falling back "
                "to default thresholds (low=%.2f, high=%.2f).",
                city,
                n,
                MIN_CITY_TRAIN_ROWS,
                default_low,
                default_high,
            )
            out[city] = {
                "low": default_low,
                "high": default_high,
                "n_train": int(n),
                "median_residual": float("nan"),
                "iqr_residual": float("nan"),
            }
            continue

        median = float(group["residual_pct"].median())
        iqr = _iqr(group["residual_pct"])
        # NaN-safe clamp: if iqr is NaN (degenerate distribution),
        # fall back to defaults.
        if not np.isfinite(iqr) or iqr == 0.0:
            logger.warning(
                "calibrate_good_deal_thresholds: city=%s has "
                "degenerate residual distribution (iqr=%s); using "
                "default thresholds.",
                city,
                iqr,
            )
            low = default_low
            high = default_high
        else:
            low = median - BAND_HALF_WIDTH * iqr
            high = median + BAND_HALF_WIDTH * iqr
            low = max(low, low_floor)
            high = min(high, high_ceiling)

        out[city] = {
            "low": float(low),
            "high": float(high),
            "n_train": int(n),
            "median_residual": median,
            "iqr_residual": float(iqr),
        }
        logger.info(
            "calibrate_good_deal_thresholds: city=%s n=%d "
            "median=%.4f iqr=%.4f → low=%.4f high=%.4f",
            city,
            n,
            median,
            iqr,
            low,
            high,
        )

    # Always emit a _default block so the persisted JSON round-trips.
    out["_default"] = {
        "low": default_low,
        "high": default_high,
        "rationale": (
            f"Used when a city has < {MIN_CITY_TRAIN_ROWS} train rows."
        ),
    }

    return out


__all__ = [
    "BAND_HALF_WIDTH",
    "CLAMP_MARGIN",
    "MIN_CITY_TRAIN_ROWS",
    "calibrate_good_deal_thresholds",
]
