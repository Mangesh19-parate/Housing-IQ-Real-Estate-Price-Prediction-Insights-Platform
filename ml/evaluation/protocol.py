"""Pinned protocol constants — single source of truth for the price
model evaluation gate (Spec 15).

These constants mirror ``02-TRD.md`` §10 + ``08-RULES.md`` §2.1 + the
PRD success-metric targets. **Any drift in these values is a
deliberate protocol revision**, not a silent change — bump
``PROTOCOL_VERSION`` and update the source doc simultaneously.

Specs 13/14's training scripts carry their own inline constants for
training-time use (no behavior change there). The gate independently
enforces the protocol at certification time so a drift in a
training script's constants does not silently ship a model that
fails the protocol.
"""

from __future__ import annotations

import logging
from typing import Final

from ml.training.candidates import RENT_MIN_ROWS

logger = logging.getLogger(__name__)

#: Semver-pinned protocol version. The ``evaluate()`` entry point
#: emits this into every ``evaluation_report_{version}.json``'s
#: ``protocol`` block so a reviewer can confirm which protocol was
#: applied. Bump on any intentional change to the constants below.
PROTOCOL_VERSION: Final[str] = "1.0.0"

#: 70/15/15 split (TRD §10). Any drift in the ratio fails a test.
SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

#: Pinned random seed (Rules §5.4 + TRD §10). Any drift fails a test.
RANDOM_STATE: Final[int] = 42

#: The four headline metrics the protocol requires (TRD §10,
#: Rules §2.1). Reported on the **original ₹ price scale** via
#: ``np.expm1`` even when the model trains on ``log1p(price)``.
METRIC_NAMES: Final[tuple[str, ...]] = ("r2", "mae", "rmse", "mape")

#: Pass/fail thresholds from ``02-TRD.md`` §10 + PRD §3:
#:
#: - ``r2_min`` — minimum acceptable test R² (PRD: ≥ 0.80,
#:   stretch 0.85).
#: - ``r2_stretch`` — the stretch goal R²; below this the model
#:   clears the floor but is flagged as "passes, below stretch"
#:   in the summary.
#: - ``mae_pct_within_15_at_least`` — fraction of test rows where
#:   the absolute prediction error is within ±15% of the actual
#:   price (PRD: ≥ 0.70).
#: - ``p95_latency_ms_max`` — ``/predict`` p95 latency cap
#:   (PRD: < 300 ms). Measured only when the gate runs against a
#:   live FastAPI instance; offline mode emits ``null``.
#: - ``rent_min_rows`` — same constant as Specs 13/14, re-exported
#:   from ``ml.training.candidates`` so the threshold lives in
#:   exactly one place.
protocol_thresholds: Final[dict[str, float]] = {
    "r2_min": 0.80,
    "r2_stretch": 0.85,
    "mae_pct_within_15_at_least": 0.70,
    "p95_latency_ms_max": 300.0,
    "rent_min_rows": float(RENT_MIN_ROWS),
}

#: Source-of-truth doc the protocol claims to mirror. Surfaced in
#: error messages and the report's ``protocol.source_doc`` field.
PROTOCOL_DOC_PATH: Final[str] = "docs/02-TRD.md"


__all__ = [
    "METRIC_NAMES",
    "PROTOCOL_DOC_PATH",
    "PROTOCOL_VERSION",
    "RANDOM_STATE",
    "SPLIT_RATIOS",
    "protocol_thresholds",
]
