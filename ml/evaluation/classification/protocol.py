"""Pinned protocol constants — single source of truth for the classification
model evaluation gate (Spec 24).

These constants mirror ``02-TRD.md`` §U-TRD-8 + ``08-RULES.md`` §2.1 + the
PRD success-metric targets. **Any drift in these values is a
deliberate protocol revision**, not a silent change — bump
``PROTOCOL_VERSION`` and update the source doc simultaneously.

Spec 23's training scripts carry their own inline constants for
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

#: 70/15/15 split (TRD §U-TRD-8). Any drift in the ratio fails a test.
SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

#: Pinned random seed (Rules §5.4 + TRD §U-TRD-8). Any drift fails a test.
RANDOM_STATE: Final[int] = 42

#: The headline metrics the protocol requires (TRD §U-TRD-8, Rules §2.1).
#: Reported in this order in all tables and JSON.
METRIC_NAMES: Final[tuple[str, ...]] = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "good_deal_recall",        # good_deal_classifier only
    "price_tier_macro_f1",     # price_tier_classifier only
)

#: Pass/fail thresholds from ``02-TRD.md`` §U-TRD-8 + PRD §3:
#:
#: - ``good_deal_recall_min`` — minimum acceptable recall on "Good Deal"
#:   class (index 0) for the primary classifier (PRD: ≥ 0.80).
#: - ``f1_macro_min`` — minimum macro-F1 floor for each classifier.
#: - ``accuracy_min`` — minimum accuracy floor for each classifier.
#: - ``recall_macro_min`` — minimum macro-recall floor for price_tier.
#: - ``rent_min_rows`` — same constant as Specs 13/14, re-exported
#:   from ``ml.training.candidates`` so the threshold lives in
#:   exactly one place.
CLASSIFIER_THRESHOLDS: Final[dict[str, dict[str, float]]] = {
    "good_deal": {
        "good_deal_recall_min": 0.80,
        "f1_macro_min": 0.70,
        "accuracy_min": 0.65,
    },
    "price_tier": {
        "f1_macro_min": 0.75,
        "accuracy_min": 0.70,
        "recall_macro_min": 0.65,
    },
}

#: Source-of-truth doc the protocol claims to mirror. Surfaced in
#: error messages and the report's ``protocol.source_doc`` field.
PROTOCOL_DOC_PATH: Final[str] = "docs/02-TRD.md"


__all__ = [
    "CLASSIFIER_THRESHOLDS",
    "METRIC_NAMES",
    "PROTOCOL_DOC_PATH",
    "PROTOCOL_VERSION",
    "RANDOM_STATE",
    "SPLIT_RATIOS",
]