"""Database row serialiser for classification_log (Spec 25).

Kept separate from ``classify_v1.py`` so the schema module stays I/O-free.
Pure function — no DB access, no side effects.

Authority:
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-11 (classification_log columns)
"""

from __future__ import annotations

import json
from typing import Any

from api.schemas.classify_v1 import ClassifyResponseV1
from api.schemas.predict_v3 import PredictRequestV3


def to_classification_log_row(
    request: PredictRequestV3,
    response: ClassifyResponseV1,
    latency_ms: int,
) -> dict[str, Any]:
    """Convert a classify request/response into a ``classification_log`` row.

    Args:
        request: The validated request payload (16 fields, no luxury_category).
        response: The classification response.
        latency_ms: Wall-clock latency for the /classify call (monotonic clock).

    Returns:
        Dict whose keys match the ``classification_log`` column names verbatim:
        - timestamp (ISO8601 string)
        - city (string)
        - input_features_json (JSON string of the request, sans luxury_category)
        - predicted_verdict (string: "Good Deal" / "Fair Deal" / "Overpriced")
        - predicted_tier (string: "Budget" / "Mid-Range" / "Premium" / "Luxury")
        - verdict_probabilities_json (JSON string of dict[VerdictLabel, float])
        - tier_probabilities_json (JSON string of dict[TierLabel, float])
        - model_version (string, e.g. "v1")
    """
    # Build the input features JSON (request dict, minus server-derived luxury_category)
    # PredictRequestV3 already excludes luxury_category via Field(exclude=True) +
    # the pre-validator, so model_dump() is clean.
    input_features = request.model_dump(mode="json")

    # Convert verdict probabilities to plain dict with string keys (use .value for enums)
    verdict_probs = {k.value: float(v) for k, v in response.verdict_probabilities.items()}
    tier_probs = {k.value: float(v) for k, v in response.tier_probabilities.items()}

    return {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "city": request.city,
        "input_features_json": json.dumps(input_features, separators=(",", ":")),
        "predicted_verdict": response.good_deal_verdict.value,
        "predicted_tier": response.price_tier.value,
        "verdict_probabilities_json": json.dumps(verdict_probs, separators=(",", ":")),
        "tier_probabilities_json": json.dumps(tier_probs, separators=(",", ":")),
        "model_version": response.model_version,
    }


__all__ = ["to_classification_log_row"]