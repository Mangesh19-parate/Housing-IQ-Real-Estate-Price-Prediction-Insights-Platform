"""DB-row serialiser for the ``POST /predict`` route.

Pure data-shape module — no I/O, no Flask imports (so the schema module
stays clean of side effects; see ``test_predict_v3_does_not_import_app_ml_or_models``
in Spec 11). This module is the only place that knows the mapping between
``PredictRequestV3`` fields and the ``prediction_log`` table columns
declared in ``app/database/db.py``.

Mapping notes:
    - ``PredictRequestV3.sector`` → ``prediction_log.locality`` (the
      existing column is singular; renaming would require a migration).
    - ``PredictRequestV3.luxury_category`` is excluded from the request
      body by ``Field(exclude=True)`` + a pre-validator (Spec 11). The
      resolved server-side value is taken from ``PredictResponseV3.luxury_category``.
    - ``PredictRequestV3.amenities`` is serialised as a JSON list in the
      log row, not flattened.

The serialised row is what the route layer passes to a parameterized
``INSERT INTO prediction_log`` statement — never f-stringed into SQL.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from api.schemas.predict_v3 import PredictRequestV3, PredictResponseV3


def to_prediction_log_row(
    request: PredictRequestV3,
    response: PredictResponseV3,
    latency_ms: int,
    *,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Return a dict whose keys match the ``prediction_log`` columns verbatim.

    Column list (from ``app/database/db.py``):
        id, timestamp, city, locality, input_features_json,
        predicted_price, predicted_range_low, predicted_range_high,
        model_version, is_outlier_input, latency_ms.

    The ``id`` column is autoincrement, so it's omitted from the dict
    (SQLite fills it on insert). ``timestamp`` defaults to UTC now; the
    caller may pass a fixed value for tests.
    """
    payload = request.model_dump()  # honours exclude=True on luxury_category
    return {
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "city": request.city,
        "locality": request.sector,
        "input_features_json": json.dumps(payload, default=str),
        "predicted_price": response.predicted_price,
        "predicted_range_low": response.range_low,
        "predicted_range_high": response.range_high,
        "model_version": response.model_version,
        "is_outlier_input": 1 if response.is_outlier_input else 0,
        "latency_ms": int(latency_ms),
    }


__all__ = ["to_prediction_log_row"]
