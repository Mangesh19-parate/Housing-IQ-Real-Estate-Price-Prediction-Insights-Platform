"""``POST /predict`` — price prediction endpoint.

This router is the FastAPI side of the price-prediction inference
contract defined in Spec 11 (Pydantic schemas), Spec 14 (v2 model
training), and Spec 16 (SHAP explainer). Per Rules §5.1, Flask never
imports or loads model files directly; Flask's ``POST /predict``
follow-on page calls this route over HTTP.

Per Rules §2.4 the route loads the exact persisted pipeline from
``models/price_model_{sale,rent}_v2.pkl`` — no re-implemented
preprocessing. Per Rules §2.6 the SHAP explainer is the same instance
the model makes predictions with.

Per Rules §5.2 a DB log failure is logged WARNING but never fails the
HTTP response — the prediction is the user-facing value; the log is
internal telemetry.

Per the ``fastapi-serving`` skill, runtime ``FileNotFoundError`` from
a missing artifact translates to ``503 Service Unavailable`` (not a
500) so the Flask caller can surface a friendly "predictions
temporarily unavailable" message.
"""

from __future__ import annotations

import logging
import re as _re
import threading
import time

from fastapi import APIRouter, HTTPException

from api.config import MODELS_DIR
from api.schemas.predict_log_entry import to_prediction_log_row
from api.schemas.predict_v3 import PredictRequestV3, PredictResponseV3
from api.services.predict_service import PredictService
from app.database.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


#: Lazy singleton — first request constructs it; subsequent requests
#: reuse it. Module-level lock guards the construction race.
_service: PredictService | None = None
_service_lock = threading.Lock()


def get_predict_service() -> PredictService:
    """Return the process-wide ``PredictService`` singleton.

    Constructed on first call; reused thereafter. The lifespan handler
    also calls :meth:`PredictService.warmup` once on startup.
    """
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = PredictService(MODELS_DIR)
    return _service


_PII_REGEX = _re.compile(r"(contact|dealer|phone|email|photo|url|spid)", _re.IGNORECASE)


@router.post("/predict", response_model=PredictResponseV3, status_code=200)
def predict(request: PredictRequestV3) -> PredictResponseV3:
    """Run the v2 price pipeline + SHAP for one request, log to ``prediction_log``.

    Hot path:
        1. Start monotonic clock for ``latency_ms``.
        2. Delegate to :class:`PredictService`.
        3. Log one row to ``prediction_log`` (DB failure → WARNING, never raise).
        4. Return the response.

    Failure modes:
        - ``FileNotFoundError`` (artifact missing at request time) → 503.
        - Any other ``RuntimeError`` from the service → 503 (the
          ``fastapi-serving`` skill's "no 500 surface to end users" rule;
          Rules §5.2).
    """
    t0 = time.perf_counter()
    service = get_predict_service()
    try:
        response = service.predict(request)
    except FileNotFoundError as exc:
        logger.error("model artifact missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"model artifact missing: {exc}",
        ) from exc
    except RuntimeError as exc:
        logger.exception("predict runtime error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"prediction service error: {exc}",
        ) from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log_prediction(request, response, latency_ms)
    return response


def log_prediction(
    request: PredictRequestV3,
    response: PredictResponseV3,
    latency_ms: int,
) -> None:
    """Insert one ``prediction_log`` row. Failure logs WARNING, never raises.

    Parameterized SQL (per Rules §1 + CLAUDE.md) — no f-strings, no
    ORM. ``to_prediction_log_row`` is the only place the column-name
    mapping lives, so this module doesn't drift if the column set
    changes.
    """
    row = to_prediction_log_row(request, response, latency_ms)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO prediction_log ("
                "  timestamp, city, locality, input_features_json,"
                "  predicted_price, predicted_range_low, predicted_range_high,"
                "  model_version, is_outlier_input, latency_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["timestamp"],
                    row["city"],
                    row["locality"],
                    row["input_features_json"],
                    row["predicted_price"],
                    row["predicted_range_low"],
                    row["predicted_range_high"],
                    row["model_version"],
                    row["is_outlier_input"],
                    row["latency_ms"],
                ),
            )
    except Exception as exc:  # noqa: BLE001 — log path must never raise
        logger.warning(
            "prediction_log insert failed (request=%s): %s",
            request.city,
            exc,
        )


__all__ = ["router", "get_predict_service", "log_prediction"]
