"""``POST /classify`` — affordability & good-deal verdict endpoint (Spec 25).

This router is the FastAPI side of the classification inference contract
defined in Spec 11 (Pydantic schemas reused), Spec 23 (v1 model training),
and Spec 24 (evaluation gate). Per Rules §5.1, Flask never imports or
loads model files directly; Flask's ``POST /classify`` follow-on page
calls this route over HTTP.

Per Rules §2.4 the route loads the exact persisted pipelines from
``models/good_deal_classifier_v1.pkl`` and ``models/price_tier_classifier_v1.pkl``
+ the shared ``models/classifier_preprocessor_v1.pkl`` — no
re-implemented preprocessing. Per Rules §2.6 the SHAP explainer is the
same instance the primary model makes predictions with.

Per Rules §5.2 a DB log failure is logged WARNING but never fails the
HTTP response — the classification is the user-facing value; the log is
internal telemetry.

Per the ``fastapi-serving`` skill, runtime ``FileNotFoundError`` from
a missing artifact translates to ``503 Service Unavailable`` (not a
500) so the Flask caller can surface a friendly "classification
temporarily unavailable" message.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, HTTPException

from api.config import MODELS_DIR
from api.schemas.classify_log_entry import to_classification_log_row
from api.schemas.classify_v1 import ClassifyResponseV1
from api.schemas.predict_v3 import PredictRequestV3
from api.services.classify_service import ClassifyService
from app.database.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

#: Lazy singleton — first request constructs it; subsequent requests
#: reuse it. Module-level lock guards the construction race. The lifespan
#: handler (Spec 20 parity) overrides this with a registry-resolved instance
#: via :func:`set_classify_service` if the registry has an active row.
_service: ClassifyService | None = None
_service_lock = threading.Lock()


def get_classify_service() -> ClassifyService:
    """Return the process-wide ``ClassifyService`` singleton.

    Constructed on first call; reused thereafter. The lifespan handler
    also calls :meth:`ClassifyService.warmup` once on startup.

    If the lifespan handler stashed a registry-resolved service via
    :func:`set_classify_service`, return that one instead — it carries
    the active ``model_version`` + ``artifact_path`` from ``model_registry``.
    """
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = ClassifyService(MODELS_DIR)
    return _service


def set_classify_service(service: ClassifyService) -> None:
    """Override the singleton with a lifespan-resolved instance (Spec 20 parity).

    Idempotent — calling twice with the same service is a no-op. Used by
    :func:`api.main.lifespan` to inject a registry-aware service after
    resolving the active row at startup.
    """
    global _service
    with _service_lock:
        _service = service


@router.post("/classify", response_model=ClassifyResponseV1, status_code=200)
def classify(request: PredictRequestV3) -> ClassifyResponseV1:
    """Run the v1 classification pipelines + SHAP for one request, log to ``classification_log``.

    Hot path:
        1. Start monotonic clock for ``latency_ms``.
        2. Delegate to :class:`ClassifyService`.
        3. Log one row to ``classification_log`` (DB failure → WARNING, never raise).
        4. Return the response.

    Failure modes:
        - ``FileNotFoundError`` (artifact missing at request time) → 503.
        - Any other ``RuntimeError`` from the service → 503 (the
          ``fastapi-serving`` skill's "no 500 surface to end users" rule;
          Rules §5.2).
    """
    t0 = time.perf_counter()
    service = get_classify_service()
    try:
        response = service.classify(request)
    except FileNotFoundError as exc:
        logger.error("model artifact missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"model artifact missing: {exc}",
        ) from exc
    except RuntimeError as exc:
        logger.exception("classify runtime error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"classification service error: {exc}",
        ) from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log_classification(request, response, latency_ms)
    return response


def log_classification(
    request: PredictRequestV3,
    response: ClassifyResponseV1,
    latency_ms: int,
) -> None:
    """Insert one ``classification_log`` row. Failure logs WARNING, never raises.

    Parameterized SQL (per Rules §1 + CLAUDE.md) — no f-strings, no
    ORM. ``to_classification_log_row`` is the only place the column-name
    mapping lives, so this module doesn't drift if the column set changes.
    """
    row = to_classification_log_row(request, response, latency_ms)
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO classification_log ("
                "  timestamp, city, input_features_json, predicted_verdict,"
                "  predicted_tier, verdict_probabilities_json,"
                "  tier_probabilities_json, model_version, latency_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["timestamp"],
                    row["city"],
                    row["input_features_json"],
                    row["predicted_verdict"],
                    row["predicted_tier"],
                    row["verdict_probabilities_json"],
                    row["tier_probabilities_json"],
                    row["model_version"],
                    latency_ms,
                ),
            )
    except Exception as exc:  # noqa: BLE001 — log path must never raise
        logger.warning(
            "classification_log insert failed (request=%s): %s",
            request.city,
            exc,
        )


__all__ = ["router", "get_classify_service", "log_classification"]