"""FastAPI inference service — model serving ONLY, never page rendering.

Per CLAUDE.md: this service is what Flask talks to over HTTP. No Jinja, no
template rendering here. Pages belong in ``app/``.

The lifespan context manager replaces the deprecated ``@app.on_event``
decorator (per the ``fastapi-serving`` skill): it runs ``init_db()``
to ensure the four operational tables exist, then warms up the price
prediction service so the first ``POST /predict`` doesn't pay the
load-from-disk cost.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import analytics, classify, insights, predict, recommend
from api.routers.predict import get_predict_service
from app.database.db import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI startup/shutdown handler.

    Startup: ``init_db()`` + ``get_predict_service().warmup()``.
    Shutdown: nothing to release (joblib-loaded artifacts are
    process-local; OS reclaims them on exit).
    """
    init_db()
    get_predict_service().warmup()
    logger.info("FastAPI startup complete — predict service warmed")
    yield


app = FastAPI(
    title="HousingIQ Inference",
    version="0.0.1",
    description="Internal model-serving microservice. Flask calls this over HTTP.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Returns ``{"status": "ok"}``. No DB ping, no model load."""
    return {"status": "ok"}


# Include the routers — ``prefix`` is empty so each router owns its full URL space.
app.include_router(predict.router)
app.include_router(classify.router)
app.include_router(analytics.router)
app.include_router(recommend.router)
app.include_router(insights.router)
