"""FastAPI inference service — model serving ONLY, never page rendering.

Per CLAUDE.md: this service is what Flask talks to over HTTP. No Jinja, no
template rendering here. Pages belong in ``app/``.

Step 01 (foundation) exposes only ``GET /health``. Later specs wire the
predict / classify / analytics / recommend / insights routers under
``api/routers/`` — they're included below as empty modules so the package
imports cleanly today.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routers import analytics, classify, insights, predict, recommend
from app.database.db import init_db

app = FastAPI(
    title="HousingIQ Inference",
    version="0.0.1",
    description="Internal model-serving microservice. Flask calls this over HTTP.",
)


@app.on_event("startup")
def _on_startup() -> None:
    """Ensure the 4 operational tables exist before serving any request."""
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Returns ``{"status": "ok"}``. No DB ping, no model load."""
    return {"status": "ok"}


# Include the (still-stub) routers — they become real in later specs.
# ``prefix`` is empty here so each router owns its full URL space; later specs
# may add prefixes like ``/api/v1`` if needed.
app.include_router(predict.router)
app.include_router(classify.router)
app.include_router(analytics.router)
app.include_router(recommend.router)
app.include_router(insights.router)
