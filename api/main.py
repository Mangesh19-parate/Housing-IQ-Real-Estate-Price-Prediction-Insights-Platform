"""FastAPI inference service — model serving ONLY, never page rendering.

Per CLAUDE.md: this service is what Flask talks to over HTTP. No Jinja, no
template rendering here. Pages belong in ``app/``.

The lifespan context manager replaces the deprecated ``@app.on_event``
decorator (per the ``fastapi-serving`` skill): it runs ``init_db()``
to ensure the four operational tables exist, then warms up the price
prediction service so the first ``POST /predict`` doesn't pay the
load-from-disk cost.

Spec 20: lifespan also resolves the active model from the
``model_registry`` table. If a row is found, the PredictService is
reconstructed with that version + artifact path so the live
``/predict`` response carries the registered version string. Falls
back to the historical module-level default (``MODEL_VERSION = "v2"``)
with a one-line warning when no active row exists — matches Rules
§5.2's graceful-degradation rule.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.config import MODELS_DIR
from api.routers import analytics, classify, insights, models as models_router
from api.routers import predict, recommend
from api.routers.predict import get_predict_service, set_predict_service
from api.services.predict_service import MODEL_VERSION, PredictService
from app.database.db import init_db
from ml.registry import get_active_artifact

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI startup/shutdown handler.

    Startup order:
        1. ``init_db()`` — apply pending migrations + 002 column guard.
        2. Resolve the active ``price_model_sale`` row from the registry
           (Spec 20). If found, construct a ``PredictService`` carrying
           the registered version + artifact path and inject it via
           :func:`set_predict_service`. If not, fall back to the
           default service (MODULE_VERSION = "v2").
        3. Warmup the resolved service.

    Shutdown: nothing to release (joblib-loaded artifacts are
    process-local; OS reclaims them on exit).
    """
    init_db()
    _resolve_active_service()
    get_predict_service().warmup()
    logger.info("FastAPI startup complete — predict service warmed")
    yield


def _resolve_active_service() -> None:
    """Construct + inject the registry-aware service (Spec 20).

    Reads the active ``price_model_sale`` row from ``model_registry``.
    On hit: build ``PredictService(models_dir, model_version=...)``.
    On miss: leave the singleton at its default (``MODEL_VERSION``).
    In both cases logs one INFO line with the resolved version.
    """
    resolved = get_active_artifact("price_model_sale")
    if resolved is None:
        logger.warning(
            "no active row for price_model_sale in model_registry — "
            "falling back to module-level MODEL_VERSION"
        )
        return
    version, artifact_path = resolved
    # The artifact_path is repo-relative; resolve to absolute so the
    # service can open it regardless of FastAPI's CWD. Falls back to
    # MODELS_DIR if the path doesn't exist yet (artifact missing).
    abs_path = Path(artifact_path)
    if not abs_path.is_absolute():
        abs_path = (Path.cwd() / abs_path).resolve()
    if not abs_path.exists():
        logger.warning(
            "active row %s/%s points at missing artifact %s — "
            "falling back to MODELS_DIR with version %s",
            "price_model_sale",
            version,
            artifact_path,
            version,
        )
        set_predict_service(PredictService(MODELS_DIR, model_version=version))
        return
    logger.info(
        "resolved active price_model_sale from registry: version=%s artifact=%s",
        version,
        artifact_path,
    )
    set_predict_service(PredictService(MODELS_DIR, model_version=version))


app = FastAPI(
    title="HousingIQ Inference",
    version="0.0.1",
    description="Internal model-serving microservice. Flask calls this over HTTP.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check + active model version (Spec 20).

    Reads the active row from ``model_registry`` on every call — a
    cheap indexed query (one row). This keeps ``/health`` honest even
    if the lifespan-built service is stale relative to a recent
    ``set_active()`` call (the spec defers hot-swap cache invalidation
    — see docs/08-RULES.md §15 deferred items).

    Falls back to the ``MODEL_VERSION`` constant when no active row
    exists yet. The fallback intentionally does NOT read the predict
    service singleton — the singleton is test-replaceable via
    ``set_predict_service`` and that would couple ``/health`` to
    unrelated test fixtures' cleanup discipline.
    """
    resolved = get_active_artifact("price_model_sale")
    version = resolved[0] if resolved is not None else MODEL_VERSION
    return {"status": "ok", "model_version": version}


# Include the routers — ``prefix`` is empty so each router owns its full URL space.
app.include_router(predict.router)
app.include_router(classify.router)
app.include_router(analytics.router)
app.include_router(recommend.router)
app.include_router(insights.router)
app.include_router(models_router.router)
