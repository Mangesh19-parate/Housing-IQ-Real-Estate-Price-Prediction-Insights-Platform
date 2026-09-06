"""Service-layer modules called by FastAPI routers.

Per CLAUDE.md, model-inference logic lives here — the routers in
``api/routers/`` are thin HTTP adapters that delegate to these services.
"""

from __future__ import annotations

from api.services.classify_service import (
    ClassifyService,
    MODEL_VERSION as CLASSIFY_MODEL_VERSION,
)
from api.services.predict_service import (
    DEFAULT_RESIDUAL_STD_PCT,
    GEO_NUMERIC_FEATURES,
    MODEL_VERSION,
    PredictService,
    SECTOR_OUTPUT_COLUMN,
)

__all__ = [
    "DEFAULT_RESIDUAL_STD_PCT",
    "GEO_NUMERIC_FEATURES",
    "MODEL_VERSION",
    "PredictService",
    "SECTOR_OUTPUT_COLUMN",
    "ClassifyService",
    "CLASSIFY_MODEL_VERSION",
]