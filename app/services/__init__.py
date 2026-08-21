"""Flask-side service helpers.

One-tiny-module-per-helper: ``fastapi_client`` for HTTP I/O,
``inr_format`` for display formatting, ``shap_format`` for the
SHAP result-page renderer (Spec 19). Re-exported here so
callers do ``from app.services import FastAPIClient, inr_format,
format_shap_for_template, summarize_direction``.
"""

from app.services.fastapi_client import (
    DEFAULT_TIMEOUT_SECONDS,
    KNOWN_CITIES,
    FastAPIClient,
    FastAPIUnavailable,
)
from app.services.inr_format import inr_format
from app.services.shap_format import (
    TOP_N_DEFAULT,
    format_shap_for_template,
    summarize_direction,
)

__all__ = [
    "FastAPIClient",
    "FastAPIUnavailable",
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWN_CITIES",
    "inr_format",
    "TOP_N_DEFAULT",
    "format_shap_for_template",
    "summarize_direction",
]
