"""Flask-side service helpers.

One-tiny-module-per-helper: ``fastapi_client`` for HTTP I/O,
``inr_format`` for display formatting. Re-exported here so callers
do ``from app.services import FastAPIClient, inr_format``.
"""

from app.services.fastapi_client import (
    DEFAULT_TIMEOUT_SECONDS,
    KNOWN_CITIES,
    FastAPIClient,
    FastAPIUnavailable,
)
from app.services.inr_format import inr_format

__all__ = [
    "FastAPIClient",
    "FastAPIUnavailable",
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWN_CITIES",
    "inr_format",
]
