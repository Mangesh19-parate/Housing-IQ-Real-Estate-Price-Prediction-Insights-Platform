"""Thin HTTP client around the FastAPI classification endpoint.

Mirrors ``app.services.fastapi_client`` pattern: short timeout, error
collapse into a single exception type, Pydantic response validation.
Per Rules §5.1 Flask never imports model code; per Rules §5.2 a stuck
FastAPI call must not freeze the UI.
"""

from __future__ import annotations

import logging
from typing import Final

import requests
from pydantic import ValidationError

from api.schemas.classify_v1 import ClassifyResponseV1, PredictRequestV3

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.5


class ClassifyUnavailable(Exception):
    """Single error type raised when the FastAPI /classify endpoint can't serve us.

    Carries the original cause via ``__cause__`` so the Flask route can
    log WARNING with the chained traceback (per Rules §5.2 spirit — log
    it, but the user gets the friendly message).
    """


class ClassifyClient:
    """Process-wide HTTP wrapper around the FastAPI classification service."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def post_classify(self, request: PredictRequestV3) -> ClassifyResponseV1:
        """POST ``request`` to ``<base_url>/classify`` and parse the response.

        Any of ``HTTPError``, ``Timeout``, ``ConnectionError``, or
        ``ValidationError`` (response shape drift) collapses into
        ``ClassifyUnavailable`` so the Flask route has exactly one
        exception to catch.
        """
        url = f"{self._base_url}/classify"
        payload = request.model_dump(mode="json")
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return ClassifyResponseV1.model_validate(resp.json())
        except (
            requests.HTTPError,
            requests.Timeout,
            requests.ConnectionError,
            ValidationError,
        ) as exc:
            logger.warning(
                "FastAPI /classify unavailable at %s: %s", url, exc
            )
            raise ClassifyUnavailable(str(exc)) from exc


__all__ = [
    "ClassifyClient",
    "ClassifyUnavailable",
    "DEFAULT_TIMEOUT_SECONDS",
]
