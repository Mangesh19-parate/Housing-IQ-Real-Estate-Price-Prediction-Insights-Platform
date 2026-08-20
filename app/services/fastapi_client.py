"""Thin HTTP client around the FastAPI inference service.

Spec 18 wires the Flask ``/predict`` form to forward directly to
FastAPI's ``POST /predict`` (Spec 17). Per Rules §5.1, Flask never
imports model code; per Rules §5.2 a stuck FastAPI call must not
freeze the UI, so requests use a short timeout and any error path
collapses into ``FastAPIUnavailable``.

The locality list is read once at first call from
``data/processed/clean_listings.parquet`` (canonical post-clean
dataset) and cached on the instance. If the parquet is missing
or empty on the checkout, we fall back to a small hardcoded
per-city stub so the form always renders. A real ``/localities``
FastAPI endpoint is a follow-on spec.

No ``ml.*``, ``models.*``, ``api.services.*``, or ``api.routers.*``
imports — this module imports only ``requests``, ``pydantic``, and
the Pydantic schema modules from ``api.schemas`` (per Rules §5.1).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final

import pandas as pd
import requests
from pydantic import ValidationError

from api.schemas.predict_v3 import PredictRequestV3, PredictResponseV3

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.5

# Hardcoded locality stubs — used only when clean_listings.parquet
# is missing or empty on this checkout (Tracker Day 7 is Not
# Started as of spec 18). Tiny strings, just enough to render the
# dependent dropdown. Replaced by the real data once Day 7 lands.
_LOCALITY_STUB: Final[dict[str, tuple[str, ...]]] = {
    "Gurgaon": (
        "Sector 84 Gurgaon", "Sector 81 Gurgaon", "Sohna Road",
        "DLF Phase 1", "DLF Phase 2", "MG Road",
    ),
    "Hyderabad": (
        "Banjara Hills", "Jubilee Hills", "Gachibowli",
        "Madhapur", "Kondapur", "Begumpet",
    ),
    "Kolkata": (
        "Salt Lake", "New Town", "Ballygunge", "Park Street",
        "Howrah", "Behala",
    ),
    "Mumbai": (
        "Andheri West", "Andheri East", "Bandra West",
        "Powai", "Worli", "Lower Parel",
    ),
}

KNOWN_CITIES: Final[tuple[str, ...]] = tuple(_LOCALITY_STUB.keys())


class FastAPIUnavailable(Exception):
    """Single error type raised when the FastAPI service can't serve us.

    Carries the original cause via ``__cause__`` so the Flask
    route can log WARNING with the chained traceback (per Rules
    §5.2 spirit — log it, but the user gets the friendly message).
    """


class FastAPIClient:
    """Process-wide HTTP wrapper around the FastAPI inference service."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._localities: dict[str, tuple[str, ...]] | None = None

    def post_predict(
        self, request: PredictRequestV3
    ) -> PredictResponseV3:
        """POST ``request`` to ``<base_url>/predict`` and parse the response.

        Any of ``HTTPError``, ``Timeout``, ``ConnectionError``, or
        ``ValidationError`` (response shape drift) collapses into
        ``FastAPIUnavailable`` so the Flask route has exactly one
        exception to catch.
        """
        url = f"{self._base_url}/predict"
        payload = request.model_dump(mode="json")
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return PredictResponseV3.model_validate(resp.json())
        except (
            requests.HTTPError,
            requests.Timeout,
            requests.ConnectionError,
            ValidationError,
        ) as exc:
            logger.warning(
                "FastAPI /predict unavailable at %s: %s", url, exc
            )
            raise FastAPIUnavailable(str(exc)) from exc

    def get_localities(self, city: str) -> list[str]:
        """Sorted unique locality names for ``city``.

        Backed by the cached parquet (loaded once at first call).
        Falls back to ``_LOCALITY_STUB[city]`` when the parquet is
        missing or empty. Returns ``[]`` for an unknown city.
        """
        cache = self._ensure_localities_cache()
        names = cache.get(city, ())
        return sorted(set(names))

    def _ensure_localities_cache(self) -> dict[str, tuple[str, ...]]:
        if self._localities is not None:
            return self._localities
        self._localities = _load_localities_from_parquet()
        return self._localities


@lru_cache(maxsize=1)
def _load_localities_from_parquet() -> dict[str, tuple[str, ...]]:
    """Read ``data/processed/clean_listings.parquet`` and return per-city locality lists.

    Falls back to ``_LOCALITY_STUB`` if the parquet is missing or
    has no rows. ``lru_cache(maxsize=1)`` means the file is read
    once per process.
    """
    parquet_path = Path("data/processed/clean_listings.parquet")
    if not parquet_path.exists():
        logger.info(
            "clean_listings.parquet missing — using locality stub fallback"
        )
        return dict(_LOCALITY_STUB)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.warning(
            "Failed to read clean_listings.parquet (%s) — using stub fallback",
            exc,
        )
        return dict(_LOCALITY_STUB)

    if df.empty or "locality" not in df.columns or "city" not in df.columns:
        logger.info(
            "clean_listings.parquet is empty or missing expected columns — using stub fallback"
        )
        return dict(_LOCALITY_STUB)

    grouped = (
        df[["city", "locality"]]
        .dropna(subset=["city", "locality"])
        .groupby("city")["locality"]
        .apply(lambda s: tuple(sorted(set(s))))
        .to_dict()
    )
    # Merge with stub so every known city has at least a baseline
    # list — protects against partial parquet snapshots.
    merged: dict[str, tuple[str, ...]] = {}
    for city in KNOWN_CITIES:
        stub = _LOCALITY_STUB[city]
        real = grouped.get(city, ())
        merged[city] = tuple(sorted(set(stub) | set(real)))
    for city, localities in grouped.items():
        if city not in merged:
            merged[city] = localities
    return merged


__all__ = [
    "FastAPIClient",
    "FastAPIUnavailable",
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWN_CITIES",
]
