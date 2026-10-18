"""GET /api/analytics/<chart_id> — serve precomputed analytics cache files.

Each cache file contains pre-shaped data for all 5 city slices (ALL + 4 cities)
plus metadata per Rules §1.3. The endpoint returns the full cache JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.schemas.analytics import (
    VALID_CHART_IDS,
    VALID_CITIES,
    AnalyticsCacheResponse,
)

router = APIRouter()

_LOG = logging.getLogger("api.routers.analytics")

# Cache directory
ANALYTICS_CACHE_DIR = Path("data/processed/analytics_cache")

# In-memory cache for loaded JSON files (loaded on first request)
_CACHE: dict[str, dict] = {}


def _load_cache_file(chart_id: str) -> dict:
    """Load a cache file from disk, with in-memory caching."""
    if chart_id in _CACHE:
        return _CACHE[chart_id]

    file_path = ANALYTICS_CACHE_DIR / f"{chart_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Cache file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _CACHE[chart_id] = data
    return data


@router.get(
    "/analytics/{chart_id}",
    response_model=AnalyticsCacheResponse,
    summary="Get precomputed analytics chart data",
    description="Returns cached chart data for a specific chart_id and city filter. "
                "Data includes pre-shaped structures for Chart.js/Leaflet rendering.",
)
async def get_analytics_chart(
    chart_id: str,
    city: str = Query("ALL", description="City filter: ALL, Gurgaon, Hyderabad, Kolkata, Mumbai"),
) -> AnalyticsCacheResponse:
    """Serve precomputed analytics chart data.

    Args:
        chart_id: One of the 13 valid chart IDs (chart_01_spatial_map through chart_13_top_builders_price)
        city: City slice to return (default: "ALL" for all 4 cities combined)

    Returns:
        AnalyticsCacheResponse with chart_id, title, chart_type, city_slices (5 keys),
        and metadata (source_parquet, hash, built_at, row_count, outlier_count)

    Raises:
        404: If chart_id is invalid or cache file not found
        400: If city parameter is invalid
    """
    # Validate chart_id
    if chart_id not in VALID_CHART_IDS:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid chart_id: {chart_id}. Valid IDs: {sorted(VALID_CHART_IDS)}"
        )

    # Validate city
    if city not in VALID_CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid city: {city}. Valid cities: {sorted(VALID_CITIES)}"
        )

    try:
        cache_data = _load_cache_file(chart_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Cache file for {chart_id} not found. Run build_analytics_cache.py first."
        )

    # Return the full cache object (includes all 5 city slices + metadata)
    # The frontend will pick the requested city slice from city_slices[city]
    return AnalyticsCacheResponse(**cache_data)


@router.get(
    "/analytics",
    summary="List available analytics charts",
    description="Returns a list of all available chart IDs with their titles and types.",
)
async def list_analytics_charts() -> dict:
    """List all available analytics charts."""
    return {
        "charts": [
            {
                "chart_id": chart_id,
                "title": title,
                "chart_type": CHART_ID_TO_TYPE[chart_id],
            }
            for chart_id, title in sorted(CHART_ID_TO_TITLE.items())
        ]
    }


# Import at module level for use in list endpoint
from api.schemas.analytics import CHART_ID_TO_TITLE, CHART_ID_TO_TYPE  # noqa: E402