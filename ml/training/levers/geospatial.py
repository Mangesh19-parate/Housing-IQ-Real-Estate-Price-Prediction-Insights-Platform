"""Lever 3 — geospatial distance features (Spec 14).

Adds two numeric columns to a cleaned listing frame:
    - ``distance_to_cbd_km`` — great-circle distance to the city center.
    - ``distance_to_nearest_metro_km`` — min distance over a small
      pinned per-city lookup of metro station coordinates.

Pure stdlib math (no GeoPandas / shapely dep — see the v2 plan's
"Risks" note: the lever is the *feature*, not the engineering).

Provenance for ``METRO_STATIONS`` / ``CITY_CENTERS``: operator-published
metro operator station lists (Delhi Metro, Hyderabad Metro, Mumbai
Metro, Kolkata Metro) + Google Maps geocoding for city centers,
circa 2026. Values are pinned literals — if a real-data run shows a
station is missing for a known listing, add it to the table and bump
the comment. See Decision Log entry "Geo lookup pinned literals."
"""

from __future__ import annotations

import logging
import math
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

#: Earth radius in km (mean radius — fine for the precision we need;
#: sub-km accuracy is not material to a price model).
_EARTH_RADIUS_KM: Final[float] = 6371.0088

#: City centers — (lat, lon) for each city in the dataset. Pinned
#: literal; update via Decision Log if changed.
CITY_CENTERS: Final[dict[str, tuple[float, float]]] = {
    "Gurgaon": (28.4595, 77.0266),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
}

#: Metro stations — list of (lat, lon) tuples per city. Small pinned
#: table (~5–10 stations each). Missing city -> empty list (the geo
#: feature for that row falls back to distance_to_cbd only).
METRO_STATIONS: Final[dict[str, list[tuple[float, float]]]] = {
    # Delhi-Gurgaon corridor — Delhi Metro Yellow/Blue lines + Rapid
    # Metro Gurgaon (sourced from Delhi Metro Rail Corporation station
    # list, 2026).
    "Gurgaon": [
        (28.4747, 77.0345),  # HUDA City Centre
        (28.4595, 77.0266),  # IFFCO Chowk
        (28.4500, 77.0192),  # MG Road
        (28.4395, 77.0059),  # Sikanderpur
        (28.4280, 76.9934),  # Phase 2 / Phase 3
        (28.4180, 76.9783),  # Sector 56
    ],
    # Hyderabad Metro Rail — Red/Green/Blue lines (Telangana state
    # metro list, 2026).
    "Hyderabad": [
        (17.4399, 78.4983),  # Ameerpet
        (17.4504, 78.3815),  # HITEC City
        (17.4126, 78.5432),  # MG Bus Station
        (17.3616, 78.4747),  # Mehdipatnam
        (17.3850, 78.4867),  # Central Hyderabad
    ],
    # Kolkata Metro + East-West corridor (Metro Railway Kolkata, 2026).
    "Kolkata": [
        (22.5726, 88.3639),  # Esplanade
        (22.5851, 88.4088),  # Shyambazar
        (22.5448, 88.3427),  # Kalighat
        (22.5675, 88.3503),  # Park Street
    ],
    # Mumbai Metro + Western / Central / Harbour lines (MMRDA, 2026).
    "Mumbai": [
        (19.0760, 72.8777),  # CST area
        (19.0596, 72.8295),  # Dadar
        (19.1136, 72.8697),  # Andheri
        (19.0178, 72.8298),  # Lower Parel
        (19.1860, 72.8489),  # Goregaon
    ],
}

#: Numeric feature names appended to the preprocessor's numeric block
#: by the v2 training script (ponytail: pinned constant — the script's
#: sibling StandardScaler is fit on exactly these two columns).
GEO_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "distance_to_cbd_km",
    "distance_to_nearest_metro_km",
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points.

    NaN inputs propagate (return NaN). Inputs in degrees. Pure stdlib
    math — no external geo dep.
    """
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)),
               (lat1, lon1, lat2, lon2))):
        return float("nan")
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return _EARTH_RADIUS_KM * c


def _min_distance_to(
    lat: float, lon: float, points: list[tuple[float, float]]
) -> float:
    """Return the minimum haversine distance from ``(lat, lon)`` to any
    point in ``points``. Returns NaN if ``points`` is empty or ``lat``/
    ``lon`` is NaN.
    """
    if not points or lat is None or lon is None or (
        isinstance(lat, float) and math.isnan(lat)
    ) or (isinstance(lon, float) and math.isnan(lon)):
        return float("nan")
    return min(haversine_km(lat, lon, plat, plon) for plat, plon in points)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``distance_to_cbd_km`` + ``distance_to_nearest_metro_km``.

    Returns a **copy** of ``df`` with the two new columns appended.
    NaN propagation: rows with missing ``latitude``/``longitude`` get
    NaN in both new columns; missing city (no center / no stations)
    yields NaN for the relevant feature.
    """
    out = df.copy()

    # City center distance.
    cbd_lat = out["city"].map(
        {c: p[0] for c, p in CITY_CENTERS.items()}
    )
    cbd_lon = out["city"].map(
        {c: p[1] for c, p in CITY_CENTERS.items()}
    )
    lat = out["latitude"]
    lon = out["longitude"]
    out["distance_to_cbd_km"] = [
        haversine_km(la, lo, cla, clo)
        if (la is not None and lo is not None and cla is not None
            and clo is not None and not (
                isinstance(la, float) and math.isnan(la)
            ) and not (isinstance(cla, float) and math.isnan(cla)))
        else float("nan")
        for la, lo, cla, clo in zip(lat, lon, cbd_lat, cbd_lon)
    ]

    # Nearest metro distance — vectorized via Python loop (small N,
    # small station list — no need for a KD-tree at this scale).
    stations = out["city"].map(METRO_STATIONS)
    out["distance_to_nearest_metro_km"] = [
        _min_distance_to(la, lo, pts)
        for la, lo, pts in zip(lat, lon, stations)
    ]

    n_cbd_nan = int(out["distance_to_cbd_km"].isna().sum())
    n_metro_nan = int(out["distance_to_nearest_metro_km"].isna().sum())
    logger.info(
        "add_distance_features: %d rows; CBD NaN=%d, metro NaN=%d",
        len(out),
        n_cbd_nan,
        n_metro_nan,
    )
    return out


__all__ = [
    "CITY_CENTERS",
    "GEO_NUMERIC_FEATURES",
    "METRO_STATIONS",
    "add_distance_features",
    "haversine_km",
]
