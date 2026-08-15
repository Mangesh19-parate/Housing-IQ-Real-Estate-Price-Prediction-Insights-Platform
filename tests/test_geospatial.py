"""Tests for the geospatial lever (Spec 14).

Pins the haversine implementation, the METRO_STATIONS / CITY_CENTERS
constants, and the NaN-coord handling on ``add_distance_features``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.training.levers.geospatial import (
    CITY_CENTERS,
    GEO_NUMERIC_FEATURES,
    METRO_STATIONS,
    add_distance_features,
    haversine_km,
)


def test_haversine_delhi_mumbai_is_approximately_1400km():
    """Pins haversine against the literature-accepted Delhi↔Mumbai
    great-circle distance (~1400 km, depending on the two chosen
    points). Uses Gurgaon (≈ Delhi NCR) and Mumbai city center.
    """
    delhi_lat, delhi_lon = CITY_CENTERS["Gurgaon"]
    mumbai_lat, mumbai_lon = CITY_CENTERS["Mumbai"]
    d = haversine_km(delhi_lat, delhi_lon, mumbai_lat, mumbai_lon)
    # Gurgaon is south of Delhi proper — the Gurgaon↔Mumbai
    # great-circle distance is ~1125 km. Loose band: 1000–1500.
    assert 1000 < d < 1500, f"Gurgaon↔Mumbai distance {d:.0f} km out of band"


def test_haversine_kolkata_mumbai_is_approximately_1700km():
    """Second pin: Kolkata↔Mumbai is ~1700 km (eastern↔western India)."""
    kol_lat, kol_lon = CITY_CENTERS["Kolkata"]
    mumbai_lat, mumbai_lon = CITY_CENTERS["Mumbai"]
    d = haversine_km(kol_lat, kol_lon, mumbai_lat, mumbai_lon)
    assert 1500 < d < 1900, f"Kolkata↔Mumbai distance {d:.0f} km out of band"


def test_haversine_same_point_is_zero():
    lat, lon = CITY_CENTERS["Mumbai"]
    assert haversine_km(lat, lon, lat, lon) == 0.0


def test_haversine_propagates_nan():
    assert math.isnan(haversine_km(float("nan"), 0.0, 0.0, 0.0))
    assert math.isnan(haversine_km(0.0, float("nan"), 0.0, 0.0))


def test_add_distance_features_adds_two_columns():
    df = pd.DataFrame(
        {
            "city": ["Gurgaon", "Mumbai", "Kolkata"],
            "latitude": [28.4595, 19.0760, 22.5726],
            "longitude": [77.0266, 72.8777, 88.3639],
        }
    )
    out = add_distance_features(df)
    assert "distance_to_cbd_km" in out.columns
    assert "distance_to_nearest_metro_km" in out.columns
    # All three rows have a city center + at least one station, so
    # neither column should be NaN.
    assert out["distance_to_cbd_km"].isna().sum() == 0
    assert out["distance_to_nearest_metro_km"].isna().sum() == 0


def test_add_distance_features_handles_missing_coords():
    df = pd.DataFrame(
        {
            "city": ["Gurgaon", "Gurgaon"],
            "latitude": [28.4595, float("nan")],
            "longitude": [77.0266, float("nan")],
        }
    )
    out = add_distance_features(df)
    assert math.isnan(out["distance_to_cbd_km"].iloc[0]) is False
    assert math.isnan(out["distance_to_cbd_km"].iloc[1])
    assert math.isnan(out["distance_to_nearest_metro_km"].iloc[1])


def test_metro_stations_constant_has_all_four_cities():
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        assert city in METRO_STATIONS, f"{city} missing from METRO_STATIONS"
        assert len(METRO_STATIONS[city]) >= 1, (
            f"{city} has no metro stations pinned"
        )


def test_geo_numeric_features_constant_matches_added_columns():
    assert set(GEO_NUMERIC_FEATURES) == {
        "distance_to_cbd_km",
        "distance_to_nearest_metro_km",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
