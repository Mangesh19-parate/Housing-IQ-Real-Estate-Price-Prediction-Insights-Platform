"""Synthetic ``data/raw/`` builder for ingestion tests.

Avoids depending on the real ~182k-row CSVs in CI. Writes 2-row CSVs that
mirror the documented facet structure closely enough to exercise every code
path in ``ml.cleaning.ingest``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 4 city CSVs, each with a small set of representative columns (including
# at least one PII-flagged and one coded column per city).
_CITY_ROWS: dict[str, list[dict[str, object]]] = {
    "gurgaon_10k.csv": [
        {
            "PROP_ID": "G1",
            "PHOTO_URL": "https://example.com/photo1.jpg",
            "CITY": "Gurgaon",
            "CITY_ID": 8,
            "PROPERTY_TYPE": 1,
            "BEDROOM_NUM": 3,
            "BATHROOM_NUM": 2,
            "FURNISH": 4,
            "AGENT_NOTES": "Contact dealer",  # PII-ish
            "LOCALITY": "Sector 84",
        },
        {
            "PROP_ID": "G2",
            "PHOTO_URL": "https://example.com/photo2.jpg",
            "CITY": "Gurgaon",
            "CITY_ID": 8,
            "PROPERTY_TYPE": 1,
            "BEDROOM_NUM": 4,
            "BATHROOM_NUM": 3,
            "FURNISH": 4,
            "AGENT_NOTES": "WhatsApp for details",
            "LOCALITY": "Sector 81",
        },
    ],
    "hyderabad.csv": [
        {
            "PROP_ID": "H1",
            "SPID": "H1",
            "CITY": "Secunderabad",
            "PROPERTY_TYPE": "Residential Apartment",
            "BEDROOM_NUM": 2,
            "BATHROOM_NUM": 2,
            "FURNISH": 2,
            "AMENITIES": "5,17,20",
            "DESCRIPTION": "Test listing",
        },
        {
            "PROP_ID": "H2",
            "SPID": "H2",
            "CITY": "Hyderabad",
            "PROPERTY_TYPE": "Residential Apartment",
            "BEDROOM_NUM": 3,
            "BATHROOM_NUM": 3,
            "FURNISH": 2,
            "AMENITIES": "5,99,999",
            "DESCRIPTION": "Test listing 2",
        },
    ],
    "kolkata.csv": [
        {
            "PROP_ID": "K1",
            "CITY": "Kolkata",
            "PROPERTY_TYPE": 1,
            "BEDROOM_NUM": 2,
            "BATHROOM_NUM": 1,
            "FURNISH": 4,
        },
        {
            "PROP_ID": "K2",
            "CITY": "Kolkata",
            "PROPERTY_TYPE": 1,
            "BEDROOM_NUM": 3,
            "BATHROOM_NUM": 2,
            "FURNISH": 1,
        },
    ],
    "mumbai.csv": [
        {
            "PROP_ID": "M1",
            "SPID": "M1",
            "CITY": "Mumbai",
            "PROPERTY_TYPE": "Residential Apartment",
            "BEDROOM_NUM": 1,
            "BATHROOM_NUM": 1,
            "FURNISH": 2,
        },
        {
            "PROP_ID": "M2",
            "SPID": "M2",
            "CITY": "Mumbai",
            "PROPERTY_TYPE": "Residential Apartment",
            "BEDROOM_NUM": 2,
            "BATHROOM_NUM": 2,
            "FURNISH": 4,
        },
    ],
}


# Minimal ``id,label`` lookup tables covering all 15 facets. Real-file shapes
# may vary (some have 2-4 columns); we keep them simple for fixture purposes.
_FACET_ROWS: dict[str, list[dict[str, object]]] = {
    "AGE": [{"id": 1, "label": "1-5 Year Old Property"}, {"id": 2, "label": "5-10"}],
    "AMENITIES": [
        {"id": 5, "label": "Gym"},
        {"id": 17, "label": "Pool"},
        {"id": 20, "label": "Club"},
    ],
    "BATHROOM_NUM": [{"id": 1, "label": "1"}, {"id": 2, "label": "2"}],
    "BEDROOM_NUM": [{"id": 1, "label": "1"}, {"id": 2, "label": "2"}, {"id": 3, "label": "3"}],
    "BUILDING_ID": [{"id": 100, "label": "Building A"}],
    "CITY": [{"id": "008", "label": "Gurgaon"}],
    "FACING_DIRECTION": [{"id": 1, "label": "North"}, {"id": 2, "label": "South"}],
    "FEATURES": [{"id": 1, "label": "Lift"}, {"id": 2, "label": "Power Backup"}],
    "FLOOR_NUM": [{"id": "B", "label": "Basement"}, {"id": "1", "label": "1"}],
    "FURNISH": [
        {"id": 1, "label": "Furnished"},
        {"id": 2, "label": "Unfurnished"},
        {"id": 4, "label": "Semifurnished"},
    ],
    "LOCALITY_ID": [{"id": 100, "label": "Sector 1"}],
    "OWNERSHIP_TYPE": [{"id": 1, "label": "Freehold"}],
    "PROPERTY_TYPE": [{"id": 1, "label": "Residential Apartment"}],
    "SUB_AVAILABILITY": [{"id": 1, "label": "Ready"}],
    "TOTAL_FLOOR": [{"id": 1, "label": "1"}, {"id": 2, "label": "2"}],
}


def build_synthetic_raw(tmp_path: Path) -> Path:
    """Write 4 city CSVs + 15 facet CSVs under ``tmp_path/raw/``.

    Returns the synthetic *data_dir* (parent of ``raw/``).
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    facets_dir = raw_dir / "facets"
    facets_dir.mkdir(parents=True, exist_ok=True)

    for filename, rows in _CITY_ROWS.items():
        pd.DataFrame(rows).to_csv(raw_dir / filename, index=False)

    for name, rows in _FACET_ROWS.items():
        pd.DataFrame(rows).to_csv(facets_dir / f"{name}.csv", index=False)

    return tmp_path


def build_synthetic_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Return (data_dir, output_dir) for end-to-end ingestion tests."""
    data_dir = build_synthetic_raw(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir
