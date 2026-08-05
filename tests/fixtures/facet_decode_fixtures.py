"""Literal facet DataFrames for ``ml.cleaning.facet_decoders`` unit tests.

Pure literal ``pd.DataFrame``s — no real-data dependency. Each fixture is
one ``_ROWS`` literal list + one ``_DF = pd.DataFrame(...)`` constant,
matching the style of ``tests/fixtures/parse_fixtures.py`` (no
``@pytest.fixture`` functions, no fixtures, just importable constants).

ID dtypes mirror the real ``data/raw/facets/*.csv`` dtypes:
  - int64 for FURNISH, FACING, AGE, PROPERTY_TYPE, OWNERSHIP_TYPE,
    LOCALITY_ID, BUILDING_ID, BATHROOM_NUM, TOTAL_FLOOR,
    SUB_AVAILABILITY, CITY, AMENITIES, FEATURES
  - str for BEDROOM_NUM and FLOOR_NUM (alphabetic floor codes)
"""

from __future__ import annotations

import pandas as pd

# Each facet gets a small set of known rows plus, where useful, a sentinel
# "high" ID the tests can use to exercise the "above max" branch.

# ---------------------------------------------------------------------------
# Single-value facets
# ---------------------------------------------------------------------------

FURNISH_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "Furnished"},
    {"id": 2, "label": "Unfurnished"},
    {"id": 4, "label": "Semifurnished"},
]
FURNISH_DF: pd.DataFrame = pd.DataFrame(FURNISH_ROWS)

FACING_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "North"},
    {"id": 2, "label": "South"},
    {"id": 3, "label": "East"},
    {"id": 4, "label": "West"},
]
FACING_DF: pd.DataFrame = pd.DataFrame(FACING_ROWS)

AGE_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "New Property"},
    {"id": 2, "label": "Relatively New"},
    {"id": 3, "label": "Moderately Old"},
    {"id": 4, "label": "Old Property"},
]
AGE_DF: pd.DataFrame = pd.DataFrame(AGE_ROWS)

PROPERTY_TYPE_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "Residential Apartment"},
    {"id": 2, "label": "Independent House/Villa"},
]
PROPERTY_TYPE_DF: pd.DataFrame = pd.DataFrame(PROPERTY_TYPE_ROWS)

OWNTYPE_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "Freehold"},
    {"id": 2, "label": "Leasehold"},
    {"id": 3, "label": "Co-operative Society"},
]
OWNTYPE_DF: pd.DataFrame = pd.DataFrame(OWNTYPE_ROWS)

LOCALITY_ID_ROWS: list[dict[str, object]] = [
    {"id": 10014, "label": "Sector 113 Gurgaon"},
    {"id": 10358, "label": "South City 2"},
]
LOCALITY_ID_DF: pd.DataFrame = pd.DataFrame(LOCALITY_ID_ROWS)

BUILDING_ID_ROWS: list[dict[str, object]] = [
    {"id": 1000501, "label": "Emaar Digihomes"},
    {"id": 10534, "label": "SS The Leaf"},
]
BUILDING_ID_DF: pd.DataFrame = pd.DataFrame(BUILDING_ID_ROWS)

# BEDROOM_NUM IDs are strings (matches real data).
BEDROOM_NUM_ROWS: list[dict[str, object]] = [
    {"id": "1", "label": "1"},
    {"id": "2", "label": "2"},
    {"id": "3", "label": "3"},
    {"id": "4", "label": "4"},
]
BEDROOM_NUM_DF: pd.DataFrame = pd.DataFrame(BEDROOM_NUM_ROWS)

BATHROOM_NUM_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "1"},
    {"id": 2, "label": "2"},
    {"id": 3, "label": "3"},
]
BATHROOM_NUM_DF: pd.DataFrame = pd.DataFrame(BATHROOM_NUM_ROWS)

# FLOOR_NUM: alphabetic string codes + numeric string floors, capped at 50
# so decode_floor_num(95, ...) reliably exercises the "above max" branch.
FLOOR_NUM_ROWS: list[dict[str, object]] = [
    {"id": "B", "label": "Basement"},
    {"id": "G", "label": "Ground"},
    {"id": "L", "label": "Lower Ground"},
    {"id": "M", "label": "Multi-Storied"},
    {"id": "1", "label": "1"},
    {"id": "5", "label": "5"},
    {"id": "10", "label": "10"},
    {"id": "50", "label": "50"},
]
FLOOR_NUM_DF: pd.DataFrame = pd.DataFrame(FLOOR_NUM_ROWS)

TOTAL_FLOOR_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "1"},
    {"id": 5, "label": "5"},
    {"id": 14, "label": "14"},
]
TOTAL_FLOOR_DF: pd.DataFrame = pd.DataFrame(TOTAL_FLOOR_ROWS)

SUB_AVAILABILITY_ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "Ready To Move"},
    {"id": 2, "label": "Under Construction"},
]
SUB_AVAILABILITY_DF: pd.DataFrame = pd.DataFrame(SUB_AVAILABILITY_ROWS)

# CITY: raw CITY is already a string label in the data; the facet CSV has
# only one row ("008" → "Gurgaon") for traceability, but the test exercises
# the decoder via the same index-lookup path.
CITY_ROWS: list[dict[str, object]] = [
    {"id": "Gurgaon", "label": "Gurgaon"},
    {"id": "Hyderabad", "label": "Hyderabad"},
    {"id": "Mumbai", "label": "Mumbai"},
    {"id": "Kolkata", "label": "Kolkata"},
]
CITY_DF: pd.DataFrame = pd.DataFrame(CITY_ROWS)

# ---------------------------------------------------------------------------
# Multi-value facets — match real 4-column shape (id,category,type,label)
# ---------------------------------------------------------------------------

# AMENITIES uses 4 columns to mirror the real file. Decoders only read the
# `id` and `label` columns, but the test asserts that the extra columns
# don't break the loader.
AMENITIES_ROWS: list[dict[str, object]] = [
    {"id": 20, "category": "Other Features", "type": "FEATURES_RESIDENTIAL", "label": "Club House"},
    {"id": 21, "category": "Other Features", "type": "FEATURES_RESIDENTIAL", "label": "Swimming Pool"},  # noqa: E501
    {"id": 23, "category": "Property Feature", "type": "FEATURES_COMMERCIAL", "label": "Power Back-up"},  # noqa: E501
    {"id": 33, "category": "Property Feature", "type": "FEATURES_RESIDENTIAL", "label": "Feng Shui Compliant"},  # noqa: E501
]
AMENITIES_DF: pd.DataFrame = pd.DataFrame(AMENITIES_ROWS)

FEATURES_ROWS: list[dict[str, object]] = [
    {"id": 12, "category": "Property Feature", "type": "FEATURES_RESIDENTIAL", "label": "Lift"},
    {"id": 23, "category": "Property Feature", "type": "FEATURES_COMMERCIAL", "label": "Power Back-up"},  # noqa: E501
    {"id": 33, "category": "Property Feature", "type": "FEATURES_RESIDENTIAL", "label": "Feng Shui Compliant"},  # noqa: E501
]
FEATURES_DF: pd.DataFrame = pd.DataFrame(FEATURES_ROWS)


# ---------------------------------------------------------------------------
# Helper: build a complete synthetic ``data/raw/facets/`` directory tree.
#
# Used by tests/test_facet_decoders.py for the load_facet_frames tests.
# Writes 15 facet CSVs (matching the real FACET_NAMES list) under
# ``tmp_path/raw/facets/`` and returns the synthetic ``data_dir`` (parent
# of ``raw/``). Mirrors the pattern of
# ``tests/fixtures/raw_snapshot_fixture.py:build_synthetic_raw`` but
# keeps the fixture local because the shared one uses 2-column tables
# only, while AMENITIES / FEATURES need 4 columns to match real data.
# ---------------------------------------------------------------------------


# Per-facet rows for the synthetic 15-file fixture (using the same content
# as the per-decoder frames above so decoder behavior is consistent across
# individual decoder tests and load_facet_frames tests).
_ALL_FACET_ROWS: dict[str, list[dict[str, object]]] = {
    "AGE": AGE_ROWS,
    "AMENITIES": AMENITIES_ROWS,
    "BATHROOM_NUM": BATHROOM_NUM_ROWS,
    "BEDROOM_NUM": BEDROOM_NUM_ROWS,
    "BUILDING_ID": BUILDING_ID_ROWS,
    "CITY": CITY_ROWS,
    "FACING_DIRECTION": FACING_ROWS,
    "FEATURES": FEATURES_ROWS,
    "FLOOR_NUM": FLOOR_NUM_ROWS,
    "FURNISH": FURNISH_ROWS,
    "LOCALITY_ID": LOCALITY_ID_ROWS,
    "OWNERSHIP_TYPE": OWNTYPE_ROWS,
    "PROPERTY_TYPE": PROPERTY_TYPE_ROWS,
    "SUB_AVAILABILITY": SUB_AVAILABILITY_ROWS,
    "TOTAL_FLOOR": TOTAL_FLOOR_ROWS,
}


def build_synthetic_facets_dir(tmp_path) -> "pd.DataFrame":
    """Write 15 facet CSVs under ``tmp_path/raw/facets/``.

    Returns the synthetic ``data_dir`` Path (parent of ``raw/``).
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    facets_dir = raw_dir / "facets"
    facets_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in _ALL_FACET_ROWS.items():
        pd.DataFrame(rows).to_csv(facets_dir / f"{name}.csv", index=False)
    return tmp_path
