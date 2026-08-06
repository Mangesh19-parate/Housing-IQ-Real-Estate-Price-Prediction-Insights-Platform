"""Literal DataFrame fixtures for Step 06 (dedup + outliers + assemble) tests.

Pure literal ``pd.DataFrame``s and builder helpers — no real-data dependency,
no ``@pytest.fixture`` decorators. Mirrors the style of
``tests/fixtures/canonical_mapping_fixtures.py``.

Each helper builds a small canonical-frame slice with explicit column values
so test bodies can assert on known inputs/outputs without fixtures hiding
shape assumptions.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# A minimal subset of CANONICAL_COLUMNS that exercises the dedup + outlier
# code paths without dragging in 39 columns per row.
CANONICAL_SUBSET: Final[tuple[str, ...]] = (
    "listing_id",
    "city",
    "property_type",
    "register_date",
    "bedRoom",
    "bathroom",
    "price_inr",
    "area_sqft",
    "price_per_sqft",
)


def make_frame(rows: list[dict], columns: tuple[str, ...] = CANONICAL_SUBSET) -> pd.DataFrame:
    """Build a DataFrame from a list of dicts, defaulting missing cells to NaN.

    Each dict must include at least the columns it cares about; missing
    columns default to ``NaN`` (matching Step 05's canonical frame behavior).
    """
    if not rows:
        return pd.DataFrame({c: [] for c in columns})
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(columns)].reset_index(drop=True)


# A "normal" Gurgaon row used by multiple tests as a baseline.
NORMAL_ROW: Final[dict] = {
    "listing_id": "G00000001",
    "city": "Gurgaon",
    "property_type": "flat",
    "register_date": "29th Sep, 2023",
    "bedRoom": 3,
    "bathroom": 3,
    "price_inr": 15_000_000.0,
    "area_sqft": 1500.0,
    "price_per_sqft": 10_000.0,
}


def make_multi_city_frame() -> pd.DataFrame:
    """Synthetic 4-city frame with mixed outlier / non-outlier / duplicate rows.

    Covers every code path exercised by the tests:
      * 2 Gurgaon rows (1 normal, 1 with extreme price_inr)
      * 1 Hyderabad normal row
      * 1 Kolkata normal row with NaN register_date
      * 1 Mumbai row with duplicate listing_id (matched against first Gurgaon row)
      * 1 Mumbai row with bedRoom > 15 (domain rule outlier)
    """
    rows = [
        NORMAL_ROW,
        {**NORMAL_ROW, "listing_id": "G00000002", "price_inr": 20_000_000.0},
        {
            "listing_id": "H00000001",
            "city": "Hyderabad",
            "property_type": "flat",
            "register_date": "1st Jan, 2024",
            "bedRoom": 2,
            "bathroom": 2,
            "price_inr": 8_000_000.0,
            "area_sqft": 1100.0,
            "price_per_sqft": 7_272.73,
        },
        {
            "listing_id": "K00000001",
            "city": "Kolkata",
            "property_type": "flat",
            "register_date": pd.NA,
            "bedRoom": 2,
            "bathroom": 1,
            "price_inr": 5_000_000.0,
            "area_sqft": 950.0,
            "price_per_sqft": 5_263.16,
        },
        # Duplicate of the first Gurgaon row → dedup should drop this.
        {**NORMAL_ROW, "listing_id": "M00000001", "city": "Mumbai"},
        # Domain-rule outlier: bedRoom > 15 with property_type=flat → flagged.
        {
            "listing_id": "M00000002",
            "city": "Mumbai",
            "property_type": "flat",
            "register_date": "15th Mar, 2025",
            "bedRoom": 25,
            "bathroom": 5,
            "price_inr": 50_000_000.0,
            "area_sqft": 4000.0,
            "price_per_sqft": 12_500.0,
        },
    ]
    return make_frame(rows)
