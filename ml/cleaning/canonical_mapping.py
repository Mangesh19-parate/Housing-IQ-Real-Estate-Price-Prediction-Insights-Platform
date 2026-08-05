"""Step 05 — Canonical Schema Mapping Per City.

The central data-assembly step of the Week 1 cleaning chain. Takes each
of the four raw city CSVs (``gurgaon_10k.csv``, ``hyderabad.csv``,
``kolkata.csv``, ``mumbai.csv``) through the building blocks shipped in
Steps 02-04 — ``load_raw_listings`` from :mod:`ml.cleaning.ingest`,
``parse_price``/``parse_area`` from :mod:`ml.cleaning.parsing`, and the
13 single-value + 2 multi-value ``decode_*`` functions from
:mod:`ml.cleaning.facet_decoders` — and emits one per-city DataFrame
conforming to the **canonical schema** below.

Schema authority (do not diverge):

- The 16 input-contract fields from ``10-FINALIZED-INPUT-SCHEMA.md`` §3
  (city, sector, property_type, transact_type, bedRoom, bathroom,
  balcony, agePossession, built_up_area, servant_room, store_room,
  furnishing_type, luxury_category, floor_category, facing,
  amenities_list). Note these names intentionally keep their
  reference-project contract spelling (``bedRoom`` camelCase etc.) — do
  NOT snake_case-ify them.
- The ~28-column extended schema from ``05-BACKEND-SCHEMA.md`` §2 +
  §U-SCHEMA-5 (which is a superset of the 16 input-contract fields plus
  downstream-only columns like ``price_per_sqft``, ``floor_ratio``,
  ``is_outlier``, ``was_missing_*``, etc.).

Per-city mappers do **not** write ``clean_listings.parquet`` — that's
Step 06's job (Day 5 of the Implementation Plan). This module only
returns in-memory ``pd.DataFrame``s. Per Rules §10.2 ``luxury_category``
is left as NaN — it is derived server-side from the amenity checklist
at prediction time, never self-reported.

Public API:
    CANONICAL_COLUMNS                       # tuple[str, ...] (~37 names)
    CITY_COLUMN_ALIASES                     # dict[str, dict[str, str]]
    CITY_FRAME_LOADERS                      # dict[str, Callable[[Path, dict], pd.DataFrame]]
    UNSAFE_COLUMNS                          # tuple[str, ...] (PII/URL/scrape-id drop list)
    clean_description(series) -> series
    normalize_columns(df, city) -> df
    map_city(name, raw_path, facet_frames) -> df
    map_gurgaon(raw_path, facet_frames) -> df
    map_hyderabad(raw_path, facet_frames) -> df
    map_kolkata(raw_path, facet_frames) -> df
    map_mumbai(raw_path, facet_frames) -> df
"""

from __future__ import annotations

import ast
import html
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd

from ml.cleaning.facet_decoders import (
    decode_age,
    decode_amenities,
    decode_bathroom_num,
    decode_bedroom_num,
    decode_building_id,
    decode_facing,
    decode_features,
    decode_floor_num,
    decode_furnish,
    decode_locality_id,
    decode_owntype,
    decode_property_type,
    decode_sub_availability,
    decode_total_floor,
)
from ml.cleaning.parsing import _log_unparseable, parse_area, parse_price

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SPEC_VERSION: Final[str] = "05-canonical-schema-mapping-v1"

_LOG: Final = logging.getLogger("ml.cleaning.canonical_mapping")

# Single grep-able list of columns that must never reach the UI or any
# export (Rules §1.1). Compiled from 02-TRD.md §4.6 + the Step 02
# `_DRAFT_CANONICAL_MAP` DROP entries + the Step 02 PII pattern catches.
# Anything matching PII_PATTERN (phone|tel|mobile|contact|dealer|agent|
# email|url|link|photo|image|img|src|media|whatsapp) is dropped, plus the
# scrape IDs and nested-dict noise columns that don't carry a PII token
# but carry no model signal either.
UNSAFE_COLUMNS: Final[tuple[str, ...]] = (
    # Photo / media URLs
    "PHOTO_URL",
    "MEDIUM_PHOTO_URL",
    "THUMBNAIL_PHOTO_URL",
    "LARGE_PHOTO_URL",
    "DEALER_PHOTO_URL",
    "PROPERTY_IMAGES",
    "THUMBNAIL_IMAGES",
    # Navigation / listing URLs
    "PROP_DETAILS_URL",
    "PD_URL",
    "PROP_URL",
    "URL",
    # Contact fields
    "CONTACT_NAME",
    "CONTACT_COMPANY_NAME",
    "DEALER_NAME",
    "DEALER_COMPANY",
    "DEALER_PHONE",
    "PHONE_NUMBER",
    "CONTACT_PHONE",
    "DEALER_EMAIL",
    "CONTACT_EMAIL",
    # Internal scrape IDs / nested-dict noise
    "SPID",
    "SCRAPED_AT",
    "RAW_JSON",
    "FSL_Data",
    "profile",
    "xid",
    "metadata",
    "COMMON_FURNISHING_ATTRIBUTES",
    "QUALITY_SCORE",
    "FURNISHING_ATTRIBUTES",
    "FSL_Data",
)

# Raw ``TRANSACT_TYPE`` values are coded ``1.0``=Sale, ``2.0``=Rent across
# all four cities — the schema calls for the human-readable string.
_TRANSACT_TYPE_CODE_TO_LABEL: Final[dict[float, str]] = {
    1.0: "Sale",
    2.0: "Rent",
}

# HTML / URL / email / whitespace normalizers used by clean_description().
_RE_HTML_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_RE_URL: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL: Final[re.Pattern[str]] = re.compile(r"\S+@\S+")
_RE_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# CANONICAL_COLUMNS — single source of truth for the column order of the
# per-city canonical DataFrame. Combines the 16 input-contract fields
# (verbatim names) and the extended-schema fields from
# 05-BACKEND-SCHEMA.md §2 + §U-SCHEMA-5.
# ---------------------------------------------------------------------------
#
# Notes on the names:
#   - The 16 input-contract fields retain their reference-project contract
#     spelling: bedRoom (camelCase), built_up_area, agePossession,
#     furnishing_type, luxury_category, floor_category, servant_room,
#     store_room, balcony (categorical string per finalized schema §1).
#     These are NOT renamed to snake_case — that would break the contract
#     with the reference project and with the FastAPI Pydantic schemas.
#   - The extended schema fields use snake_case throughout.
#   - ``bedrooms`` / ``bathrooms`` (plural) and ``bedRoom`` / ``bathroom``
#     (singular-camelCase) BOTH appear because both 02-BACKEND-SCHEMA §2
#     and §U-SCHEMA-5 call them out separately. The mappers populate both
#     columns from the same source value (BEDROOM_NUM / BATHROOM_NUM)
#     where the city has those columns; cities missing the source get
#     NaN for both.
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    # Identity / locality (extended schema)
    "listing_id",
    "city",
    "sector",
    "locality",
    # Transaction / ownership
    "transact_type",
    "ownership_type",
    "property_type",
    # Numeric core (extended-schema plural form)
    "bedrooms",
    "bathrooms",
    "balconies",
    # Numeric core (16-input-contract singular-camelCase form)
    "bedRoom",
    "bathroom",
    "balcony",
    "servant_room",
    "store_room",
    # Categorical core
    "furnish",
    "furnishing_type",
    "facing",
    "age_bucket",
    "agePossession",
    "floor_num",
    "total_floor",
    "floor_category",
    "luxury_category",
    # Areas / prices (extended)
    "area_sqft",
    "built_up_area",
    "price_inr",
    "price_per_sqft",
    "floor_ratio",
    # Multi-value / derived
    "features_list",
    "amenities_list",
    "n_amenities",
    "n_features",
    # Building / geo
    "building_name",
    "building_id",
    "latitude",
    "longitude",
    # Free text
    "description_clean",
    # Temporal / lineage
    "register_date",
    "is_outlier",
)

# ---------------------------------------------------------------------------
# CITY_COLUMN_ALIASES — per-city {canonical_name: raw_column_name}.
# Only canonical columns the city actually has a source for are listed.
# The mapper fills missing canonicals with NaN by reindexing against
# CANONICAL_COLUMNS at the end.
#
# Sources cross-referenced against the actual raw CSV bytes (Step 02
# inventory) and the Step 02 _DRAFT_CANONICAL_MAP.
# ---------------------------------------------------------------------------

CITY_COLUMN_ALIASES: Final[dict[str, dict[str, str]]] = {
    # Gurgaon — 67-col superset, has flat LOCALITY and PII/url columns.
    "Gurgaon": {
        "listing_id": "PROP_ID",
        "city": "CITY",
        "locality": "LOCALITY",
        "property_type": "PROPERTY_TYPE",
        "transact_type": "TRANSACT_TYPE",
        "ownership_type": "OWNTYPE",
        "bedrooms": "BEDROOM_NUM",
        "bathrooms": "BATHROOM_NUM",
        "balconies": "BALCONY_NUM",
        "balcony": "BALCONY_NUM",
        "furnish": "FURNISH",
        "furnishing_type": "FURNISH",
        "facing": "FACING",
        "age_bucket": "AGE",
        "agePossession": "AGE",
        "floor_num": "FLOOR_NUM",
        "total_floor": "TOTAL_FLOOR",
        "area_sqft": "AREA",
        "built_up_area": "AREA",
        "price_inr": "PRICE",
        "features_list": "FEATURES",
        "amenities_list": "AMENITIES",
        "building_name": "BUILDING_NAME",
        "building_id": "BUILDING_ID",
        "latitude": "MAP_DETAILS",
        "longitude": "MAP_DETAILS",
        "description_clean": "DESCRIPTION",
        "register_date": "REGISTER_DATE",
    },
    # Hyderabad — 55 cols; location is a nested dict (no flat LOCALITY),
    # ownership is VALUE_LABEL (string), no BATHROOM_NUM column.
    "Hyderabad": {
        "listing_id": "PROP_ID",
        "city": "CITY",
        "property_type": "PROPERTY_TYPE",
        "transact_type": "TRANSACT_TYPE",
        "ownership_type": "VALUE_LABEL",
        "bedrooms": "BEDROOM_NUM",
        "balconies": "BALCONY_NUM",
        "balcony": "BALCONY_NUM",
        "furnish": "FURNISH",
        "furnishing_type": "FURNISH",
        "facing": "FACING",
        "age_bucket": "AGE",
        "agePossession": "AGE",
        "floor_num": "FLOOR_NUM",
        "total_floor": "TOTAL_FLOOR",
        "area_sqft": "AREA",
        "built_up_area": "AREA",
        "price_inr": "PRICE",
        "features_list": "FEATURES",
        "amenities_list": "AMENITIES",
        "building_name": "BUILDING_NAME",
        "building_id": "BUILDING_ID",
        "latitude": "MAP_DETAILS",
        "longitude": "MAP_DETAILS",
        "description_clean": "DESCRIPTION",
        "register_date": "REGISTER_DATE",
    },
    # Kolkata — 35 cols (smallest schema); no BUILDING_ID, no
    # REGISTER_DATE, no BATHROOM_NUM; location nested dict only.
    "Kolkata": {
        "listing_id": "PROP_ID",
        "city": "CITY",
        "property_type": "PROPERTY_TYPE",
        "transact_type": "TRANSACT_TYPE",
        "ownership_type": "OWNTYPE",
        "bedrooms": "BEDROOM_NUM",
        "balconies": "BALCONY_NUM",
        "balcony": "BALCONY_NUM",
        "furnish": "FURNISH",
        "furnishing_type": "FURNISH",
        "facing": "FACING",
        "age_bucket": "AGE",
        "agePossession": "AGE",
        "floor_num": "FLOOR_NUM",
        "total_floor": "TOTAL_FLOOR",
        "area_sqft": "AREA",
        "built_up_area": "AREA",
        "price_inr": "PRICE",
        "features_list": "FEATURES",
        "amenities_list": "AMENITIES",
        "building_name": "BUILDING_NAME",
        "latitude": "MAP_DETAILS",
        "longitude": "MAP_DETAILS",
        "description_clean": "DESCRIPTION",
    },
    # Mumbai — 55 cols; same shape as Hyderabad (location nested dict,
    # VALUE_LABEL ownership).
    "Mumbai": {
        "listing_id": "PROP_ID",
        "city": "CITY",
        "property_type": "PROPERTY_TYPE",
        "transact_type": "TRANSACT_TYPE",
        "ownership_type": "VALUE_LABEL",
        "bedrooms": "BEDROOM_NUM",
        "balconies": "BALCONY_NUM",
        "balcony": "BALCONY_NUM",
        "furnish": "FURNISH",
        "furnishing_type": "FURNISH",
        "facing": "FACING",
        "age_bucket": "AGE",
        "agePossession": "AGE",
        "floor_num": "FLOOR_NUM",
        "total_floor": "TOTAL_FLOOR",
        "area_sqft": "AREA",
        "built_up_area": "AREA",
        "price_inr": "PRICE",
        "features_list": "FEATURES",
        "amenities_list": "AMENITIES",
        "building_name": "BUILDING_NAME",
        "building_id": "BUILDING_ID",
        "latitude": "MAP_DETAILS",
        "longitude": "MAP_DETAILS",
        "description_clean": "DESCRIPTION",
        "register_date": "REGISTER_DATE",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_description(series: pd.Series) -> pd.Series:
    """Vectorized DESCRIPTION cleanup: HTML-decode, strip tags, drop URLs
    and emails, lowercase, collapse whitespace.

    Per Rules §1.1 — even though DESCRIPTION itself is not PII, raw
    agent copy frequently contains embedded contact info (URLs, emails).
    The minimum needed for downstream TF-IDF (Week 5); no stemming /
    lemmatization / stop-word removal here — that's a training-time
    concern. NaN passes through unchanged.
    """
    if series.empty:
        return series

    result = series.astype("object").copy()
    is_na = series.isna()
    str_mask = ~is_na

    if str_mask.any():
        values = series[str_mask].astype(str)
        # Order matters: HTML-decode first (so &#x...; entities become real
        # chars), then strip tags (the tags won't have those entities anymore).
        values = values.map(html.unescape)
        values = values.str.replace(_RE_HTML_TAG, " ", regex=True)
        values = values.str.replace(_RE_URL, " ", regex=True)
        values = values.str.replace(_RE_EMAIL, " ", regex=True)
        values = values.str.lower()
        values = values.str.replace(_RE_WHITESPACE_RUN, " ", regex=True)
        values = values.str.strip()
        result.loc[str_mask] = values

    result.loc[is_na] = pd.NA
    return result


def _parse_map_details(value: object) -> tuple[float | None, float | None]:
    """Extract (latitude, longitude) from a raw MAP_DETAILS dict-string.

    Shape: ``"{'LATITUDE': '28.40...', 'LONGITUDE': '76.96...', ...}"``.

    Step 03 did not ship ``parse_map_details`` (only ``parse_price`` and
    ``parse_area``), so this helper lives in Step 05 instead. Mirrors the
    Step 03 convention: never raises on bad input — returns
    ``(None, None)`` for NaN, malformed, missing keys, or non-numeric
    coords. Failure is logged via ``_log_unparseable`` so log lines stay
    consistent with Step 03/04.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (None, None)
    if not isinstance(value, str):
        return (None, None)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        _log_unparseable("MAP_DETAILS", value)
        return (None, None)
    if not isinstance(parsed, dict):
        _log_unparseable("MAP_DETAILS", value)
        return (None, None)
    lat_raw = parsed.get("LATITUDE")
    lon_raw = parsed.get("LONGITUDE")
    try:
        lat = float(lat_raw) if lat_raw is not None else None
        lon = float(lon_raw) if lon_raw is not None else None
    except (TypeError, ValueError):
        _log_unparseable("MAP_DETAILS", value)
        return (None, None)
    return (lat, lon)


def _extract_location_dict(value: object) -> dict | None:
    """Decode the ``location`` nested-dict string used by Hyderabad /
    Mumbai / Kolkata. Returns ``None`` for NaN / malformed / non-dict.
    Never raises.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        _log_unparseable("location", value)
        return None
    return parsed if isinstance(parsed, dict) else None


def _locality_from_location(value: object) -> str | None:
    loc = _extract_location_dict(value)
    if loc is None:
        return None
    name = loc.get("LOCALITY_NAME")
    return str(name).strip() if name is not None else None


def _building_from_location(value: object) -> tuple[str | None, int | None]:
    loc = _extract_location_dict(value)
    if loc is None:
        return (None, None)
    name = loc.get("BUILDING_NAME")
    bid = loc.get("BUILDING_ID")
    name = str(name).strip() if name else None
    if not name:
        name = None
    try:
        bid_int = int(bid) if bid is not None and bid != "" else None
    except (TypeError, ValueError):
        bid_int = None
    return (name, bid_int)


# ---------------------------------------------------------------------------
# normalize_columns
# ---------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Apply CITY_COLUMN_ALIASES[city] rename + UNSAFE_COLUMNS drop to a
    raw per-city DataFrame.

    Drops any column whose name is in ``UNSAFE_COLUMNS`` (PII / URL /
    scrape-id) so unsafe columns never enter the canonical frame in any
    form. Renames the remaining raw columns via the city's alias dict.
    Columns not in the alias dict are passed through unchanged under
    their raw name (they get dropped by the reindex-to-CANONICAL_COLUMNS
    step in the per-city mapper, but only if they're not in the canonical
    set — this preserves ``SOME_GURGAON_SPECIFIC_FIELD`` semantics if a
    caller wants to inspect pre-reindex output).

    The function does NOT add new canonical columns (e.g.,
    ``description_clean``), apply decoders, parse prices, or reindex to
    ``CANONICAL_COLUMNS`` — those are the per-city mapper's job.
    """
    if city not in CITY_COLUMN_ALIASES:
        raise ValueError(f"Unknown city: {city!r}. Expected one of {list(CITY_COLUMN_ALIASES)}.")
    alias = CITY_COLUMN_ALIASES[city]

    # Drop unsafe columns first (rename never accidentally re-introduces one).
    cols_to_drop = [c for c in df.columns if c in UNSAFE_COLUMNS]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Rename only columns that are present; leave everything else alone.
    present_renames = {raw: canon for raw, canon in alias.items() if raw in df.columns}
    if present_renames:
        df = df.rename(columns=present_renames)

    return df


# ---------------------------------------------------------------------------
# Per-city mappers
# ---------------------------------------------------------------------------


def _decode_series(series: pd.Series, decoder: Callable, facet_name: str) -> pd.Series:
    """Vectorized apply of a single-value Step-04 decoder to a Series.

    Decoders expect ``(value, facet_df)``; we pass the relevant facet
    frame in via a closure. NaN values pass through as None per the
    decoder contract.
    """
    return series.map(lambda v: decoder(v, _FACET_LOOKUP[facet_name]))


def _build_decoders(facet_frames: dict[str, pd.DataFrame]) -> dict[str, Callable]:
    """Bind each Step-04 decoder to its facet frame for fast lookup."""
    return {
        "FURNISH": lambda v: decode_furnish(v, facet_frames["FURNISH"]),
        "FACING": lambda v: decode_facing(v, facet_frames["FACING_DIRECTION"]),
        "AGE": lambda v: decode_age(v, facet_frames["AGE"]),
        "PROPERTY_TYPE": lambda v: decode_property_type(v, facet_frames["PROPERTY_TYPE"]),
        "OWNERSHIP_TYPE": lambda v: decode_owntype(v, facet_frames["OWNERSHIP_TYPE"]),
        "LOCALITY_ID": lambda v: decode_locality_id(v, facet_frames["LOCALITY_ID"]),
        "BUILDING_ID": lambda v: decode_building_id(v, facet_frames["BUILDING_ID"]),
        "BEDROOM_NUM": lambda v: decode_bedroom_num(v, facet_frames["BEDROOM_NUM"]),
        "BATHROOM_NUM": lambda v: decode_bathroom_num(v, facet_frames["BATHROOM_NUM"]),
        "FLOOR_NUM": lambda v: decode_floor_num(v, facet_frames["FLOOR_NUM"]),
        "TOTAL_FLOOR": lambda v: decode_total_floor(v, facet_frames["TOTAL_FLOOR"]),
        "SUB_AVAILABILITY": lambda v: decode_sub_availability(v, facet_frames["SUB_AVAILABILITY"]),
    }


# Pre-built decoders (one per call). The mapper accepts a `facet_frames`
# dict and constructs its bound decoders inside each map_<city>() call so
# the public signature stays explicit (no hidden module-level state).
_FACET_LOOKUP: dict[str, pd.DataFrame] = {}


def _populate_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical columns that are intentionally NaN at this stage.

    Per the spec: imputation flags, outlier flags, engineered features,
    and luxury_category are NOT computed here. Step 06 owns imputation,
    Step 07 owns outlier flagging, Step 08 owns feature engineering, and
    luxury_category is server-derived at prediction time per Rules §10.2.
    """
    df["floor_ratio"] = pd.NA
    df["price_per_sqft"] = pd.NA
    df["n_amenities"] = pd.NA
    df["n_features"] = pd.NA
    df["sector"] = pd.NA
    df["floor_category"] = pd.NA
    df["luxury_category"] = pd.NA
    df["is_outlier"] = False
    return df


def _apply_shared_decoding(
    df: pd.DataFrame,
    decoders: dict[str, Callable],
    facet_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply the cross-city decoders + parsers to ``df`` (after
    normalize_columns). Each per-city mapper calls this with its
    city-specific quirks already layered on top (or skipped).
    """
    df["bedrooms"] = (
        df["BEDROOM_NUM"].map(decoders["BEDROOM_NUM"])
        if "BEDROOM_NUM" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["bedRoom"] = df["bedrooms"]
    df["bathrooms"] = (
        df["BATHROOM_NUM"].map(decoders["BATHROOM_NUM"])
        if "BATHROOM_NUM" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["bathroom"] = df["bathrooms"]
    df["balconies"] = (
        df["BALCONY_NUM"]
        if "BALCONY_NUM" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["balcony"] = df["balconies"]

    df["furnish"] = (
        df["FURNISH"].map(decoders["FURNISH"])
        if "FURNISH" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["furnishing_type"] = df["furnish"]
    df["facing"] = (
        df["FACING"].map(decoders["FACING"])
        if "FACING" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["age_bucket"] = (
        df["AGE"].map(decoders["AGE"])
        if "AGE" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["agePossession"] = df["age_bucket"]
    df["floor_num"] = (
        df["FLOOR_NUM"].map(decoders["FLOOR_NUM"])
        if "FLOOR_NUM" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["total_floor"] = (
        df["TOTAL_FLOOR"].map(decoders["TOTAL_FLOOR"])
        if "TOTAL_FLOOR" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )

    df["property_type"] = (
        df["PROPERTY_TYPE"].map(decoders["PROPERTY_TYPE"])
        if "PROPERTY_TYPE" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )

    df["features_list"] = (
        df["FEATURES"].map(lambda v: decode_features(v, facet_frames["FEATURES"]))
        if "FEATURES" in df.columns
        else pd.Series([[] for _ in range(len(df))], index=df.index)
    )
    df["amenities_list"] = (
        df["AMENITIES"].map(lambda v: decode_amenities(v, facet_frames["AMENITIES"]))
        if "AMENITIES" in df.columns
        else pd.Series([[] for _ in range(len(df))], index=df.index)
    )

    df["price_inr"] = (
        df["PRICE"].map(parse_price)
        if "PRICE" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["area_sqft"] = (
        df["AREA"].map(parse_area)
        if "AREA" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    df["built_up_area"] = df["area_sqft"]

    # Latitude / longitude from MAP_DETAILS (parsed via the local helper).
    if "MAP_DETAILS" in df.columns:
        coords = df["MAP_DETAILS"].map(_parse_map_details)
        df["latitude"] = coords.map(lambda c: c[0])
        df["longitude"] = coords.map(lambda c: c[1])
    else:
        df["latitude"] = pd.NA
        df["longitude"] = pd.NA

    # Transaction type: decode 1.0/2.0 → "Sale"/"Rent".
    if "TRANSACT_TYPE" in df.columns:
        df["transact_type"] = df["TRANSACT_TYPE"].map(
            lambda v: _TRANSACT_TYPE_CODE_TO_LABEL.get(float(v), None) if pd.notna(v) else None
        )
    else:
        df["transact_type"] = pd.NA

    return df


def _load_city_csv(raw_path: Path) -> pd.DataFrame:
    """Read the raw city CSV. Step 06's orchestrator can call
    ``load_raw_listings`` directly and dispatch by name; for tests we
    accept a Path so the synthetic-fixture builder can pass ``tmp_path``
    files directly.
    """
    return pd.read_csv(raw_path)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the city-agnostic tail of every mapper:
    engineered-column NaN fill + description cleanup + reindex to
    ``CANONICAL_COLUMNS``.
    """
    if "DESCRIPTION" in df.columns:
        df["description_clean"] = clean_description(df["DESCRIPTION"])
    elif "description_clean" not in df.columns:
        df["description_clean"] = pd.NA
    df = _populate_engineered_columns(df)
    df = df.reindex(columns=list(CANONICAL_COLUMNS), fill_value=pd.NA)
    return df


def map_gurgaon(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Canonical-schema mapper for the Gurgaon CSV."""
    df = _load_city_csv(raw_path)
    df = normalize_columns(df, "Gurgaon")
    decoders = _build_decoders(facet_frames)
    df = _apply_shared_decoding(df, decoders, facet_frames)

    # Gurgaon-specific: flat LOCALITY (already aliased by normalize_columns);
    # OWNTYPE is the ownership code (decode via Step 04); no nested
    # location dict (Gurgaon uses the flat columns).
    if "OWNERSHIP_TYPE" in df.columns:
        df["ownership_type"] = df["OWNERSHIP_TYPE"].map(decoders["OWNERSHIP_TYPE"])
    else:
        df["ownership_type"] = pd.NA

    # City column from the flat CITY field.
    if "CITY" in df.columns:
        df["city"] = df["CITY"]

    # listing_id from PROP_ID.
    if "PROP_ID" in df.columns:
        df["listing_id"] = df["PROP_ID"].astype("string")

    # building_id from BUILDING_ID facet decode.
    if "BUILDING_ID" in df.columns:
        df["building_id"] = df["BUILDING_ID"].map(decoders["BUILDING_ID"])
    else:
        df["building_id"] = pd.NA

    # building_name from the flat BUILDING_NAME column.
    if "BUILDING_NAME" in df.columns:
        df["building_name"] = df["BUILDING_NAME"]

    # register_date passes through (parse later).
    if "REGISTER_DATE" in df.columns:
        df["register_date"] = df["REGISTER_DATE"]

    return _finalize(df)


def map_hyderabad(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Canonical-schema mapper for the Hyderabad CSV.

    Per-city quirks:
      - ``location`` is a nested dict string; extract LOCALITY_NAME,
        BUILDING_NAME, BUILDING_ID from it.
      - ``VALUE_LABEL`` is the ownership string (not OWNTYPE).
      - No BATHROOM_NUM column (Hyderabad skipped it).
    """
    df = _load_city_csv(raw_path)
    df = normalize_columns(df, "Hyderabad")
    decoders = _build_decoders(facet_frames)
    df = _apply_shared_decoding(df, decoders, facet_frames)

    # No flat LOCALITY for Hyderabad; extract from nested location dict.
    if "location" in df.columns:
        df["locality"] = df["location"].map(_locality_from_location)
        bld = df["location"].map(_building_from_location)
        df["building_name"] = bld.map(lambda t: t[0])
        df["building_id"] = bld.map(lambda t: t[1])
    else:
        df["locality"] = pd.NA
        df["building_name"] = pd.NA
        df["building_id"] = pd.NA

    # ownership_type comes from VALUE_LABEL (string pass-through).
    if "VALUE_LABEL" in df.columns:
        df["ownership_type"] = df["VALUE_LABEL"]
    else:
        df["ownership_type"] = pd.NA

    if "CITY" in df.columns:
        df["city"] = df["CITY"]
    if "PROP_ID" in df.columns:
        df["listing_id"] = df["PROP_ID"].astype("string")
    if "REGISTER_DATE" in df.columns:
        df["register_date"] = df["REGISTER_DATE"]

    return _finalize(df)


def map_kolkata(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Canonical-schema mapper for the Kolkata CSV.

    Per-city quirks:
      - No BUILDING_ID, no REGISTER_DATE, no BATHROOM_NUM column.
      - location nested dict only (no BUILDING_NAME inside the dict for
        Kolkata; building_name comes from the flat BUILDING_NAME column
        when present, else NaN).
      - FEATURES is mostly ``"N"`` or NaN; AMENITIES mostly NaN.
    """
    df = _load_city_csv(raw_path)
    df = normalize_columns(df, "Kolkata")
    decoders = _build_decoders(facet_frames)
    df = _apply_shared_decoding(df, decoders, facet_frames)

    # Locality from nested location dict (Kolkata has no flat LOCALITY).
    if "location" in df.columns:
        df["locality"] = df["location"].map(_locality_from_location)
    else:
        df["locality"] = pd.NA

    # Kolkata's location dict lacks BUILDING_NAME in the inspected rows;
    # fall back to the flat BUILDING_NAME column.
    if "location" in df.columns:
        bld = df["location"].map(_building_from_location)
        bld_from_loc = bld.map(lambda t: t[0])
        if "BUILDING_NAME" in df.columns:
            df["building_name"] = bld_from_loc.fillna(df["BUILDING_NAME"])
        else:
            df["building_name"] = bld_from_loc
    elif "BUILDING_NAME" in df.columns:
        df["building_name"] = df["BUILDING_NAME"]
    else:
        df["building_name"] = pd.NA

    df["building_id"] = pd.NA  # No BUILDING_ID column at all in Kolkata.

    # OWNTYPE is the ownership code (no VALUE_LABEL for Kolkata).
    if "OWNERSHIP_TYPE" in df.columns:
        df["ownership_type"] = df["OWNERSHIP_TYPE"].map(decoders["OWNERSHIP_TYPE"])
    else:
        df["ownership_type"] = pd.NA

    if "CITY" in df.columns:
        df["city"] = df["CITY"]
    if "PROP_ID" in df.columns:
        df["listing_id"] = df["PROP_ID"].astype("string")
    # register_date has no source column for Kolkata — leave as NaN.

    return _finalize(df)


def map_mumbai(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Canonical-schema mapper for the Mumbai CSV.

    Per-city quirks (same shape as Hyderabad):
      - ``location`` is a nested dict string.
      - ``VALUE_LABEL`` is the ownership string.
      - FLOOR_NUM is dtype=str (contains B/G/L/M tokens), handled by the
        same ``decode_floor_num`` decoder — see Step 04 contract.
      - No BATHROOM_NUM column.
    """
    df = _load_city_csv(raw_path)
    df = normalize_columns(df, "Mumbai")
    decoders = _build_decoders(facet_frames)
    df = _apply_shared_decoding(df, decoders, facet_frames)

    if "location" in df.columns:
        df["locality"] = df["location"].map(_locality_from_location)
        bld = df["location"].map(_building_from_location)
        df["building_name"] = bld.map(lambda t: t[0])
        df["building_id"] = bld.map(lambda t: t[1])
    else:
        df["locality"] = pd.NA
        df["building_name"] = pd.NA
        df["building_id"] = pd.NA

    if "VALUE_LABEL" in df.columns:
        df["ownership_type"] = df["VALUE_LABEL"]
    else:
        df["ownership_type"] = pd.NA

    if "CITY" in df.columns:
        df["city"] = df["CITY"]
    if "PROP_ID" in df.columns:
        df["listing_id"] = df["PROP_ID"].astype("string")
    if "BUILDING_ID" in df.columns:
        df["building_id"] = df["BUILDING_ID"].map(decoders["BUILDING_ID"])
    if "REGISTER_DATE" in df.columns:
        df["register_date"] = df["REGISTER_DATE"]

    return _finalize(df)


# ---------------------------------------------------------------------------
# City-keyed dispatch table + map_city entry point
# ---------------------------------------------------------------------------


CITY_FRAME_LOADERS: Final[dict[str, Callable[[Path, dict[str, pd.DataFrame]], pd.DataFrame]]] = {
    "Gurgaon": map_gurgaon,
    "Hyderabad": map_hyderabad,
    "Kolkata": map_kolkata,
    "Mumbai": map_mumbai,
}


def map_city(
    name: str,
    raw_path: Path,
    facet_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Dispatch to the per-city mapper for ``name``.

    Raises ``ValueError`` if ``name`` is not one of the four cities in
    ``CITY_FRAME_LOADERS`` — no silent default.
    """
    if name not in CITY_FRAME_LOADERS:
        raise ValueError(
            f"Unknown city: {name!r}. Expected one of {sorted(CITY_FRAME_LOADERS)}."
        )
    return CITY_FRAME_LOADERS[name](raw_path, facet_frames)


# Silence unused-private-helper warning for _populate_engineered_columns
# alias kept for clarity. Not exported (leading underscore).
_ = _FACET_LOOKUP
