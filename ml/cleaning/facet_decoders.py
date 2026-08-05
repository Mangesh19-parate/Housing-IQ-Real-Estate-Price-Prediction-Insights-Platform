"""Step 04 — Facet Decoding Joins.

Per-facet ID→label decoders for the 15 coded columns in the raw city CSVs.
13 single-value decoders (FURNISH, FACING, AGE, etc.) + 2 multi-value
decoders (FEATURES, AMENITIES).

Pure functions, deterministic, no I/O beyond CSV reads at load time. The
runtime decoder functions never raise on bad input — missing/NaN returns
``None`` for single-value decoders and ``[]`` for multi-value decoders;
unknown-but-present IDs return ``DEFAULT_UNKNOWN_LABEL`` ("unknown") for
single-value decoders and are silently dropped (with a log warning) for
multi-value decoders, per the facet-decoding skill's binding rule.

Public API:
    load_facet_frames(facets_dir) -> dict[str, pd.DataFrame]
    decode_row(row, facet_frames) -> dict
    decode_furnish / decode_facing / decode_age / decode_property_type /
    decode_owntype / decode_locality_id / decode_building_id /
    decode_bedroom_num / decode_bathroom_num / decode_floor_num /
    decode_total_floor / decode_sub_availability / decode_city
        -> str | None
    decode_features / decode_amenities
        -> list[str]
    SINGLE_VALUE_FACETS, MULTI_VALUE_FACETS,
    DEFAULT_UNKNOWN_LABEL, MULTI_VALUE_DELIMITER

Source columns handled (per Step 02 inventory + 05-BACKEND-SCHEMA.md §3):
    FURNISH → furnishing_type, FACING → facing, AGE → agePossession,
    PROPERTY_TYPE → property_type, OWNTYPE → ownership_type,
    LOCALITY_ID → locality (note: many cities use free-text LOCALITY
    instead), BUILDING_ID → building_id,
    BEDROOM_NUM → bedRoom, BATHROOM_NUM → bathroom,
    FLOOR_NUM → floor_num (with string codes B/G/L/M and "95+"
    domain-rule fallback for codes above the facet's max),
    TOTAL_FLOOR → total_floor, SUB_AVAILABILITY → sub_availability,
    CITY → city (raw CITY is already a string label; this decoder is
    future-proofing), FEATURES → features_list, AMENITIES → amenities_list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ml.cleaning.parsing import _log_unparseable

_log: Final = logging.getLogger("ml.cleaning.facet_decoders")

# Single source of truth for the spec version this module ships.
SPEC_VERSION: Final[str] = "04-facet-decoding-v1"

# 13 single-value facets in the order they're applied by decode_row().
# Keys match ml.cleaning.ingest.FACET_NAMES (the canonical 15-facet list
# from Step 02 — e.g. "FACING_DIRECTION" is the canonical facet file name,
# even though the raw listing column is "FACING"). Keeping this in sync
# with FACET_NAMES is the contract for the synthetic-fixture builder.
SINGLE_VALUE_FACETS: Final[tuple[str, ...]] = (
    "FURNISH",
    "FACING_DIRECTION",
    "AGE",
    "PROPERTY_TYPE",
    "OWNERSHIP_TYPE",
    "LOCALITY_ID",
    "BUILDING_ID",
    "BEDROOM_NUM",
    "BATHROOM_NUM",
    "FLOOR_NUM",
    "TOTAL_FLOOR",
    "SUB_AVAILABILITY",
    "CITY",
)

# 2 multi-value facets (comma-separated ID lists in the raw data).
MULTI_VALUE_FACETS: Final[tuple[str, ...]] = ("FEATURES", "AMENITIES")

# Per facet-decoding skill binding rule: unknown IDs decode to "unknown",
# not NaN, not silent drop.
DEFAULT_UNKNOWN_LABEL: Final[str] = "unknown"

# Raw multi-value strings are single-comma separated (confirmed from data:
# "33,23,12,46" — no whitespace). The decoder strips defensively anyway.
MULTI_VALUE_DELIMITER: Final[str] = ","

# Domain-rule fallback for floor codes that exceed the facet's labeled max
# (Indian real-estate regularly has 40-floor buildings while FLOOR_NUM.csv
# labels fewer rows). Numeric codes above the facet max return this label.
FLOOR_NUM_ABOVE_MAX_LABEL: Final[str] = "95+"

# Pre-validate multi-value strings: a single regex that accepts any
# non-empty comma-separated list of integers with optional whitespace.
# Empty string fails this regex — the empty-raw guard runs first.
_RE_MULTI_VALUE: Final[re.Pattern[str]] = re.compile(r"[0-9]+(?:\s*,\s*[0-9]+)*")

# Canonical lowercase log-field tag per facet. Used by every decoder when
# calling _log_unparseable so log lines are grep-able by canonical name
# (not by raw uppercase column name like "FURNISH").
_LOG_FIELD_BY_FACET: Final[dict[str, str]] = {
    "FURNISH": "furnish",
    "FACING_DIRECTION": "facing",
    "AGE": "age",
    "PROPERTY_TYPE": "property_type",
    "OWNERSHIP_TYPE": "ownership_type",
    "LOCALITY_ID": "locality_id",
    "BUILDING_ID": "building_id",
    "BEDROOM_NUM": "bedroom_num",
    "BATHROOM_NUM": "bathroom_num",
    "FLOOR_NUM": "floor_num",
    "TOTAL_FLOOR": "total_floor",
    "SUB_AVAILABILITY": "sub_availability",
    "CITY": "city",
    "FEATURES": "features",
    "AMENITIES": "amenities",
}

# Maps the raw listing column name → facet name (inverted form of Step 02's
# CODED_COLUMNS_BY_FACET, picking only the first candidate column per facet
# since decode_row iterates one column at a time). The raw column "FACING"
# maps to the canonical facet name "FACING_DIRECTION".
_RAW_COLUMN_TO_FACET: Final[dict[str, str]] = {
    "FURNISH": "FURNISH",
    "FACING": "FACING_DIRECTION",
    "AGE": "AGE",
    "PROPERTY_TYPE": "PROPERTY_TYPE",
    "OWNTYPE": "OWNERSHIP_TYPE",
    "LOCALITY_ID": "LOCALITY_ID",
    "BUILDING_ID": "BUILDING_ID",
    "BEDROOM_NUM": "BEDROOM_NUM",
    "BATHROOM_NUM": "BATHROOM_NUM",
    "FLOOR_NUM": "FLOOR_NUM",
    "TOTAL_FLOOR": "TOTAL_FLOOR",
    "SUB_AVAILABILITY": "SUB_AVAILABILITY",
    "CITY": "CITY",
    "FEATURES": "FEATURES",
    "AMENITIES": "AMENITIES",
}

# Key on the returned dict holding the per-run unknown-ID counters that the
# decoders increment. Step 05 reads this to write _unmapped_ids.csv.
_STATS_KEY: Final[str] = "decode_stats"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_key(value: object) -> str | None:
    """Normalize a raw listing value into a join-key string.

    - Missing/NaN → None (caller treats as missing).
    - bool → rejected (bool is technically int in Python; we don't want
      True/False to silently decode).
    - int / float → string of the int form ("4" not "4.0").
    - Anything else → str(value) (catches FLOOR_NUM's "B"/"G"/"L"/"M" codes
      and any pandas Quirks where the column dtype is mixed).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    # numpy scalars, strings, etc. — fall through to str().
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _build_index(facet_df: pd.DataFrame) -> dict[str, str]:
    """Build a {normalized_key: label} index from a facet DataFrame.

    For all facets, the join key is ``str(value)`` of the id column. This
    works uniformly for int ids (4 → "4") and string codes ("B" → "B").
    """
    index: dict[str, str] = {}
    for _, row in facet_df.iterrows():
        raw_id = row["id"]
        if pd.isna(raw_id):
            continue
        key = str(raw_id)
        index[key] = str(row["label"])
    return index


def _increment_stat(stats: dict[str, int], key: str) -> None:
    """Bump an unknown-ID counter on the stats sidecar. Best-effort."""
    stats[key] = stats.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Public API — single-value decoders
# ---------------------------------------------------------------------------


def decode_furnish(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single FURNISH int ID → label."""
    return _decode_single(value, facet_df, "FURNISH")


def decode_facing(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single FACING int ID → label."""
    return _decode_single(value, facet_df, "FACING_DIRECTION")


def decode_age(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single AGE int ID → label."""
    return _decode_single(value, facet_df, "AGE")


def decode_property_type(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single PROPERTY_TYPE int ID → label."""
    return _decode_single(value, facet_df, "PROPERTY_TYPE")


def decode_owntype(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single OWNTYPE int ID → label."""
    return _decode_single(value, facet_df, "OWNERSHIP_TYPE")


def decode_locality_id(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single LOCALITY_ID int ID → locality label."""
    return _decode_single(value, facet_df, "LOCALITY_ID")


def decode_building_id(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single BUILDING_ID int ID → building label."""
    return _decode_single(value, facet_df, "BUILDING_ID")


def decode_bedroom_num(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single BEDROOM_NUM ID → label (e.g., ``"3"``)."""
    return _decode_single(value, facet_df, "BEDROOM_NUM")


def decode_bathroom_num(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single BATHROOM_NUM ID → label (e.g., ``"2"``)."""
    return _decode_single(value, facet_df, "BATHROOM_NUM")


def decode_total_floor(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single TOTAL_FLOOR ID → label."""
    return _decode_single(value, facet_df, "TOTAL_FLOOR")


def decode_sub_availability(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single SUB_AVAILABILITY ID → label."""
    return _decode_single(value, facet_df, "SUB_AVAILABILITY")


def decode_city(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single CITY value → label.

    In the current dataset, raw CITY is already a free-text label like
    ``"Gurgaon"``. This decoder still routes through the facet index for
    consistency, returning ``None`` if the label is not present (so future
    data that arrives coded gets handled the same way).
    """
    return _decode_single(value, facet_df, "CITY")


def decode_floor_num(value: object, facet_df: pd.DataFrame) -> str | None:
    """Decode a single FLOOR_NUM value → label.

    Accepts both integer floors (``1``, ``2``, ..., ``95``) and the four
    alphabetic codes in the raw data: ``"B"`` (Basement), ``"G"`` (Ground),
    ``"L"`` (Lower Ground), ``"M"`` (Multi-Storied).

    Numeric codes above the facet's labeled max return
    ``FLOOR_NUM_ABOVE_MAX_LABEL`` (``"95+"``) — Indian real-estate
    listings regularly have buildings taller than the FLOOR_NUM.csv's
    labeled inventory, so we don't want those to decode to ``"unknown"``.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    index = facet_df.attrs.get("_index", {})
    if isinstance(value, str):
        return index.get(value, DEFAULT_UNKNOWN_LABEL)

    key = _coerce_key(value)
    if key is None:
        return None
    if key in index:
        return index[key]

    # Domain-rule fallback: numeric codes above the facet's max return "95+".
    if key.isdigit():
        max_key = max((int(k) for k in index if k.isdigit()), default=None)
        if max_key is not None and int(key) > max_key:
            return FLOOR_NUM_ABOVE_MAX_LABEL

    _log_unparseable(_LOG_FIELD_BY_FACET["FLOOR_NUM"], value)
    return DEFAULT_UNKNOWN_LABEL


def _decode_single(value: object, facet_df: pd.DataFrame, facet_name: str) -> str | None:
    """Single-value decoder template used by 12 of 13 facets.

    Missing/NaN input → ``None``; unknown-but-present ID → logged warning
    + ``DEFAULT_UNKNOWN_LABEL``; known ID → label.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    key = _coerce_key(value)
    if key is None:
        return None

    index = facet_df.attrs.get("_index")
    label = index.get(key)
    if label is not None:
        return label

    _log_unparseable(_LOG_FIELD_BY_FACET[facet_name], value)
    # Track the unknown on the per-run stats sidecar if present.
    stats = facet_df.attrs.get("decode_stats")
    if isinstance(stats, dict):
        _increment_stat(stats, f"{facet_name}_unknown_single")
    return DEFAULT_UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Public API — multi-value decoders
# ---------------------------------------------------------------------------


def decode_features(value: object, facet_df: pd.DataFrame) -> list[str]:
    """Decode a comma-separated FEATURES ID list → list of labels.

    Unknown IDs are silently dropped (with a per-ID log warning) — they
    never become ``"unknown"`` in the returned list, because "unknown" is
    not a real amenity and would corrupt downstream ``n_amenities``
    counts.
    """
    return _decode_multi(value, facet_df, "FEATURES")


def decode_amenities(value: object, facet_df: pd.DataFrame) -> list[str]:
    """Decode a comma-separated AMENITIES ID list → list of labels.

    Same semantics as ``decode_features``: unknown IDs dropped silently
    (with warning).
    """
    return _decode_multi(value, facet_df, "AMENITIES")


def _decode_multi(value: object, facet_df: pd.DataFrame, facet_name: str) -> list[str]:
    """Multi-value decoder template used by FEATURES and AMENITIES."""
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        return []

    raw = str(value).strip()
    if not raw or not _RE_MULTI_VALUE.fullmatch(raw):
        _log_unparseable(_LOG_FIELD_BY_FACET[facet_name], value)
        return []

    index = facet_df.attrs.get("_index", {})
    stats = facet_df.attrs.get("decode_stats")
    labels: list[str] = []
    for token in raw.split(MULTI_VALUE_DELIMITER):
        key = token.strip()
        if not key:
            continue
        label = index.get(key)
        if label is None:
            _log_unparseable(_LOG_FIELD_BY_FACET[facet_name], key)
            if isinstance(stats, dict):
                _increment_stat(stats, f"{facet_name}_dropped_multi")
            continue
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Public API — driver
# ---------------------------------------------------------------------------


def load_facet_frames(facets_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all 15 facet CSVs and attach a per-facet ``_index`` column.

    Returns a dict keyed by canonical facet name (``"FURNISH"``, ``"AGE"``,
    etc.). Each value is the raw facet DataFrame augmented with:

    - ``_index`` column: a ``{str(id): str(label)}`` dict for O(1) lookups
      at decode time. Built once at load time; decoders never re-scan.
    - ``attrs["decode_stats"]``: a mutable counter dict (initialized to
      ``{f"{facet}_unknown_single": 0, f"{facet}_dropped_multi": 0}``)
      shared across all decoders for this load. Step 05 reads it to write
      ``_unmapped_ids.csv``.

    Per Rule §1.1: ``data/raw/`` is immutable. The function asserts that
    ``data/raw/`` was not modified during the load via the existing
    ``ml.cleaning.ingest._snapshot_raw_files`` primitive.
    """
    facets_dir = Path(facets_dir)
    if not facets_dir.exists():
        raise FileNotFoundError(f"facet dir missing: {facets_dir}")

    # Immutability guard — snapshot before, snapshot after, assert equal.
    # Match Step 02's load_raw_listings pattern (ml/cleaning/ingest.py).
    import ml.cleaning.ingest as ingest  # local import to avoid circulars at module load

    raw_root = facets_dir.parent  # facets_dir = data/raw/facets → parent = data/raw
    before = ingest._snapshot_raw_files(raw_root.parent)

    out: dict[str, pd.DataFrame] = {}
    for name in SINGLE_VALUE_FACETS + MULTI_VALUE_FACETS:
        path = facets_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"facet file missing: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise RuntimeError(f"facet file is empty: {path}")
        # Index is stored on df.attrs (DataFrame metadata dict) so it's
        # accessible by the decoder as facet_df.attrs["_index"] without
        # being part of the tabular data the DataFrame represents.
        df.attrs["_index"] = _build_index(df)
        df.attrs["decode_stats"] = {
            f"{name}_unknown_single": 0,
            f"{name}_dropped_multi": 0,
        }
        out[name] = df
        _log.info("facet_decoders.loaded name=%s rows=%d", name, len(df))

    after = ingest._snapshot_raw_files(raw_root.parent)
    if before != after:
        raise RuntimeError(
            "data/raw/ was modified during load_facet_frames "
            "(Rules §1.1 raw immutability violated)"
        )

    _log.info(
        "facet_decoders.load_done facets=%d spec_version=%s",
        len(out),
        SPEC_VERSION,
    )
    return dict(out)


def decode_row(row: pd.Series, facet_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Apply every facet decoder to *row* and return a dict of decoded values.

    Single-value decoders emit ``str | None``; multi-value decoders emit
    ``list[str]``. Missing raw columns are silently skipped (their decoded
    output is omitted from the result). Step 05 will call this per row
    while building ``clean_listings.parquet``.
    """
    decoded: dict[str, Any] = {}
    for facet_name in SINGLE_VALUE_FACETS:
        raw_col = _RAW_COLUMN_TO_FACET[facet_name]
        if raw_col not in row.index:
            continue
        decoder = _SINGLE_DECODERS[facet_name]
        decoded[facet_name.lower()] = decoder(row[raw_col], facet_frames[facet_name])

    for facet_name in MULTI_VALUE_FACETS:
        raw_col = _RAW_COLUMN_TO_FACET[facet_name]
        if raw_col not in row.index:
            continue
        decoder = _MULTI_DECODERS[facet_name]
        decoded[facet_name.lower()] = decoder(row[raw_col], facet_frames[facet_name])

    return decoded


# Internal dispatch tables for decode_row — kept private so test code
# imports decoders by name (matching the spec's public API contract).
_SINGLE_DECODERS: Final[dict[str, Any]] = {
    "FURNISH": decode_furnish,
    "FACING_DIRECTION": decode_facing,
    "AGE": decode_age,
    "PROPERTY_TYPE": decode_property_type,
    "OWNERSHIP_TYPE": decode_owntype,
    "LOCALITY_ID": decode_locality_id,
    "BUILDING_ID": decode_building_id,
    "BEDROOM_NUM": decode_bedroom_num,
    "BATHROOM_NUM": decode_bathroom_num,
    "FLOOR_NUM": decode_floor_num,
    "TOTAL_FLOOR": decode_total_floor,
    "SUB_AVAILABILITY": decode_sub_availability,
    "CITY": decode_city,
}

_MULTI_DECODERS: Final[dict[str, Any]] = {
    "FEATURES": decode_features,
    "AMENITIES": decode_amenities,
}


__all__ = [
    "SPEC_VERSION",
    "SINGLE_VALUE_FACETS",
    "MULTI_VALUE_FACETS",
    "DEFAULT_UNKNOWN_LABEL",
    "MULTI_VALUE_DELIMITER",
    "load_facet_frames",
    "decode_row",
    "decode_furnish",
    "decode_facing",
    "decode_age",
    "decode_property_type",
    "decode_owntype",
    "decode_locality_id",
    "decode_building_id",
    "decode_bedroom_num",
    "decode_bathroom_num",
    "decode_floor_num",
    "decode_total_floor",
    "decode_sub_availability",
    "decode_city",
    "decode_features",
    "decode_amenities",
]
