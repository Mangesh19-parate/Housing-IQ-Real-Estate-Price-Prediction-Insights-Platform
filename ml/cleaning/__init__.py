"""``ml.cleaning`` — data preparation pipeline (TRD §6).

Step ownership:
- Step 02: ``ingest`` (raw CSV loading + raw-data immutability snapshot)
- Step 03: ``parsing`` (price/area/string parsing primitives)
- Step 04: ``facet_decoders`` (15 facet ID→label decoders)
- Step 05: ``canonical_mapping`` (per-city canonical-schema mapping)
"""

from ml.cleaning.canonical_mapping import (
    CANONICAL_COLUMNS,
    CITY_COLUMN_ALIASES,
    CITY_FRAME_LOADERS,
    UNSAFE_COLUMNS,
    clean_description,
    map_city,
    normalize_columns,
)
from ml.cleaning.facet_decoders import (
    DEFAULT_UNKNOWN_LABEL,
    decode_age,
    decode_amenities,
    decode_bathroom_num,
    decode_bedroom_num,
    decode_building_id,
    decode_city,
    decode_facing,
    decode_features,
    decode_floor_num,
    decode_furnish,
    decode_locality_id,
    decode_owntype,
    decode_property_type,
    decode_sub_availability,
    decode_total_floor,
    load_facet_frames,
)
from ml.cleaning.ingest import (
    CODED_COLUMNS_BY_FACET,
    FACET_NAMES,
    PII_PATTERN,
    RAW_FILE_TO_CITY,
    _snapshot_raw_files,
    load_raw_listings,
)
from ml.cleaning.parsing import (
    parse_area,
    parse_price,
)

__all__ = [
    # Step 02
    "CODED_COLUMNS_BY_FACET",
    "FACET_NAMES",
    "PII_PATTERN",
    "RAW_FILE_TO_CITY",
    "_snapshot_raw_files",
    "load_raw_listings",
    # Step 03
    "parse_area",
    "parse_price",
    # Step 04
    "DEFAULT_UNKNOWN_LABEL",
    "decode_age",
    "decode_amenities",
    "decode_bathroom_num",
    "decode_bedroom_num",
    "decode_building_id",
    "decode_city",
    "decode_facing",
    "decode_features",
    "decode_floor_num",
    "decode_furnish",
    "decode_locality_id",
    "decode_owntype",
    "decode_property_type",
    "decode_sub_availability",
    "decode_total_floor",
    "load_facet_frames",
    # Step 05
    "CANONICAL_COLUMNS",
    "CITY_COLUMN_ALIASES",
    "CITY_FRAME_LOADERS",
    "UNSAFE_COLUMNS",
    "clean_description",
    "map_city",
    "normalize_columns",
]
