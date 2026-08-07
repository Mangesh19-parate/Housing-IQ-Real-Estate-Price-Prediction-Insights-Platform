"""``ml.cleaning`` — data preparation pipeline (TRD §6).

Step ownership:
- Step 02: ``ingest`` (raw CSV loading + raw-data immutability snapshot)
- Step 03: ``parsing`` (price/area/string parsing primitives)
- Step 04: ``facet_decoders`` (15 facet ID→label decoders)
- Step 05: ``canonical_mapping`` (per-city canonical-schema mapping)
- Step 06: ``dedup`` + ``outliers`` + ``assemble`` (dedup + outlier-flag orchestrator)
- Step 07: ``imputation`` + ``writers`` + ``pipeline`` (TRD §5 4-tier missing-value
  imputation, Parquet writer, end-to-end orchestrator producing
  ``data/processed/clean_listings.parquet``)
"""

from ml.cleaning.assemble import (
    ASSEMBLE_CITY_FILES,
    assemble_cleaned_frame,
)
from ml.cleaning.imputation import (
    IMPUTATION_CATEGORICAL_LOW,
    IMPUTATION_DROP_THRESHOLD,
    IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS,
    IMPUTATION_HIGH_TIER_COLUMNS,
    IMPUTATION_NUMERIC_LOW,
    MISSINGNESS_HIGH_THRESHOLD,
    MISSINGNESS_LOW_THRESHOLD,
    MISSINGNESS_MEDIUM_THRESHOLD,
    add_was_missing_flags,
    classify_missingness_tiers,
    drop_high_missing_columns,
    impute_high_tier,
    impute_low_tier,
    impute_medium_tier,
    impute_missing_values,
)
from ml.cleaning.pipeline import (
    PIPELINE_REPORT_FIELDS,
    run_clean_listings_pipeline,
)
from ml.cleaning.writers import (
    CLEAN_LISTINGS_DATASET_VERSION,
    CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER,
    CLEAN_LISTINGS_PARQUET_PATH,
    build_clean_listings_columns_order,
    read_clean_listings_parquet,
    verify_clean_listings_parquet,
    write_clean_listings_parquet,
)
from ml.cleaning.canonical_mapping import (
    CANONICAL_COLUMNS,
    CITY_COLUMN_ALIASES,
    CITY_FRAME_LOADERS,
    UNSAFE_COLUMNS,
    clean_description,
    map_city,
    normalize_columns,
)
from ml.cleaning.dedup import (
    CONFLICT_TIEBREAKER_ORDER,
    DEDUP_KEY_COLUMN,
    compute_nonnull_field_count,
    deduplicate_listings,
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
    assert_raw_readonly,
    load_raw_city_frames,
    load_raw_listings,
)
from ml.cleaning.outliers import (
    IQR_MULTIPLIER,
    OUTLIER_DOMAIN_RULES,
    OUTLIER_NUMERIC_COLUMNS,
    OUTLIER_PROPERTY_TYPE_EXEMPTIONS,
    OUTLIER_REASON_COLUMN,
    PERCENTILE_LOWER,
    PERCENTILE_UPPER,
    flag_all_outliers,
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
    "assert_raw_readonly",
    "load_raw_city_frames",
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
    # Step 06 — re-exports added below
    "DEDUP_KEY_COLUMN",
    "CONFLICT_TIEBREAKER_ORDER",
    "compute_nonnull_field_count",
    "deduplicate_listings",
    "OUTLIER_NUMERIC_COLUMNS",
    "OUTLIER_DOMAIN_RULES",
    "OUTLIER_PROPERTY_TYPE_EXEMPTIONS",
    "PERCENTILE_LOWER",
    "PERCENTILE_UPPER",
    "IQR_MULTIPLIER",
    "flag_all_outliers",
    "OUTLIER_REASON_COLUMN",
    "assemble_cleaned_frame",
    "ASSEMBLE_CITY_FILES",
    # Step 07 — missing-value imputation + Parquet writer + orchestrator
    "IMPUTATION_CATEGORICAL_LOW",
    "IMPUTATION_DROP_THRESHOLD",
    "IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS",
    "IMPUTATION_HIGH_TIER_COLUMNS",
    "IMPUTATION_NUMERIC_LOW",
    "MISSINGNESS_HIGH_THRESHOLD",
    "MISSINGNESS_LOW_THRESHOLD",
    "MISSINGNESS_MEDIUM_THRESHOLD",
    "add_was_missing_flags",
    "classify_missingness_tiers",
    "drop_high_missing_columns",
    "impute_high_tier",
    "impute_low_tier",
    "impute_medium_tier",
    "impute_missing_values",
    "CLEAN_LISTINGS_PARQUET_PATH",
    "CLEAN_LISTINGS_DATASET_VERSION",
    "CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER",
    "build_clean_listings_columns_order",
    "read_clean_listings_parquet",
    "verify_clean_listings_parquet",
    "write_clean_listings_parquet",
    "PIPELINE_REPORT_FIELDS",
    "run_clean_listings_pipeline",
]
