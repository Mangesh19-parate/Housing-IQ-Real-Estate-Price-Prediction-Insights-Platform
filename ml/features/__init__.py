"""Feature engineering for the price-prediction model (Spec 12).

Authority:
    - docs/02-TRD.md §8 (engineered columns) + §U-TRD-3 (ColumnTransformer shape)
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-5 (16-field canonical schema)
    - docs/10-FINALIZED-INPUT-SCHEMA.md §1 + §2 (field contract)
    - docs/08-RULES.md §2.3, §2.4 (leakage + train/serve consistency)

Public API:
    build_feature_frame(df)              # deterministic DataFrame builder
    derive_row_features(df)              # pure row-level derivations
    select_top_amenities(df, k=10)       # top-K amenity labels by frequency
    slugify_amenity(label)               # "Swimming Pool" -> "swimming_pool"
    ENGINEERED_COLUMNS                   # 11-element tuple, locked order
    LocalityAggregator                   # leakage-safe locality aggregates (Phase 2)
    make_preprocessor / fit_preprocessor / transform_with_preprocessor  # Phase 3
"""

from ml.features.feature_frame import (
    ENGINEERED_COLUMNS,
    build_feature_frame,
    derive_row_features,
    select_top_amenities,
    slugify_amenity,
)

# Phase 2 + 3 modules are imported lazily so each phase can land + be
# tested independently. Test files for an earlier phase don't import
# through this __init__ to avoid forcing later modules to exist.
try:  # pragma: no cover - exercised only when phase 2 has landed
    from ml.features.locality_aggregator import LocalityAggregator
except ImportError:  # noqa: F401
    LocalityAggregator = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised only when phase 3 has landed
    from ml.features.preprocessor import (
        fit_preprocessor,
        make_preprocessor,
        transform_with_preprocessor,
    )
except ImportError:  # noqa: F401
    make_preprocessor = None  # type: ignore[assignment]
    fit_preprocessor = None  # type: ignore[assignment]
    transform_with_preprocessor = None  # type: ignore[assignment]


__all__ = [
    "ENGINEERED_COLUMNS",
    "build_feature_frame",
    "derive_row_features",
    "select_top_amenities",
    "slugify_amenity",
    "LocalityAggregator",
    "make_preprocessor",
    "fit_preprocessor",
    "transform_with_preprocessor",
]
