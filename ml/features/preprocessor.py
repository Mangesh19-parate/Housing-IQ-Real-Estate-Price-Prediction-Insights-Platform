"""ColumnTransformer factory + fit/transform helpers (Spec 12 Phase 3).

Authority:
    - docs/02-TRD.md §U-TRD-3 (ColumnTransformer shape)
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-5 (16-field canonical schema)
    - docs/10-FINALIZED-INPUT-SCHEMA.md §1 + §2 (field contract)

Three branches:
    numeric  — StandardScaler
    ordinal  — OrdinalEncoder with pinned category orderings
    one-hot  — OneHotEncoder(handle_unknown="ignore", drop="first")

Encoding decisions (pinned):
    - ``sector`` is **target-encoded**, not one-hot. TRD §U-TRD-3 leaves
      this as an open choice; we pick smoothed target encoding via the
      LocalityAggregator columns (``locality_smoothed_price`` /
      ``locality_avg_price_sqft``). ``sector`` itself does not appear
      in any input tuple.
    - ``transact_type`` is a **routing key**, not a feature (TRD
      §U-TRD-4). FastAPI dispatches on it before preprocessing. It is
      absent from all three input tuples.
    - The 10 ``has_<amenity>`` flags are not module constants — they
      are added at fit time from ``df.columns[df.columns.str.startswith(
      "has_")]``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from ml.features.feature_frame import (
    BALCONY_ORDINAL,
    FLOOR_CATEGORY_ORDINAL,
    FURNISHING_TYPE_ORDINAL,
    LUXURY_CATEGORY_ORDINAL,
)
from ml.features.locality_aggregator import LocalityAggregator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned input tuples
# ---------------------------------------------------------------------------

#: 15 fixed numeric columns + the 10 ``has_<amenity>`` flags resolved at
#: fit time from the input frame's columns. Total at fit time is 25.
NUMERIC_FEATURES: tuple[str, ...] = (
    "bedRoom",
    "bathroom",
    "built_up_area",
    "servant_room",
    "store_room",
    "n_amenities",
    "n_features",
    "floor_ratio",
    "age_bucket_ord",
    "bath_bed_ratio",
    "area_per_bedroom",
    "locality_avg_price_sqft",
    "locality_listing_count",
    "locality_smoothed_price",
    "top_amenities_count",
)

#: 4 ordinal features. Each has an explicit category ordering passed to
#: ``OrdinalEncoder`` so train/serve agree.
ORDINAL_FEATURES: tuple[str, ...] = (
    "luxury_category",
    "floor_category",
    "furnishing_type",
    "balcony",
)

#: 4 one-hot groups. ``sector`` and ``transact_type`` are deliberately
#: absent (target-encoded / routing key respectively — see module
#: docstring).
ONEHOT_FEATURES: tuple[str, ...] = (
    "city",
    "property_type",
    "agePossession",
    "facing",
)

#: Explicit category ordering for each ordinal column. Train/serve
#: skew if this ever drifts — pinned by ``test_ordinal_category_orderings_are_pinned``.
ORDINAL_CATEGORY_ORDERINGS: dict[str, list[str]] = {
    "luxury_category": list(LUXURY_CATEGORY_ORDINAL.keys()),
    "floor_category": list(FLOOR_CATEGORY_ORDINAL.keys()),
    "furnishing_type": list(FURNISHING_TYPE_ORDINAL.keys()),
    "balcony": list(BALCONY_ORDINAL.keys()),
}


def _numeric_columns_at_fit(df: pd.DataFrame) -> list[str]:
    """Resolve the numeric columns for a given input frame.

    Base 15 (NUMERIC_FEATURES) + any ``has_<amenity>`` columns the
    frame has.
    """
    has_cols = sorted(c for c in df.columns if c.startswith("has_"))
    return list(NUMERIC_FEATURES) + has_cols


def make_preprocessor(numeric_cols: Sequence[str] | None = None) -> ColumnTransformer:
    """Return an unfitted ``ColumnTransformer``.

    Args:
        numeric_cols: Numeric column names (base 15 + any ``has_*``
            flags). If ``None``, uses :data:`NUMERIC_FEATURES` (no
            amenity flags). Pass the resolved list at fit time via
            :func:`fit_preprocessor` for the full set.

    The returned transformer is unfitted: ``hasattr(prep,
    "transformers_") == False`` until :meth:`ColumnTransformer.fit` is
    called.
    """
    if numeric_cols is None:
        numeric_cols = list(NUMERIC_FEATURES)

    ordinal_categories = [
        ORDINAL_CATEGORY_ORDERINGS[name] for name in ORDINAL_FEATURES
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_cols)),
            (
                "ord",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype="int64",
                ),
                list(ORDINAL_FEATURES),
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                    dtype="float64",
                ),
                list(ONEHOT_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense output
    )


def fit_preprocessor(
    train_feature_frame: pd.DataFrame,
    locality_aggregator: LocalityAggregator,
) -> ColumnTransformer:
    """Fit the preprocessor on the training feature frame.

    Steps:
        1. Apply ``locality_aggregator.transform`` to materialize the
           three locality columns (they're not in the training frame
           directly — the aggregator produces them).
        2. Resolve numeric columns (base 15 + ``has_*`` flags).
        3. Fit a fresh preprocessor on the result.

    Pure with respect to ``train_feature_frame``: no I/O.
    """
    df = locality_aggregator.transform(train_feature_frame)
    numeric_cols = _numeric_columns_at_fit(df)
    prep = make_preprocessor(numeric_cols=numeric_cols)
    # Subset to the columns the preprocessor expects so sklearn doesn't
    # complain about extras.
    expected = list(numeric_cols) + list(ORDINAL_FEATURES) + list(ONEHOT_FEATURES)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"fit_preprocessor: input frame missing columns the "
            f"preprocessor expects: {missing}"
        )
    prep.fit(df[expected])
    logger.info(
        "Preprocessor fitted on %d rows, %d columns -> output dim = %d",
        len(df),
        len(expected),
        _expected_output_dim(len(numeric_cols)),
    )
    return prep


def _expected_output_dim(n_numeric: int) -> int:
    """Return the expected post-transform column count.

    - numeric: n_numeric columns (passthrough)
    - ordinal: len(ORDINAL_FEATURES) = 4
    - one-hot: variable per category cardinality; this is a rough
      estimate used only for logging. Real count comes from
      ``fitted.get_feature_names_out()``.
    """
    return n_numeric + len(ORDINAL_FEATURES)


def transform_with_preprocessor(
    fitted: ColumnTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a fitted preprocessor; return the expanded DataFrame.

    Columns are named via ``fitted.get_feature_names_out()``.
    """
    # Apply the aggregator if locality columns are missing on this frame.
    if "locality_avg_price_sqft" not in df.columns:
        raise ValueError(
            "transform_with_preprocessor: input frame is missing locality_* "
            "columns — call LocalityAggregator.transform() first."
        )
    # Resolve the actual numeric column list the fitted transformer was
    # built with (the fitted form stores columns at index 2 of each
    # transformer tuple, not by string-keyed index).
    numeric_cols: list[str] = []
    for name, _transformer, cols in fitted.transformers_:
        if name == "num":
            numeric_cols = list(cols)
            break
    ordinal_cols = list(ORDINAL_FEATURES)
    onehot_cols = list(ONEHOT_FEATURES)
    expected = list(numeric_cols) + ordinal_cols + onehot_cols
    arr = fitted.transform(df[expected])
    names = fitted.get_feature_names_out(expected)
    return pd.DataFrame(arr, columns=names, index=df.index)


__all__ = [
    "NUMERIC_FEATURES",
    "ORDINAL_FEATURES",
    "ONEHOT_FEATURES",
    "ORDINAL_CATEGORY_ORDERINGS",
    "make_preprocessor",
    "fit_preprocessor",
    "transform_with_preprocessor",
]
