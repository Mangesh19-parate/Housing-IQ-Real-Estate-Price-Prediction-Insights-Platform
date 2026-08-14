"""Tests for ml.features.preprocessor (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from ml.features.feature_frame import (
    BALCONY_ORDINAL,
    FLOOR_CATEGORY_ORDINAL,
    FURNISHING_TYPE_ORDINAL,
    LUXURY_CATEGORY_ORDINAL,
)
from ml.features.locality_aggregator import LocalityAggregator
from ml.features.preprocessor import (
    NUMERIC_FEATURES,
    ONEHOT_FEATURES,
    ORDINAL_CATEGORY_ORDERINGS,
    ORDINAL_FEATURES,
    fit_preprocessor,
    make_preprocessor,
    transform_with_preprocessor,
)


def _make_frame(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """A synthetic DataFrame covering all NUMERIC/ORDINAL/ONEHOT columns."""
    rng = np.random.default_rng(seed)
    cities = ["Gurgaon", "Hyderabad", "Mumbai", "Kolkata"]
    rows = []
    for i in range(n):
        city = cities[i % len(cities)]
        rows.append(
            {
                "city": city,
                "locality": f"L{i // 4}",
                "property_type": "flat" if i % 2 == 0 else "house",
                "agePossession": "Relatively New",
                "facing": "North" if i % 2 == 0 else "East",
                "bedRoom": int(rng.integers(1, 6)),
                "bathroom": int(rng.integers(1, 6)),
                "built_up_area": float(rng.uniform(500, 3000)),
                "servant_room": bool(i % 3 == 0),
                "store_room": bool(i % 4 == 0),
                "n_amenities": int(rng.integers(0, 10)),
                "n_features": int(rng.integers(0, 5)),
                "floor_ratio": float(rng.uniform(0, 1)),
                "age_bucket_ord": int(rng.integers(0, 5)),
                "bath_bed_ratio": float(rng.uniform(0.5, 2.0)),
                "area_per_bedroom": float(rng.uniform(200, 1500)),
                "top_amenities_count": int(rng.integers(0, 10)),
                "luxury_category": "Medium",
                "floor_category": "Mid Floor",
                "furnishing_type": "Semifurnished",
                "balcony": "2",
                "price_inr": float(rng.uniform(5_000_000, 50_000_000)),
                "price_per_sqft": float(rng.uniform(5000, 25000)),
                "is_outlier": False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constant pinning
# ---------------------------------------------------------------------------


def test_numeric_features_constant_matches_trd_section_utrd3() -> None:
    """Exact equality against the 15-name tuple."""
    assert NUMERIC_FEATURES == (
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


def test_ordinal_features_constant_has_four_entries() -> None:
    assert ORDINAL_FEATURES == (
        "luxury_category",
        "floor_category",
        "furnishing_type",
        "balcony",
    )


def test_onehot_features_constant_has_four_entries() -> None:
    assert ONEHOT_FEATURES == ("city", "property_type", "agePossession", "facing")


def test_sector_not_in_column_transformer_inputs() -> None:
    """``sector`` is target-encoded, not in the ColumnTransformer."""
    assert "sector" not in NUMERIC_FEATURES
    assert "sector" not in ORDINAL_FEATURES
    assert "sector" not in ONEHOT_FEATURES


def test_transact_type_not_in_column_transformer_inputs() -> None:
    """``transact_type`` is a routing key, not a feature."""
    assert "transact_type" not in NUMERIC_FEATURES
    assert "transact_type" not in ORDINAL_FEATURES
    assert "transact_type" not in ONEHOT_FEATURES


def test_ordinal_category_orderings_are_pinned() -> None:
    """Exact equality against the pinned dict."""
    assert ORDINAL_CATEGORY_ORDERINGS == {
        "luxury_category": list(LUXURY_CATEGORY_ORDINAL.keys()),
        "floor_category": list(FLOOR_CATEGORY_ORDINAL.keys()),
        "furnishing_type": list(FURNISHING_TYPE_ORDINAL.keys()),
        "balcony": list(BALCONY_ORDINAL.keys()),
    }


# ---------------------------------------------------------------------------
# make_preprocessor
# ---------------------------------------------------------------------------


def test_make_preprocessor_returns_unfitted_column_transformer() -> None:
    """An unfitted ColumnTransformer has no `transformers_` attribute."""
    prep = make_preprocessor()
    assert isinstance(prep, ColumnTransformer)
    assert hasattr(prep, "transformers")  # always present
    # ``transformers_`` (with trailing underscore) is the *fitted* form;
    # before fit, it should be absent.
    assert not hasattr(prep, "transformers_")


# ---------------------------------------------------------------------------
# fit + transform round-trip
# ---------------------------------------------------------------------------


def test_preprocessor_fit_transform_round_trip() -> None:
    """Fit on synthetic frame, transform a second frame."""
    train = _make_frame(n=50)
    other = _make_frame(n=20, seed=99)
    agg = LocalityAggregator().fit(train)
    prep = fit_preprocessor(train, agg)
    out = transform_with_preprocessor(prep, agg.transform(other))
    assert isinstance(out, pd.DataFrame)
    # Numeric branch produces len(NUMERIC_FEATURES) scaled columns.
    # Ordinal branch produces 4 columns.
    # One-hot branch produces n_cities-1 + n_property_types-1 + n_age-1
    # + n_facing-1 = (4-1) + (2-1) + 1 + (2-1) = 5 columns.
    # So expected: 15 + 4 + 5 = 24 columns.
    assert out.shape[1] == 24
    assert out.shape[0] == 20


def test_preprocessor_handle_unknown_ignore_for_onehot() -> None:
    """Unseen ``facing`` value at transform time -> all-zero one-hot row, no raise."""
    train = _make_frame(n=50)
    other = _make_frame(n=5, seed=99).copy()
    agg = LocalityAggregator().fit(train)
    prep = fit_preprocessor(train, agg)
    # Set facing to an unseen value on one row.
    other.loc[0, "facing"] = "NORTHEAST_BY_NORTH"  # not in training set
    # Should not raise.
    out = transform_with_preprocessor(prep, agg.transform(other))
    # The unseen row should have zeros in all one-hot columns
    # (handle_unknown="ignore" means the row is filled with zeros for
    # that branch).
    row = out.iloc[0]
    onehot_names = [c for c in out.columns if c.startswith("cat__")]
    assert (row[onehot_names] == 0).all()


def test_preprocessor_n_amenity_flags_resolved_at_fit() -> None:
    """``has_<amenity>`` columns are picked up at fit time."""
    train = _make_frame(n=50)
    train["has_swimming_pool"] = train["n_amenities"] > 3
    train["has_club_house"] = train["n_amenities"] > 5
    agg = LocalityAggregator().fit(train)
    prep = fit_preprocessor(train, agg)
    # The fitted transformer's numeric branch should include both
    # ``has_*`` columns. Access by iteration (not by string index).
    num_cols: list[str] = []
    for name, _transformer, cols in prep.transformers_:
        if name == "num":
            num_cols = list(cols)
            break
    assert "has_swimming_pool" in num_cols
    assert "has_club_house" in num_cols
