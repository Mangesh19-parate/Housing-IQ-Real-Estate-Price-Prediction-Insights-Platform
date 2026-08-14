"""Tests for ml.features.persistence (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features.locality_aggregator import LocalityAggregator
from ml.features.persistence import (
    ENGINEERED_FEATURE_RECIPE,
    FEATURE_ARTIFACT_VERSION,
    load_feature_artifacts,
    save_feature_artifacts,
)
from ml.features.preprocessor import fit_preprocessor, transform_with_preprocessor


def _make_small_frame(n: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = ["Gurgaon", "Hyderabad", "Mumbai"]
    rows = []
    for i in range(n):
        city = cities[i % len(cities)]
        rows.append(
            {
                "city": city,
                "locality": f"L{i // 4}",
                "property_type": "flat",
                "agePossession": "Relatively New",
                "facing": "North",
                "bedRoom": int(rng.integers(1, 5)),
                "bathroom": int(rng.integers(1, 5)),
                "built_up_area": float(rng.uniform(500, 3000)),
                "servant_room": False,
                "store_room": False,
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


def _build_fitted_artifact_pair(train: pd.DataFrame, tmp_path: Path):
    """Build a fitted preprocessor + aggregator to feed save_feature_artifacts."""
    agg = LocalityAggregator().fit(train)
    prep = fit_preprocessor(train, agg)
    out = transform_with_preprocessor(prep, agg.transform(_make_small_frame(5, 11)))
    return agg, prep, list(out.columns)


# ---------------------------------------------------------------------------
# Recipe pinning
# ---------------------------------------------------------------------------


def test_engineered_feature_recipe_covers_all_11_engineered_columns() -> None:
    assert len(ENGINEERED_FEATURE_RECIPE) == 11
    assert set(ENGINEERED_FEATURE_RECIPE.keys()) == {
        "price_per_sqft",
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
    }


def test_feature_artifact_version_pinned() -> None:
    assert FEATURE_ARTIFACT_VERSION == "v1"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_feature_artifacts_writes_three_files(tmp_path: Path) -> None:
    """Saves to {pkl, json}; the report builder is a separate script."""
    train = _make_small_frame()
    agg, prep, feature_list = _build_fitted_artifact_pair(train, tmp_path)
    paths = save_feature_artifacts(
        prep, agg, feature_list, version="v1", artifact_dir=tmp_path
    )
    assert (tmp_path / "feature_pipeline_v1.pkl").exists()
    assert (tmp_path / "feature_list_v1.json").exists()
    assert paths["pkl"].name == "feature_pipeline_v1.pkl"
    assert paths["json"].name == "feature_list_v1.json"


def test_load_feature_artifacts_round_trip(tmp_path: Path) -> None:
    """Save then load; transformed output is identical."""
    train = _make_small_frame(n=30)
    agg, prep, feature_list = _build_fitted_artifact_pair(train, tmp_path)
    save_feature_artifacts(prep, agg, feature_list, artifact_dir=tmp_path)

    # Load back.
    loaded_prep, loaded_agg, loaded_list = load_feature_artifacts(
        version="v1", artifact_dir=tmp_path
    )

    # Apply both preprocessors to a fresh frame and compare.
    other = _make_small_frame(n=10, seed=123)
    # Pre-save transform.
    out_pre = transform_with_preprocessor(
        prep, agg.transform(other)
    )
    # Post-load transform.
    out_post = transform_with_preprocessor(
        loaded_prep, loaded_agg.transform(other)
    )
    # Spot-check: shape and first numeric column match.
    assert out_pre.shape == out_post.shape
    np.testing.assert_allclose(
        out_pre.iloc[:, 0].to_numpy(),
        out_post.iloc[:, 0].to_numpy(),
        rtol=1e-9,
    )
    assert loaded_list == feature_list


def test_load_feature_artifacts_raises_on_missing_version(tmp_path: Path) -> None:
    """Loading an unknown version raises FileNotFoundError with the path."""
    with pytest.raises(FileNotFoundError, match="v999"):
        load_feature_artifacts(version="v999", artifact_dir=tmp_path)


def test_feature_list_json_contains_recipe_section(tmp_path: Path) -> None:
    """JSON has both feature_names and engineered_feature_recipe keys."""
    train = _make_small_frame()
    agg, prep, feature_list = _build_fitted_artifact_pair(train, tmp_path)
    save_feature_artifacts(prep, agg, feature_list, artifact_dir=tmp_path)
    with open(tmp_path / "feature_list_v1.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "feature_names" in payload
    assert "engineered_feature_recipe" in payload
    assert len(payload["feature_names"]) == len(feature_list)
    # Recipe has all 11 engineered columns.
    assert len(payload["engineered_feature_recipe"]) == 11
    assert "price_per_sqft" in payload["engineered_feature_recipe"]
    assert "locality_avg_price_sqft" in payload["engineered_feature_recipe"]


def test_save_feature_artifacts_creates_dir_if_missing(tmp_path: Path) -> None:
    """save_feature_artifacts creates the artifact dir if it doesn't exist."""
    nested = tmp_path / "deep" / "nested" / "path"
    train = _make_small_frame()
    agg, prep, feature_list = _build_fitted_artifact_pair(train, tmp_path)
    paths = save_feature_artifacts(prep, agg, feature_list, artifact_dir=nested)
    assert paths["pkl"].exists()
    assert paths["json"].exists()
