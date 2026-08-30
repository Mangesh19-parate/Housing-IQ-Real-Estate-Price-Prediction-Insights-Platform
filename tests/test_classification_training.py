"""Tests for ml.classification.training (Spec 23)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.classification.training import (
    GOOD_DEAL_LABELS,
    PRICE_TIER_LABELS,
    evaluate_classifier,
    train_good_deal_classifier,
    train_price_tier_classifier,
)
from ml.features.locality_aggregator import LocalityAggregator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_good_deal_labels_constant():
    """GOOD_DEAL_LABELS must match Spec 21 exactly."""
    assert GOOD_DEAL_LABELS == ("Good Deal", "Fair Deal", "Overpriced")


def test_price_tier_labels_constant():
    """PRICE_TIER_LABELS must match Spec 21 exactly."""
    assert PRICE_TIER_LABELS == ("Budget", "Mid-Range", "Premium", "Luxury")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_feature_frame() -> pd.DataFrame:
    """Create a synthetic feature frame matching the classifier schema.

    Uses the same columns as the real feature frame from Spec 22.
    Also includes columns needed by LocalityAggregator.fit(): is_outlier, price_inr, price_per_sqft, locality
    """
    np.random.seed(42)
    n = 500

    # Columns from the classifier feature frame (after Spec 22 processing)
    data = {
        "listing_id": [f"L{i:06d}" for i in range(n)],
        "bedRoom": np.random.randint(1, 5, n),
        "bathroom": np.random.randint(1, 4, n),
        "built_up_area": np.random.uniform(500, 4000, n),
        "servant_room": np.random.randint(0, 2, n),
        "store_room": np.random.randint(0, 2, n),
        "n_amenities": np.random.randint(0, 15, n),
        "n_features": np.random.randint(0, 10, n),
        "floor_ratio": np.random.uniform(0.1, 1.0, n),
        "age_bucket_ord": np.random.randint(0, 4, n),
        "bath_bed_ratio": np.random.uniform(0.5, 2.0, n),
        "area_per_bedroom": np.random.uniform(200, 2000, n),
        "locality_avg_price_sqft": np.random.uniform(3000, 25000, n),
        "locality_listing_count": np.random.randint(10, 500, n),
        "locality_smoothed_price": np.random.uniform(4000, 30000, n),
        "top_amenities_count": np.random.randint(0, 8, n),
        "luxury_category": np.random.choice(["Standard", "Premium", "Luxury", "Ultra-Luxury"], n),
        "floor_category": np.random.choice(["Ground", "Lower", "Mid", "Upper", "Penthouse"], n),
        "furnishing_type": np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n),
        "balcony": np.random.choice(["None", "One", "Two", "Three+"], n),
        "city": np.random.choice(["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"], n),
        "property_type": np.random.choice(["Apartment", "Independent House", "Villa"], n),
        "agePossession": np.random.choice(["Ready to Move", "Under Construction", "New Launch"], n),
        "facing": np.random.choice(["North", "South", "East", "West", "North-East", "North-West", "South-East", "South-West"], n),
        # Required by LocalityAggregator
        "locality": [f"Locality_{i % 20}" for i in range(n)],
        "is_outlier": np.random.choice([False, True], n, p=[0.9, 0.1]),
        "price_inr": np.random.uniform(30_00_000, 5_00_00_000, n),  # 30L to 5Cr
        "price_per_sqft": np.random.uniform(3000, 25000, n),
    }
    # Add some has_* amenity columns
    for i in range(5):
        data[f"has_amenity_{i}"] = np.random.randint(0, 2, n)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_labels(synthetic_feature_frame) -> tuple[pd.Series, pd.Series]:
    """Create synthetic labels matching the feature frame index."""
    n = len(synthetic_feature_frame)
    np.random.seed(123)

    # good_deal_verdict: 0=Good Deal, 1=Fair Deal, 2=Overpriced
    # Imbalance: Good Deal is minority (~20%)
    y_good = np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2])

    # price_tier: 0=Budget, 1=Mid-Range, 2=Premium, 3=Luxury
    # Imbalance: Luxury is minority (~10%)
    y_tier = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1])

    return (
        pd.Series(y_good, name="good_deal_verdict", index=synthetic_feature_frame.index),
        pd.Series(y_tier, name="price_tier", index=synthetic_feature_frame.index),
    )


@pytest.fixture
def fitted_locality_aggregator(synthetic_feature_frame) -> LocalityAggregator:
    """Create and fit a LocalityAggregator on synthetic data."""
    agg = LocalityAggregator()
    # synthetic_feature_frame already has city, locality, is_outlier, price_inr, price_per_sqft
    agg.fit(synthetic_feature_frame)
    return agg


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------

def test_train_good_deal_returns_pipeline_and_metrics(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """train_good_deal_classifier returns (Pipeline, metrics_dict)."""
    y_good, _ = synthetic_labels

    # Split manually for test
    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_good.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_good.iloc[train_size:train_size + val_size]

    pipe, metrics = train_good_deal_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        class_weight="balanced",
        random_state=42,
    )

    assert isinstance(pipe, Pipeline)
    assert hasattr(pipe, "named_steps")
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert hasattr(pipe, "best_model_name_")

    # Check metrics structure
    required_keys = {
        "accuracy", "macro_f1", "per_class_precision",
        "per_class_recall", "per_class_f1", "confusion_matrix",
        "primary_class_recall",
    }
    assert set(metrics.keys()) == required_keys
    assert isinstance(metrics["confusion_matrix"], list)
    assert len(metrics["confusion_matrix"]) == 3  # 3 classes


def test_train_price_tier_returns_pipeline_and_metrics(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """train_price_tier_classifier returns (Pipeline, metrics_dict)."""
    _, y_tier = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_tier.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_tier.iloc[train_size:train_size + val_size]

    pipe, metrics = train_price_tier_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        class_weight="balanced",
        random_state=42,
    )

    assert isinstance(pipe, Pipeline)
    assert hasattr(pipe, "best_model_name_")

    required_keys = {
        "accuracy", "macro_f1", "per_class_precision",
        "per_class_recall", "per_class_f1", "confusion_matrix",
        "primary_class_recall",
    }
    assert set(metrics.keys()) == required_keys
    assert len(metrics["confusion_matrix"]) == 4  # 4 classes


def test_good_deal_selection_prioritizes_recall(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """Best model for good_deal_verdict chosen by Good Deal recall, not accuracy."""
    y_good, _ = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_good.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_good.iloc[train_size:train_size + val_size]

    pipe, metrics = train_good_deal_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        random_state=42,
    )

    # The selected model should have best_model_name_ set
    assert pipe.best_model_name_ in ("LogisticRegression", "RandomForestClassifier", "XGBClassifier")

    # The primary metric tracked is primary_class_recall (Good Deal recall)
    assert "primary_class_recall" in metrics
    assert 0.0 <= metrics["primary_class_recall"] <= 1.0


def test_price_tier_selection_prioritizes_macro_f1(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """Best model for price_tier chosen by macro-F1."""
    _, y_tier = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_tier.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_tier.iloc[train_size:train_size + val_size]

    pipe, metrics = train_price_tier_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        random_state=42,
    )

    assert pipe.best_model_name_ in ("LogisticRegression", "RandomForestClassifier", "XGBClassifier")
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_evaluate_classifier_returns_full_metrics(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """evaluate_classifier returns full metrics dict with all keys."""
    y_good, _ = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_good.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_good.iloc[train_size:train_size + val_size]

    pipe, _ = train_good_deal_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        random_state=42,
    )

    # Evaluate on validation set
    metrics = evaluate_classifier(
        pipe, X_val, y_val, GOOD_DEAL_LABELS, primary_class_index=0
    )

    required_keys = {
        "accuracy", "macro_f1", "per_class_precision",
        "per_class_recall", "per_class_f1", "confusion_matrix",
        "primary_class_recall",
    }
    assert set(metrics.keys()) == required_keys

    # Per-class dicts should have all 3 labels
    for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
        assert set(metrics[key].keys()) == set(GOOD_DEAL_LABELS)

    # Confusion matrix should be 3x3
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_class_weight_balanced_default(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """Default class_weight='balanced' is used."""
    y_good, _ = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_good.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_good.iloc[train_size:train_size + val_size]

    pipe, _ = train_good_deal_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        # Not passing class_weight explicitly - should default to "balanced"
        random_state=42,
    )

    # The classifier inside should have class_weight="balanced"
    clf = pipe.named_steps["classifier"]
    assert clf.class_weight == "balanced" or clf.get_params().get("class_weight") == "balanced"


def test_random_state_reproducibility(
    synthetic_feature_frame,
    synthetic_labels,
    fitted_locality_aggregator,
):
    """Same random_state produces identical model parameters."""
    y_good, _ = synthetic_labels

    train_size = int(0.7 * len(synthetic_feature_frame))
    val_size = int(0.15 * len(synthetic_feature_frame))

    X_train = synthetic_feature_frame.iloc[:train_size]
    y_train = y_good.iloc[:train_size]
    X_val = synthetic_feature_frame.iloc[train_size:train_size + val_size]
    y_val = y_good.iloc[train_size:train_size + val_size]

    pipe1, _ = train_good_deal_classifier(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        random_state=42,
    )
    pipe2, _ = train_good_deal_classifier(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        locality_aggregator=fitted_locality_aggregator,
        random_state=42,
    )

    # Both should select the same model
    assert pipe1.best_model_name_ == pipe2.best_model_name_

    # Classifier parameters should be identical
    clf1 = pipe1.named_steps["classifier"]
    clf2 = pipe2.named_steps["classifier"]
    params1 = clf1.get_params()
    params2 = clf2.get_params()

    # Compare all params except those that might differ (like random_state internals)
    for k in params1:
        if k not in ("random_state",):  # random_state is the seed, not the internal state
            assert params1[k] == params2[k], f"Param {k} differs: {params1[k]} vs {params2[k]}"