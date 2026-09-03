"""Tests for classification gate evaluation (Spec 24)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.evaluation.classification.gate import (
    ClassificationEvaluationResult,
    evaluate,
    format_summary,
)
from ml.evaluation.classification.protocol import CLASSIFIER_THRESHOLDS


@pytest.fixture
def temp_artifacts():
    """Create temporary directory structure with synthetic artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        processed = tmp / "data" / "processed" / "classifier"
        models = tmp / "models"
        processed.mkdir(parents=True)
        models.mkdir(parents=True)

        # Create synthetic feature frame (matching Spec 22 schema)
        np.random.seed(42)
        n = 1000
        features = pd.DataFrame({
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
            "facing": np.random.choice(["North", "South", "East", "West"], n),
            "locality": [f"Locality_{i % 20}" for i in range(n)],
            "is_outlier": np.random.choice([False, True], n, p=[0.9, 0.1]),
            "price_inr": np.random.uniform(30_00_000, 5_00_00_000, n),
            "price_per_sqft": np.random.uniform(3000, 25000, n),
        })
        for i in range(3):
            features[f"has_amenity_{i}"] = np.random.randint(0, 2, n)

        feature_path = processed / "feature_frame.parquet"
        features.to_parquet(feature_path, index=False)

        # Labels
        good_deal_labels = pd.DataFrame({
            "listing_id": features["listing_id"],
            "good_deal_verdict": np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2]),
        })
        price_tier_labels = pd.DataFrame({
            "listing_id": features["listing_id"],
            "price_tier": np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1]),
        })

        good_deal_labels.to_parquet(processed / "good_deal_labels.parquet", index=False)
        price_tier_labels.to_parquet(processed / "price_tier_labels.parquet", index=False)

        # Create a fitted ColumnTransformer matching the real preprocessor structure
        # (numeric + ordinal + one-hot) -- WITHOUT leakage columns
        numeric_cols = [
            "bedRoom", "bathroom", "built_up_area", "servant_room", "store_room",
            "n_amenities", "n_features", "floor_ratio", "age_bucket_ord",
            "bath_bed_ratio", "area_per_bedroom", "locality_listing_count",
            "top_amenities_count",
            "has_amenity_0", "has_amenity_1", "has_amenity_2",
        ]
        ordinal_cols = ["luxury_category", "floor_category", "furnishing_type", "balcony"]
        onehot_cols = ["city", "property_type", "agePossession", "facing"]

        ordinal_categories = [
            ["Standard", "Premium", "Luxury", "Ultra-Luxury"],
            ["Ground", "Lower", "Mid", "Upper", "Penthouse"],
            ["Unfurnished", "Semi-Furnished", "Furnished"],
            ["None", "One", "Two", "Three+"],
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("ord", OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype="int64",
                ), ordinal_cols),
                ("cat", OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                    dtype="float64",
                ), onehot_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

        # Fit on the feature frame (excluding leakage columns - as gate expects)
        expected_cols = numeric_cols + ordinal_cols + onehot_cols
        preprocessor.fit(features[expected_cols])

        # Create dummy classifiers that can predict 3 and 4 classes
        clf_3 = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        X_dummy_3 = preprocessor.transform(features[expected_cols])
        y_dummy_3 = np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2])
        clf_3.fit(X_dummy_3, y_dummy_3)

        clf_4 = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        y_dummy_4 = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1])
        clf_4.fit(X_dummy_3, y_dummy_4)

        # Save artifacts
        joblib.dump(preprocessor, models / "classifier_preprocessor_v1.pkl")
        joblib.dump(clf_3, models / "good_deal_classifier_v1.pkl")
        joblib.dump(clf_4, models / "price_tier_classifier_v1.pkl")

        yield {
            "root": tmp,
            "processed": processed,
            "models": models,
            "features": feature_path,
            "labels_dir": processed,
            "n": n,
        }


def test_evaluate_returns_classification_evaluation_result(temp_artifacts):
    """evaluate() returns ClassificationEvaluationResult with all fields populated."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,  # data/processed
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )

    assert isinstance(result, ClassificationEvaluationResult)
    assert result.version == "v1"
    assert result.target == "good_deal"
    assert result.protocol_version == "1.0.0"
    assert result.dataset_version.startswith("feature_frame.parquet-")
    assert len(result.git_commit) == 12 or result.git_commit == "unknown"
    assert set(result.split_sizes.keys()) == {"train", "val", "test"}
    # Total should be close to n minus outliers (10% with some variance)
    total = sum(result.split_sizes.values())
    assert 850 <= total <= 950
    assert "test" in result.metrics
    assert "train" in result.metrics
    assert "val" in result.metrics
    assert set(result.metrics["test"].keys()) >= {"accuracy", "f1_macro", "good_deal_recall"}
    assert isinstance(result.per_class_test, dict)
    assert "matrix" in result.confusion_matrix
    assert "labels" in result.confusion_matrix
    assert isinstance(result.thresholds_passed, dict)
    assert isinstance(result.overall_passed, bool)
    # evaluated_at should be valid ISO8601
    from datetime import datetime
    datetime.fromisoformat(result.evaluated_at.replace("Z", "+00:00"))
    assert result.evaluator_version == "1.0.0"


def test_evaluate_good_deal_target(temp_artifacts):
    """evaluate() works for good_deal target."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )
    assert result.target == "good_deal"
    assert "good_deal_recall" in result.metrics["test"]


def test_evaluate_price_tier_target(temp_artifacts):
    """evaluate() works for price_tier target."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "price_tier_classifier_v1.pkl",
        version="v1",
        target="price_tier",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )
    assert result.target == "price_tier"
    assert "price_tier_macro_f1" in result.metrics["test"]


def test_evaluate_leakage_guard_drops_columns(temp_artifacts, caplog):
    """evaluate() drops price-derived columns and logs warning instead of raising."""
    d = temp_artifacts
    # The feature frame already has price-derived columns
    # evaluate() should drop them and log a warning, not raise
    with caplog.at_level("WARNING"):
        result = evaluate(
            model_path=d["models"] / "good_deal_classifier_v1.pkl",
            version="v1",
            target="good_deal",
            processed_dir=d["processed"].parent.parent,
            models_dir=d["models"],
            feature_frame_path=d["features"],
            labels_dir=d["labels_dir"],
        )
    # Should complete successfully
    assert isinstance(result, ClassificationEvaluationResult)
    # Should log warning about dropping columns
    assert any("Dropping price-derived columns" in record.message for record in caplog.records)


def test_evaluate_thresholds_checked_correctly(temp_artifacts):
    """Thresholds checked correctly: overall_passed is True iff all thresholds pass."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )

    # Check that thresholds_passed has entries for all CLASSIFIER_THRESHOLDS[good_deal]
    expected_thresholds = set(CLASSIFIER_THRESHOLDS["good_deal"].keys())
    assert set(result.thresholds_passed.keys()) == expected_thresholds

    # overall_passed should be all(thresholds_passed.values())
    assert result.overall_passed == all(result.thresholds_passed.values())


def test_evaluate_evaluated_at_iso8601(temp_artifacts):
    """evaluated_at is valid ISO8601 UTC."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(result.evaluated_at.replace("Z", "+00:00"))
    assert dt.tzinfo is not None  # timezone aware


def test_evaluate_git_commit_format(temp_artifacts):
    """git_commit is 12-char or 'unknown'."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )
    assert len(result.git_commit) == 12 or result.git_commit == "unknown"


def test_evaluate_dataset_fingerprint_changes(temp_artifacts):
    """Dataset fingerprint changes when feature_frame.parquet content changes."""
    d = temp_artifacts
    result1 = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )

    # Modify feature frame
    features = pd.read_parquet(d["features"])
    features.loc[0, "bedRoom"] = 999
    features.to_parquet(d["features"], index=False)

    result2 = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )

    assert result1.dataset_version != result2.dataset_version


def test_format_summary_output(temp_artifacts):
    """format_summary returns correct one-line format."""
    d = temp_artifacts
    result = evaluate(
        model_path=d["models"] / "good_deal_classifier_v1.pkl",
        version="v1",
        target="good_deal",
        processed_dir=d["processed"].parent.parent,
        models_dir=d["models"],
        feature_frame_path=d["features"],
        labels_dir=d["labels_dir"],
    )
    summary = format_summary(result)
    assert summary.startswith("[PASS]") or summary.startswith("[FAIL]")
    assert "good_deal_v1" in summary
    assert "f1_macro=" in summary
    assert "good_deal_recall=" in summary
    assert "accuracy=" in summary


def test_evaluate_raises_on_missing_model(temp_artifacts):
    """evaluate() raises FileNotFoundError for missing model artifact."""
    d = temp_artifacts
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        evaluate(
            model_path=d["models"] / "missing_model.pkl",
            version="v1",
            target="good_deal",
            processed_dir=d["processed"].parent.parent,
            models_dir=d["models"],
            feature_frame_path=d["features"],
            labels_dir=d["labels_dir"],
        )


def test_evaluate_raises_on_missing_preprocessor(temp_artifacts):
    """evaluate() raises FileNotFoundError for missing preprocessor."""
    d = temp_artifacts
    # Remove preprocessor
    (d["models"] / "classifier_preprocessor_v1.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="classifier preprocessor not found"):
        evaluate(
            model_path=d["models"] / "good_deal_classifier_v1.pkl",
            version="v1",
            target="good_deal",
            processed_dir=d["processed"].parent.parent,
            models_dir=d["models"],
            feature_frame_path=d["features"],
            labels_dir=d["labels_dir"],
        )


def test_evaluate_raises_on_missing_labels(temp_artifacts):
    """evaluate() raises FileNotFoundError for missing labels."""
    d = temp_artifacts
    with pytest.raises(FileNotFoundError, match="labels not found"):
        evaluate(
            model_path=d["models"] / "good_deal_classifier_v1.pkl",
            version="v1",
            target="good_deal",
            processed_dir=d["processed"].parent.parent,
            models_dir=d["models"],
            feature_frame_path=d["features"],
            labels_dir=d["processed"].parent / "missing_labels",
        )


def test_evaluate_invalid_target():
    """evaluate() raises ValueError for invalid target."""
    with pytest.raises(ValueError, match="target must be 'good_deal' or 'price_tier'"):
        evaluate(
            model_path="dummy.pkl",
            version="v1",
            target="invalid",
        )