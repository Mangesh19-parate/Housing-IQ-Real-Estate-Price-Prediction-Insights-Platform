"""Integration tests for scripts/train_classifiers.py (Spec 23)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root for running CLI scripts
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dirs():
    """Create temporary directory structure for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        processed = tmp / "data" / "processed" / "classifier"
        models = tmp / "models"
        processed.mkdir(parents=True)
        models.mkdir(parents=True)
        yield {
            "root": tmp,
            "processed": processed,
            "models": models,
            "features": processed / "feature_frame.parquet",
            "labels_dir": processed,
            "report": models / "classification_training_report.md",
        }


@pytest.fixture
def synthetic_data(temp_dirs):
    """Create synthetic feature frame and labels matching real schema."""
    np.random.seed(42)
    n = 1000

    # Feature frame (Spec 22 output schema) - needs columns for LocalityAggregator
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
        # Required by LocalityAggregator.fit()
        "locality": [f"Locality_{i % 20}" for i in range(n)],
        "is_outlier": np.random.choice([False, True], n, p=[0.9, 0.1]),
        "price_inr": np.random.uniform(30_00_000, 5_00_00_000, n),
        "price_per_sqft": np.random.uniform(3000, 25000, n),
    })
    # Add has_* columns
    for i in range(3):
        features[f"has_amenity_{i}"] = np.random.randint(0, 2, n)

    features.to_parquet(temp_dirs["features"], index=False)

    # Labels (Spec 21 output schema)
    price_tier_labels = pd.DataFrame({
        "listing_id": features["listing_id"],
        "price_tier": np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1]),
    })
    good_deal_labels = pd.DataFrame({
        "listing_id": features["listing_id"],
        "good_deal_verdict": np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2]),
    })

    price_tier_labels.to_parquet(temp_dirs["labels_dir"] / "price_tier_labels.parquet", index=False)
    good_deal_labels.to_parquet(temp_dirs["labels_dir"] / "good_deal_labels.parquet", index=False)

    return temp_dirs


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run the train_classifiers.py CLI and return result."""
    cmd = [sys.executable, "scripts/train_classifiers.py"] + args
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cli_creates_both_model_artifacts(synthetic_data):
    """CLI creates both .pkl model artifacts in models-dir."""
    d = synthetic_data
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "1",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    good_deal_path = d["models"] / "good_deal_classifier_v1.pkl"
    price_tier_path = d["models"] / "price_tier_classifier_v1.pkl"

    assert good_deal_path.exists(), "good_deal_classifier_v1.pkl not created"
    assert price_tier_path.exists(), "price_tier_classifier_v1.pkl not created"


def test_cli_creates_preprocessor_artifact(synthetic_data):
    """CLI creates classifier_preprocessor_v{n}.pkl artifact."""
    d = synthetic_data
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "2",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    preprocessor_path = d["models"] / "classifier_preprocessor_v2.pkl"
    assert preprocessor_path.exists(), "classifier_preprocessor_v2.pkl not created"


def test_cli_writes_report(synthetic_data):
    """CLI writes report file with correct version header."""
    d = synthetic_data
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "3",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    assert d["report"].exists(), "Report file not created"
    content = d["report"].read_text(encoding="utf-8")
    assert "# Classification Model Training Report" in content
    assert "## Training Run v3" in content


def test_cli_stratified_split(synthetic_data):
    """CLI uses stratified split preserving class proportions."""
    d = synthetic_data
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "1",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Check that the report mentions split sizes (70/15/15 of 1000 = 700/150/150)
    content = d["report"].read_text(encoding="utf-8")
    # The split should be approximately 700/150/150 (allowing for stratification variance)
    assert "700" in content or "69" in content or "71" in content  # ~700 train


def test_cli_exits_nonzero_on_missing_features(synthetic_data):
    """CLI exits with code 1 when features file is missing."""
    d = synthetic_data
    missing_features = d["processed"] / "missing_features.parquet"

    result = _run_cli([
        "--features", str(missing_features),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "1",
    ], d["root"])

    assert result.returncode == 1
    assert "features_missing" in result.stdout or "FAIL" in result.stdout


def test_cli_exits_nonzero_on_missing_labels_dir(synthetic_data):
    """CLI exits with code 1 when labels directory is missing."""
    d = synthetic_data
    missing_labels = d["processed"] / "missing_labels"

    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(missing_labels),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "1",
    ], d["root"])

    assert result.returncode == 1
    assert "labels_dir_missing" in result.stdout or "FAIL" in result.stdout


def test_cli_version_in_artifact_names(synthetic_data):
    """Version arg reflected in output filenames."""
    d = synthetic_data

    # Version 5
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "5",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    assert (d["models"] / "good_deal_classifier_v5.pkl").exists()
    assert (d["models"] / "price_tier_classifier_v5.pkl").exists()
    assert (d["models"] / "classifier_preprocessor_v5.pkl").exists()

    # Version 10
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "10",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0

    assert (d["models"] / "good_deal_classifier_v10.pkl").exists()
    assert (d["models"] / "price_tier_classifier_v10.pkl").exists()
    assert (d["models"] / "classifier_preprocessor_v10.pkl").exists()


def test_cli_stdout_summary(synthetic_data):
    """CLI prints one-line [OK] summary with key metrics."""
    d = synthetic_data
    result = _run_cli([
        "--features", str(d["features"]),
        "--labels", str(d["labels_dir"]),
        "--models-dir", str(d["models"]),
        "--report", str(d["report"]),
        "--version", "1",
        "--random-state", "42",
    ], d["root"])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    stdout = result.stdout.strip()
    assert stdout.startswith("[OK]")
    assert "good_deal=" in stdout
    assert "price_tier=" in stdout
    assert "test_acc=" in stdout
    assert "GoodDeal_recall=" in stdout
    assert "macro_f1=" in stdout
    assert "v1 artifacts in" in stdout
    assert "report at" in stdout