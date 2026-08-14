"""Tests for ml.training.persistence (Spec 13 Phase A)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ml.training.persistence import (
    MODEL_REGISTRY_FIELDS,
    append_model_registry,
    save_metrics,
    save_price_model,
)

# ---------------------------------------------------------------------------
# save_price_model
# ---------------------------------------------------------------------------


def test_save_price_model_writes_versioned_filename(tmp_path):
    pipe = LinearRegression()
    path = save_price_model(pipe, "Sale", artifact_dir=tmp_path)
    assert path.name == "price_model_sale_v1.pkl"
    assert path.parent == tmp_path
    # Round-trip load works.
    loaded = joblib.load(path)
    assert isinstance(loaded, LinearRegression)


def test_save_price_model_lowercases_transact_type(tmp_path):
    pipe = LinearRegression()
    path = save_price_model(pipe, "RENT", artifact_dir=tmp_path)
    assert path.name == "price_model_rent_v1.pkl"


def test_save_price_model_creates_artifact_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    pipe = LinearRegression()
    path = save_price_model(pipe, "Sale", artifact_dir=nested)
    assert nested.exists()
    assert path.exists()


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------


def test_save_metrics_writes_versioned_filename(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3]}
    path = save_metrics(payload, artifact_dir=tmp_path)
    assert path.name == "metrics_v1.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == payload


def test_save_metrics_handles_numpy_scalars(tmp_path):
    payload = {"r2": np.float64(0.85), "t": datetime(2026, 8, 14, tzinfo=timezone.utc)}
    path = save_metrics(payload, artifact_dir=tmp_path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["r2"] == pytest.approx(0.85)
    assert "2026-08-14" in on_disk["t"]


def test_save_metrics_sorted_keys(tmp_path):
    """Key order is deterministic (sort_keys=True) so JSON diffs stay sane."""
    payload = {"z": 1, "a": 2, "m": 3}
    path = save_metrics(payload, artifact_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"m"') < text.index('"z"')


# ---------------------------------------------------------------------------
# append_model_registry
# ---------------------------------------------------------------------------


def test_append_model_registry_writes_header_on_first_call(tmp_path):
    csv_path = tmp_path / "registry.csv"
    appended = append_model_registry(
        {
            "model_name": "price_model_sale",
            "version": "v1",
            "training_dataset_version": "clean_listings.parquet",
            "git_commit": "abc123",
            "training_date": "2026-08-14T00:00:00Z",
            "rmse": 1.0,
            "mae": 0.5,
            "r2": 0.85,
            "hyperparameters": "{}",
            "feature_hash": "deadbeef",
        },
        csv_path=csv_path,
    )
    assert appended is True
    text = csv_path.read_text(encoding="utf-8")
    # Header is the first line.
    header_line = text.splitlines()[0]
    assert header_line == ",".join(MODEL_REGISTRY_FIELDS)


def test_append_model_registry_is_idempotent_on_rerun(tmp_path):
    csv_path = tmp_path / "registry.csv"
    row = {
        "model_name": "price_model_sale",
        "version": "v1",
        "training_dataset_version": "clean_listings.parquet",
        "git_commit": "abc123",
        "training_date": "2026-08-14T00:00:00Z",
        "rmse": 1.0,
        "mae": 0.5,
        "r2": 0.85,
        "hyperparameters": "{}",
        "feature_hash": "deadbeef",
    }
    first = append_model_registry(row, csv_path=csv_path)
    second = append_model_registry(row, csv_path=csv_path)
    assert first is True
    assert second is False
    # 1 header + 1 data row.
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_append_model_registry_appends_distinct_rows(tmp_path):
    csv_path = tmp_path / "registry.csv"
    append_model_registry(
        {
            "model_name": "price_model_sale",
            "version": "v1",
            "git_commit": "aaa",
            "training_date": "2026-08-14T00:00:00Z",
            "rmse": 1.0,
            "mae": 0.5,
            "r2": 0.85,
            "training_dataset_version": "clean_listings.parquet",
            "hyperparameters": "{}",
            "feature_hash": "x",
        },
        csv_path=csv_path,
    )
    append_model_registry(
        {
            "model_name": "price_model_rent",
            "version": "v1",
            "git_commit": "bbb",  # distinct
            "training_date": "2026-08-14T00:00:00Z",
            "rmse": 1.0,
            "mae": 0.5,
            "r2": 0.85,
            "training_dataset_version": "clean_listings.parquet",
            "hyperparameters": "{}",
            "feature_hash": "y",
        },
        csv_path=csv_path,
    )
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # header + 2 rows


def test_model_registry_csv_columns_match_backend_schema():
    """Exact match against Backend Schema §U-SCHEMA-13 column order."""
    expected = (
        "model_name",
        "version",
        "training_dataset_version",
        "git_commit",
        "training_date",
        "rmse",
        "mae",
        "r2",
        "hyperparameters",
        "feature_hash",
    )
    assert MODEL_REGISTRY_FIELDS == expected
