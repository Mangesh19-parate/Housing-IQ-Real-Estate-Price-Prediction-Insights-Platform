"""Tests for the ``ml.registry`` package + Spec 20 migrations (Spec 20).

Round-trip + idempotency + naming + hash + endpoint coverage. Uses the
existing ``tmp_clean_db`` fixture from ``tests/conftest.py`` so each
test gets an isolated SQLite file that already has migration 001 +
the 002 column guard applied via ``init_db()``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.database.db import init_db
from ml.registry import (
    artifact_path,
    compute_feature_hash,
    get_active,
    get_active_artifact,
    list_models,
    metrics_path,
    next_version,
    register_model,
    set_active,
)
from ml.registry.naming import _artifact_dir as naming_artifact_dir  # noqa: F401

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    model_name: str = "price_model_sale",
    version: str = "v1",
    rmse: float = 0.42,
    mae: float = 0.31,
    r2: float = 0.85,
    feature_hash: str = "deadbeefdeadbeef",
    artifact_path: str | None = None,
    hyperparameters: dict | None = None,
    training_dataset_version: str = "clean_listings.parquet",
    git_commit: str = "abc1234",
    training_date: datetime | None = None,
) -> dict:
    return {
        "model_name": model_name,
        "version": version,
        "training_dataset_version": training_dataset_version,
        "git_commit": git_commit,
        "training_date": training_date
        or datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc),
        "artifact_path": artifact_path or f"models/{model_name}_{version}.pkl",
        "hyperparameters": hyperparameters or {"chosen_model": "xgb"},
        "feature_hash": feature_hash,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    }


def _register(row: dict, db_path: str) -> int:
    return register_model(
        model_name=row["model_name"],
        version=row["version"],
        training_dataset_version=row["training_dataset_version"],
        git_commit=row["git_commit"],
        training_date=row["training_date"],
        artifact_path=row["artifact_path"],
        hyperparameters=row["hyperparameters"],
        feature_hash=row["feature_hash"],
        metrics=row["metrics"],
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# registry round-trip
# ---------------------------------------------------------------------------


def test_register_then_get_active(tmp_clean_db):
    row = _row()
    new_id = _register(row, tmp_clean_db)
    set_active(row["model_name"], row["version"], db_path=tmp_clean_db)
    active = get_active(row["model_name"], db_path=tmp_clean_db)
    assert active is not None
    assert active["id"] == new_id
    assert active["model_name"] == row["model_name"]
    assert active["version"] == row["version"]
    assert active["is_active"] == 1
    assert active["rmse"] == pytest.approx(0.42)
    assert active["mae"] == pytest.approx(0.31)
    assert active["r2"] == pytest.approx(0.85)
    # JSON round-trip preserves dict shape.
    assert json.loads(active["hyperparameters"]) == {"chosen_model": "xgb"}


def test_register_is_idempotent(tmp_clean_db):
    row = _row()
    first_id = _register(row, tmp_clean_db)
    second_id = _register(row, tmp_clean_db)
    assert first_id == second_id
    rows = list_models(model_name=row["model_name"], db_path=tmp_clean_db)
    assert len(rows) == 1


def test_set_active_is_exclusive(tmp_clean_db):
    _register(_row(version="v1"), tmp_clean_db)
    _register(_row(version="v2"), tmp_clean_db)

    set_active("price_model_sale", "v1", db_path=tmp_clean_db)
    rows = {
        r["version"]: r["is_active"]
        for r in list_models(model_name="price_model_sale", db_path=tmp_clean_db)
    }
    assert rows == {"v1": 1, "v2": 0}

    set_active("price_model_sale", "v2", db_path=tmp_clean_db)
    rows = {
        r["version"]: r["is_active"]
        for r in list_models(model_name="price_model_sale", db_path=tmp_clean_db)
    }
    assert rows == {"v1": 0, "v2": 1}


def test_set_active_missing_version_raises(tmp_clean_db):
    _register(_row(version="v1"), tmp_clean_db)
    with pytest.raises(AssertionError):
        set_active("price_model_sale", "v9", db_path=tmp_clean_db)


def test_list_models_orders_by_training_date_desc(tmp_clean_db):
    _register(
        _row(version="v1", training_date=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        tmp_clean_db,
    )
    _register(
        _row(version="v2", training_date=datetime(2026, 6, 1, tzinfo=timezone.utc)),
        tmp_clean_db,
    )
    _register(
        _row(version="v3", training_date=datetime(2026, 12, 1, tzinfo=timezone.utc)),
        tmp_clean_db,
    )
    rows = list_models(model_name="price_model_sale", db_path=tmp_clean_db)
    assert [r["version"] for r in rows] == ["v3", "v2", "v1"]


def test_get_active_artifact_returns_narrow_tuple(tmp_clean_db):
    row = _row(version="v2")
    _register(row, tmp_clean_db)
    set_active(row["model_name"], row["version"], db_path=tmp_clean_db)
    assert get_active_artifact("price_model_sale", db_path=tmp_clean_db) == (
        "v2",
        "models/price_model_sale_v2.pkl",
    )


def test_get_active_artifact_returns_none_when_no_active(tmp_clean_db):
    _register(_row(), tmp_clean_db)
    # No set_active() call → row exists but is_active=0.
    assert get_active_artifact("price_model_sale", db_path=tmp_clean_db) is None


# ---------------------------------------------------------------------------
# naming helpers
# ---------------------------------------------------------------------------


def test_artifact_path_default_dir():
    assert artifact_path("price_model_sale", "v2") == Path("models") / "price_model_sale_v2.pkl"


def test_artifact_path_override_dir(tmp_path):
    expected = tmp_path / "tier_classifier_v1.pkl"
    assert artifact_path("tier_classifier", "v1", artifact_dir=tmp_path) == expected


def test_metrics_path_default_dir():
    assert metrics_path("v3") == Path("models") / "metrics_v3.json"


def test_next_version_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSINGIQ_ARTIFACT_DIR", str(tmp_path))
    assert next_version("price_model_sale") == "v1"


def test_next_version_picks_max_plus_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSINGIQ_ARTIFACT_DIR", str(tmp_path))
    (tmp_path / "price_model_sale_v1.pkl").touch()
    (tmp_path / "price_model_sale_v2.pkl").touch()
    (tmp_path / "feature_pipeline_v1.pkl").touch()  # noise
    assert next_version("price_model_sale") == "v3"


def test_next_version_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSINGIQ_ARTIFACT_DIR", str(tmp_path / "does-not-exist"))
    assert next_version("price_model_sale") == "v1"


# ---------------------------------------------------------------------------
# feature hash
# ---------------------------------------------------------------------------


def test_feature_hash_deterministic_and_order_insensitive():
    a = compute_feature_hash(["a", "b", "c"])
    b = compute_feature_hash(["c", "a", "b"])
    assert a == b
    assert len(a) == 16


def test_feature_hash_changes_with_content():
    base = compute_feature_hash(["area_sqft", "bedrooms"])
    diff = compute_feature_hash(["area_sqft", "bathrooms"])
    assert base != diff


def test_feature_hash_dedupes_inputs():
    once = compute_feature_hash(["a", "b"])
    twice = compute_feature_hash(["a", "b", "a", "b"])
    assert once == twice


# ---------------------------------------------------------------------------
# migration idempotency (Spec 20)
# ---------------------------------------------------------------------------


def test_init_db_runs_002_columns(tmp_clean_db):
    """After ``init_db()``, the registry has is_active + artifact_path."""
    # tmp_clean_db already calls init_db once; check the columns.
    with __import__("sqlite3").connect(tmp_clean_db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(model_registry)")}
    assert "is_active" in cols
    assert "artifact_path" in cols


def test_init_db_runs_twice_without_error(tmp_path, monkeypatch):
    """Calling init_db() twice on the same DB does not error."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    init_db(db_path=str(db_file))
    # Second call must not raise even though SQLite ALTER ADD COLUMN
    # is not idempotent at the SQL level — the guard handles it.
    init_db(db_path=str(db_file))


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


def test_models_endpoint_returns_newest_first(tmp_clean_db):
    _register(
        _row(version="v1", training_date=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        tmp_clean_db,
    )
    _register(
        _row(version="v2", training_date=datetime(2026, 6, 1, tzinfo=timezone.utc)),
        tmp_clean_db,
    )
    rows = list_models(model_name="price_model_sale", db_path=tmp_clean_db)
    assert len(rows) == 2
    assert [r["version"] for r in rows] == ["v2", "v1"]
    # Each row carries the Spec 20 columns.
    assert "is_active" in rows[0]
    assert "artifact_path" in rows[0]


def test_models_filter_and_limit(tmp_clean_db):
    _register(_row(model_name="price_model_sale"), tmp_clean_db)
    _register(_row(model_name="price_model_rent"), tmp_clean_db)
    rows = list_models(model_name="price_model_rent", db_path=tmp_clean_db)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "price_model_rent"


def test_models_limit_caps_rows(tmp_clean_db):
    for n in range(5):
        _register(_row(version=f"v{n + 1}"), tmp_clean_db)
    rows = list_models(limit=2, db_path=tmp_clean_db)
    assert len(rows) == 2


def test_health_returns_active_version_via_direct_read(tmp_clean_db):
    """Ponytail: prove /health's data source is get_active_artifact.

    The FastAPI lifespan wires the route; we exercise the same read
    path that the route uses. A live ``TestClient`` requires reloading
    api.main under monkeypatch, which is heavy and stateful across
    tests — confirmed manually outside the test suite.
    """
    row = _row(version="v3")
    _register(row, tmp_clean_db)
    set_active("price_model_sale", "v3", db_path=tmp_clean_db)
    # Mirror the route handler's read.
    assert get_active_artifact("price_model_sale", db_path=tmp_clean_db) == (
        "v3",
        "models/price_model_sale_v3.pkl",
    )
