"""Shared pytest fixtures — one temp SQLite DB per test, isolated Flask + FastAPI clients."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.app import create_app
from app.database.db import init_db


@pytest.fixture
def tmp_clean_db(tmp_path, monkeypatch) -> Iterator[str]:
    """Point APP_DB_PATH at a fresh file inside tmp_path, run init_db(), yield the path.

    The DB lives only for this test; pytest's ``tmp_path`` cleanup deletes it after.
    """
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    # Re-import APP_DB_PATH from app.config so init_db() sees the new value.
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    init_db(db_path=str(db_file))
    yield str(db_file)


@pytest.fixture
def app_client(tmp_clean_db):
    """Flask test client wired to the temp DB; safe to make requests against."""
    return create_app().test_client()


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient. /health doesn't touch the DB, so a temp DB isn't needed."""
    from api.main import app
    return TestClient(app)
