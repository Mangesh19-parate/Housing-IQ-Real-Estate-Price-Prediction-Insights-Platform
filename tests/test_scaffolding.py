"""Step 01 smoke tests.

Anchored to the spec at ``.claude/specs/01-repo-scaffolding-and-environment-setup.md``
§1. Every test corresponds to one bullet in the spec's "Definition of done" list.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.database.db import _EXPECTED_TABLES, existing_tables, get_db, init_db


# 1. Imports — required by spec DoD §1 test 1 & 2.
def test_app_imports():
    from app.app import create_app
    assert create_app() is not None


def test_api_imports():
    from api.main import app
    assert app.title == "HousingIQ Inference"


# 2. Flask landing page renders — spec DoD §1 test 3.
def test_flask_landing_renders(app_client):
    resp = app_client.get("/")
    assert resp.status_code == 200
    assert "HousingIQ" in resp.get_data(as_text=True)


# 3. Landing template extends base.html — spec DoD §1 test 4.
def test_flask_landing_extends_base():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "app" / "templates" / "landing.html").read_text(encoding="utf-8")
    assert text.lstrip().startswith('{% extends "base.html" %}')


# 4. FastAPI /health returns 200 + {"status": "ok"} — spec DoD §1 test 5.
def test_fastapi_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# 5. init_db creates all 4 tables — spec DoD §1 test 6.
def test_init_db_creates_tables(tmp_clean_db):
    assert _EXPECTED_TABLES.issubset(existing_tables(tmp_clean_db))


# 6. get_db uses parameterized SQL — spec DoD §1 test 7.
def test_get_db_uses_parameterized_query(tmp_clean_db):
    with get_db(tmp_clean_db) as conn:
        conn.execute(
            "INSERT INTO prediction_log "
            "(timestamp, city, predicted_price, predicted_range_low, predicted_range_high) "
            "VALUES (datetime('now'), ?, ?, ?, ?)",
            ("Gurgaon", 1.0, 0.5, 1.5),
        )
        row = conn.execute(
            "SELECT city, predicted_price FROM prediction_log WHERE city = ?",
            ("Gurgaon",),
        ).fetchone()
    assert row["city"] == "Gurgaon"
    assert row["predicted_price"] == 1.0


# 7. APP_DB_PATH env var changes where the DB is created — spec DoD §1 test 8.
def test_db_path_is_configurable(tmp_path, monkeypatch):
    from app import config as app_config
    target = tmp_path / "custom.db"
    monkeypatch.setenv("APP_DB_PATH", str(target))
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(target))
    init_db(db_path=str(target))
    assert target.exists()
    assert app_config.APP_DB_PATH == str(target)


# 8. CSS outside the :root block has no hardcoded hex — spec DoD §1 test 9.
def test_css_has_no_hardcoded_hex():
    repo = Path(__file__).resolve().parents[1]
    css_dir = repo / "app" / "static" / "css"
    hex_re = re.compile(r"#[0-9a-fA-F]{3,6}\b")
    for path in css_dir.glob("*.css"):
        text = path.read_text(encoding="utf-8")
        # Strip the :root { ... } token block — that's where hex is allowed.
        cleaned = re.sub(r":root\s*\{[^}]*\}", "", text, flags=re.DOTALL)
        matches = hex_re.findall(cleaned)
        assert not matches, f"{path.name} contains hardcoded hex: {matches}"


# 9. Landing template has no hardcoded hex — spec DoD §1 test 10.
def test_landing_template_no_hardcoded_hex():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "app" / "templates" / "landing.html").read_text(encoding="utf-8")
    hex_re = re.compile(r"#[0-9a-fA-F]{3,6}\b")
    assert not hex_re.findall(text), f"landing.html contains hardcoded hex: {text}"


# 10. Conftest fixtures resolve — spec DoD §1 test 11.
def test_conftest_fixtures(tmp_clean_db, app_client, api_client):
    assert Path(tmp_clean_db).exists()
    assert app_client.get("/").status_code == 200
    assert api_client.get("/health").status_code == 200
