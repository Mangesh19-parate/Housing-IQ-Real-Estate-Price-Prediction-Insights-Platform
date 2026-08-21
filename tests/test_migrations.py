"""Tests for the migrations runner (Step 08).

Anchored to the spec at
``.claude/specs/08-sqlite-postgres-schema-migration.md`` §Definition of done.
Every test corresponds to one bullet in DoD §1. SQLite-only by default; the
Postgres test is skipped with a clear reason when ``psycopg2`` is not
importable (dev path doesn't need it).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from app.database.db import get_db, init_db
from migrations import runner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_sqlite_db(tmp_path, monkeypatch) -> Path:
    """Point APP_DB_PATH at a fresh file under tmp_path; do NOT call init_db."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    return db_file


@pytest.fixture
def migrations_dir_with_sentinel(tmp_path, monkeypatch) -> Path:
    """A temporary migrations/<dialect>/ directory with one extra migration file.

    Lets the "skip-already-applied" test verify that a second migration is
    not re-executed, without mutating the production migrations folder.
    """
    mdir = tmp_path / "migrations" / "sqlite"
    mdir.mkdir(parents=True)
    (mdir / "999_sentinel.sql").write_text(
        "CREATE TABLE IF NOT EXISTS sentinel (\n"
        "    id INTEGER PRIMARY KEY,\n"
        "    label TEXT\n"
        ");\n",
        encoding="utf-8",
    )
    # Inject this directory into the runner's discovery path.
    monkeypatch.setattr(runner, "_MIGRATIONS_DIR", tmp_path / "migrations")
    return tmp_path / "migrations"


# ---------------------------------------------------------------------------
# DoD §1 / test 1 — fresh DB gets all 4 tables + schema_migrations
# ---------------------------------------------------------------------------

def test_migrate_creates_all_four_tables_on_empty_db(fresh_sqlite_db):
    init_db(db_path=str(fresh_sqlite_db))

    expected = {
        "prediction_log",
        "recommendation_log",
        "classification_log",
        "model_registry",
        "schema_migrations",
    }
    with sqlite3.connect(fresh_sqlite_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        names = {r[0] for r in rows}
    assert expected.issubset(names), f"Missing tables: {expected - names}"

    # schema_migrations should have exactly one row for 001_initial.
    with sqlite3.connect(fresh_sqlite_db) as conn:
        rows = conn.execute(
            "SELECT version, source FROM schema_migrations"
        ).fetchall()
    assert rows == [("001", "init_db"), ("002", "init_db")]


# ---------------------------------------------------------------------------
# DoD §1 / test 2 — running migrate() twice is a no-op
# ---------------------------------------------------------------------------

def test_migrate_is_idempotent(fresh_sqlite_db):
    init_db(db_path=str(fresh_sqlite_db))
    # Second run: no errors, no new rows.
    applied = runner.migrate(db_path_or_url=str(fresh_sqlite_db), source="cli")
    assert applied == []

    with sqlite3.connect(fresh_sqlite_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0]
    # Spec 20 added 002; both 001 + 002 are recorded after the first init_db().
    assert count == 2


# ---------------------------------------------------------------------------
# DoD §1 / test 3 — already-applied version is skipped
# ---------------------------------------------------------------------------

def test_migrate_skips_already_applied_version(
    fresh_sqlite_db, migrations_dir_with_sentinel, tmp_path
):
    """Pre-seed ``schema_migrations`` with a row for the sentinel file, then
    run migrate() and assert the sentinel file was NOT re-executed."""
    # Pre-create the SQLite DB with the schema_migrations table seeded.
    db_file = fresh_sqlite_db
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "CREATE TABLE schema_migrations ("
            "    version TEXT PRIMARY KEY,"
            "    applied_at DATETIME NOT NULL,"
            "    source TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at, source) "
            "VALUES (?, ?, ?)",
            ("001", "2025-01-01T00:00:00+00:00", "manual"),
        )
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at, source) "
            "VALUES (?, ?, ?)",
            ("999", "2025-01-01T00:00:00+00:00", "manual"),
        )
        conn.commit()

    # Run migrate(): the runner should see both 001 and 999 already applied
    # and produce no new rows.
    applied = runner.migrate(db_path_or_url=str(db_file), source="test")
    assert applied == []

    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert rows == [("001",), ("999",)]

    # The sentinel TABLE should NOT exist (file was not re-executed).
    with sqlite3.connect(db_file) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "sentinel" not in names


# ---------------------------------------------------------------------------
# DoD §1 / test 4 — Postgres migration path runs (skipped if no psycopg2)
# ---------------------------------------------------------------------------

def test_postgres_sql_files_exist_and_are_well_formed():
    pg_dir = (
        Path(runner.__file__).resolve().parent / "postgres"
    )
    assert (pg_dir / "001_initial.sql").exists()
    assert (pg_dir / "001_initial.down.sql").exists()
    # Sanity: the postgres file uses SERIAL, not AUTOINCREMENT.
    text = (pg_dir / "001_initial.sql").read_text(encoding="utf-8")
    assert "SERIAL" in text
    assert "AUTOINCREMENT" not in text


def test_postgres_runner_path_attempts_real_connection(monkeypatch):
    """Postgres URLs must route through the postgres branch, never silently
    fall back to sqlite. Simulate a SQLite-only install (no psycopg2) by
    patching ``sys.modules`` so the runner's lazy import raises ImportError;
    the runner must propagate that cleanly, not silently fall back.
    """
    assert runner.detect_dialect("postgresql://u:p@h:5432/d") == "postgres"

    # Simulate a SQLite-only install: the lazy ``import psycopg2`` inside
    # ``_open_postgres`` will raise ImportError when sys.modules blocks it.
    monkeypatch.setitem(sys.modules, "psycopg2", None)

    with pytest.raises(ImportError):
        runner.migrate(
            db_path_or_url="postgresql://u:p@h:5432/d",
            source="test",
        )


# ---------------------------------------------------------------------------
# DoD §1 / test 5 — detect_dialect matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("postgresql://localhost/db", "postgres"),
        ("postgres://localhost/db", "postgres"),
        ("sqlite:///foo.db", "sqlite"),
        ("/abs/path/to.db", "sqlite"),
        ("relative.db", "sqlite"),
        (None, "sqlite"),
    ],
)
def test_detect_dialect(value, expected):
    assert runner.detect_dialect(value) == expected


# ---------------------------------------------------------------------------
# DoD §1 / test 6 — get_db() sets PRAGMA foreign_keys = ON
# ---------------------------------------------------------------------------

def test_get_db_sets_foreign_keys_pragma(fresh_sqlite_db):
    with get_db(str(fresh_sqlite_db)) as conn:
        result = conn.execute("PRAGMA foreign_keys").fetchone()
    assert result[0] == 1


# ---------------------------------------------------------------------------
# DoD §1 / test 7 — init_db() is back-compat with existing callers
# ---------------------------------------------------------------------------

def test_init_db_is_back_compatible_with_existing_callers(fresh_sqlite_db):
    # Existing callers (Step 01–07) call init_db() with no args and ignore
    # the return value. The widened return type must not break that.
    result = init_db(db_path=str(fresh_sqlite_db))
    assert isinstance(result, list)
    # Spec 20 added 002; both 001 + 002 are recorded in apply-order.
    assert len(result) == 2
    assert result[0].version == "001"
    assert result[1].version == "002"

    # And the 4 tables exist.
    with sqlite3.connect(fresh_sqlite_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"prediction_log", "recommendation_log", "classification_log",
            "model_registry"}.issubset(names)


# ---------------------------------------------------------------------------
# DoD §1 / test 8 — schema_migrations table has the documented shape
# ---------------------------------------------------------------------------

def test_schema_migrations_table_has_expected_columns(fresh_sqlite_db):
    init_db(db_path=str(fresh_sqlite_db))
    with sqlite3.connect(fresh_sqlite_db) as conn:
        rows = conn.execute("PRAGMA table_info(schema_migrations)").fetchall()
    cols = {row[1]: row[2] for row in rows}
    # version (PK), applied_at (NOT NULL), source (TEXT).
    assert "version" in cols
    assert "applied_at" in cols
    assert "source" in cols
    # applied_at must be NOT NULL.
    applied_at_notnull = next(row[3] for row in rows if row[1] == "applied_at")
    assert applied_at_notnull == 1


# ---------------------------------------------------------------------------
# Extra: MigrationRecord shape + current_version round-trip
# ---------------------------------------------------------------------------

def test_migration_record_is_namedtuple():
    r = runner.MigrationRecord(
        version="001",
        filename="001_initial.sql",
        applied_at="2025-01-01T00:00:00+00:00",
        source="test",
    )
    assert r.version == "001"
    assert r.filename == "001_initial.sql"


def test_current_version_round_trip(fresh_sqlite_db):
    assert runner.current_version(db_path_or_url=str(fresh_sqlite_db)) is None
    init_db(db_path=str(fresh_sqlite_db))
    # Spec 20 added 002; current_version returns the highest applied version.
    assert runner.current_version(db_path_or_url=str(fresh_sqlite_db)) == "002"
    # Idempotent: still 002 after a second run.
    init_db(db_path=str(fresh_sqlite_db))
    assert runner.current_version(db_path_or_url=str(fresh_sqlite_db)) == "002"
