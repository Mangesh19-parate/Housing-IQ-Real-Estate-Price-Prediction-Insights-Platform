"""CLI smoke test for ``scripts/init_db.py``.

Anchored to the spec at
``.claude/specs/08-sqlite-postgres-schema-migration.md`` §Definition of done §2.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "init_db.py"


def _run_cli(*args: str, db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--db-path", str(db_path), *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )


def test_cli_first_run_prints_applied_and_creates_db(tmp_path):
    db_file = tmp_path / "app.db"
    assert not db_file.exists()

    result = _run_cli("--print-version", db_path=db_file)
    # --print-version runs on a fresh DB which has no migrations yet.
    assert result.returncode == 0
    assert "no migrations applied" in result.stdout

    # Real migration run.
    result = _run_cli(db_path=db_file)
    assert result.returncode == 0, result.stderr
    assert "applied 001" in result.stdout
    assert db_file.exists()


def test_cli_second_run_says_already_up_to_date(tmp_path):
    db_file = tmp_path / "app.db"
    first = _run_cli(db_path=db_file)
    assert first.returncode == 0

    second = _run_cli(db_path=db_file)
    assert second.returncode == 0
    assert "already up to date" in second.stdout


def test_cli_print_version_after_migration(tmp_path):
    db_file = tmp_path / "app.db"
    _run_cli(db_path=db_file)  # initial run

    result = _run_cli("--print-version", db_path=db_file)
    assert result.returncode == 0
    # Spec 20 added migration 002; current_version() returns the highest.
    assert result.stdout.strip() == "002"


def test_cli_applies_002_columns_to_model_registry(tmp_path):
    """Spec 20: after init_db, model_registry has is_active + artifact_path columns."""
    db_file = tmp_path / "app.db"
    result = _run_cli(db_path=db_file)
    assert result.returncode == 0, result.stderr
    import sqlite3
    with sqlite3.connect(db_file) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(model_registry)")}
    assert "is_active" in cols
    assert "artifact_path" in cols
    # Both Spec 20 indexes exist.
    idx_rows = conn.execute("PRAGMA index_list(model_registry)").fetchall()
    index_names = {row[1] for row in idx_rows}
    assert "idx_model_registry_name_version" in index_names
    assert "idx_model_registry_active" in index_names


def test_cli_idempotent_002_even_if_runner_runs_twice(tmp_path):
    """Running init_db twice on the same DB does not raise on the 002 ALTER guard."""
    db_file = tmp_path / "app.db"
    first = _run_cli(db_path=db_file)
    assert first.returncode == 0, first.stderr
    second = _run_cli(db_path=db_file)
    assert second.returncode == 0, second.stderr
    # Second run prints "already up to date" (no migration applied).
    assert "already up to date" in second.stdout


def test_cli_returns_nonzero_on_init_failure(tmp_path, monkeypatch):
    """Force a migration failure by pointing the runner at a bogus path.

    The Postgres-without-psycopg2 branch is the only built-in failure mode
    the CLI can hit in SQLite-only dev; force it by passing a postgres URL.
    """
    if "psycopg2" in sys.modules:
        pytest.skip("psycopg2 is importable; cannot force the missing-driver path.")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--db-url", "postgresql://u:p@h:5432/d"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode != 0
    assert "ERROR" in result.stderr
