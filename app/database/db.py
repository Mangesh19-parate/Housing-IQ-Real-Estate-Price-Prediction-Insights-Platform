"""SQLite helpers for the HousingIQ application DB.

Operational/logging tables only — no user data, no auth (per CLAUDE.md).
Schema follows ``docs/05-BACKEND-SCHEMA.md`` §5 + §U-SCHEMA-11 + §U-SCHEMA-13.

This module is a thin shim over ``migrations.runner.migrate()``. The actual
DDL lives in versioned SQL files under ``migrations/sqlite/`` and
``migrations/postgres/``; ``init_db()`` delegates to the runner so that
convergence over time (add/rename columns) is a forward migration, not a
hand-edited ``_DDL`` tuple. The migration runner is the single source of
truth for what tables exist.

Parameterized SQL only (``?`` placeholders). Per CLAUDE.md and 08-RULES §1:
no SQLAlchemy, no ORM, no f-strings or ``.format()`` into a query string.

Every new SQLite connection opened via ``get_db()`` runs
``PRAGMA foreign_keys = ON`` (per the ``sqlite-postgres-schema`` skill's
"Gotchas" note — SQLite has FKs off by default, and turning them on only
at init time would silently no-op for connections that bypass ``init_db``).
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from app.config import APP_DB_PATH

# ---------------------------------------------------------------------------
# DDL — re-exported for back-compat. The real source of truth is the
# versioned SQL files under ``migrations/<dialect>/001_initial.sql``; the
# runner is what ``init_db()`` invokes. Kept importable so existing tests
# (``tests/test_scaffolding.py``) and any caller that introspects the
# in-code schema keep working unchanged.
# ---------------------------------------------------------------------------

_DDL: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        city TEXT,
        locality TEXT,
        input_features_json TEXT,
        predicted_price REAL,
        predicted_range_low REAL,
        predicted_range_high REAL,
        model_version TEXT,
        is_outlier_input INTEGER,
        latency_ms INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        seed_features_json TEXT,
        returned_listing_ids TEXT,
        used_fallback INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS classification_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        city TEXT,
        input_features_json TEXT,
        predicted_verdict TEXT,
        predicted_tier TEXT,
        tier_probabilities_json TEXT,
        model_version TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        version TEXT,
        training_dataset_version TEXT,
        git_commit TEXT,
        training_date DATETIME,
        rmse REAL,
        mae REAL,
        r2 REAL,
        hyperparameters TEXT,
        feature_hash TEXT
    )
    """,
)

_EXPECTED_TABLES: Final[frozenset[str]] = frozenset({
    "prediction_log",
    "recommendation_log",
    "classification_log",
    "model_registry",
})


def _ensure_parent_dir(path: Path) -> None:
    """Create the directory holding the SQLite file if it doesn't exist."""
    parent = path.parent
    if parent and not parent.exists():
        os.makedirs(parent, exist_ok=True)


@contextmanager
def get_db(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with Row factory; commit on success, rollback on error.

    Caller commits via the connection (``conn.commit()``); the context manager
    only handles rollback on exception. Pass ``db_path`` to override the env
    var (used by tests).

    Every connection has ``PRAGMA foreign_keys = ON`` set on open — see the
    module docstring for why.
    """
    path = Path(db_path) if db_path is not None else Path(APP_DB_PATH)
    _ensure_parent_dir(path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> list:
    """Run all undiscovered migrations against the SQLite DB.

    Delegates to ``migrations.runner.migrate(source="init_db")``. Returns
    the list of ``MigrationRecord``s applied this run (empty list = already
    up to date). Existing callers ignore the return value, so the widened
    return type is a non-breaking change.
    """
    # Lazy import keeps ``app.database.db`` importable without committing
    # to the migrations package's runtime cost in every context.
    from migrations.runner import migrate

    return migrate(db_path_or_url=db_path, source="init_db")


def existing_tables(db_path: str | None = None) -> set[str]:
    """Return the set of table names currently present in the DB. Test helper."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


__all__ = [
    "get_db",
    "init_db",
    "existing_tables",
    "_EXPECTED_TABLES",
    "_DDL",
    "MigrationRecord",
]
