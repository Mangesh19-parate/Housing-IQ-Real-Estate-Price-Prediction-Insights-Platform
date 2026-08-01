"""SQLite helpers for the HousingIQ application DB.

Operational/logging tables only — no user data, no auth (per CLAUDE.md).
Schema follows ``docs/05-BACKEND-SCHEMA.md`` §5 + §U-SCHEMA-11 + §U-SCHEMA-13.

Parameterized SQL only (``?`` placeholders). Per CLAUDE.md and 08-RULES §1:
no SQLAlchemy, no ORM, no f-strings or ``.format()`` into a query string.
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
# Schema DDL — single source of truth in code, tracks 05-BACKEND-SCHEMA.md.
# Add new tables here when the schema doc adds them. Do not edit ad-hoc.
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
    """
    path = Path(db_path) if db_path is not None else Path(APP_DB_PATH)
    _ensure_parent_dir(path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the four operational tables if they don't already exist.

    Idempotent — safe to call from every Flask request or FastAPI startup.
    """
    with get_db(db_path) as conn:
        for stmt in _DDL:
            conn.execute(stmt)


def existing_tables(db_path: str | None = None) -> set[str]:
    """Return the set of table names currently present in the DB. Test helper."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


__all__ = ["get_db", "init_db", "existing_tables", "_EXPECTED_TABLES"]
