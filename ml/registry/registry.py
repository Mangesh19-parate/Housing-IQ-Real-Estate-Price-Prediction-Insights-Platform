"""SQLite-backed model registry (Spec 20).

Thin wrapper over the ``model_registry`` table created by Spec 08's
``migrations/<dialect>/001_initial.sql``. The 002 migration (this spec)
adds ``is_active`` + ``artifact_path`` and the unique index on
``(model_name, version)`` — without those two columns the registry
can't actually answer "which model is live?", so this module assumes
migration 002 has run.

All SQL is parameterized (no f-strings / ``.format()`` into queries —
per CLAUDE.md and 08-RULES §1). Connection management goes through
``app.database.db.get_db()`` so a test can ``monkeypatch`` the DB path
via the existing ``tmp_clean_db`` fixture without touching this file.

Concurrency: ``set_active`` runs both UPDATEs inside one
``get_db()`` context — the context manager commits on success and
rolls back on any failure, so an interrupted activation never leaves
two rows marked active.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime

from app.database.db import get_db

logger = logging.getLogger(__name__)

#: Numeric columns that should be ``float``-cast on read. Listed
#: explicitly so a future schema addition doesn't silently round-trip
#: a NULL into the Python ``None`` (which is correct) while accidentally
#: keeping a stored string as-is.
_NUMERIC_COLUMNS: frozenset[str] = frozenset({"rmse", "mae", "r2"})


def _row_to_dict(row) -> dict:
    """Coerce a sqlite3.Row into a JSON-friendly dict.

    Datetimes → ISO 8601 strings, NULL numerics stay ``None``,
    hyperparameters JSON text → ``dict`` (already JSON on the way in).
    """
    out: dict = {}
    for key in row.keys():
        value = row[key]
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif key in _NUMERIC_COLUMNS:
            out[key] = float(value) if value is not None else None
        else:
            out[key] = value
    return out


def register_model(
    *,
    model_name: str,
    version: str,
    training_dataset_version: str,
    git_commit: str,
    training_date: datetime,
    artifact_path: str,
    hyperparameters: dict,
    feature_hash: str,
    metrics: dict,
    db_path: str | None = None,
) -> int:
    """Insert a registry row keyed on ``(model_name, version)``.

    Idempotent: re-registering the same ``(model_name, version)`` returns
    the existing rowid instead of raising. Metrics + hyperparameters
    are serialised as JSON so the SQLite round-trip preserves dict shape.

    Returns the rowid of the row (existing or newly inserted).
    """
    payload = (
        model_name,
        version,
        training_dataset_version,
        git_commit,
        training_date.isoformat() if isinstance(training_date, datetime) else training_date,
        artifact_path,
        _safe_float(metrics.get("rmse")),
        _safe_float(metrics.get("mae")),
        _safe_float(metrics.get("r2")),
        json.dumps(hyperparameters, default=str, sort_keys=True),
        feature_hash,
    )
    with get_db(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM model_registry WHERE model_name = ? AND version = ?",
            (model_name, version),
        ).fetchone()
        if existing is not None:
            logger.info(
                "register_model: (%s, %s) already present (rowid=%d) — no-op",
                model_name,
                version,
                existing["id"],
            )
            return int(existing["id"])
        cur = conn.execute(
            """
            INSERT INTO model_registry (
                model_name, version, training_dataset_version, git_commit,
                training_date, artifact_path, rmse, mae, r2,
                hyperparameters, feature_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        new_id = int(cur.lastrowid)
    logger.info(
        "register_model: inserted (%s, %s) rowid=%d", model_name, version, new_id
    )
    return new_id


def set_active(
    model_name: str,
    version: str,
    db_path: str | None = None,
) -> None:
    """Mark ``(model_name, version)`` as the active row.

    Exclusivity enforced inside one transaction: first clear
    ``is_active`` for every row of ``model_name``, then set it on the
    target row. Asserts exactly one row updated in the second UPDATE
    — a missing target raises so the caller can spot a typo'd version
    string instead of silently leaving the registry with no active row.

    Raises ``AssertionError`` if no row matches the requested
    ``(model_name, version)``. Raises ``ValueError`` if multiple rows
    share that pair (impossible while the unique index holds; defensive).
    """
    with get_db(db_path) as conn:
        conn.execute(
            "UPDATE model_registry SET is_active = 0 "
            "WHERE model_name = ? AND is_active = 1",
            (model_name,),
        )
        cur = conn.execute(
            "UPDATE model_registry SET is_active = 1 "
            "WHERE model_name = ? AND version = ?",
            (model_name, version),
        )
        if cur.rowcount != 1:
            raise AssertionError(
                f"set_active({model_name!r}, {version!r}) updated "
                f"{cur.rowcount} rows; expected exactly 1"
            )


def get_active(
    model_name: str,
    db_path: str | None = None,
) -> dict | None:
    """Return the active row as a dict, or ``None`` if no row is active."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_registry "
            "WHERE model_name = ? AND is_active = 1 LIMIT 1",
            (model_name,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def get_active_artifact(
    model_name: str,
    db_path: str | None = None,
) -> tuple[str, str] | None:
    """Narrow read for the FastAPI loader: ``(version, artifact_path)``.

    Avoids pulling every column when the caller only needs to
    construct a ``PredictService``. Returns ``None`` if no row is
    active — caller falls back to the ``MODEL_VERSION`` constant.
    """
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT version, artifact_path FROM model_registry "
            "WHERE model_name = ? AND is_active = 1 LIMIT 1",
            (model_name,),
        ).fetchone()
    if row is None:
        return None
    return (str(row["version"]), str(row["artifact_path"]))


def list_models(
    *,
    model_name: str | None = None,
    limit: int = 100,
    db_path: str | None = None,
) -> list[dict]:
    """Return registry rows newest-first (training_date DESC).

    ``limit`` is hard-capped at 1000 so an unbounded caller doesn't
    pull the entire table into memory.
    """
    effective_limit = min(max(int(limit), 1), 1000)
    sql = (
        "SELECT * FROM model_registry "
        "ORDER BY training_date DESC, id DESC LIMIT ?"
    )
    params: tuple = (effective_limit,)
    if model_name is not None:
        sql = (
            "SELECT * FROM model_registry WHERE model_name = ? "
            "ORDER BY training_date DESC, id DESC LIMIT ?"
        )
        params = (model_name, effective_limit)
    with get_db(db_path) as conn:
        rows: Iterable = conn.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def _safe_float(value) -> float | None:
    """Coerce ``value`` to ``float``; ``None``/empty stays ``None``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "get_active",
    "get_active_artifact",
    "list_models",
    "register_model",
    "set_active",
]
