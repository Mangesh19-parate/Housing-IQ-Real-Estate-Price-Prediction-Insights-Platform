"""Versioned, idempotent migration runner for the HousingIQ application DB.

Replaces the hardcoded ``_DDL`` tuple that previously lived in
``app/database/db.py``. Reads plain SQL migration files from
``migrations/<dialect>/`` and records applied versions in a
``schema_migrations`` bookkeeping table so a fresh DB or an upgraded
existing DB converges to the same schema deterministically.

Public API:
    MIGRATIONS_TABLE, SUPPORTED_DIALECTS, MigrationRecord
    detect_dialect, ensure_migrations_table, applied_versions,
    discover_migrations, apply_migration, migrate, current_version

Dialect handling:
    - SQLite: stdlib ``sqlite3`` (always available).
    - Postgres: ``psycopg2`` (lazy import; only required when dialect is
      ``postgres``). ``psycopg2-binary`` is already pinned in
      ``requirements.txt`` from Step 01.

Per the spec (``.claude/specs/08-sqlite-postgres-schema-migration.md``) and
the ``sqlite-postgres-schema`` skill: parameterized SQL only, no
SQLAlchemy/ORM, no f-string interpolation into queries, idempotent.
"""

from __future__ import annotations

import re
import sqlite3
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from app.config import APP_DB_PATH

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MIGRATIONS_TABLE: Final[str] = "schema_migrations"
SUPPORTED_DIALECTS: Final[tuple[str, ...]] = ("sqlite", "postgres")

# Default connection timeout for postgres (libpq honors seconds).
_PG_CONNECT_TIMEOUT_SECONDS: Final[int] = 5

# Path layout: <repo>/migrations/<dialect>/NNN_*.sql
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR: Final[Path] = _REPO_ROOT / "migrations"

# MigrationRecord is the row shape for ``schema_migrations`` plus the on-disk
# filename for traceability. ``source`` is the caller that triggered the run
# ("init_db", "cli", "test", "manual").
MigrationRecord = namedtuple(
    "MigrationRecord",
    ["version", "filename", "applied_at", "source"],
)

Dialect = Literal["sqlite", "postgres"]


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

def detect_dialect(db_path_or_url: str | None) -> Dialect:
    """Pick a dialect from the connection string / path.

    ``postgres://`` and ``postgresql://`` → ``"postgres"``. Everything else
    (including ``None``, absolute paths, ``sqlite:///foo.db``, plain
    filenames) → ``"sqlite"``.
    """
    if db_path_or_url is None:
        return "sqlite"
    lowered = db_path_or_url.lower().lstrip()
    if lowered.startswith(("postgresql://", "postgres://")):
        return "postgres"
    return "sqlite"


# ---------------------------------------------------------------------------
# Connection openers — one per dialect. Both are parameterized; no SQL is
# built by string interpolation.
# ---------------------------------------------------------------------------

def _open_sqlite(db_path_or_url: str | None) -> sqlite3.Connection:
    path = Path(db_path_or_url) if db_path_or_url is not None else Path(APP_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _open_postgres(db_path_or_url: str) -> "object":  # psycopg2 connection
    # Lazy import so a dev install that only runs SQLite never imports the
    # Postgres driver. ``psycopg2-binary`` is in requirements.txt.
    import psycopg2  # type: ignore[import-untyped]

    return psycopg2.connect(
        dsn=db_path_or_url,
        connect_timeout=_PG_CONNECT_TIMEOUT_SECONDS,
    )


def _open_connection(db_path_or_url: str | None, dialect: Dialect):
    if dialect == "sqlite":
        return _open_sqlite(db_path_or_url)
    if dialect == "postgres":
        return _open_postgres(str(db_path_or_url))
    raise ValueError(f"Unsupported dialect: {dialect!r}")


# ---------------------------------------------------------------------------
# Bookkeeping table
# ---------------------------------------------------------------------------

def ensure_migrations_table(conn) -> None:
    """Create ``schema_migrations`` if it doesn't already exist. Idempotent."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
            version TEXT PRIMARY KEY,
            applied_at DATETIME NOT NULL,
            source TEXT
        )
        """
    )
    conn.commit()


def applied_versions(conn, dialect: Dialect) -> list[str]:
    """Return the sorted list of ``version`` strings already applied."""
    rows = conn.execute(
        f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY version"
    ).fetchall()
    if dialect == "sqlite":
        return [row["version"] for row in rows]
    # psycopg2 returns tuples by default; support both for the postgres branch.
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Filename pattern: NNN_description.sql where NNN is a zero-padded version.
_MIGRATION_FILENAME_RE = re.compile(r"^(\d{3,})_.+\.sql$")


def discover_migrations(dialect: Dialect) -> list[Path]:
    """Return the sorted list of forward migration files for ``dialect``.

    Scans ``migrations/<dialect>/`` for files matching ``NNN_*.sql`` and
    excludes ``*.down.sql`` (rollbacks exist for humans only; not auto-run).
    """
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported dialect: {dialect!r}. Supported: {SUPPORTED_DIALECTS}"
        )
    dialect_dir = _MIGRATIONS_DIR / dialect
    if not dialect_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(dialect_dir.iterdir()):
        if path.suffix != ".sql":
            continue
        if path.name.endswith(".down.sql"):
            continue
        if _MIGRATION_FILENAME_RE.match(path.name):
            files.append(path)
    return files


def _version_from_path(path: Path) -> str:
    """Extract the ``NNN`` prefix from ``NNN_description.sql``."""
    match = _MIGRATION_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid migration filename: {path.name}")
    return match.group(1)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _strip_line_comments(sql: str) -> str:
    """Remove ``--`` line comments from SQL. Strips the whole line if any
    ``--`` starts a comment, even mid-line — enough for our migration files,
    none of which embed comments inside string literals.
    """
    cleaned: list[str] = []
    for line in sql.splitlines():
        # Strip everything from the first ``--`` onward (within the line).
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        cleaned.append(line)
    return "\n".join(cleaned)


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL file body into individual statements on top-level semicolons.

    Comments are stripped first so a comment that contains a ``;`` (e.g. a
    prose note) doesn't break the split. Naive beyond that: enough for the
    simple ``CREATE TABLE`` / ``DROP TABLE`` chain in our migrations; if a
    future migration embeds ``;`` inside string literals, swap this for a
    real SQL splitter (e.g. ``sqlparse``).
    """
    cleaned = _strip_line_comments(sql)
    statements: list[str] = []
    for raw in cleaned.split(";"):
        chunk = raw.strip()
        if chunk:
            statements.append(chunk)
    return statements


def apply_migration(
    conn,
    dialect: Dialect,
    path: Path,
    source: str = "manual",
) -> MigrationRecord:
    """Apply one migration file in a transaction and record it.

    Skips silently if the version is already in ``schema_migrations``.
    Returns the ``MigrationRecord`` written; raises on any SQL error.
    """
    version = _version_from_path(path)
    sql = path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)

    try:
        for stmt in statements:
            conn.execute(stmt)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute(
            f"INSERT INTO {MIGRATIONS_TABLE} (version, applied_at, source) "
            "VALUES (?, ?, ?)",
            (version, now, source),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return MigrationRecord(
        version=version,
        filename=path.name,
        applied_at=now,
        source=source,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def migrate(
    db_path_or_url: str | None = None,
    source: str = "init_db",
) -> list[MigrationRecord]:
    """Run all undiscovered migrations. Returns the records applied this run.

    Empty list means the DB is already up to date.
    """
    dialect = detect_dialect(db_path_or_url)
    conn = _open_connection(db_path_or_url, dialect)
    try:
        ensure_migrations_table(conn)
        already = set(applied_versions(conn, dialect))
        applied: list[MigrationRecord] = []
        for path in discover_migrations(dialect):
            version = _version_from_path(path)
            if version in already:
                continue
            applied.append(apply_migration(conn, dialect, path, source=source))
        return applied
    finally:
        conn.close()


def current_version(db_path_or_url: str | None = None) -> str | None:
    """Return the highest applied version, or ``None`` if no migrations ran."""
    dialect = detect_dialect(db_path_or_url)
    conn = _open_connection(db_path_or_url, dialect)
    try:
        ensure_migrations_table(conn)
        versions = applied_versions(conn, dialect)
    finally:
        conn.close()
    if not versions:
        return None
    return versions[-1]


__all__ = [
    "MIGRATIONS_TABLE",
    "SUPPORTED_DIALECTS",
    "MigrationRecord",
    "detect_dialect",
    "ensure_migrations_table",
    "applied_versions",
    "discover_migrations",
    "apply_migration",
    "migrate",
    "current_version",
]
