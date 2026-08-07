# Plan: SQLite → Postgres Schema Migration

## Context

`app/database/db.py` currently hardcodes the 4 operational tables
(`prediction_log`, `recommendation_log`, `classification_log`,
`model_registry`) as a `_DDL` tuple and re-runs every `CREATE TABLE IF NOT
EXISTS` on every `init_db()` call. There is no version history, no
separation between SQLite (dev) and Postgres (prod-path), and no
forward-migration story when those tables need to change in Week 4+
(training scripts will start writing to `model_registry`).

This plan replaces that with a **versioned, idempotent migration runner**
that reads plain `.sql` files from `migrations/<dialect>/`, records applied
versions in a `schema_migrations` bookkeeping table, and converges a fresh
DB or an upgraded existing DB to the same schema deterministically. The
runner is wired in below the existing `init_db()` entry point so that all
existing callers (Flask routes, FastAPI startup, `tests/conftest.py`,
Step 07's pipeline) keep working unchanged.

Execution is anchored to `.claude/specs/08-sqlite-postgres-schema-migration.md`.

## Key deviation from the spec (flagged)

The spec says *"add `psycopg[binary]>=3.1`"* to `requirements.txt`. The
project already pins **`psycopg2-binary==2.9.9`** (Step 01) for the same
Postgres-driver role. Adding a second Postgres driver would create two
uncoordinated deps for one connection. **Plan uses `psycopg2-binary`**
(already present, no new install needed for the dev path). If the user
prefers `psycopg` v3 over `psycopg2`, swap is one line in `runner.py` and
one line in `requirements.txt` — call it out in the PR.

## Files to create

### `migrations/__init__.py`

Empty marker (makes the dir a Python package so `migrations/runner.py` is
importable across boundaries).

### `migrations/runner.py`

The migration engine. Single file, ~150 lines. Public API:

- `MIGRATIONS_TABLE = "schema_migrations"`
- `SUPPORTED_DIALECTS = ("sqlite", "postgres")`
- `MigrationRecord = namedtuple(...)` — row shape for `schema_migrations`.
- `detect_dialect(db_path_or_url: str | None) -> Literal["sqlite", "postgres"]`
  — `None` and any path that doesn't start with `postgres://` / `postgresql://`
  → `sqlite`. Uses the existing `Literal` import from `typing`.
- `ensure_migrations_table(conn) -> None` — idempotent `CREATE TABLE IF NOT
  EXISTS schema_migrations(version TEXT PRIMARY KEY, applied_at DATETIME
  NOT NULL, source TEXT)`. Parameterized; the SQL is a single static
  string, no interpolation.
- `applied_versions(conn) -> list[str]` — sorted list of `version` strings
  already in `schema_migrations`.
- `discover_migrations(dialect: str) -> list[Path]` — glob
  `migrations/<dialect>/NNN_*.sql` (excludes `.down.sql`), sorted by
  filename so order is deterministic.
- `apply_migration(conn, dialect, path, source="manual") -> MigrationRecord`
  — read the file, split on `;`, execute each non-empty statement as
  `conn.execute(sql)`, then `INSERT INTO schema_migrations(version, applied_at,
  source) VALUES (?, ?, ?)` with the row's own values. Wraps in a
  transaction (commit on success, rollback on exception).
- `migrate(db_path_or_url=None, source="init_db") -> list[MigrationRecord]`
  — top-level entry. Detects dialect, opens a connection (sqlite via
  stdlib `sqlite3`; postgres via lazy `import psycopg2` + `psycopg2.connect`),
  `ensure_migrations_table`, walks undiscovered files in order, applies
  each. Returns the list of records applied this run (empty = already up
  to date). Lazy import of `psycopg2` so SQLite-only dev runs never need
  the driver to import.
- `current_version(db_path_or_url=None) -> str | None` — highest applied
  version, or `None`. Helper for diagnostics / tests.

Postgres connection string is parsed out of the URL when given
(`postgresql://user:pass@host:port/dbname` → `psycopg2.connect(
dsn=...)`); no in-house URL parser — `psycopg2` accepts libpq-style
strings directly.

### `migrations/sqlite/001_initial.sql`

Verbatim DDL from today's `db.py:_DDL` (4 tables, in declaration order:
`prediction_log`, `recommendation_log`, `classification_log`,
`model_registry`). One statement per table, semicolon-terminated. Drops
`AUTOINCREMENT` per U-RULES-3 (Postgres-compat rule from the spec); uses
`INTEGER PRIMARY KEY` with a header comment noting the runner translation:
> `-- computation_date: n/a | source_dataset_version: n/a
>    -- SQLite note: INTEGER PRIMARY KEY without AUTOINCREMENT is
>    -- rowid-aliased; Postgres casts to SERIAL at runtime.`

### `migrations/sqlite/001_initial.down.sql`

Manual rollback only — `DROP TABLE IF EXISTS` for the 4 tables, NOT auto-run
by `migrate()`. Header comment marks it as human-only.

### `migrations/postgres/001_initial.sql`

Postgres equivalent of the SQLite file: `SERIAL PRIMARY KEY` instead of
`INTEGER PRIMARY KEY AUTOINCREMENT`, `TIMESTAMP` instead of `DATETIME`,
`JSONB` for JSON-shaped columns (`input_features_json`,
`returned_listing_ids`, `tier_probabilities_json`, `hyperparameters`).
Same 4 tables, same column order, same column names. Header comment
records `computation_date: n/a` / `source_dataset_version: n/a`.

### `migrations/postgres/001_initial.down.sql`

Postgres `DROP TABLE IF EXISTS` set.

### `scripts/init_db.py`

CLI entry point. Mirrors the precedent set by `scripts/ingest_raw.py`:
- `_REPO_ROOT` sys.path tweak so it runs from anywhere.
- `argparse` with `--db-path PATH` (default = `APP_DB_PATH` from `app.config`)
  and `--print-version` flag.
- Imports `from migrations.runner import migrate, current_version`.
- Prints `applied 001_initial` on first run, `already up to date` on
  subsequent runs, exits 0.
- `--print-version` prints the current `current_version()` and exits 0.
- Any migration failure → non-zero exit with the failed filename and
  exception class on stderr (no traceback spam).

### `tests/test_migrations.py`

Eight tests, one per spec DoD §1 bullet (see Definition of done below).
All use `tmp_path` + `monkeypatch` of `app.config.APP_DB_PATH`, no network.
The Postgres test is skipped with a clear reason if `psycopg2` is not
importable (the dev path doesn't need it installed). The "skip-already-applied"
test uses a `tmp_path` sentinel migration (2nd file with a `CREATE TABLE
sentinel (...)` extension) so production SQL is never mutated by tests.

### `tests/test_init_db_cli.py`

Subprocess smoke test using `subprocess.run([sys.executable, "scripts/init_db.py",
"--db-path", str(tmp_path/"app.db"), "--print-version"])`. Asserts exit 0
and that the stdout contains the version string. Mirrors the style of
existing CLI tests in the foundation.

## Files to modify

### `app/database/db.py`

Rewrite the existing module (~140 lines today → ~110 lines) to delegate:

- Keep `_DDL` and `_EXPECTED_TABLES` as module-level constants for
  back-compat (existing tests in `test_scaffolding.py` import them).
  `_DDL` becomes a re-export tuple pointing to the same 4 statements —
  it can be removed in a future sweep.
- `init_db(db_path=None)` now imports `migrations.runner.migrate` and
  calls it with `source="init_db"`. Returns the list of
  `MigrationRecord`s (callers ignoring the return are unaffected).
- Add `conn.execute("PRAGMA foreign_keys = ON")` at the top of every
  `get_db()` open — binding per the sqlite-postgres-schema skill's
  "Gotchas" note. Today's `db.py` never sets this.
- `get_db()` and `existing_tables()` unchanged in shape.
- `__all__` extended to keep `MigrationRecord` importable from
  `app.database.db` for downstream modules.

### `requirements.txt`

No new dependency. Comment-update the existing `psycopg2-binary==2.9.9`
line to note the new role (driver for the migrations runner's Postgres
path). Lines 33–34 are the only edit.

## Existing functions/utilities to reuse

- `app.config.APP_DB_PATH` — the standard "where does the DB live"
  resolver. `scripts/init_db.py` and `migrations/runner.py` both read it
  as the default; no second config mechanism.
- `tests/conftest.py::tmp_clean_db` — already does the `monkeypatch` +
  `init_db(...)` boilerplate. New migration tests reuse it where
  possible; new `tmp_path`-based tests stay self-contained because the
  fixture calls `init_db` itself, which is what the tests are testing.
- `scripts/ingest_raw.py` — the CLI pattern (`_REPO_ROOT` sys.path hack,
  `_parse_args`, `if __name__ == "__main__"`) is the precedent for
  `scripts/init_db.py`.
- `app/database/db.py` — kept as the public DB surface; the only file
  Flask routes / FastAPI startup need to import. The migration runner
  is an implementation detail.

## Verification

Run from the repo root on a fresh checkout:

1. `pytest tests/test_migrations.py -v` — all 8 tests pass (sqlite 4-table
   create, idempotency, skip-already-applied, detect_dialect matrix,
   `PRAGMA foreign_keys = 1` after `get_db()`, `init_db()` back-compat,
   `schema_migrations` shape, postgres with skip-if-missing-psycopg2).
2. `pytest tests/test_init_db_cli.py -v` — subprocess CLI smoke test.
3. `python scripts/init_db.py` — prints `applied 001_initial`, exits 0.
4. `python scripts/init_db.py` — second run, prints `already up to date`,
   exits 0.
5. `pytest tests/ -v` — full pre-existing suite still passes (Step 01–07
   tests, in particular `test_init_db_creates_tables` and
   `test_get_db_uses_parameterized_query` in `test_scaffolding.py`).
6. `git grep -nE "AUTOINCREMENT|PRAGMA foreign_keys" app/database/db.py`
   — confirms `AUTOINCREMENT` is gone and `PRAGMA foreign_keys = ON`
   is present.
7. `python -c "from migrations.runner import migrate; print(migrate())"` —
   returns a list of `MigrationRecord` namedtuples on a fresh DB.

## Out of scope (deferred, per Rules §13)

- A real Postgres instance for CI — the Postgres test path is
  write-then-skip-if-missing; it would need a `pytest-postgresql` plugin
  + `docker run` story which is a separate scope commitment.
- `psycopg` v3 migration — `psycopg2-binary` is already the project pin.
- Auto-running `.down.sql` — manual rollback only, by design.
- Renaming `_DDL` / `_EXPECTED_TABLES` — kept as re-exports for back-compat
  with `tests/test_scaffolding.py`.
