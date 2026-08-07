# Spec: SQLite → Postgres Schema Migration

## Overview
Replace the current hardcoded `_DDL` tuple inside `app/database/db.py` (which
creates the four operational tables — `prediction_log`,
`recommendation_log`, `classification_log`, `model_registry` — on every
`init_db()` call) with a **versioned, idempotent migration system** that
runs the same set of DDL against SQLite (dev) and Postgres (prod-path), and
records applied versions in a `schema_migrations` bookkeeping table so a
fresh DB or an upgraded existing DB converges to the same schema
deterministically. This is Step 08 of the foundation module, sitting between
the Parquet-pipeline checkpoint (Step 07, data-side) and the upcoming
model-training / API-serving weeks (which will write to these tables for the
first time). It executes the migration guidance in
`.claude/skills/sqlite-postgres-schema/SKILL.md` and the Postgres-prod-path
commitment in `02-TRD.md` §1 and `05-BACKEND-SCHEMA.md` §5. Module:
**foundation**.

## Depends on
- **Step 01** — `01-repo-scaffolding-and-environment-setup` — already wired
  `APP_DB_PATH`, `pytest.ini`, `tests/conftest.py`, `.gitignore` (which
  excludes `data/app.db`).
- **Step 07** — `07-clean-listings-parquet-pipeline` — first offline
  pipeline that touches the application DB transitively via `init_db()`
  during tests; confirms the DDL tuple currently in `db.py` is reachable.
- Existing `app/database/db.py` with its current `_DDL` tuple +
  `_EXPECTED_TABLES` set — this spec **rewrites** that module, it does not
  bolt a parallel system on the side.

## Routes / Endpoints
No new routes/endpoints. Migrations are an operational/CLI concern; the
migration runner is invoked from `scripts/init_db.py` (new) and from
`app/database/db.py:init_db()` (existing entry point, now delegates to the
runner). Flask routes and FastAPI startup that already call `init_db()`
keep working unchanged.

## Data / Schema changes
- **New table** `schema_migrations(version TEXT PRIMARY KEY, applied_at
  DATETIME NOT NULL, source TEXT)` — bookkeeping only, never queried by
  app code. Single row per applied migration file.
- **Existing tables** `prediction_log`, `recommendation_log`,
  `classification_log`, `model_registry` — column shape unchanged from
  today's `db.py:_DDL` tuples. Migration `001_initial.py` re-creates them
  verbatim, so an existing `data/app.db` with those tables already in it
  converges to the same end state after running migrations (idempotent).
- **No writes to `data/raw/`** (binding per Rules §1.2).
- **No writes to `data/processed/`** (the Parquet pipeline is Step 07's
  territory, untouched here).
- Migration files live under `migrations/sqlite/` and `migrations/postgres/`
  in versioned pairs (`NNN_description.sql` + `NNN_description.down.sql`).
  Files are **plain SQL**, not Python — the runner is dialect-aware, the
  files are not.
- A header comment in each migration file records its `computation_date`
  and `source_dataset_version` placeholder (Rules §1.3) — these are
  application-DB migrations, not derived data tables, so the fields are
  stamped `n/a` here, but the rule that "every new derived table states its
  computation date and source dataset version" is honored as a written
  comment for forward consistency.

## Templates / UI
None.

## Files to change / Files to create

**Create:**
- `migrations/__init__.py` — empty marker so the directory is a Python
  package (lets us import helpers like `migrations.runner`).
- `migrations/runner.py` — the migration engine. Public API:
  - `MIGRATIONS_TABLE: str = "schema_migrations"`.
  - `SUPPORTED_DIALECTS: tuple[str, ...] = ("sqlite", "postgres")`.
  - `MigrationRecord = namedtuple("MigrationRecord", ["version", "filename", "applied_at", "source"])` — the row shape for `schema_migrations`.
  - `detect_dialect(db_path_or_url: str) -> Literal["sqlite", "postgres"]` — picks the dialect from the connection string prefix (`postgresql://` / `postgres://` → `postgres`, anything else → `sqlite`).
  - `ensure_migrations_table(conn: sqlite3.Connection | psycopg.Connection) -> None` — `CREATE TABLE IF NOT EXISTS schema_migrations(...)`; idempotent.
  - `applied_versions(conn, dialect: str) -> list[str]` — returns the sorted list of `version` strings already applied.
  - `discover_migrations(dialect: str) -> list[Path]` — returns the sorted list of `migrations/<dialect>/NNN_*.sql` files (forward-only; `.down.sql` files exist but are not auto-run by `migrate()`).
  - `apply_migration(conn, dialect: str, path: Path, source: str = "manual") -> None` — runs the SQL file in one transaction, inserts the `schema_migrations` row, commits. Skips if already applied.
  - `migrate(db_path_or_url: str | None = None, source: str = "init_db") -> list[MigrationRecord]` — top-level entry point. Opens a connection, detects dialect, ensures the bookkeeping table, walks undiscovered files in order, applies each. Returns the records that were applied this run (empty list = already up to date).
  - `current_version(db_path_or_url: str | None = None) -> str | None` — helper for `/health`-style diagnostics and tests; returns the highest applied `version` or `None`.
- `migrations/sqlite/001_initial.sql` — exact DDL copied from today's
  `db.py:_DDL` for the 4 operational tables (CREATE TABLE IF NOT EXISTS
  for each, in declaration order: `prediction_log`,
  `recommendation_log`, `classification_log`, `model_registry`). One
  statement per table, semicolon-terminated, Postgres-compatible syntax
  (no `AUTOINCREMENT` — uses `INTEGER PRIMARY KEY` which Postgres maps to
  `serial`/`GENERATED BY DEFAULT AS IDENTITY`; the runner translates it).
- `migrations/sqlite/001_initial.down.sql` — `DROP TABLE IF EXISTS`
  statements for the 4 tables (manual rollback only; never auto-run).
- `migrations/postgres/001_initial.sql` — Postgres equivalent of
  `001_initial.sql`: uses `SERIAL PRIMARY KEY` instead of `INTEGER
  PRIMARY KEY AUTOINCREMENT`, `TIMESTAMP` instead of `DATETIME`, `JSONB`
  instead of `TEXT` for JSON columns (cast at write time). Same 4 tables,
  same column order, same column names.
- `migrations/postgres/001_initial.down.sql` — Postgres `DROP TABLE IF
  EXISTS` set.
- `scripts/init_db.py` — CLI entry point. Usage: `python
  scripts/init_db.py [--db-path PATH|--db-url URL] [--print-version]`.
  Defaults to `APP_DB_PATH` from `app.config`. Exits non-zero on any
  migration failure with the failed filename + exception class on stderr.
- `tests/test_migrations.py` — pytest coverage (see Definition of done
  below).
- `tests/test_init_db_cli.py` — CLI smoke test (subprocess invocation,
  exit code, stdout/stderr shape).

**Modify:**
- `app/database/db.py` — rewrite `_DDL` to be a thin re-export shim that
  delegates to `migrations.runner.migrate()`. Keep the existing public
  surface (`get_db`, `init_db`, `existing_tables`, `_EXPECTED_TABLES`)
  byte-compatible so nothing else in the repo (tests, scripts, future
  Flask routes) needs to change. Specifically:
  - `init_db(db_path=None)` now calls `migrate(db_path,
    source="init_db")` and returns the list of `MigrationRecord`s
    applied (the return type widens from `None` to
    `list[MigrationRecord]`; existing callers ignore it).
  - `_DDL` and `_EXPECTED_TABLES` are kept as module-level constants
    (still exported, still used by `tests/test_init_db_cli.py` style
    contract tests) so removing them later is one sweep, not a breaking
    change in this PR.
  - Add `PRAGMA foreign_keys = ON` on every SQLite connection opened via
    `get_db()` (binding per the skill's "Gotchas" note — today this is
    not set anywhere in `db.py`).
- `requirements.txt` — add `psycopg[binary]>=3.1` (Postgres driver for
  the runner's `postgres` dialect path; gated by import so dev without it
  installed still works for SQLite-only runs). Flagged as a new
  dependency per Rules §5.7 — see "New dependencies" below.

## New dependencies
- **`psycopg[binary]>=3.1`** — Postgres driver used by
  `migrations/runner.py` only when the detected dialect is `postgres`. The
  import is lazy (`from __future__ import annotations` +
  `import psycopg` inside the postgres-only branch) so a dev install that
  only ever runs SQLite does not break. Production deployments that target
  Postgres get it via `pip install -r requirements.txt` like every other
  dep. Listed in `requirements.txt` with a short comment explaining the
  lazy-import rationale.

## Rules for implementation
- **No SQLAlchemy / ORM** — raw `sqlite3` + raw `psycopg`, parameterized
  queries only. The runner splits `.sql` files on `;` and executes each
  statement with `conn.execute(sql)`; there is no string interpolation
  into SQL anywhere.
- **No dealer/contact/media-URL fields ever reach the UI or an export** —
  these tables hold operational logs and registry metadata only. This
  spec does not add such fields; reviewers should reject any PR that
  does.
- **CSS variables only, never hardcoded hex values** — n/a (no UI in
  this spec).
- **All templates extend `base.html`** — n/a (no templates in this spec).
- **Model changes must reference the fixed evaluation protocol** — n/a
  (no models touched; the existing `model_registry` table shape is
  preserved verbatim so the Week 4+ training scripts that will write to
  it keep working without modification).
- **Idempotency** — running `migrate()` twice in a row is a no-op the
  second time (verified by a test). Bookkeeping via `schema_migrations`
  is the single source of truth; never check "does table X exist" as a
  proxy.
- **Dialect-aware file lookup** — `discover_migrations()` reads only
  `migrations/<dialect>/NNN_*.sql`; the `.down.sql` files exist on disk
  for human rollback but are not auto-run by `migrate()`.
- **Schema doc coupling** — every column added/renamed by a future
  migration must be reflected in `05-BACKEND-SCHEMA.md` in the same PR
  (Rules §11/U-RULES-3 documentation-consistency rule). This spec does
  not add columns, so no doc changes are needed for this PR.
- **`PRAGMA foreign_keys = ON`** must run on every new SQLite connection
  in `get_db()` — explicit, not buried in `init_db()`. FK constraints
  are part of the schema, not an init-time convenience.

## Definition of done
A reviewer should be able to verify every item below by running the listed
command on a fresh clone:

1. `pytest tests/test_migrations.py -v` — passes all of:
   - `test_migrate_creates_all_four_tables_on_empty_db` — start with
     `data/app.db` deleted, run `init_db()`, assert the 4 tables + the
     `schema_migrations` table all exist and have the documented columns
     and types.
   - `test_migrate_is_idempotent` — run `migrate()` twice, assert no
     `OperationalError`, assert `schema_migrations` still has exactly
     one row per migration file.
   - `test_migrate_skips_already_applied_version` — manually insert a
     fake `schema_migrations(version='001_initial')` row, run
     `migrate()`, assert `001_initial.sql` was not re-executed (verified
     via a sentinel column added by a *second* migration file in the
     test, not in production SQL — see `tmp_path` fixtures).
   - `test_postgres_migration_runs_against_psycopg_connection` — uses
     `pytest-postgresql` (already in `requirements.txt` from Step 01
     test deps) or a stubbed `psycopg.Connection` if the plugin isn't
     available, asserts the postgres `001_initial.sql` DDL produces the
     4 tables with matching column names. The test is skipped with a
     clear reason if `psycopg` is not importable.
   - `test_detect_dialect` — table-driven test for `detect_dialect()`
     covering `postgresql://`, `postgres://`, `sqlite:///foo.db`,
     `/abs/path.db`, and `None` (defaults to sqlite).
   - `test_get_db_sets_foreign_keys_pragma` — opens a SQLite connection
     via `get_db()`, runs `PRAGMA foreign_keys`, asserts result is `1`.
   - `test_init_db_is_back_compatible_with_existing_callers` — calls
     `init_db()` with no args, asserts it returns without raising and
     that the 4 tables exist (matches the Step 07 + earlier test
     expectations).
   - `test_schema_migrations_table_has_expected_columns` — verifies the
     bookkeeping table shape (version PK, applied_at NOT NULL,
     source TEXT) per the Rules §1.3 spirit.
2. `pytest tests/test_init_db_cli.py -v` — passes the CLI subprocess
   smoke test (exit 0, stdout contains the applied version string).
3. `python scripts/init_db.py` — runs cleanly on a fresh DB, prints
   `applied 001_initial`, exits 0.
4. `python scripts/init_db.py` — run a second time, prints `already up
   to date`, exits 0.
5. `pytest tests/ -v` — the full pre-existing test suite (Step 01–07
   tests) still passes; in particular any test that called
   `init_db()` or asserted on the 4 table shapes must keep working
   unmodified.
6. `git grep -nE "AUTOINCREMENT|PRAGMA foreign_keys" app/database/db.py`
   — confirms `AUTOINCREMENT` is gone (Postgres-compat) and the
   `PRAGMA foreign_keys = ON` line is present.
7. A short note in the PR description listing the one new dependency
   (`psycopg[binary]>=3.1`) and the one behavior change visible to
   existing callers (`init_db()` now returns a `list[MigrationRecord]`
   instead of `None`; existing callers ignore the return value and are
   unaffected).
