# Skill: SQLite → Postgres Schema Management

**Trigger:** Any schema change to the application DB (users, prediction logs, locality/city reference tables).

## Use this skill when
- Adding/altering a table used by the Flask app or logged by the API
- Preparing the Postgres migration path referenced in the TRD

## Key conventions (binding for this project)
- Dev uses SQLite; the schema must remain Postgres-compatible (no SQLite-only types/pragmas baked into app logic beyond `PRAGMA foreign_keys = ON`)
- All queries are parameterized — never f-string interpolation into SQL
- Every new derived table states its computation date and source dataset version in a header/metadata field (per Rules doc §1.3)

## Workflow
1. Write the schema change as a versioned migration script, not a manual ad hoc `ALTER TABLE`
2. Test the migration against a copy of the dev DB before applying
3. Update `05-BACKEND-SCHEMA.md` to reflect the change

## Gotchas / things that have bitten us before
- SQLite foreign keys are off by default — `get_db()` must run `PRAGMA foreign_keys = ON` on every connection or FK constraints silently no-op

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
