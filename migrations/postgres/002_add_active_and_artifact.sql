-- Migration: 002_add_active_and_artifact (postgres)
-- computation_date: 2026-08-21 | source_dataset_version: n/a
--
-- Postgres mirror of the sqlite 002 migration (Spec 20).
--
-- Postgres ADD COLUMN IF NOT EXISTS is idempotent natively, so no
-- application-layer guard is required on this dialect — the runner
-- can apply this file unconditionally on a fresh database.
--
-- is_active is BOOLEAN here (SQLite stores it as INTEGER 0/1 for
-- portability); ml.registry.set_active() writes literal 1/0 so a
-- psycopg2-backed caller would need to coerce — Spec 20 stays on
-- SQLite, the postgres path is for prod parity only.

ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS artifact_path TEXT;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_name_version
    ON model_registry(model_name, version);

CREATE INDEX IF NOT EXISTS idx_model_registry_active
    ON model_registry(model_name, is_active);
