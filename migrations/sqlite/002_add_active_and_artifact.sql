-- Migration: 002_add_active_and_artifact (sqlite)
-- computation_date: 2026-08-21 | source_dataset_version: n/a
--
-- Adds two columns to model_registry so the registry can answer
-- "which model is live?" + "where is its .pkl?":
--   - is_active (INTEGER NOT NULL DEFAULT 0): exactly one row per
--     model_name should have is_active=1; enforced application-side
--     in ml.registry.set_active().
--   - artifact_path (TEXT): repo-relative path the FastAPI loader
--     resolves through to construct PredictService.
-- Plus two indexes:
--   - unique (model_name, version) — idempotency guarantee for
--     register_model().
--   - (model_name, is_active) — cheap lookup for get_active().
--
-- SQLite ALTER TABLE ADD COLUMN is NOT idempotent — re-running this
-- file raises "duplicate column name". The application-layer guard in
-- app/database/db.py init_db() reads PRAGMA table_info(model_registry)
-- and only applies the body if the is_active column is missing.

ALTER TABLE model_registry ADD COLUMN artifact_path TEXT;
ALTER TABLE model_registry ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0;

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_name_version
    ON model_registry(model_name, version);

CREATE INDEX IF NOT EXISTS idx_model_registry_active
    ON model_registry(model_name, is_active);
