-- Migration: 002_add_active_and_artifact.down (postgres)
-- Manual rollback only; never auto-run by `migrate()`.

DROP INDEX IF EXISTS idx_model_registry_active;
DROP INDEX IF EXISTS idx_model_registry_name_version;

ALTER TABLE model_registry DROP COLUMN IF EXISTS is_active;
ALTER TABLE model_registry DROP COLUMN IF EXISTS artifact_path;
