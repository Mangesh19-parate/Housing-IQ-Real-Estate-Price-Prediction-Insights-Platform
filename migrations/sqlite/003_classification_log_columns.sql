-- Migration: 003_classification_log_columns (sqlite)
-- computation_date: 2026-09-05 | source_dataset_version: n/a
--
-- Adds missing columns to classification_log per Spec 25:
--   - verdict_probabilities_json (TEXT): JSON string of good-deal verdict probabilities
--   - latency_ms (INTEGER): wall-clock latency for the /classify call
--
-- SQLite ALTER TABLE ADD COLUMN is NOT idempotent — re-running this
-- file raises "duplicate column name". The application-layer guard in
-- app/database/db.py init_db() reads PRAGMA table_info(classification_log)
-- and only applies the body if the verdict_probabilities_json column is missing.

ALTER TABLE classification_log ADD COLUMN verdict_probabilities_json TEXT;
ALTER TABLE classification_log ADD COLUMN latency_ms INTEGER;