-- Migration: 001_initial.down (sqlite)
-- computation_date: n/a | source_dataset_version: n/a
--
-- Manual rollback only; never auto-run by `migrate()`. Drop order is
-- reverse of creation so foreign keys (when added in future migrations) are
-- unset before their parents.

DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS classification_log;
DROP TABLE IF EXISTS recommendation_log;
DROP TABLE IF EXISTS prediction_log;
