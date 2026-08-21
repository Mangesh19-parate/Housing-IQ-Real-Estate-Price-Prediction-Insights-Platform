-- Migration: 002_add_active_and_artifact.down (sqlite)
-- Manual rollback only; never auto-run by `migrate()`.
--
-- SQLite < 3.35 has no `ALTER TABLE ... DROP COLUMN`. The pragmatic
-- rollback pattern (same as Spec 08's `_initial.down.sql`) is to drop
-- and recreate the table — but only after dropping dependent indexes,
-- otherwise the recreate step inherits them and we leak column shape.
--
-- Down-migrations are humans-only. If you need to actually roll back,
-- back up the DB file first.

DROP INDEX IF EXISTS idx_model_registry_active;
DROP INDEX IF EXISTS idx_model_registry_name_version;

CREATE TABLE model_registry__backup (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    version TEXT,
    training_dataset_version TEXT,
    git_commit TEXT,
    training_date DATETIME,
    rmse REAL,
    mae REAL,
    r2 REAL,
    hyperparameters TEXT,
    feature_hash TEXT
);

INSERT INTO model_registry__backup
    (id, model_name, version, training_dataset_version, git_commit,
     training_date, rmse, mae, r2, hyperparameters, feature_hash)
SELECT id, model_name, version, training_dataset_version, git_commit,
       training_date, rmse, mae, r2, hyperparameters, feature_hash
FROM model_registry;

DROP TABLE model_registry;
ALTER TABLE model_registry__backup RENAME TO model_registry;
