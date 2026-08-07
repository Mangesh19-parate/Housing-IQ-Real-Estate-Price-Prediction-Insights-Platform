-- Migration: 001_initial (postgres)
-- computation_date: n/a | source_dataset_version: n/a
--
-- Application DB baseline: 4 operational tables (no user data, no auth).
-- Column shapes match docs/05-BACKEND-SCHEMA.md §5 + §U-SCHEMA-11 + §U-SCHEMA-13.
--
-- Postgres-specific shapes:
--   SERIAL PRIMARY KEY (SQLite uses INTEGER PRIMARY KEY with rowid alias)
--   TIMESTAMP        (instead of DATETIME)
--   JSONB            (for JSON-shaped columns; cast at write time)

CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    city TEXT,
    locality TEXT,
    input_features_json JSONB,
    predicted_price DOUBLE PRECISION,
    predicted_range_low DOUBLE PRECISION,
    predicted_range_high DOUBLE PRECISION,
    model_version TEXT,
    is_outlier_input BOOLEAN,
    latency_ms INTEGER
);

CREATE TABLE IF NOT EXISTS recommendation_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    seed_features_json JSONB,
    returned_listing_ids JSONB,
    used_fallback BOOLEAN
);

CREATE TABLE IF NOT EXISTS classification_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    city TEXT,
    input_features_json JSONB,
    predicted_verdict TEXT,
    predicted_tier TEXT,
    tier_probabilities_json JSONB,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    version TEXT,
    training_dataset_version TEXT,
    git_commit TEXT,
    training_date TIMESTAMP,
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    hyperparameters JSONB,
    feature_hash TEXT
);
