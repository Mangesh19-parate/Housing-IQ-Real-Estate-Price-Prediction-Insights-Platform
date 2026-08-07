-- Migration: 001_initial (sqlite)
-- computation_date: n/a | source_dataset_version: n/a
--
-- Application DB baseline: 4 operational tables (no user data, no auth).
-- Column shapes match docs/05-BACKEND-SCHEMA.md §5 + §U-SCHEMA-11 + §U-SCHEMA-13.
--
-- SQLite note: INTEGER PRIMARY KEY without AUTOINCREMENT is rowid-aliased
-- and Postgres-compatible; the migrations runner does no translation here.
-- AUTOINCREMENT is intentionally omitted so the same SQL body works on both
-- dialects once the postgres counterpart substitutes SERIAL/TIMESTAMP.

CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    city TEXT,
    locality TEXT,
    input_features_json TEXT,
    predicted_price REAL,
    predicted_range_low REAL,
    predicted_range_high REAL,
    model_version TEXT,
    is_outlier_input INTEGER,
    latency_ms INTEGER
);

CREATE TABLE IF NOT EXISTS recommendation_log (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    seed_features_json TEXT,
    returned_listing_ids TEXT,
    used_fallback INTEGER
);

CREATE TABLE IF NOT EXISTS classification_log (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    city TEXT,
    input_features_json TEXT,
    predicted_verdict TEXT,
    predicted_tier TEXT,
    tier_probabilities_json TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
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
