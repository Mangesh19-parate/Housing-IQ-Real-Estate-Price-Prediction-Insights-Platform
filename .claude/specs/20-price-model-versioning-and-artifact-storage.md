# Spec: Price Model Versioning and Artifact Storage

## Overview
Establish a versioning convention and lightweight registry for the price-prediction model artifacts (`.pkl` files and their `metrics.json` siblings) so that every trained version is reproducible, traceable, and loadable by name from a single canonical location. This spec implements the `model_registry` table already documented in Backend Schema §U-SCHEMA-13 and the version-suffix rule in Rules §2.5 — currently, artifacts are saved ad-hoc by training scripts (Spec 13, Spec 14) and the FastAPI service (Spec 17) just reads whatever file is on disk. This module belongs to the **deployment** track: it touches no model logic, only the storage, naming, and lookup layer that the inference service depends on. It exists now because the v2 boosted-tree run (Spec 14) produced `price_model_v2` artifacts alongside the existing `price_model_v1` set, and a future v3 / classification / recommender registration will pile on more `_*_v{n}.pkl` files; without a single registry + naming contract, "which model is the live one?" becomes guesswork.

## Depends on
- Spec 13 (`baseline-regression-model-training`) — produces `price_model_v1.pkl` + `metrics_v1.json`
- Spec 14 (`xgboost-lightgbm-price-model-training`) — produces `price_model_v2.pkl` + v2 metrics, and the `PRICE_MODEL_VERSION_V2` constant in `ml/training/__init__.py`
- Spec 15 (`price-model-evaluation-protocol`) — defines the `metrics_v{n}.json` schema (R², MAE, RMSE, MAPE on original ₹ scale)
- Spec 17 (`price-prediction-fastapi-endpoint`) — consumes a `price_model_v{n}.pkl` file path and a `model_version` string at request time
- Backend Schema §U-SCHEMA-13 — the `model_registry` table contract this spec implements

## Routes / Endpoints
- FastAPI: `GET /health` (modify) — extend the existing response to include `model_version` from the registry (already in the §7 contract as `{"status": "ok", "model_version": "..."}` but not yet wired)
- FastAPI: `GET /models` (new) — list registered model versions and their metadata (read-only, no auth)
- Flask: none

## Data / Schema changes
- **New SQLite table** `model_registry` (per Backend Schema §U-SCHEMA-13). Schema:
  | Column | Type | Notes |
  |---|---|---|
  | `id` | int (PK, autoincrement) | |
  | `model_name` | string | e.g. `price_model_sale`, `price_model_rent`, `tier_classifier`, `good_deal_classifier` |
  | `version` | string | e.g. `v1`, `v2` |
  | `training_dataset_version` | string | source parquet filename/hash, e.g. `clean_listings_2026-08-14.parquet` |
  | `git_commit` | string | `git rev-parse HEAD` at training time |
  | `training_date` | datetime | ISO 8601 UTC |
  | `rmse` | float (nullable) | for regressors |
  | `mae` | float (nullable) | for regressors |
  | `r2` | float (nullable) | for regressors |
  | `accuracy` | float (nullable) | for classifiers |
  | `macro_f1` | float (nullable) | for classifiers |
  | `roc_auc_ovr` | float (nullable) | for classifiers |
  | `hyperparameters` | text/JSON | serialized estimator params |
  | `feature_hash` | string | SHA-256 of sorted final feature list |
  | `artifact_path` | string | repo-relative path to the `.pkl` |
  | `is_active` | bool | exactly one row per `model_name` is `is_active=1` |
- Migration script applied through `app/database/db.py` `init_db()` (idempotent CREATE TABLE IF NOT EXISTS) — same channel Spec 08 already uses for `prediction_log`.
- No new model artifacts created by this spec (no retraining); existing files under `models/` are registered, not regenerated.

## Templates / UI
No UI changes. The `model_version` field is already passed through `predict_result.html` (Spec 19); it will now reflect the registered, active row rather than a hardcoded string.

## Files to change / Files to create
- **Create** `app/database/migrations/002_model_registry.sql` — `CREATE TABLE model_registry IF NOT EXISTS` + unique index on `(model_name, version)`
- **Create** `ml/registry/__init__.py` — public surface re-exporting the helpers below
- **Create** `ml/registry/registry.py` — `register_model(...)`, `set_active(model_name, version)`, `get_active(model_name)`, `list_models()`. Uses `app.database.db.get_db()` (parameterized queries only) for reads/writes.
- **Create** `ml/registry/naming.py` — `artifact_path(model_name, version) -> Path`, `metrics_path(model_name, version) -> Path`, `next_version(model_name) -> str` (scans `models/` + the registry to find the highest existing `v{n}` and returns `v{n+1}`).
- **Create** `ml/registry/feature_hash.py` — `compute_feature_hash(feature_list: list[str]) -> str` (sorted, SHA-256 hex, first 16 chars — short enough to scan, long enough to be unique within a project)
- **Create** `tests/test_model_registry.py` — round-trip tests for register/activate/list, naming conventions, feature-hash determinism, and migration idempotency
- **Modify** `app/database/db.py` — add `model_registry` DDL to `init_db()` so fresh dev DBs get the table
- **Modify** `api/services/model_loader.py` (or wherever Spec 17 reads the artifact path — see `api/routers/predict.py` import of `price_model_v2.pkl`) — replace the hardcoded filename with `get_active("price_model_sale")["artifact_path"]` / `get_active("price_model_rent")["artifact_path"]`, and surface `model_version` to the response + `/health`
- **Modify** `api/routers/predict.py` — read `model_version` from the registry row, not from a constant
- **Modify** `api/main.py` (or `routers/health.py` if present) — return `model_version` from `GET /health`
- **Modify** `ml/training/__init__.py` — keep `PRICE_MODEL_VERSION_V2` as the default for **new** training runs, but the FastAPI side now resolves through the registry
- **Modify** `scripts/train_price_model_v2.py` — call `register_model(...)` once training finishes (one extra block at the end of the script, idempotent — re-running an already-registered `(model_name, version)` is a no-op)
- **Modify** `scripts/seed_registry.py` (new tiny script) — one-shot backfill: register the existing `price_model_v1.pkl` + `price_model_v2.pkl` and their `metrics_*.json` siblings so the table is non-empty on day one; `python scripts/seed_registry.py` is the documented bootstrap command

## New dependencies
No new dependencies. `hashlib` (stdlib) for the feature hash, `sqlite3` (stdlib, already used via `app/database/db.py`), `pathlib` (stdlib).

## Rules for implementation
- Parameterized SQL only (`?` placeholders via `app/database/db.py`) — no f-strings or `.format()` into any query, in `ml/registry/registry.py` or anywhere else.
- No SQLAlchemy/ORM — the project uses raw sqlite3 throughout; this spec follows suit.
- The `models/` directory stays the source of truth for the actual `.pkl` bytes; the registry is a metadata index, not a blob store. Files are never moved or renamed by this spec — only their path is recorded.
- Re-registering an existing `(model_name, version)` pair must be a no-op (idempotent), so re-running a training script after a crash does not duplicate rows.
- `is_active` is enforced application-side: `set_active(model_name, version)` first clears `is_active` for all rows of that `model_name`, then sets it for the chosen row — both inside one transaction.
- The registry never holds any PII. Model metadata is derived from training-time inputs only (no listing IDs, no dealer/contact fields).
- Model evaluation metrics written here come from the already-validated `metrics_v{n}.json` produced under Spec 15's fixed protocol — this spec does not re-compute or re-evaluate anything; it just copies the numbers.
- The version string in the live FastAPI response must match the version in the registry; if a model file on disk has no registry row, the loader logs a warning and falls back to the on-disk file (one-line warning, not a crash — matches Rules §5.2's graceful-degradation spirit, applied to model resolution rather than HTTP calls).
- No new CSS, no template changes — this is a backend-only spec.

## Definition of done
- `pytest tests/test_model_registry.py` passes; covers: register → list → set_active → get_active round-trip; naming convention for sale/rent/tier/good_deal; feature_hash determinism; idempotent re-registration; migration runs twice without error.
- `python scripts/seed_registry.py` registers `price_model_v1` and `price_model_v2` (and their rent-pipeline siblings if present) without error and prints a summary table.
- `curl http://localhost:8000/health` returns `{"status": "ok", "model_version": "<value from registry>"}` where the value matches the `is_active=1` row for `price_model_sale`.
- `curl http://localhost:8000/models` returns a JSON array of registry rows, newest training_date first.
- `POST /predict` response `model_version` field matches the active registry row's `version`, not a hardcoded string.
- `ruff check ml/registry app/database tests/test_model_registry.py` is clean.
- Existing 379+ tests still pass (full suite).
