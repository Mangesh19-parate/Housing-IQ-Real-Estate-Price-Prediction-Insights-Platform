# Spec: Repo Scaffolding And Environment Setup

## Overview
Lock down the HousingIQ development environment so Week 1 (data cleaning) and every later step run reproducibly. This spec turns the empty scaffold (`app/app.py`, `api/main.py`, all `routers/*.py`, all `tests/*.py`, all `templates/*.html`, all `static/css/*.css`, all `static/js/*.js`) into importable Python packages and a working two-service dev loop (Flask `:5000` + FastAPI `:8000`) with config from `.env`, the dependency set already pinned in `requirements.txt`, and the tooling hooks (`pytest`, `ruff`) wired up. It also creates the SQLite application-DB schema (operational tables only — no user accounts) and a one-command pipeline entry point per TRD §13 ("`make pipeline`" or equivalent) so future steps have a single `python -m` command to call. Module: **foundation**. This is Step 01 — everything in the 7-week roadmap depends on it landing cleanly.

## Depends on
Nothing. This is the first spec.

## Routes / Endpoints
**No new routes/endpoints.** Stub placeholders for the routes listed in `CLAUDE.md`'s "Implemented vs stub routes" table remain stubs after this spec — filling them in is the job of later specs (price prediction, classification, analytics, recommender, insights, map).

What this spec *does* add, in service of making those routes testable later:
- FastAPI: `GET /health` returns `{"status": "ok"}` so the Flask app's "is FastAPI up?" check has something to call. No model loading, no inference logic.
- Flask: `GET /` renders `landing.html` (a real landing template, not the empty stub) so the dev loop is visible end-to-end.

## Data / Schema changes
SQLite application DB at `data/app.db` (path configurable via `APP_DB_PATH` env var), created lazily by `app/database/db.py:init_db()`. Per CLAUDE.md, this DB is operational/logging only — no user data.

Tables created in `init_db()`. Column names follow `docs/05-BACKEND-SCHEMA.md` §5 + §U-SCHEMA-11 + §U-SCHEMA-13 (the schema doc is authoritative; this spec tracks it):
- `prediction_log` — anonymous prediction request/response records: `id` INTEGER PK, `timestamp` DATETIME, `city` TEXT, `locality` TEXT, `input_features_json` TEXT, `predicted_price` REAL, `predicted_range_low` REAL, `predicted_range_high` REAL, `model_version` TEXT, `is_outlier_input` INTEGER (bool), `latency_ms` INTEGER. No PII.
- `recommendation_log` — seed + top-N returned: `id` INTEGER PK, `timestamp` DATETIME, `seed_features_json` TEXT, `returned_listing_ids` TEXT (JSON-encoded array), `used_fallback` INTEGER (bool).
- `classification_log` — verdict + tier per request (U-SCHEMA-11 — verdict is the primary output, tier is supporting): `id` INTEGER PK, `timestamp` DATETIME, `city` TEXT, `input_features_json` TEXT, `predicted_verdict` TEXT, `predicted_tier` TEXT, `tier_probabilities_json` TEXT, `model_version` TEXT.
- `model_registry` — lightweight training-trace registry (U-SCHEMA-13, no MLflow): `id` INTEGER PK, `model_name` TEXT, `version` TEXT, `training_dataset_version` TEXT, `git_commit` TEXT, `training_date` DATETIME, `rmse` REAL NULL, `mae` REAL NULL, `r2` REAL NULL, `hyperparameters` TEXT (JSON), `feature_hash` TEXT. Populated by future training scripts, not by this spec.

Schema is referenced by CLAUDE.md; cross-checked against `05-BACKEND-SCHEMA.md` (read of that doc is out of scope for this spec — Week 1's cleaning spec owns that alignment).

No parquet cache files, no analytics cache files, no model artifacts touched — those come later.

## Templates / UI
**Create:**
- `app/templates/landing.html` — minimal landing page: city quick-filter buttons (4 cities, `data-city` attribute, no logic wired yet — that's the predictor page's job), module cards linking to `/predict`, `/classify`, `/analytics`, `/recommend`, `/insights`, `/map`. Extends `base.html`. Uses CSS variables only — no hardcoded hex. Static layout; the visual system arrives in a later `ui-ux` spec.

**Modify:** none (other templates stay as empty stubs).

The stub `app/static/css/style.css` gets a minimal token set so the landing page renders without inline hex — just CSS custom properties for `--color-bg`, `--color-fg`, `--color-accent`, `--space-1`..`--space-4`, `--radius`, `--font-sans`. No page-specific layout CSS yet.

## Files to change / Files to create

**Create:**
- `housingiq/__init__.py` — empty, makes the project root a package. (Project lives at repo root, not in a subfolder — see `scaffold.ps1` which created `housingiq/` only inside the script's cwd; since we're already at the repo root, the package is the repo root.)
- `app/__init__.py` — empty, makes `app/` a package.
- `app/database/__init__.py` — empty.
- `api/__init__.py` — empty.
- `api/routers/__init__.py` — empty.
- `api/schemas/__init__.py` — empty.
- `api/services/__init__.py` — empty.
- `ml/__init__.py`, `ml/cleaning/__init__.py`, `ml/features/__init__.py`, `ml/training/__init__.py`, `ml/evaluation/__init__.py`, `ml/recommender/__init__.py` — empty packages.
- `tests/__init__.py` — empty.
- `app/database/db.py` — `get_db()` context manager + `init_db()` that creates the 4 tables above. Parameterized SQL only (`?` placeholders). No ORM.
- `app/config.py` — reads `APP_DB_PATH`, `FASTAPI_BASE_URL`, `FLASK_SECRET_KEY` from env (via `python-dotenv`). Defaults: `data/app.db`, `http://localhost:8000`, dev secret.
- `api/config.py` — same env-reader, but for `MODELS_DIR`, `ANALYTICS_CACHE_DIR`, `PROCESSED_DATA_DIR`.
- `api/main.py` — current stub → real FastAPI app: `GET /health` returns `{"status": "ok"}`, includes the (still-stub) routers, calls `init_db()` on startup.
- `api/routers/predict.py`, `classify.py`, `analytics.py`, `recommend.py`, `insights.py` — currently empty stubs; leave them empty for now (later specs fill them). Just need the package to import.
- `app/app.py` — current empty stub → real Flask app: `GET /` renders `landing.html`, registers a `before_request` hook that ensures `init_db()` has run, reads `FLASK_DEBUG` and `FASTAPI_BASE_URL` from config, error handler that uses `abort()` per CLAUDE.md.
- `app/templates/landing.html` — landing page described above.
- `app/templates/base.html` — currently empty stub → real base layout with `{% block content %}` and a `<link>` to `/static/css/style.css`. All later templates extend this.
- `app/static/css/style.css` — currently empty stub → CSS custom properties + minimal reset.
- `tests/conftest.py` — currently empty stub → pytest fixtures: `app_client` (Flask test client with a temp SQLite DB per test), `api_client` (FastAPI TestClient), `tmp_clean_db` (yields a fresh DB path).
- `tests/test_scaffolding.py` — minimum smoke tests listed in "Definition of done" below.
- `scripts/run_pipeline.py` — stub of the `make pipeline`/`python -m` entry point. Single function `def main(): print("pipeline not implemented yet — see 02-data-cleaning spec")`. This is the only artifact from TRD §13 that lands in Step 01; later specs add the actual stage functions.
- `.env.example` — documents required + optional env vars (`APP_DB_PATH`, `FASTAPI_BASE_URL`, `FLASK_SECRET_KEY`, `MODELS_DIR`, `ANALYTICS_CACHE_DIR`, `PROCESSED_DATA_DIR`). Real `.env` stays out of git per `.gitignore`.
- `pytest.ini` — sets `pythonpath = .` and `testpaths = tests`. (Python 3.10+ assumed per CLAUDE.md.)
- `ruff.toml` (or `[tool.ruff]` in a new `pyproject.toml`) — minimal config: line-length 100, target Python 3.10, select `E,F,W,I` (pycodestyle + pyflakes + warnings + isort).
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/processed/analytics_cache/.gitkeep`, `data/stats/.gitkeep`, `models/.gitkeep`, `notebooks/.gitkeep` — keep the empty tracked directories in git so the layout is visible.

**Modify:**
- `requirements.txt` — already complete; no changes needed. (Pinning policy: don't bump versions in this spec.)
- `.gitignore` — already written by hand in this session; needs the additions: `data/app.db`, `data/app.db-journal`, `.env`, `coverage.xml`, `.mypy_cache/`, `.ruff_cache/`. (`docs.zip` already excluded.)

## New dependencies
No new dependencies. Everything needed is already in `requirements.txt`: `flask`, `fastapi`, `uvicorn[standard]`, `pydantic`, `pytest`, `httpx`, `python-dotenv`, `ruff`. SQLite ships with Python stdlib (`sqlite3`).

## Rules for implementation
- No SQLAlchemy/ORM — `sqlite3` stdlib + `?`-parameterized queries only, per CLAUDE.md "No SQLAlchemy/ORM unless already in use."
- No dealer/contact/media-URL fields ever reach the UI or an export — N/A for this spec (no data flows yet), but the rule is fixed and binding for every later spec.
- CSS variables only — `style.css` defines tokens, no hardcoded hex anywhere in the landing page or base layout.
- All templates extend `base.html` — `landing.html` is the first template and must extend it.
- Model changes must reference the fixed evaluation protocol — N/A for this spec (no model artifacts touched).
- Repo is a Python package from this spec onward. `python -m pytest`, `python -m app.app`, `uvicorn api.main:app` must all work from repo root without any `PYTHONPATH` games. `pytest.ini`'s `pythonpath = .` covers the test runner; for `python -m app.app` and `uvicorn api.main:app` the implicit cwd-at-repo-root is enough since `app/` and `api/` are packages at root.
- Flask never imports model code or touches `.pkl` files — `app/app.py` and `app/config.py` only talk to SQLite + FastAPI via HTTP. The `requests` dep already in `requirements.txt` covers the internal FastAPI calls.
- No real secrets in `.env` or `.env.example`. `.env.example` documents keys only; the real `.env` is gitignored.

## Definition of done
A specific, testable checklist verifiable by running the app or the test suite.

1. `python -m pytest` from repo root runs and passes. `tests/test_scaffolding.py` contains exactly these tests:
   - `test_app_imports` — `from app.app import create_app; app = create_app()` succeeds.
   - `test_api_imports` — `from api.main import app as fastapi_app` succeeds.
   - `test_flask_landing_renders` — GET `/` returns 200, response body contains "HousingIQ" (landing page heading text).
   - `test_flask_landing_extends_base` — landing template starts with `{% extends "base.html" %}`.
   - `test_fastapi_health` — GET `/health` via FastAPI TestClient returns 200 and JSON `{"status": "ok"}`.
   - `test_init_db_creates_tables` — `init_db()` on a temp DB creates all 4 tables (`prediction_log`, `recommendation_log`, `classification_log`, `model_registry`); verified by querying `sqlite_master`.
   - `test_get_db_uses_parameterized_query` — `get_db()` opens a connection; a single parameterized insert + select round-trips.
   - `test_db_path_is_configurable` — setting `APP_DB_PATH` env var changes where the DB is created.
   - `test_css_has_no_hardcoded_hex` — scan `app/static/css/*.css` for `#[0-9a-fA-F]{3,6}` outside the `:root` token block; assert zero matches.
   - `test_landing_template_no_hardcoded_hex` — same scan against `app/templates/landing.html`.
   - `test_conftest_fixtures` — `app_client`, `api_client`, `tmp_clean_db` fixtures all resolve and produce working test clients.
2. `python -m app.app` from repo root starts Flask on `:5000`; visiting `http://localhost:5000/` in a browser shows the landing page with 4 city buttons and 6 module cards.
3. `uvicorn api.main:app --reload --port 8000` from repo root starts FastAPI; `curl http://localhost:8000/health` returns `{"status":"ok"}`.
4. `python scripts/run_pipeline.py` runs without error and prints the placeholder message.
5. `ruff check .` from repo root reports zero issues on the new code (existing stubs remain empty — they're ignored by ruff).
6. `.gitignore` excludes `.env`, `data/app.db`, `data/app.db-journal`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `venv/`, `.venv/`, `docs.zip`. Verified by `git check-ignore -v .env data/app.db`.
7. `git status` clean after a fresh `git add . && git commit`. (Repo state — not a code check.)
8. `CLAUDE.md`'s "Implemented vs stub routes" table is updated: `GET /` moves from "Stub" to "Landing page rendered (this spec)", `GET /health` (FastAPI) moves from "Stub" to "Returns `{'status': 'ok'}` (this spec)"; everything else stays "Stub". This update is part of the PR for this spec.
