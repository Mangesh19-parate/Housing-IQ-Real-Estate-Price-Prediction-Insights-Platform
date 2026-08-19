# Spec: Price Prediction FastAPI Endpoint

## Overview
Wire the v2 price regression model (Spec 14) + the precomputed SHAP
explainer (Spec 16) into the FastAPI `POST /predict` route using the
frozen Pydantic `PredictRequestV3` / `PredictResponseV3` contract from
Spec 11. The route dispatches on `transact_type` (`Sale` → sale
pipeline, `Rent` → rent pipeline) *before* any preprocessing, applies
the v1 preprocessor (`models/feature_pipeline_v1.pkl`) + the v2
geo/sector-encoder sidecar, computes the point estimate + the
`-1/+1` std residual-band range, attaches the top-N SHAP
contributions from the precomputed `shap_explainer_*.pkl`, and
appends one row to `prediction_log` with the per-request latency.
Module: **price-prediction**.

This is the single point where the offline ML pipeline meets the
serving layer. It does **not** train anything new; it consumes the
artifacts Spec 13/14/16 produced and the schemas Spec 11 froze.
Per Rules §2.4 the route loads the exact persisted `Pipeline`
instance — no re-implementing preprocessing in the API layer.

## Depends on
- **Spec 11** (`11-price-prediction-input-schema-v3.md`) — provides
  `PredictRequestV3`, `PredictResponseV3`, `PredictResponseV3`'s
  `ShapContribution` shape, plus `INPUT_FIELDS_V3` /
  `INPUT_FIELD_TYPES_V3` and the 8 enums.
- **Spec 13** (`13-baseline-regression-model-training.md`) —
  `models/price_model_{sale,rent}_v1.pkl` + `models/feature_pipeline_v1.pkl`
  + `models/feature_list_v1.json` + `models/metrics_v1.json`.
- **Spec 14** (`14-xgboost-lightgbm-price-model-training.md`) —
  `models/price_model_{sale,rent}_v2.pkl` (the winner) +
  `models/sector_target_encoder_v2.pkl` (Lever 4 sidecar) +
  `models/metrics_v2.json`. The default version served is `v2`,
  per Spec 15's gate certification.
- **Spec 15** (`15-price-model-evaluation-protocol.md`) — the
  evaluation gate that certified v2 is shippable; the route's
  `MODEL_VERSION` default pulls from the gate's `active_version`.
- **Spec 16** (`16-shap-explainability-price-model.md`) —
  `models/shap_explainer_{sale,rent}_v2.pkl` + `models/feature_label_map_v2.json`
  + `ml.explainability.explain_one` / `global_summary`.
- **`api/schemas/predict_v3.py`** — re-exported via `api.schemas`.
- **`api/config.py`** — `MODELS_DIR`, `APP_DB_PATH`,
  `PROCESSED_DATA_DIR` env vars.
- **`app/database/db.py`** — `get_db()` for `prediction_log` writes.
- **`docs/05-BACKEND-SCHEMA.md` §7** — `POST /predict` response
  contract.
- **`docs/05-BACKEND-SCHEMA.md` §5** — `prediction_log` columns.
- **`docs/02-TRD.md` §U-TRD-4** — `transact_type` routing rule.
- **`docs/02-TRD.md` §U-TRD-3** — `ColumnTransformer` block layout
  the preprocessor emits.
- **`docs/08-RULES.md` §2.4** — model in production is the exact
  same `Pipeline` used at evaluation; no re-implemented
  preprocessing.
- **`docs/08-RULES.md` §2.6** — SHAP comes from the same model
  instance making the prediction.
- **`docs/08-RULES.md` §5.1, §5.2** — Flask never loads models;
  FastAPI degrades gracefully on Flask-side failures (this spec is
  the FastAPI side; the Flask-call path is a later spec).
- **`docs/08-RULES.md` §1.3, §2.5** — versioned artifacts never
  overwritten; every `prediction_log` row is dated.
- **`shap-explainability` skill** — precomputed-explainer-at-startup
  convention + top-N=7 rule.
- **`fastapi-serving` skill** — FastAPI lifespan vs on_event, lazy
  model load, and `HTTPException` usage.

## Routes / Endpoints
- **FastAPI:** `POST /predict` — price prediction. Accepts a
  `PredictRequestV3` body, returns a `PredictResponseV3`. Loads the
  v2 sale/rent pipeline + the precomputed SHAP explainer at FastAPI
  startup (lifespan), keeps a single in-memory cache; no I/O on the
  hot path. Access: internal (Flask → FastAPI over HTTP, not public
  internet).
- **Flask:** none added. The Flask `/predict` page wiring is a
  follow-on spec (Day 38 in the Implementation Plan).

## Data / Schema changes
- **No new application DB tables.** `prediction_log` already exists
  (`app/database/db.py`) with the full column set from Backend
  Schema §5 — the route writes one row per successful request.
- **No new model artifacts.** This spec consumes the v2 model +
  SHAP explainer + preprocessor + sidecar encoder from earlier
  specs.
- **No `data/raw/` or `data/processed/` writes.**
- **No new SQL migrations.**
- **No `analytics_cache/*.json` additions.**

## Templates / UI
None. This spec is the FastAPI serving layer. The Flask `/predict`
form + the SHAP chart on the result page are follow-on specs.

## Files to change / Files to create

**Create:**
- `api/services/predict_service.py` — the inference service that
  the route delegates to. Public API:
  - `class PredictService:` — lazily-loaded model + explainer
    cache, keyed by `(transact_type, model_version)`. Uses a
    `threading.Lock` per key for the first-load race (the
    `fastapi-serving` skill's "load once, cache forever" pattern).
    - `def __init__(self, models_dir: Path, *, model_version:
      str = "v2") -> None:` — does **not** load anything;
      model loads happen on the first `predict()` call (or on
      `warmup()`).
    - `def warmup(self) -> None:` — loads all `transact_type`
      pipelines + matched explainers + the shared preprocessor +
      the v2 sector-encoder sidecar. Called from FastAPI's
      lifespan handler. Idempotent. If `models/price_model_rent_v2.pkl`
      is missing (Rent was skipped at training), logs INFO and
      leaves the rent key unset.
    - `def predict(self, request: PredictRequestV3) ->
      PredictResponseV3:` — the single hot path. Steps:
      1. Resolve `(transact_type, model_version)` → load the
         pipeline + explainer if not cached.
      2. Resolve `luxury_category` from the `amenities` list
         (server-derived, never client-supplied — `Rules §10.2`).
         Helper: `_resolve_luxury_category(amenities:
         list[str]) -> LuxuryCategory` — pure function; threshold
         rules pinned by tests (e.g., `n_amenities >= 5` →
         `HIGH`, `>= 2` → `MEDIUM`, else `LOW`). *(ponytail: small
         deterministic lookup table; over-engineering this into a
         trained classifier is out of scope for one endpoint.)*
      3. Build a 1-row DataFrame in the same column order the
         preprocessor expects (per `models/feature_list_v1.json`).
         `'amenities'` list is converted to `n_amenities` +
         `has_<amenity>` flags via the pinned top-15 amenity set
         from `ml/features/feature_frame.py`. Missing preprocessor
         columns are filled with `NaN` (the fitted
         `ColumnTransformer` expects them to be present, even if
         all-NaN for a row).
      4. Apply the v1 preprocessor (loaded, not refit — Rules
         §2.4): `X = preprocessor.transform(df)`.
      5. If `model_version == "v2"`, append the v2 sibling
         features (geo + `sector_smoothed_price`) via the same
         `np.hstack` pattern `scripts/train_price_model_v2.py`
         used at training time. *(ponytail: copy the single
         `StandardScaler` + `hstack` block from the training
         script — one screen, no helper class.)*
      6. Call `model.predict(X)` → `y_log_pred` (log-price).
         `price_pred = float(np.expm1(y_log_pred))`.
      7. Compute the range band. Pinned rule: `range_low =
         max(0.0, price_pred * (1 - std_pct))`, `range_high =
         price_pred * (1 + std_pct)`, where `std_pct` is the
         per-model **residual relative std** from
         `metrics_v{N}.json.chosen_metrics.test_residual_std_pct`
         (a new field added by training; default `0.15` if the
         field is missing). *(ponytail: simplest band — the
         PRD §6.1 `FR2` calls ±1 std "or quantile bounds"; the
         ±15% choice matches the MAE-within-15% gateway and needs
         no new code path.)*
      8. Compute SHAP via
         `ml.explainability.explain_one(model, explainer, X,
         feature_names, label_map, top_n=SHAP_TOP_N)`. Map to
         `list[ShapContribution]` (Backend Schema §7 shape).
      9. Build `PredictResponseV3(...)` and return.
    - `def _is_outlier_input(self, X: np.ndarray) -> bool:` —
      cheap heuristic: returns `True` if any feature is more than
      6σ from the training-distribution mean (Pinned by tests).
      *(ponytail: simplest defensible outlier flag — the full
      training-set distance check is a later spec; the route just
      needs a boolean to echo in the response per Backend Schema
      §7.)*
    - `MODEL_VERSION: str = "v2"` — pinned module constant;
      overridable via the `PredictService(..., model_version=...)`
      constructor arg (used by tests).

- `api/schemas/predict_log_entry.py` — the DB-row serialiser (kept
  separate from `predict_v3.py` so the schema module stays
  I/O-free — `test_predict_v3_does_not_import_app_ml_or_models`
  from Spec 11 would otherwise break). Public API:
  - `def to_prediction_log_row(request: PredictRequestV3,
    response: PredictResponseV3, latency_ms: int) -> dict[str,
    Any]:` — returns a dict whose keys match the `prediction_log`
    column names verbatim (Backend Schema §5). Pure function, no
    I/O. Pinned by tests.

- `tests/test_predict_service.py` — pytest tests for the service
  layer. Required tests (exact names):
  - `test_predict_service_warmup_loads_sale_pipeline`
  - `test_predict_service_warmup_loads_rent_pipeline_when_present`
  - `test_predict_service_warmup_skips_rent_when_artifact_missing`
  - `test_predict_service_warmup_is_idempotent`
  - `test_predict_service_predict_returns_response_v3`
  - `test_predict_service_predict_routes_by_transact_type` — same
    request shape but `Sale` vs `Rent` resolve to different
    models in the cache.
  - `test_predict_service_predict_attaches_shap_contributions` —
    hot-path SHAP returns a `list[ShapContribution]` capped at
    `SHAP_TOP_N`.
  - `test_predict_service_predict_excludes_outlier_input_flag` —
    the response always has a boolean `is_outlier_input`.
  - `test_predict_service_predict_resolves_luxury_category` —
    `amenities=[]` → `LOW`; `["Clubhouse"]` → `MEDIUM`; ≥ 5
    amenities → `HIGH` (pinned lookup).
  - `test_predict_service_predict_uses_loaded_preprocessor` —
    pinned: monkeypatch the preprocessor's `transform` and
    assert the service called it (proves Rules §2.4 — no
    re-implementation).
  - `test_predict_service_predict_uses_loaded_explainer` —
    pinned: monkeypatch the explainer and assert `shap_values`
    was called.
  - `test_predict_service_predict_logs_to_prediction_log` — uses
    an in-memory SQLite DB (`tmp_path`) + the existing
    `app.database.db.init_db` to create the tables, runs
    `predict()`, asserts exactly one row inserted into
    `prediction_log` with the expected columns.
  - `test_predict_service_predict_does_not_log_pii_fields` —
    greps the captured row's `input_features_json` for the
    regex `(contact|dealer|phone|email|photo|url|spid)` — must
    be absent.
  - `test_predict_service_predict_latency_ms_is_positive_int` —
    assert the logged `latency_ms` is a non-negative int.

- `tests/test_predict_endpoint.py` — pytest tests for the FastAPI
  route via `TestClient`. Required tests (exact names):
  - `test_predict_endpoint_returns_200_for_valid_payload` — full
    minimal payload with a 3BHK Gurgaon flat → 200 +
    `PredictResponseV3`-shaped body.
  - `test_predict_endpoint_returns_422_on_missing_field` — omit
    `built_up_area` → 422 (Pydantic, not a 500).
  - `test_predict_endpoint_returns_422_on_extra_field` — send
    `{"bedrooms": 3}` (typo) → 422 (confirms `extra="forbid"`).
  - `test_predict_endpoint_returns_422_on_bedroom_bathroom_violation`
  - `test_predict_endpoint_returns_422_on_area_over_20000`
  - `test_predict_endpoint_returns_503_when_model_artifact_missing` —
    patch the service to raise `FileNotFoundError`; route must
    translate to `HTTPException(503, "model not loaded")`, not
    a 500.
  - `test_predict_endpoint_returns_503_no_500_on_runtime_error` —
    any unexpected `RuntimeError` from the service → 503, not
    500.
  - `test_predict_endpoint_response_includes_model_version` —
    the body has `model_version == "v2"` (or the override).
  - `test_predict_endpoint_response_includes_shap_contributions` —
    the body has a non-empty `shap_contributions` list.
  - `test_predict_endpoint_does_not_log_contact_fields` — captures
    the inserted `prediction_log` row + greps for the PII regex.

- `tests/test_predict_log_entry.py` — pytest tests for the
  DB-row serialiser. Required tests (exact names):
  - `test_to_prediction_log_row_keys_match_db_columns` — the
    returned dict's keys are exactly the `prediction_log` columns
    (no extras, no missing).
  - `test_to_prediction_log_row_serialises_features_as_json` —
    the `input_features_json` value is a JSON string parsable by
    `json.loads`.
  - `test_to_prediction_log_row_excludes_luxury_category_from_input`
    — the dumped input JSON does **not** contain the
    `luxury_category` key (the client-supplied value was dropped
    on parse; this serialisation is the standard schema dump).
  - `test_to_prediction_log_row_latency_ms_is_int` — the
    `latency_ms` field is a Python `int`.

**Modify:**
- `api/routers/predict.py` — replace the Step-01 stub with a real
  route. Public API:
  - `router = APIRouter()` (unchanged).
  - `@router.post("/predict", response_model=PredictResponseV3,
    status_code=200)` —
    `def predict(request: PredictRequestV3) -> PredictResponseV3:`
    body:
    1. `t0 = time.perf_counter()`.
    2. `service = get_predict_service()` (a process-global helper
       defined in the same module below).
    3. `response = service.predict(request)` inside
       `try / except FileNotFoundError as e: raise HTTPException(503, f"model artifact missing: {e}")`.
    4. `latency_ms = int((time.perf_counter() - t0) * 1000)`.
    5. `log_prediction(request, response, latency_ms)` — try
       around the DB write; a DB failure logs WARNING but does
       **not** fail the request (the prediction is more valuable
       than the log; this matches Rules §5.2's "graceful
       degradation" spirit).
    6. `return response`.
  - `def get_predict_service() -> PredictService:` — module-level
    lazy singleton. First call instantiates the service; later
    calls return the same instance. Pinned by a test.
  - `def log_prediction(request: PredictRequestV3, response:
    PredictResponseV3, latency_ms: int) -> None:` — calls
    `to_prediction_log_row(...)` and writes via `get_db()` with a
    parameterized SQL insert. Idempotent re: schema (reuses
    `init_db()` once at startup).
  - Module docstring cites Spec 11 (schemas) + Spec 14 (model) +
    Spec 16 (explainer) + Rules §2.4 / §2.6 / §5.1 / §5.2.

- `api/main.py` — replace the `@app.on_event("startup")` handler
  with a FastAPI **`lifespan`** context manager (the
  `fastapi-serving` skill's preferred pattern; `@on_event` is
  deprecated in modern FastAPI). The lifespan:
  1. `init_db()` (unchanged).
  2. `get_predict_service().warmup()` — loads all
     `transact_type` pipelines + explainers at startup per the
     `shap-explainability` skill's "load precomputed explainer at
     startup" rule.
  Drop the old `@app.on_event("startup")` decorator. The
  `@app.get("/health")` handler stays unchanged.

- `api/__init__.py` — no change (already empty).

- `requirements.txt` — no new packages. `fastapi`, `pydantic`,
  `joblib`, `numpy`, `pandas`, `scikit-learn`, `shap` are already
  pinned by Steps 01/13/16.

**No changes** to:
- `app/`, `data/`, `models/`, `migrations/`, `notebooks/`,
  `tests/conftest.py` (existing fixtures remain; new fixtures live
  in the new test files).
- `scripts/run_pipeline.py` — the offline pipeline does not
  import from `api/`; this spec changes only the serving layer.
- `CLAUDE.md`'s "Implemented vs stub routes" table — the
  `POST /predict` route moves from **Stub** to **Implemented**.
  All other rows stay unchanged. The Flask `POST /predict` row
  stays **Stub** (that's a follow-on spec).

## New dependencies
**No new dependencies.** `fastapi`, `pydantic`, `joblib`, `numpy`,
`pandas`, `scikit-learn`, `shap`, `threading` (stdlib) are all
already pinned or stdlib. No `pip install` required.

## Rules for implementation

- **No SQLAlchemy/ORM.** All DB access is parameterized via
  `conn.execute("INSERT INTO prediction_log (...) VALUES (?, ...,
  ...)", (...))` — never f-strings into SQL. Pinned by
  `test_predict_endpoint_does_not_log_contact_fields` + the
  existing `app/database/db.py` style.
- **No dealer/contact/media-URL fields ever reach the UI or an
  export.** The `predict_service` I/O path does not pass any column
  matching the regex `(contact|dealer|phone|email|photo|url|spid)`
  anywhere serialised. Pinned by `test_predict_service_predict_does_not_log_pii_fields`
  + `test_predict_endpoint_does_not_log_contact_fields`.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.**
  The route loads the v2 pipeline from `models/price_model_*.pkl`
  — the same artifact Spec 15's gate certified. No retraining, no
  hyperparameter tweaks, no preprocessing re-implementation.
- **Same model instance for prediction + SHAP (Rules §2.6).**
  `predict_service.predict` passes the exact loaded `Pipeline` to
  both `model.predict(X)` and `ml.explainability.explain_one(...,
  model, ...)`. No proxy model, no simplified surrogate.
- **Preprocessor loaded, not refit (Rules §2.4).** The service
  loads `models/feature_pipeline_v1.pkl` once and reuses it. No
  `ColumnTransformer(...)` construction in the API layer.
- **`transact_type` is a routing key (Rules §10.3, TRD §U-TRD-4).**
  The route resolves `service.predict(...)` against the cached
  `(transact_type, model_version)` lookup *before* any
  preprocessing. Mixing Sale + Rent in a single pipeline is a hard
  rule violation; the cache key prevents it.
- **Precomputed explainer at startup (skill).** The lifespan
  handler calls `service.warmup()` which loads every
  `shap_explainer_*.pkl` exactly once. The route never calls
  `shap.TreeExplainer(model)` inside the request handler.
- **Versioned artifacts, never overwritten (Rules §2.5).** The
  service's `model_version` is pinned at constructor time; flipping
  `v2` → `v3` is a single env var change, not a code change.
- **Graceful degradation on DB failure (Rules §5.2).** A failed
  `prediction_log` write is logged WARNING but does not fail the
  HTTP response. The prediction is the user-facing value; the log
  is internal telemetry.
- **503, not 500, on model-artifact missing.** Missing models at
  startup are a deployment bug, not a user error → `503 Service
  Unavailable` with a clear message. The Flask caller will surface
  "predictions temporarily unavailable" per Rules §5.2.
- **No `app/` imports inside `api/services/` or `api/routers/`.**
  `api/services/predict_service.py` imports from `app.database.db`
  *only* for the DB write path (this is the documented exception
  — the FastAPI service legitimately writes to the shared SQLite
  file). It does **not** import Flask, Jinja, or anything from
  `app/templates/`. Pinned by a test that introspects
  `sys.modules` for `app.*` at service import time and asserts
  no Flask-specific symbols are imported.
- **No `ml/training/` imports.** The service imports from
  `ml.explainability` (the per-prediction helper, Spec 16) and
  `ml.features.feature_frame` (for the pinned top-15 amenity set,
  Spec 12). It does **not** import from `ml/training/*` —
  training code is offline-only.
- **Logging uses stdlib `logging` only.** One module-level logger
  per file (`logger = logging.getLogger(__name__)`). INFO for
  startup warmup + per-request start/finish; WARNING for DB log
  failure + label-map fallthrough; ERROR for hard failures
  (artifact missing, preprocessor missing).
- **All randomness is seeded.** N/A — no randomness in the hot
  path. The preprocessor / model / SHAP outputs are deterministic
  given the same input.
- **Per-request latency is measured via `time.perf_counter()`**
  (the stdlib monotonic clock per FastAPI's docs). `latency_ms`
  is `int(...)` truncated; the column is `INTEGER` in SQLite.
- **No notebook-only steps.** Everything is reproducible via
  `python -m uvicorn api.main:app --reload` from repo root. No
  Jupyter cell trains, loads, or computes a SHAP value the script
  can't reproduce.

## Definition of done

1. `python -m pytest tests/test_predict_service.py
   tests/test_predict_endpoint.py tests/test_predict_log_entry.py
   -v` from repo root runs and passes. Tests required (exact
   names):
   - **Service** (`test_predict_service.py`):
     - `test_predict_service_warmup_loads_sale_pipeline`
     - `test_predict_service_warmup_loads_rent_pipeline_when_present`
     - `test_predict_service_warmup_skips_rent_when_artifact_missing`
     - `test_predict_service_warmup_is_idempotent`
     - `test_predict_service_predict_returns_response_v3`
     - `test_predict_service_predict_routes_by_transact_type`
     - `test_predict_service_predict_attaches_shap_contributions`
     - `test_predict_service_predict_excludes_outlier_input_flag`
     - `test_predict_service_predict_resolves_luxury_category`
     - `test_predict_service_predict_uses_loaded_preprocessor`
     - `test_predict_service_predict_uses_loaded_explainer`
     - `test_predict_service_predict_logs_to_prediction_log`
     - `test_predict_service_predict_does_not_log_pii_fields`
     - `test_predict_service_predict_latency_ms_is_positive_int`
   - **Endpoint** (`test_predict_endpoint.py`):
     - `test_predict_endpoint_returns_200_for_valid_payload`
     - `test_predict_endpoint_returns_422_on_missing_field`
     - `test_predict_endpoint_returns_422_on_extra_field`
     - `test_predict_endpoint_returns_422_on_bedroom_bathroom_violation`
     - `test_predict_endpoint_returns_422_on_area_over_20000`
     - `test_predict_endpoint_returns_503_when_model_artifact_missing`
     - `test_predict_endpoint_returns_503_no_500_on_runtime_error`
     - `test_predict_endpoint_response_includes_model_version`
     - `test_predict_endpoint_response_includes_shap_contributions`
     - `test_predict_endpoint_does_not_log_contact_fields`
   - **Log entry** (`test_predict_log_entry.py`):
     - `test_to_prediction_log_row_keys_match_db_columns`
     - `test_to_prediction_log_row_serialises_features_as_json`
     - `test_to_prediction_log_row_excludes_luxury_category_from_input`
     - `test_to_prediction_log_row_latency_ms_is_int`
2. `python -m pytest -m "not realdata"` from repo root still
   passes — no real-data dependency introduced (the service tests
   use a tiny synthetic `clean_listings.parquet` + a fitted
   XGBoost pipeline written to `tmp_path`, then monkeypatch the
   service cache).
3. `ruff check api/routers/predict.py api/services/predict_service.py
   api/schemas/predict_log_entry.py api/main.py
   tests/test_predict_service.py tests/test_predict_endpoint.py
   tests/test_predict_log_entry.py` reports zero issues.
4. `python -c "from api.services.predict_service import
   PredictService; from api.routers.predict import
   get_predict_service; print('ok')"` from repo root prints
   `ok` without error — public API imports cleanly.
5. `python -m uvicorn api.main:app --port 8000` from repo root
   starts cleanly (FastAPI lifespan runs `init_db()` +
   `service.warmup()` without error against the artifacts Spec
   13/14/16 produced). Manual smoke test of the startup path.
6. With the server running from step 5, a `curl -X POST
   http://localhost:8000/predict -H "Content-Type:
   application/json" -d '{"city": "Gurgaon", "sector": "sector
   84", "property_type": "flat", "transact_type": "Sale",
   "bedRoom": 3, "bathroom": 3, "balcony": "2",
   "agePossession": "Relatively New", "built_up_area": 1450,
   "servant_room": true, "store_room": false,
   "furnishing_type": "Semifurnished", "floor_category": "Mid
   Floor", "facing": "North", "amenities": ["Clubhouse",
   "Swimming Pool"]}'` returns `200` with a JSON body containing
   `predicted_price`, `range_low`, `range_high`, a non-empty
   `shap_contributions` array, `is_outlier_input: false`, and
   `model_version: "v2"`. Then a `SELECT * FROM prediction_log
   ORDER BY id DESC LIMIT 1;` against `data/app.db` returns one
   row with the request's `input_features_json` parsed back
   cleanly and a `latency_ms` integer. Manual smoke test of the
   full route + DB write path.
7. After step 6, the `data/app.db` `prediction_log` row's
   `input_features_json` (when parsed back) does **not** contain
   any key matching `(contact|dealer|phone|email|photo|url|spid)`
   — confirmed manually by the smoke test reading the row.
8. `git status` after committing shows only the new files listed
   above, the modified `api/routers/predict.py`, the modified
   `api/main.py`, and the modified `api/services/__init__.py`
   (re-exports `PredictService`). No accidental additions to
   `data/`, `models/`, `app/`, `migrations/`, or
   `requirements.txt`.
9. `CLAUDE.md`'s "Implemented vs stub routes" table is updated
   to flip the `POST /predict` (FastAPI) row from **Stub** to
   **Implemented**. The Flask `POST /predict` row stays **Stub**
   (that's a follow-on spec). `GET /health` stays **Implemented**.
10. `07-TRACKER.md` is updated via `/update-tracker` to mark Day
    27 ("FastAPI /predict route + schemas + smoke test") as
    **Done** with the actual date and a one-line summary of the
    v2 model + SHAP wiring. The Decision Log gets one new entry:
    "Replaced `@app.on_event('startup')` with FastAPI lifespan
    context manager per the `fastapi-serving` skill — the
    `@on_event` decorator is deprecated in current FastAPI."
