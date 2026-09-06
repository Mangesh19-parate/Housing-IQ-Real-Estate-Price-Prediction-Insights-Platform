# Spec: Classification FastAPI Endpoint

## Overview
Wire the two trained classification models from Spec 23 (`good_deal_classifier_v{n}.pkl` and `price_tier_classifier_v{n}.pkl`) + the shared fitted preprocessor (`classifier_preprocessor_v{n}.pkl`) into the FastAPI `POST /classify` route. The route uses the frozen 16-field input contract from Spec 11 (same as `/predict`), applies the leakage-safe preprocessor, runs both classifiers, and returns the primary `good_deal_verdict` + secondary `price_tier` with probabilities and SHAP-style contributions. A row is appended to `classification_log` with per-request latency.

Module: **classification**.

This is the serving layer for the Classification module — it consumes artifacts from Specs 21/22/23/24 and the schemas from Spec 11. Per Rules §2.4 the route loads the exact persisted `Pipeline` instances — no re-implemented preprocessing. Per Rules §5.1 Flask never loads models; FastAPI is the sole inference surface.

## Depends on
- **Spec 11** (`11-price-prediction-input-schema-v3.md`) — provides `PredictRequestV3` (the 16-field contract, reused here). The classification request is identical in shape.
- **Spec 21** (`21-classification-target-definition.md`) — label definitions: `GOOD_DEAL_LABELS` ("Good Deal", "Fair Deal", "Overpriced"), `PRICE_TIER_LABELS` ("Budget", "Mid-Range", "Premium", "Luxury").
- **Spec 22** (`22-classification-feature-reuse-from-price-schema.md`) — leakage-safe feature frame and `DROPPED_FOR_LEAKAGE` list.
- **Spec 23** (`23-classification-model-training.md`) — produces:
  - `models/good_deal_classifier_v{n}.pkl` (primary, 3-class)
  - `models/price_tier_classifier_v{n}.pkl` (secondary, 4-class)
  - `models/classifier_preprocessor_v{n}.pkl` (shared fitted `ColumnTransformer`)
- **Spec 24** (`24-classification-model-evaluation.md`) — gate certifies the artifacts; only certified versions should be served.
- **`api/schemas/predict_v3.py`** — reuses `PredictRequestV3`, `INPUT_FIELDS_V3`, enums.
- **`api/config.py`** — `MODELS_DIR`, `APP_DB_PATH` env vars.
- **`app/database/db.py`** — `get_db()` for `classification_log` writes.
- **`docs/05-BACKEND-SCHEMA.md` §U-SCHEMA-10** — `POST /classify` response contract (verdict leads).
- **`docs/02-TRD.md` §U-TRD-8** — classification targets + selection metrics.
- **`docs/08-RULES.md` §2.4** — model in production is the exact same `Pipeline` used at evaluation; no re-implemented preprocessing.
- **`docs/08-RULES.md` §5.1, §5.2** — Flask never loads models; FastAPI degrades gracefully.
- **`docs/08-RULES.md` §1.3, §2.5** — versioned artifacts never overwritten; every `classification_log` row is dated.
- **`fastapi-serving` skill** — FastAPI lifespan vs on_event, lazy model load, `HTTPException` usage.

## Routes / Endpoints
- **FastAPI:** `POST /classify` — classification (affordability + good-deal verdict). Accepts a `PredictRequestV3` body (same 16 fields as `/predict`), returns a `ClassifyResponseV1`. Loads both classifier pipelines + the shared preprocessor at FastAPI startup (lifespan), keeps a single in-memory cache keyed by `model_version`; no I/O on the hot path. Access: internal (Flask → FastAPI over HTTP).
- **Flask:** none added. The Flask `/classify` page wiring is a follow-on spec.

## Data / Schema changes
- **No new application DB tables.** `classification_log` already exists (Backend Schema §U-SCHEMA-11) with columns: `id`, `timestamp`, `city`, `input_features_json`, `predicted_verdict`, `predicted_tier`, `verdict_probabilities_json`, `tier_probabilities_json`, `model_version`.
- **No new model artifacts.** This spec consumes the v1 classifiers + preprocessor from Spec 23.
- **No `data/raw/` or `data/processed/` writes.**
- **No new SQL migrations.**
- **No `analytics_cache/*.json` additions.**

## Templates / UI
None. This spec is the FastAPI serving layer. The Flask `/classify` form + result page are follow-on specs.

## Files to change / Files to create

**Create:**
- `api/schemas/classify_v1.py` — Pydantic request/response models for `/classify`. Public API:
  - Re-exports `PredictRequestV3` from `predict_v3` as `ClassifyRequestV1` (alias, same shape).
  - `VerdictLabel(str, Enum)` — "Good Deal", "Fair Deal", "Overpriced" (pinned from Spec 21).
  - `TierLabel(str, Enum)` — "Budget", "Mid-Range", "Premium", "Luxury" (pinned from Spec 21).
  - `ClassifyResponseV1` — response model matching Backend Schema §U-SCHEMA-10:
    ```python
    class ClassifyResponseV1(BaseModel):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())
        good_deal_verdict: VerdictLabel
        verdict_probabilities: dict[VerdictLabel, float]  # sums to 1.0
        price_tier: TierLabel
        tier_probabilities: dict[TierLabel, float]  # sums to 1.0
        shap_contributions: list[ShapContribution] = Field(default_factory=list)  # reuse from predict_v3
        model_version: str
        luxury_category: LuxuryCategory  # server-derived, echoed back
    ```
  - `SHAP_TOP_N: int = 7` — pinned constant (mirrors price model's top-N rule).
  - Constants: `PRIMARY_TARGET = "good_deal_verdict"`, `SECONDARY_TARGET = "price_tier"`.

- `api/services/classify_service.py` — the inference service. Public API:
  - `class ClassifyService:` — lazily-loaded model + preprocessor cache, keyed by `model_version`. Uses a `threading.Lock` per key for first-load race (same pattern as `PredictService`).
    - `def __init__(self, models_dir: Path, *, model_version: str = "v1") -> None:` — does **not** load anything; loads happen on first `classify()` call (or `warmup()`).
    - `def warmup(self) -> None:` — loads both pipelines + shared preprocessor. Called from FastAPI's lifespan handler. Idempotent. If either artifact is missing, logs WARNING and leaves that key unset.
    - `def classify(self, request: PredictRequestV3) -> ClassifyResponseV1:` — the single hot path. Steps:
      1. Resolve `model_version` → load pipelines + preprocessor if not cached.
      2. Resolve `luxury_category` from the `amenities` list (server-derived, never client-supplied — Rules §10.2). Reuses the same helper `_resolve_luxury_category` from `predict_service.py` (or a shared util).
      3. Build a 1-row DataFrame in the same column order the preprocessor expects (pinned feature order from training). The `amenities` list is converted to `n_amenities` + `has_<amenity>` flags via the pinned top-15 amenity set from `ml/features/feature_frame.py`. Missing preprocessor columns are filled with `NaN`.
      4. Apply the fitted preprocessor (loaded, not refit — Rules §2.4): `X = preprocessor.transform(df)`.
      5. Run both classifiers: `good_deal_pipe.predict(X)` → class index (0/1/2), `good_deal_pipe.predict_proba(X)` → probabilities; `tier_pipe.predict(X)` → class index (0/1/2/3), `tier_pipe.predict_proba(X)` → probabilities.
      6. Map indices to labels using `GOOD_DEAL_LABELS` / `PRICE_TIER_LABELS`.
      7. Compute SHAP-style contributions: reuse `ml.explainability.explain_one` with the **primary** classifier (good_deal) since it's the primary output. Map to `list[ShapContribution]`.
      8. Build `ClassifyResponseV1(...)` and return.
    - `MODEL_VERSION: str = "v1"` — pinned module constant; overridable via constructor arg (used by tests).

- `api/schemas/classify_log_entry.py` — the DB-row serialiser (kept separate from `classify_v1.py` so the schema module stays I/O-free). Public API:
  - `def to_classification_log_row(request: PredictRequestV3, response: ClassifyResponseV1, latency_ms: int) -> dict[str, Any]:` — returns a dict whose keys match the `classification_log` column names verbatim (Backend Schema §U-SCHEMA-11). Pure function, no I/O. Pinned by tests.

- `tests/test_classify_service.py` — pytest tests for the service layer. Required tests (exact names):
  - `test_classify_service_warmup_loads_both_pipelines`
  - `test_classify_service_warmup_is_idempotent`
  - `test_classify_service_classify_returns_response_v1`
  - `test_classify_service_classify_routes_to_both_classifiers` — verifies both `good_deal_verdict` and `price_tier` are present in response.
  - `test_classify_service_classify_attaches_shap_contributions` — hot-path SHAP returns a `list[ShapContribution]` capped at `SHAP_TOP_N`.
  - `test_classify_service_classify_resolves_luxury_category` — `amenities=[]` → `LOW`; `["Clubhouse"]` → `MEDIUM`; ≥ 5 amenities → `HIGH` (pinned lookup).
  - `test_classify_service_classify_uses_loaded_preprocessor` — pinned: monkeypatch the preprocessor's `transform` and assert the service called it (proves Rules §2.4 — no re-implementation).
  - `test_classify_service_classify_logs_to_classification_log` — uses an in-memory SQLite DB (`tmp_path`) + the existing `app.database.db.init_db` to create the tables, runs `classify()`, asserts exactly one row inserted into `classification_log` with the expected columns.
  - `test_classify_service_classify_does_not_log_pii_fields` — greps the captured row's `input_features_json` for the regex `(contact|dealer|phone|email|photo|url|spid)` — must be absent.
  - `test_classify_service_classify_latency_ms_is_positive_int` — assert the logged `latency_ms` is a non-negative int.

- `tests/test_classify_endpoint.py` — pytest tests for the FastAPI route via `TestClient`. Required tests (exact names):
  - `test_classify_endpoint_returns_200_for_valid_payload` — full minimal payload with a 3BHK Gurgaon flat → 200 + `ClassifyResponseV1`-shaped body.
  - `test_classify_endpoint_returns_422_on_missing_field` — omit `built_up_area` → 422 (Pydantic, not a 500).
  - `test_classify_endpoint_returns_422_on_extra_field` — send `{"bedrooms": 3}` (typo) → 422 (confirms `extra="forbid"`).
  - `test_classify_endpoint_returns_422_on_bedroom_bathroom_violation`
  - `test_classify_endpoint_returns_422_on_area_over_20000`
  - `test_classify_endpoint_returns_503_when_model_artifact_missing` — patch the service to raise `FileNotFoundError`; route must translate to `HTTPException(503, "model not loaded")`, not a 500.
  - `test_classify_endpoint_returns_503_no_500_on_runtime_error` — any unexpected `RuntimeError` from the service → 503, not 500.
  - `test_classify_endpoint_response_includes_model_version` — the body has `model_version == "v1"` (or the override).
  - `test_classify_endpoint_response_includes_shap_contributions` — the body has a non-empty `shap_contributions` list.
  - `test_classify_endpoint_response_verdict_leads` — `good_deal_verdict` is the first field in the response JSON (Backend Schema §U-SCHEMA-10).
  - `test_classify_endpoint_does_not_log_contact_fields` — captures the inserted `classification_log` row + greps for the PII regex.

- `tests/test_classify_log_entry.py` — pytest tests for the DB-row serialiser. Required tests (exact names):
  - `test_to_classification_log_row_keys_match_db_columns` — the returned dict's keys are exactly the `classification_log` columns (no extras, no missing).
  - `test_to_classification_log_row_serialises_features_as_json` — the `input_features_json` value is a JSON string parsable by `json.loads`.
  - `test_to_classification_log_row_excludes_luxury_category_from_input` — the dumped input JSON does **not** contain the `luxury_category` key (the client-supplied value was dropped on parse).
  - `test_to_classification_log_row_latency_ms_is_int` — the `latency_ms` field is a Python `int`.

**Modify:**
- `api/routers/classify.py` — replace the Step-01 stub with a real route. Public API:
  - `router = APIRouter()` (unchanged).
  - `@router.post("/classify", response_model=ClassifyResponseV1, status_code=200)` —
    `def classify(request: PredictRequestV3) -> ClassifyResponseV1:` body:
    1. `t0 = time.perf_counter()`.
    2. `service = get_classify_service()` (a process-global helper defined in the same module below).
    3. `response = service.classify(request)` inside `try / except FileNotFoundError as e: raise HTTPException(503, f"model artifact missing: {e}")`.
    4. `latency_ms = int((time.perf_counter() - t0) * 1000)`.
    5. `log_classification(request, response, latency_ms)` — try around the DB write; a DB failure logs WARNING but does **not** fail the request (the classification is the user-facing value; the log is internal telemetry; matches Rules §5.2).
    6. `return response`.
  - `def get_classify_service() -> ClassifyService:` — module-level lazy singleton. First call instantiates the service; later calls return the same instance. Pinned by a test.
  - `def set_classify_service(service: ClassifyService) -> None:` — override the singleton with a lifespan-resolved instance (mirrors `set_predict_service`). Used by lifespan if registry has an active row.
  - `def log_classification(request: PredictRequestV3, response: ClassifyResponseV1, latency_ms: int) -> None:` — calls `to_classification_log_row(...)` and writes via `get_db()` with a parameterized SQL insert into `classification_log`. Idempotent re: schema (reuses `init_db()` once at startup).
  - Module docstring cites Spec 11 (schemas) + Spec 23 (models) + Spec 24 (gate) + Rules §2.4 / §5.1 / §5.2.

- `api/main.py` — update the lifespan handler to also warm up the classify service:
  1. Keep existing `init_db()` + `get_predict_service().warmup()`.
  2. Add `get_classify_service().warmup()` — loads both classifier pipelines + shared preprocessor at startup.
  3. (Optional, Spec 20 parity) Add `_resolve_active_classify_service()` mirroring `_resolve_active_service()` to read the active `good_deal_classifier` row from `model_registry` and inject a versioned service. If not implemented now, add a TODO comment — the default `MODEL_VERSION = "v1"` fallback covers v1.
  4. Keep the `@app.get("/health")` handler unchanged (it only reports price model version per Spec 20).

- `api/services/__init__.py` — re-export `ClassifyService` and any constants:
  ```python
  from api.services.classify_service import ClassifyService, MODEL_VERSION as CLASSIFY_MODEL_VERSION, SHAP_TOP_N
  __all__ = [..., "ClassifyService", "CLASSIFY_MODEL_VERSION", "SHAP_TOP_N"]
  ```

- `api/__init__.py` — no change (already empty).

- `requirements.txt` — no new packages. `fastapi`, `pydantic`, `joblib`, `numpy`, `pandas`, `scikit-learn`, `shap` are already pinned by Steps 01/13/16/17/23.

**No changes** to:
- `app/`, `data/`, `models/`, `migrations/`, `notebooks/`, `tests/conftest.py` (existing fixtures remain; new fixtures live in the new test files).
- `scripts/run_pipeline.py` — the offline pipeline does not import from `api/`; this spec changes only the serving layer.
- `CLAUDE.md`'s "Implemented vs stub routes" table — the `POST /classify` (FastAPI) row moves from **Stub** to **Implemented**. All other rows stay unchanged. The Flask `POST /classify` row stays **Stub** (that's a follow-on spec).

## New dependencies
**No new dependencies.** `fastapi`, `pydantic`, `joblib`, `numpy`, `pandas`, `scikit-learn`, `shap`, `threading` (stdlib) are all already pinned or stdlib. No `pip install` required.

## Rules for implementation

- **No SQLAlchemy/ORM.** All DB access is parameterized via `conn.execute("INSERT INTO classification_log (...) VALUES (?, ..., ...)", (...))` — never f-strings into SQL. Pinned by `test_classify_endpoint_does_not_log_contact_fields` + the existing `app/database/db.py` style.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** The `classify_service` I/O path does not pass any column matching the regex `(contact|dealer|phone|email|photo|url|spid)` anywhere serialised. Pinned by `test_classify_service_classify_does_not_log_pii_fields` + `test_classify_endpoint_does_not_log_contact_fields`.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** The route loads the v1 pipelines from `models/good_deal_classifier_v1.pkl` / `models/price_tier_classifier_v1.pkl` — the same artifacts Spec 24's gate certified. No retraining, no hyperparameter tweaks, no preprocessing re-implementation.
- **Same model instance for prediction + SHAP (Rules §2.6).** `classify_service.classify` passes the exact loaded `Pipeline` to both `model.predict(X)` / `model.predict_proba(X)` and `ml.explainability.explain_one(..., model, ...)`. No proxy model, no simplified surrogate.
- **Preprocessor loaded, not refit (Rules §2.4).** The service loads `models/classifier_preprocessor_v1.pkl` once and reuses it. No `ColumnTransformer(...)` construction in the API layer.
- **Precomputed explainer at startup (skill).** The lifespan handler calls `service.warmup()` which loads every classifier artifact exactly once. The route never calls `shap.TreeExplainer(model)` inside the request handler — SHAP is computed via the shared `ml.explainability.explain_one` helper which accepts the loaded model.
- **Versioned artifacts, never overwritten (Rules §2.5).** The service's `model_version` is pinned at constructor time; flipping `v1` → `v2` is a single env var change, not a code change.
- **Graceful degradation on DB failure (Rules §5.2).** A failed `classification_log` write is logged WARNING but does not fail the HTTP response. The classification is the user-facing value; the log is internal telemetry.
- **503, not 500, on model-artifact missing.** Missing models at startup are a deployment bug, not a user error → `503 Service Unavailable` with a clear message. The Flask caller will surface "classification temporarily unavailable" per Rules §5.2.
- **No `app/` imports inside `api/services/` or `api/routers/`.** `api/services/classify_service.py` imports from `app.database.db` *only* for the DB write path (this is the documented exception — the FastAPI service legitimately writes to the shared SQLite file). It does **not** import Flask, Jinja, or anything from `app/templates/`. Pinned by a test that introspects `sys.modules` for `app.*` at service import time and asserts no Flask-specific symbols are imported.
- **No `ml/training/` imports.** The service imports from `ml.explainability` (the per-prediction helper, Spec 16) and `ml.features.feature_frame` (for the pinned top-15 amenity set, Spec 12). It does **not** import from `ml/training/*` — training code is offline-only.
- **Logging uses stdlib `logging` only.** One module-level logger per file (`logger = logging.getLogger(__name__)`). INFO for startup warmup + per-request start/finish; WARNING for DB log failure + label-map fallthrough; ERROR for hard failures (artifact missing, preprocessor missing).
- **All randomness is seeded.** N/A — no randomness in the hot path. The preprocessor / model / SHAP outputs are deterministic given the same input.
- **Per-request latency is measured via `time.perf_counter()`** (the stdlib monotonic clock per FastAPI's docs). `latency_ms` is `int(...)` truncated; the column is `INTEGER` in SQLite.
- **No notebook-only steps.** Everything is reproducible via `python -m uvicorn api.main:app --reload` from repo root. No Jupyter cell trains, loads, or computes a SHAP value the script can't reproduce.
- **Shared helper for luxury_category resolution.** The `_resolve_luxury_category(amenities: list[str]) -> LuxuryCategory` logic is identical to `predict_service.py`. Either factor it into a shared `api/services/utils.py` or duplicate the 5-line lookup table (ponytail: duplicate for now; factor when a third consumer appears).

## Definition of done

1. `python -m pytest tests/test_classify_service.py tests/test_classify_endpoint.py tests/test_classify_log_entry.py -v` from repo root runs and passes. Tests required (exact names):
   - **Service** (`test_classify_service.py`):
     - `test_classify_service_warmup_loads_both_pipelines`
     - `test_classify_service_warmup_is_idempotent`
     - `test_classify_service_classify_returns_response_v1`
     - `test_classify_service_classify_routes_to_both_classifiers`
     - `test_classify_service_classify_attaches_shap_contributions`
     - `test_classify_service_classify_resolves_luxury_category`
     - `test_classify_service_classify_uses_loaded_preprocessor`
     - `test_classify_service_classify_logs_to_classification_log`
     - `test_classify_service_classify_does_not_log_pii_fields`
     - `test_classify_service_classify_latency_ms_is_positive_int`
   - **Endpoint** (`test_classify_endpoint.py`):
     - `test_classify_endpoint_returns_200_for_valid_payload`
     - `test_classify_endpoint_returns_422_on_missing_field`
     - `test_classify_endpoint_returns_422_on_extra_field`
     - `test_classify_endpoint_returns_422_on_bedroom_bathroom_violation`
     - `test_classify_endpoint_returns_422_on_area_over_20000`
     - `test_classify_endpoint_returns_503_when_model_artifact_missing`
     - `test_classify_endpoint_returns_503_no_500_on_runtime_error`
     - `test_classify_endpoint_response_includes_model_version`
     - `test_classify_endpoint_response_includes_shap_contributions`
     - `test_classify_endpoint_response_verdict_leads`
     - `test_classify_endpoint_does_not_log_contact_fields`
   - **Log entry** (`test_classify_log_entry.py`):
     - `test_to_classification_log_row_keys_match_db_columns`
     - `test_to_classification_log_row_serialises_features_as_json`
     - `test_to_classification_log_row_excludes_luxury_category_from_input`
     - `test_to_classification_log_row_latency_ms_is_int`
2. `python -m pytest -m "not realdata"` from repo root still passes — no real-data dependency introduced (the service tests use a tiny synthetic `clean_listings.parquet` + fitted classifier pipelines written to `tmp_path`, then monkeypatch the service cache).
3. `ruff check api/routers/classify.py api/services/classify_service.py api/schemas/classify_v1.py api/schemas/classify_log_entry.py api/main.py api/services/__init__.py tests/test_classify_service.py tests/test_classify_endpoint.py tests/test_classify_log_entry.py` reports zero issues.
4. `python -c "from api.services.classify_service import ClassifyService; from api.routers.classify import get_classify_service; from api.schemas.classify_v1 import ClassifyResponseV1, VerdictLabel, TierLabel; print('ok')"` from repo root prints `ok` without error — public API imports cleanly.
5. `python -m uvicorn api.main:app --port 8000` from repo root starts cleanly (FastAPI lifespan runs `init_db()` + `get_predict_service().warmup()` + `get_classify_service().warmup()` without error against the artifacts Spec 23 produced). Manual smoke test of the startup path.
6. With the server running from step 5, a `curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"city": "Gurgaon", "sector": "sector 84", "property_type": "flat", "transact_type": "Sale", "bedRoom": 3, "bathroom": 3, "balcony": "2", "agePossession": "Relatively New", "built_up_area": 1450, "servant_room": true, "store_room": false, "furnishing_type": 1, "floor_category": "Mid Floor", "facing": "North", "amenities": ["Clubhouse", "Swimming Pool"]}'` returns `200` with a JSON body containing `good_deal_verdict` (one of "Good Deal"/"Fair Deal"/"Overpriced"), `verdict_probabilities` (dict summing to 1.0), `price_tier` (one of "Budget"/"Mid-Range"/"Premium"/"Luxury"), `tier_probabilities` (dict summing to 1.0), a non-empty `shap_contributions` array, `model_version: "v1"`, and `luxury_category`. Then a `SELECT * FROM classification_log ORDER BY id DESC LIMIT 1;` against `data/app.db` returns one row with the request's `input_features_json` parsed back cleanly and a `latency_ms` integer. Manual smoke test of the full route + DB write path.
7. After step 6, the `data/app.db` `classification_log` row's `input_features_json` (when parsed back) does **not** contain any key matching `(contact|dealer|phone|email|photo|url|spid)` — confirmed manually by the smoke test reading the row.
8. `git status` after committing shows only the new files listed above, the modified `api/routers/classify.py`, the modified `api/main.py`, and the modified `api/services/__init__.py`. No accidental additions to `data/`, `models/`, `app/`, `migrations/`, or `requirements.txt`.
9. `CLAUDE.md`'s "Implemented vs stub routes" table is updated to flip the `POST /classify` (FastAPI) row from **Stub** to **Implemented**. The Flask `POST /classify` row stays **Stub** (that's a follow-on spec). `GET /health` stays **Implemented**.
10. `07-TRACKER.md` is updated via `/update-tracker` to mark Day 54 ("FastAPI /classify route + schemas + smoke test") as **Done** with the actual date and a one-line summary of the v1 classifier + preprocessor wiring. The Decision Log gets one new entry: "Reused `PredictRequestV3` as classification input contract (same 16 fields) — no separate schema needed. Classification response leads with `good_deal_verdict` per Backend Schema §U-SCHEMA-10 (good-deal-first reframing)."