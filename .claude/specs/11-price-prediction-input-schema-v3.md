# Spec: Price Prediction Input Schema v3 (Pydantic Contracts)

## Overview
Codify the finalized 16-field input contract from `10-FINALIZED-INPUT-SCHEMA.md` §3 + `05-BACKEND-SCHEMA.md` §U-SCHEMA-5 as **versioned Pydantic request/response models** in `api/schemas/`, plus a single source of truth `INPUT_FIELDS_V3` constant (column order + types) that the inference pipeline, training script, and tests all import. This is the bridge between the canonical-cleaned DataFrame columns emitted by Step 05 (`ml/cleaning/canonical_mapping.py`) and the FastAPI `/predict` route — the layer that makes "the 16-field input contract" machine-checkable instead of just documented. Module: **price-prediction**.

Step 11 is a deliberate jump ahead of the planned sequence (the foundation chain runs Steps 06–10: dedup, Parquet pipeline, log-transform, feature engineering, classification tier derivation). Reason: the input contract is frozen and reference-project-locked, so pinning it as code now — while Steps 06–10 are still landing — prevents the training scripts and the FastAPI route from drifting on field names/typing. Step 11 only defines schemas; it does **not** wire them into `/predict` (that's a later spec, after the model pipeline lands).

## Depends on
- **Step 05** — `05-canonical-schema-mapping-per-city` (`ml/cleaning/canonical_mapping.py`) — defines `CANONICAL_COLUMNS` (28-column extended schema) and the 16 input-contract fields with their canonical reference-project names (`bedRoom`, `built_up_area`, `agePossession`, etc.).
- **Step 01** — `01-repo-scaffolding-and-environment-setup` — FastAPI + Pydantic v2 are pinned in `requirements.txt`; the project's Pydantic version is the v2 line (no v1-only APIs allowed).
- **`10-FINALIZED-INPUT-SCHEMA.md`** — authoritative field list, allowed values, types. This spec encodes that document, it does not re-design it.
- **`05-BACKEND-SCHEMA.md` §U-SCHEMA-5, §U-SCHEMA-6** — the canonical field table + `transact_type` routing rule.

## Routes / Endpoints
No new routes/endpoints. This spec defines Pydantic models only. FastAPI's `/predict` route (Step 09+ in the plan, still pending) will later `from api.schemas.predict_v3 import PredictRequestV3, PredictResponseV3` — but that wiring is **out of scope here**.

## Data / Schema changes
- **No new application DB tables.** The `prediction_log.input_features_json` column (Step 08's `08-sqlite-postgres-schema-migration.md` §5) is the destination for these payloads; no DDL change.
- **No new model artifacts.**
- **No `data/raw/` or `data/processed/` writes.**
- **No `analytics_cache/*.json` additions.**
- **No new SQL migration.** The Pydantic schemas describe HTTP payloads, not DB rows; the DB row contract is Step 08's.

## Templates / UI
None. This spec is API-layer code only — no Flask templates, no static assets.

## Files to change / Files to create

**Create:**
- `api/schemas/predict_v3.py` — Pydantic v2 models. Public API:
  - `class TransactType(str, Enum):` — values `SALE = "Sale"`, `RENT = "Rent"`. Matches the raw `TRANSACT_TYPE` strings in the cleaned dataset (case-sensitive per `05-BACKEND-SCHEMA.md` §U-SCHEMA-6).
  - `class PropertyType(str, Enum):` — values `FLAT = "flat"`, `HOUSE = "house"` (extendable per `10-FINALIZED-INPUT-SCHEMA.md` §1 row 1; explicit enum prevents typos at the API boundary).
  - `class Balcony(str, Enum):` — values `ZERO = "0"`, `ONE = "1"`, `TWO = "2"`, `THREE = "3"`, `THREE_PLUS = "3+"`. Matches the reference project's `balcony` column categorical/ordinal type.
  - `class AgePossession(str, Enum):` — `NEW = "New Property"`, `RELATIVELY_NEW = "Relatively New"`, `MODERATELY_OLD = "Moderately Old"`, `OLD = "Old Property"`, `UNDER_CONSTRUCTION = "Under Construction"`.
  - `class FurnishingType(str, Enum):` — `UNFURNISHED = "Unfurnished"`, `SEMIFURNISHED = "Semifurnished"`, `FURNISHED = "Furnished"`. (The `0/1/2` encoding is the `ColumnTransformer`'s job per `02-TRD.md` §U-TRD-3, not the API's.)
  - `class LuxuryCategory(str, Enum):` — `LOW = "Low"`, `MEDIUM = "Medium"`, `HIGH = "High"`. Server-derived per Rules §10.2 — the field appears in the **response** as the resolved value, but a request that supplies it is rejected (see `PredictRequestV3` below).
  - `class FloorCategory(str, Enum):` — `LOW = "Low Floor"`, `MID = "Mid Floor"`, `HIGH = "High Floor"`.
  - `class FacingDirection(str, Enum):` — the 9 standard compass values from `facets/FACING_DIRECTION.csv`: `NORTH`, `SOUTH`, `EAST`, `WEST`, `NORTH_EAST`, `NORTH_WEST`, `SOUTH_EAST`, `SOUTH_WEST`, `UNFURNISHED` placeholder if the facet includes it (verified at implementation time; if a facet value is missing from the enum, the model validator raises a clear 422).
  - `class PredictRequestV3(BaseModel):` — the 16 input fields, with strict types and per-field constraints. Field-by-field (using exact canonical names from `10-FINALIZED-INPUT-SCHEMA.md`):
    1. `city: str` — required, `min_length=1`, regex `^[A-Za-z][A-Za-z\s]+$` (must start with a letter; allows multi-word like `"New Delhi"` if a future city is added, but the current set is single-word).
    2. `sector: str` — required, `min_length=1`. Documented as city-scoped in the docstring (city + sector pair is the locality key).
    3. `property_type: PropertyType` — required.
    4. `transact_type: TransactType` — required. Routing key per `02-TRD.md` §U-TRD-4; the `/predict` handler dispatches on this field.
    5. `bedRoom: int` — required, `ge=1`, `le=15` (15+ is the outlier-domain-rule cap from `02-TRD.md` §6.3 — the API rejects 16+ before the model sees it).
    6. `bathroom: int` — required, `ge=1`, `le=15`.
    7. `balcony: Balcony` — required (categorical enum, not int — matches reference project).
    8. `agePossession: AgePossession` — required.
    9. `built_up_area: float` — required, `gt=0`, `le=20000` (sanity ceiling — anything above 20k sqft is almost certainly commercial / data error; `FR4` from PRD).
    10. `servant_room: bool` — required.
    11. `store_room: bool` — required.
    12. `furnishing_type: FurnishingType` — required.
    13. `floor_category: FloorCategory` — required.
    14. `facing: FacingDirection` — required.
    15. `amenities: list[str]` — optional, default `[]`. Free-text list of amenity labels (must match decoded labels from `facets/AMENITIES.csv`; the API does not validate against the facet table in this spec — Step 14+ will add a venue-against-facet validator when the recommender uses the same model). Stored as-is for downstream `n_amenities` and `has_<amenity>` feature engineering (which happens in the training pipeline, not the API).
    16. `luxury_category: LuxuryCategory | None = Field(default=None, exclude=True)` — **excluded from the JSON payload** via `exclude=True`. The API does **not** accept `luxury_category` from the client (Rules §10.2: server-derived from the amenity checklist, never self-reported). The field exists on the model only so the **response** can echo it back after the server has resolved it (see `PredictResponseV3`). A client that sends `luxury_category` in the request body is silently dropped at validation time, not 422'd — this avoids leaking the field-existence to API consumers who might try to set it. (Documented in the model docstring.)
    - `model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)` — **no extra fields accepted** (catches typos like `"Bedroom"` instead of `"bedRoom"` early with a clean 422), all string fields are stripped on parse.
    - `@model_validator(mode="after")` cross-field rule: `bedRoom <= bathroom + 3` (PRD `FR4` sanity flag) — if violated, raise `ValueError("bathroom too low for bedroom count")` so FastAPI emits a friendly 422, not a bare stack trace.
  - `class ShapContribution(BaseModel):` — single SHAP feature contribution. Fields: `feature: str`, `impact: float`. Mirrors `05-BACKEND-SCHEMA.md` §7's `/predict` response shape.
  - `class PredictResponseV3(BaseModel):` — fields:
    - `predicted_price: float` — point estimate, INR (no log scale to the client).
    - `range_low: float` and `range_high: float` — ±1 std of residuals or quantile bounds (per `FR2`); both `>= 0`.
    - `shap_contributions: list[ShapContribution]` — top-N contributors (default top-10), sorted by `abs(impact)` descending.
    - `is_outlier_input: bool` — flag from distance-to-distribution check (per `05-BACKEND-SCHEMA.md` §7).
    - `model_version: str` — e.g. `"price_model_v1"`.
    - `luxury_category: LuxuryCategory` — the **server-resolved** value, echoed back so the UI can show "Luxury Category: High" without a second roundtrip.
    - `model_config = ConfigDict(extra="forbid")`.
  - `INPUT_FIELDS_V3: tuple[str, ...]` — the **ordered tuple** of the 16 input field names (in the exact order they appear in `10-FINALIZED-INPUT-SCHEMA.md` §1 + §2). This is the single source of truth for:
    - "what columns does the model's preprocessing pipeline expect, in what order?"
    - "what does the FastAPI request body validate against?"
    The training script (future spec) will import this constant to build its `ColumnTransformer` feature list — guaranteeing the model's training column order matches the API's request body order. Order matches the reference project's CSV header (`property_type, sector, price, bedRoom, bathroom, balcony, agePossession, built_up_area, servant room, store room, furnishing_type, luxury_category, floor_category`) **plus** the 4 added fields (`city`, `facing`, `amenities_list`, `transact_type`) inserted at positions 1, 14, 15, 4 respectively, per `05-BACKEND-SCHEMA.md` §U-SCHEMA-5 + §U-SCHEMA-6. The module docstring cites both references.
  - `INPUT_FIELD_TYPES_V3: dict[str, type]` — companion mapping of field name → Python type, used by tests to assert the tuple above against the actual model annotations.
  - Module docstring that:
    - Names `10-FINALIZED-INPUT-SCHEMA.md` and `05-BACKEND-SCHEMA.md` §U-SCHEMA-5 as the schema authority.
    - States the field names are locked to the reference project's contract (camelCase `bedRoom`, lowercase-with-space `servant room` style preserved where applicable — but note: this spec uses **snake_case** `servant_room` / `store_room` in the API because Pydantic field names become JSON keys; the training-time mapping back to the reference project's column name is documented but is a Step-08+ concern, not this spec's). *(ponytail: kept snake_case at the API boundary — Pydantic aliases can map to reference names during model ingestion later, this keeps the wire format readable.)*
    - States `transact_type` is a routing key, not a model feature (cross-reference `02-TRD.md` §U-TRD-4).
    - States `luxury_category` is server-derived, never client-supplied (Rules §10.2).
    - States this spec defines the API contract only — the actual model pipeline wiring is out of scope.

- `tests/test_predict_v3_schema.py` — pytest unit tests, no real-data or DB dependency. Required tests in "Definition of done".

**Modify:**
- `api/schemas/__init__.py` — re-export `PredictRequestV3`, `PredictResponseV3`, `ShapContribution`, `INPUT_FIELDS_V3`, `INPUT_FIELD_TYPES_V3`, plus all 8 enums so future specs can `from api.schemas import PredictRequestV3` without reaching into `predict_v3`. Empty `__init__.py` already exists — fill it with these re-exports.
- `scripts/run_pipeline.py` — no change. This spec does not touch the offline pipeline (the cleaning chain is Steps 02–07; this is the serving layer). *(ponytail: no placeholder import here — the pipeline script never imports from `api/`, and adding a no-op import would suggest coupling that isn't there yet.)*

**No changes** to:
- `requirements.txt` — `pydantic` v2 and `fastapi` are already pinned in Step 01; no new packages.
- `app/`, `api/main.py`, `api/routers/*`, `api/services/*`, `ml/`, `tests/conftest.py`, `data/`, `models/`, `notebooks/`, `migrations/`.
- `CLAUDE.md`'s "Implemented vs stub routes" table — this spec adds **no routes**.

## New dependencies
No new dependencies. `pydantic` (v2) and `enum` are stdlib/Pydantic-core; no `pip install` required.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** N/A — this spec defines request/response shapes; both contain zero PII fields. The `PredictRequestV3` and `PredictResponseV3` models include no `contact_*`, `dealer_*`, `phone`, `email`, `url`, `photo_*` columns — verified by the `test_no_pii_or_contact_fields` test.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no trained model artifact produced. The `model_version` field on `PredictResponseV3` is a string label only; no actual `.pkl` is loaded by this spec.
- **Pydantic v2 only — no v1 patterns.** Use `ConfigDict`, `Field(default=..., exclude=True)`, `model_validator(mode="after")`. No `class Config:` (v1 syntax), no `@validator` (v1 syntax), no `orm_mode` (v1 syntax; v2 equivalent is `from_attributes=True`).
- **Field names are case-sensitive at the API boundary, snake_case at the boundary, camelCase at the training-pipeline boundary.** The 16 input field names in `INPUT_FIELDS_V3` are the **wire-format** snake_case names (`bedRoom`, `built_up_area`, `agePossession`, `servant_room`, `store_room`, `furnishing_type`, `luxury_category`, `floor_category` are passed through as-is from the reference project's camelCase). This is a deliberate one-time exception: Pydantic field names become JSON keys, and renaming `bedRoom` → `bed_room` would lose the contract with the reference project. The mapping back to whatever the training pipeline eventually consumes is a Step-08+ concern. *(ponytail: this is the one place snake_case vs. camelCase collides — keeping it locks the contract.)*
- **Field count is exactly 16.** The contract is locked at `10-FINALIZED-INPUT-SCHEMA.md` §3. Adding a 17th field requires updating that doc first (Rules §10.1 — schema-change-first rule). The `test_input_fields_v3_has_exactly_sixteen_entries` test enforces this.
- **Field order in `INPUT_FIELDS_V3` matches the reference project + 4 additions.** Specifically: `("property_type", "sector", "city", "transact_type", "bedRoom", "bathroom", "balcony", "agePossession", "built_up_area", "servant_room", "store_room", "furnishing_type", "luxury_category", "floor_category", "facing", "amenities")`. Order matters because the training pipeline will iterate this tuple to build feature columns; reorder = train/serve skew. The `test_input_fields_v3_order_matches_reference_project` test pins the order.
- **`luxury_category` is excluded from request bodies.** `Field(..., exclude=True)` means Pydantic drops it on parse — a client sending `{"luxury_category": "High"}` in the JSON body sees a clean validation that ignores the field, not a 422. This is intentional per Rules §10.2 (server-derived to avoid self-report bias). The `test_luxury_category_excluded_from_request` test pins this behavior; if it ever flips, it's a deliberate change to allow client-side self-report and must come with a Rules-doc update first.
- **`extra="forbid"` is on every model.** A request body with `{"bedrooms": 3}` (typo, missing the `R`) is rejected with a 422 listing the unknown field. This catches client bugs at the API boundary rather than letting them silently produce wrong predictions. The `test_extra_fields_rejected` test pins this on both request and response models.
- **Enums use string values that match the cleaned dataset verbatim.** `TransactType.SALE == "Sale"` (capital S, lowercase rest) — not `"sale"` or `"SALE"`. The Step 05 mappers pass raw `TRANSACT_TYPE` strings through; if those are uppercase `"SALE"` in any city file, that's a Step 05 bug to fix at the cleaning layer, not an API-layer coercion. *(ponytail: don't paper over upstream casing.)*
- **Cross-field validators are minimal and well-named.** Only `bedRoom <= bathroom + 3` is enforced at this layer — other rules (e.g., `area_sqft >= bedrooms * 100`) are heuristics and belong in the model service, not the API. Adding more cross-field rules is a deliberate expansion of API surface; reject premature additions.
- **No HTTP status code constants are defined here.** N/A — FastAPI handles status codes via its own `HTTPException` (per CLAUDE.md "FastAPI relies on Pydantic validation errors and explicit HTTPExceptions"); this spec relies on Pydantic's automatic 422 for validation failures.
- **No imports from `app/`, `ml/`, `models/`, or `notebooks/`.** `api/schemas/predict_v3.py` is a pure data-shape module; it imports only from `pydantic`, `enum`, and stdlib. The `test_predict_v3_does_not_import_app_ml_or_models` test enforces this.
- **`INPUT_FIELDS_V3` is a tuple, not a list.** Tuples are immutable; this prevents downstream code from accidentally reordering or appending to it. The `test_input_fields_v3_is_tuple` test pins this.
- **All randomness is seeded.** N/A — no randomness used.
- **Logging uses stdlib `logging` only if a logger is needed.** This spec has no runtime side effects — it defines classes; nothing to log. No logger is instantiated.

## Definition of done

1. `python -m pytest tests/test_predict_v3_schema.py -v` from repo root runs and passes. Tests required (exact names):
   - `test_input_fields_v3_has_exactly_sixteen_entries` — `len(INPUT_FIELDS_V3) == 16`. Catches accidental field additions/removals.
   - `test_input_fields_v3_is_tuple` — `isinstance(INPUT_FIELDS_V3, tuple)`. Catches accidental list mutation.
   - `test_input_fields_v3_order_matches_reference_project` — exact equality against the pinned 16-tuple order documented in "Rules for implementation" above.
   - `test_input_fields_v3_names_match_input_schema_doc` — every name in `INPUT_FIELDS_V3` appears verbatim in `10-FINALIZED-INPUT-SCHEMA.md` §1 + §2's two field tables (read the doc and assert set equality). Locks the API contract to the frozen doc.
   - `test_input_field_types_v3_covers_all_input_fields` — `INPUT_FIELD_TYPES_V3.keys() == set(INPUT_FIELDS_V3)`.
   - `test_predict_request_v3_minimal_valid_payload` — a dict containing all 15 visible (non-excluded) fields with one valid value each deserializes into `PredictRequestV3` without error.
   - `test_predict_request_v3_rejects_missing_required_field` — a payload missing `built_up_area` raises `pydantic.ValidationError`.
   - `test_predict_request_v3_rejects_extra_fields` — a payload with `"unknown_field": "x"` raises `pydantic.ValidationError` (confirms `extra="forbid"`).
   - `test_predict_request_v3_rejects_bedroom_zero` — `bedRoom=0` raises `ValidationError` (catches `ge=1`).
   - `test_predict_request_v3_rejects_bedroom_over_15` — `bedRoom=16` raises `ValidationError` (catches `le=15`).
   - `test_predict_request_v3_rejects_negative_area` — `built_up_area=-100` raises `ValidationError` (catches `gt=0`).
   - `test_predict_request_v3_rejects_area_over_20000` — `built_up_area=50000` raises `ValidationError` (catches `le=20000`).
   - `test_predict_request_v3_bedroom_bathroom_sanity_check` — `bedRoom=5, bathroom=1` raises `ValidationError` with the message containing `"bathroom"` (confirms the `bedRoom <= bathroom + 3` validator).
   - `test_predict_request_v3_balcony_accepts_three_plus` — `balcony="3+"` parses successfully (the `3+` enum value is preserved).
   - `test_predict_request_v3_transact_type_enum_values` — both `"Sale"` and `"Rent"` parse successfully; `"sale"` (lowercase) raises `ValidationError` (case-sensitive enum).
   - `test_predict_request_v3_strips_string_whitespace` — `city=" Gurgaon "` parses to `city="Gurgaon"` (confirms `str_strip_whitespace=True`).
   - `test_predict_request_v3_amenities_defaults_to_empty_list` — omitting the `amenities` field yields `amenities=[]`, not a `ValidationError`.
   - `test_predict_request_v3_amenities_accepts_list_of_strings` — `amenities=["Clubhouse", "Swimming Pool"]` parses to a `list[str]` of length 2 in order.
   - `test_predict_request_v3_luxury_category_excluded_from_request` — a payload containing `"luxury_category": "High"` parses successfully, but the resulting object's `luxury_category` is `None` (the `exclude=True` flag drops it on parse — proves the field is invisible to clients).
   - `test_predict_request_v3_dump_excludes_luxury_category` — calling `.model_dump()` on a `PredictRequestV3` does **not** contain the key `"luxury_category"` (proves `exclude=True` on the dump side too).
   - `test_predict_response_v3_minimal_valid_payload` — a dict with all 7 response fields deserializes into `PredictResponseV3` without error.
   - `test_predict_response_v3_rejects_negative_price` — `predicted_price=-1` raises `ValidationError`.
   - `test_predict_response_v3_rejects_extra_fields` — confirms `extra="forbid"` on the response model too.
   - `test_shap_contribution_accepts_float_impact` — `ShapContribution(feature="area_sqft", impact=0.18)` parses; `impact="high"` raises `ValidationError`.
   - `test_no_pii_or_contact_fields` — `PredictRequestV3.model_fields.keys()` and `PredictResponseV3.model_fields.keys()` contain no substring from the regex `(contact|dealer|phone|email|photo|url|spid)`, case-insensitive. Locks the no-PII rule at the API boundary.
   - `test_predict_v3_does_not_import_app_ml_or_models` — `api.schemas.predict_v3` module-level imports (via `sys.modules` introspection) do not include `app.*`, `ml.*`, or `models.*`. Locks the layering rule.
   - `test_enums_have_expected_string_values` — every enum's string values match the documented strings from `10-FINALIZED-INPUT-SCHEMA.md` §1 + the cleaned-dataset raw column values (e.g., `TransactType.SALE.value == "Sale"`, `FurnishingType.SEMIFURNISHED.value == "Semifurnished"`). Pin the case-sensitive strings.
   - `test_schemas_init_reexports_public_api` — `from api.schemas import PredictRequestV3, PredictResponseV3, INPUT_FIELDS_V3, INPUT_FIELD_TYPES_V3, TransactType, PropertyType, Balcony, AgePossession, FurnishingType, LuxuryCategory, FloorCategory, FacingDirection, ShapContribution` succeeds without `ImportError`.
2. `python -m pytest -m "not realdata"` from repo root still passes — no real-data dependency was introduced.
3. `ruff check api/schemas/predict_v3.py tests/test_predict_v3_schema.py api/schemas/__init__.py` reports zero issues.
4. `python -c "from api.schemas import PredictRequestV3, INPUT_FIELDS_V3; print(len(INPUT_FIELDS_V3)); req = PredictRequestV3(city='Gurgaon', sector='Sector 84', property_type='flat', transact_type='Sale', bedRoom=3, bathroom=3, balcony='2', agePossession='Relatively New', built_up_area=1450, servant_room=True, store_room=False, furnishing_type='Semifurnished', floor_category='Mid Floor', facing='North'); print(req.model_dump_json())"` from repo root prints `16` and a valid JSON payload with no `luxury_category` key — confirms public API imports cleanly and `exclude=True` works end-to-end.
5. `git status` after committing this spec, the new schema module, the tests, and the modified `api/schemas/__init__.py` shows only those files changed (plus this spec). No accidental additions to `data/`, `models/`, `app/`, `migrations/`, or `requirements.txt`.
6. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds **no routes**.
7. `07-TRACKER.md` does **not** need updating for this spec — the planned sequence (Days 6–22 of the Implementation Plan) doesn't include "Pydantic schema definition" as a numbered day; it's an enabling step that lands as part of the inference API wiring (Day 27 / Week 4). The Decision Log entry for "Pydantic schema layer added ahead of route wiring" can be filed via `/update-tracker` if the user wants the decision logged, but it is not required for this spec to merge.
