# Spec: Price Prediction Flask Form Page

## Overview
Wire the Flask `GET /predict` route to render the finalized 16-field
Price Prediction form (from `docs/10-FINALIZED-INPUT-SCHEMA.md` §3 +
`docs/04-UI-UX-DESIGN.md` §U-UX-6) and the `POST /predict` route to
forward the submitted form to the already-implemented FastAPI
`POST /predict` endpoint (Spec 17). The page is the user-facing
counterpart to the API; Flask never loads models itself (Rules §5.1),
never imports anything from `ml/` or `models/`, and degrades
gracefully when the FastAPI service is unreachable (Rules §5.2).
Module: **price-prediction**.

This spec closes the "Flask `POST /predict` row = Stub" line in
`CLAUDE.md`'s route table. The SHAP chart on the result screen, the
inline `VerdictBadge` + `AffordabilityChip`, and the inline insight
cards are follow-on specs (Day 38+ in the Implementation Plan) — this
spec ships the form + a result page that shows the price hero +
range, with placeholders for the SHAP/insight/tier widgets.

## Depends on
- **Spec 11** (`11-price-prediction-input-schema-v3.md`) —
  `PredictRequestV3`, `PredictResponseV3`, `ShapContribution`, plus
  the 8 enums (`TransactType`, `PropertyType`, `Balcony`,
  `AgePossession`, `FurnishingType`, `LuxuryCategory`,
  `FloorCategory`, `FacingDirection`). The Flask route forwards
  directly to these typed Pydantic models.
- **Spec 17** (`17-price-prediction-fastapi-endpoint.md`) — FastAPI
  `POST /predict` returns a `PredictResponseV3`. The Flask side
  HTTP-calls that endpoint and renders the result.
- **`docs/10-FINALIZED-INPUT-SCHEMA.md`** — authoritative field list,
  allowed values, types. The form's field order, labels, and enums
  all derive from this doc.
- **`docs/04-UI-UX-DESIGN.md` §U-UX-6** — finalized form layout (5
  steps), §U-UX-7 (guided luxury-category derivation), §U-UX-8
  (Sale/Rent radio prominence).
- **`docs/03-APP-FLOW.md` §2.2** — page-level flow.
- **`docs/05-BACKEND-SCHEMA.md` §U-SCHEMA-5, §U-SCHEMA-6** —
  canonical field names + `transact_type` routing rule.
- **`docs/08-RULES.md` §5.1** — Flask never imports model code.
- **`docs/08-RULES.md` §5.2** — graceful degradation when FastAPI is
  unreachable.
- **`docs/08-RULES.md` §10.2** — `luxury_category` is server-derived,
  never self-reported; the form therefore uses a guided mini-checklist,
  not a raw dropdown.
- **`app/config.py`** — `FASTAPI_BASE_URL` (used by the HTTP client).
- **`flask-routing` skill** — one-route-one-responsibility,
  `url_for()` only, DB-free route, graceful degradation.
- **`frontend-design` skill** — opinionated palette/type/layout
  choices; restraint; the SHAP/insight widgets are placeholders in
  this spec, not the full visual design.
- **`css-design-tokens-and-card-system` skill** — all colors/spacing
  via CSS variables in `style.css`; no inline hex values.
- **`api-schema-design-pydantic` skill** — request/response models
  already exist; the Flask side does not redefine them.

## Routes / Endpoints
- **Flask:** `GET /predict` — render the 16-field prediction form
  with city-scoped locality dropdown bootstrap data, dependent
  dropdown wiring stub, and the Sale/Rent radio group. Access:
  public.
- **Flask:** `POST /predict` — accept the submitted form (multipart
  form-encoded), translate to a `PredictRequestV3` JSON payload,
  HTTP-POST to `http://<FASTAPI_BASE_URL>/predict`, render the
  result template with the `PredictResponseV3` body. Access:
  public.

## Data / Schema changes
- **No new application DB tables.** `prediction_log` is written by
  the FastAPI side (Spec 17); Flask does not duplicate the write.
- **No new model artifacts.**
- **No `data/raw/` or `data/processed/` writes.**
- **No new SQL migrations.**
- **No new analytics cache files.**

## Templates / UI

**Create:**
- `app/templates/predict.html` — the GET form. Extends `base.html`.
  Five steps per UI/UX §U-UX-6, with collapsed Step 5 (amenities)
  by default. Locality dropdown is populated client-side from a
  small JSON blob the route injects (4 cities → list of locality
  strings). The form is server-rendered; no client-side React-style
  build step.
- `app/templates/predict_result.html` — the POST result. Extends
  `base.html`. Shows the predicted price hero, the price range,
  the locality/property summary line, a "Why this price?" section
  with the SHAP bars stubbed for the follow-on spec, a
  "See similar properties →" CTA (links to `/recommend`, pre-fill
  is a later spec), and a clear failure state when FastAPI is
  unreachable. If `is_outlier_input == true`, an amber banner sits
  above the hero price (UI/UX §4.2's confidence flag).

**Modify:**
- `app/templates/base.html` — add a `<nav>` block with module links
  (Predict / Analytics / Recommend / Insights / Map) under the
  brand header, using `url_for()`. Currently the header is brand
  only; the spec adds the shared navigation that every page
  expects per UI/UX §3.
- `app/templates/landing.html` — turn the module-card markup into
  clickable links (`<a href="{{ url_for('predict') }}">` etc.) so
  the landing-page module cards route to real pages. No copy
  change; just wire the links.

**CSS:**
- `app/static/css/predict.css` — layout for the form: 2-column grid
  on desktop, 1-column on mobile, step dividers, sticky submit
  button per UI/UX §4.1. Uses CSS variables only (no hex
  hardcodes). Layout-specific rules only — colors/spacing live in
  `style.css`.

## Files to change / Files to create

**Create:**
- `app/services/fastapi_client.py` — small HTTP client wrapper
  around the FastAPI inference service. Public API:
  - `class FastAPIClient:` — single-purpose wrapper.
    - `def __init__(self, base_url: str, *, timeout_seconds: float =
      2.5) -> None:` — stores the base URL + a per-request timeout.
      Default timeout chosen per Rules §5.2's "graceful degradation"
      spirit: a stuck FastAPI call must not freeze the UI.
    - `def post_predict(self, request: PredictRequestV3) ->
      PredictResponseV3:` — POSTs the request body to
      `<base_url>/predict`, parses the JSON response into a
      `PredictResponseV3`. On any of `requests.HTTPError`,
      `requests.Timeout`, `requests.ConnectionError`, or
      `pydantic.ValidationError` (response shape drift), raises a
      single `FastAPIUnavailable` exception. Pinned by tests.
    - `def get_localities(self, city: str) -> list[str]:` — returns
      a sorted unique list of locality names for the city. Backed
      by a tiny in-process lookup table that loads once from
      `data/processed/clean_listings.parquet` at first call (cached
      on the instance) — a real FastAPI endpoint for this is a
      later spec; the in-process loader is the simplest way to
      populate the dependent dropdown for v1. *(ponytail: avoids
      adding a `/localities` route to FastAPI for one form
      dependency; the parquet is already canonical and small
      enough to read once. Add a FastAPI endpoint only if the
      locality list grows beyond what fits in memory or needs to
      be filtered server-side.)*
  - `class FastAPIUnavailable(Exception):` — single error type the
    Flask route catches. Carries the original cause via
    `__cause__`.
  - Module docstring citing Spec 17 (target endpoint), Rules §5.1
    + §5.2, and the `flask-routing` skill.

- `app/templates/predict.html` — the GET form. Block overview:
  ```
  {% extends "base.html" %}
  {% block content %}
  <form method="post" action="{{ url_for('predict') }}" class="predict-form">
    <!-- Step 1: Location & Deal Type -->
    <fieldset class="form-step">
      <legend>Step 1 — Location & Deal Type</legend>
      <label>City <select name="city" required>...</select></label>
      <label>Sector / Locality
        <select name="sector" required data-city=""></select>
      </label>
      <label>Property Type <select name="property_type" required>...</select></label>
      <fieldset class="transact-type">
        <legend>Transaction Type</legend>
        <label><input type="radio" name="transact_type" value="Sale" required checked> Sale</label>
        <label><input type="radio" name="transact_type" value="Rent"> Rent</label>
      </fieldset>
    </fieldset>
    <!-- Step 2: Core Structure -->
    <fieldset class="form-step">
      <legend>Step 2 — Core Structure</legend>
      <label>Bedrooms <input type="number" name="bedRoom" min="1" max="15" required></label>
      <label>Bathrooms <input type="number" name="bathroom" min="1" max="15" required></label>
      <label>Balconies <select name="balcony" required>...</select></label>
      <label>Built-up Area (sqft) <input type="number" name="built_up_area" min="1" max="20000" required></label>
    </fieldset>
    <!-- Step 3: Details -->
    <fieldset class="form-step">
      <legend>Step 3 — Details</legend>
      <label>Property Age <select name="agePossession" required>...</select></label>
      <label>Furnishing <select name="furnishing_type" required>...</select></label>
      <label>Facing <select name="facing" required>...</select></label>
      <label>Floor Category <select name="floor_category" required>...</select></label>
    </fieldset>
    <!-- Step 4: Extras (guided luxury derivation) -->
    <fieldset class="form-step">
      <legend>Step 4 — Extras</legend>
      <label><input type="checkbox" name="servant_room" value="1"> Servant Room</label>
      <label><input type="checkbox" name="store_room" value="1"> Store Room</label>
      <fieldset class="luxury-checklist">
        <legend>Finish level (auto-derives Luxury Category)</legend>
        <label><input type="checkbox" name="luxury_finish" value="branded_developer"> Branded developer</label>
        <label><input type="checkbox" name="luxury_finish" value="imported_fittings"> Imported fittings</label>
        <label><input type="checkbox" name="luxury_finish" value="clubhouse_access"> Clubhouse access</label>
      </fieldset>
    </fieldset>
    <!-- Step 5: Amenities (collapsed) -->
    <details class="form-step form-step--optional">
      <summary>Step 5 — Add amenities (optional)</summary>
      <label><input type="checkbox" name="amenities" value="Swimming Pool"> Swimming Pool</label>
      <label><input type="checkbox" name="amenities" value="Gym"> Gym</label>
      <label><input type="checkbox" name="amenities" value="Club house"> Clubhouse</label>
      <label><input type="checkbox" name="amenities" value="Power Backup"> Power Backup</label>
      <label><input type="checkbox" name="amenities" value="Lift"> Lift</label>
    </details>
    <button type="submit" class="predict-form__submit">Predict Price</button>
  </form>
  {% endblock %}
  ```
  The locality `<select>` options are populated client-side by a
  tiny inline `<script>` reading the `localitiesByCity` JSON
  injected by the route, switching options when `city` changes.

- `app/templates/predict_result.html` — the POST result. Block
  overview:
  ```
  {% extends "base.html" %}
  {% block content %}
  {% if unavailable %}
    <section class="empty-state" role="alert">
      <h2>Prediction service is temporarily unavailable</h2>
      <p>Please try again in a moment.</p>
    </section>
  {% else %}
    {% if is_outlier_input %}
      <div class="confidence-banner" role="status">
        Low confidence — input is outside the typical range for this locality.
      </div>
    {% endif %}
    <section class="price-hero">
      <div class="price-hero__amount">₹ {{ predicted_price_inr | inr_format }}</div>
      <div class="price-hero__range">({{ range_low_inr | inr_format }} – {{ range_high_inr | inr_format }})</div>
      <div class="price-hero__summary">{{ bedRoom }} BHK · {{ built_up_area }} sqft · {{ sector }}, {{ city }}</div>
      <div class="price-hero__model">model: {{ model_version }}</div>
    </section>
    <section class="why-this-price">
      <h2>Why this price?</h2>
      <!-- SHAP bar chart placeholder — wired in a follow-on spec -->
      <div class="chart-placeholder" aria-label="SHAP explanation chart placeholder">
        SHAP explanation goes here in a follow-on spec.
      </div>
    </section>
    <a class="cta-recommend" href="{{ url_for('recommend') }}">See similar properties →</a>
  {% endif %}
  {% endblock %}
  ```
  A Jinja `inr_format` filter (registered in `create_app()`) turns
  `14200000.0` into `"1.42 Cr"` (Sale) or `"1,42,000 / month"`
  (Rent) — see the `inr_format` helper in `app/services/`.

- `app/static/css/predict.css` — form layout. Pure layout rules
  (grid, spacing, sticky button). No hex colors, no pixel magic
  numbers outside `--space-*` tokens; the file's first line cites
  `style.css` for shared tokens.

- `tests/test_predict_route.py` — Flask route tests via
  `app.test_client()`. Required tests:
  - `test_predict_get_renders_form` — `GET /predict` → 200 +
    contains all 16 input field names from `INPUT_FIELDS_V3` minus
    `luxury_category` (excluded per Rules §10.2) in the rendered
    HTML.
  - `test_predict_get_injects_localities_by_city` — the rendered
    HTML contains a `data-localities` attribute (or a `<script>`
    JSON blob) with one entry per known city.
  - `test_predict_post_forwards_to_fastapi_and_renders_result` —
    monkeypatches `FastAPIClient.post_predict` to return a
    canned `PredictResponseV3`, POSTs the form, asserts the
    rendered HTML contains the formatted price hero + the
    `model_version` string.
  - `test_predict_post_returns_unavailable_state_when_fastapi_down`
    — monkeypatches `post_predict` to raise `FastAPIUnavailable`;
    the response is 200 (the page itself is fine) and the
    template contains the "temporarily unavailable" copy.
  - `test_predict_post_returns_400_on_missing_field` — submitting
    a form missing `built_up_area` redirects with a flash message
    OR re-renders the form with an inline error (per
    `flask-routing` skill: no inline business logic in the route,
    validation lives in the Pydantic schema on the API side — the
    Flask route catches `pydantic.ValidationError` from the
    forward and re-renders with a friendly message). Pinned by
    a test.
  - `test_predict_post_returns_400_on_invalid_bedroom` —
    `bedRoom=20` → same friendly re-render.
  - `test_predict_post_returns_400_on_bedroom_bathroom_violation`
    — `bedRoom=5, bathroom=1` → friendly re-render with a
    "bathroom too low for bedroom count" message.
  - `test_predict_post_renders_outlier_banner_when_flagged` —
    `post_predict` returns `is_outlier_input=True`; the result
    HTML contains the "Low confidence" banner.
  - `test_predict_post_passes_transact_type_to_fastapi` — submit
    with `transact_type=Rent`; the monkeypatched client receives
    `transact_type="Rent"`.
  - `test_predict_post_does_not_import_ml_or_models` — introspect
    `sys.modules` after the request; no `ml.*` or `models.*`
    modules appear. Pinned to enforce Rules §5.1.

**Modify:**
- `app/app.py` — add the two routes + the `inr_format` Jinja
  filter. Public additions:
  - `def _get_client() -> FastAPIClient:` — module-level lazy
    singleton (mirrors `get_predict_service` in `api/routers/predict.py`).
  - `@app.route("/predict", methods=["GET"])` →
    `def predict_get() -> str:` — renders `predict.html` with
    `cities`, `localities_by_city`, `enum_options` (a dict of
    each enum's allowed values, sourced from `api.schemas.predict_v3`'s
    enums).
  - `@app.route("/predict", methods=["POST"])` →
    `def predict_post() -> Any:` — reads `request.form`,
    constructs `PredictRequestV3`, calls
    `client.post_predict(request)`, renders `predict_result.html`.
    On `FastAPIUnavailable` → renders `predict_result.html` with
    `unavailable=True`. On `pydantic.ValidationError` → flash
    + redirect to `predict_get` (the friendly UX path).
  - `app.jinja_env.filters["inr_format"] = inr_format` — registers
    the currency formatter.
  - Updated module docstring cites Spec 11 (schemas), Spec 17
    (target endpoint), Rules §5.1 / §5.2 / §10.2, and the
    `flask-routing` skill.
  - The existing `landing()` route is unchanged in behavior; it
    still passes `cities` + `modules` to the template. (The
    template itself is modified below to wire the module-card
    links.)

- `app/services/__init__.py` — empty package marker; created here
  so `app.services.fastapi_client` is importable. (If the file
  already exists, fill with `from .fastapi_client import
  FastAPIClient, FastAPIUnavailable` re-export.)

- `app/templates/base.html` — add a `<nav class="site-nav">` block
  between `.brand` and `{% block content %}` containing links to
  `/predict`, `/analytics`, `/recommend`, `/insights`, `/map` via
  `url_for()`. The links are inert stubs that route to the
  placeholder pages for now (those pages already exist per
  `CLAUDE.md`'s route table) — this spec doesn't add them.

- `app/templates/landing.html` — wrap each `.module-card` in an
  `<a href="{{ url_for(endpoint) }}">` (one-liner, no copy change).
  Existing `module-card` class is preserved.

- `app/static/css/style.css` — add a `--color-warning-bg`,
  `--color-warning-fg`, and `--color-cta-bg` token (for the
  confidence banner + sticky CTA button) — all as HSL values,
  matching the existing token style. *(ponytail: extend the
  token system once here, not inline per component — keeps the
  shared visual language consistent.)*

**No changes** to:
- `requirements.txt` — `requests` (HTTP client) is already pinned
  by Step 01; no new packages.
- `api/`, `ml/`, `models/`, `data/`, `migrations/`, `notebooks/`.
- The FastAPI side of the predict contract (Spec 17) — this spec
  is strictly Flask.
- `tests/conftest.py` (existing fixtures remain).

## New dependencies
**No new dependencies.** `requests` (HTTP client), `pydantic` (v2),
and `flask` are already pinned in Step 01. The `inr_format` helper
uses stdlib `math` + `re` only.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — this spec has no DB writes (the
  FastAPI side handles `prediction_log` per Spec 17).
- **No dealer/contact/media-URL fields ever reach the UI or an
  export.** The `FastAPIClient.post_predict` forwards only the
  15 visible fields from `PredictRequestV3` (excluding
  `luxury_category` which is server-derived). The result template
  does not include any PII column from the response (the response
  shape has none). Pinned by
  `test_predict_post_does_not_import_ml_or_models` + the existing
  Pydantic no-PII guarantees from Spec 11.
- **CSS variables only, never hardcoded hex values.** All new
  colors live in `style.css` as HSL tokens. `predict.css` uses
  only `var(--color-*)` / `var(--space-*)` / `var(--radius)` /
  `var(--font-sans)` references. Pinned by a `ruff`/`grep` check
  in CI (existing convention).
- **All templates extend `base.html`.** Both `predict.html` and
  `predict_result.html` start with `{% extends "base.html" %}`.
- **Model changes must reference the fixed evaluation protocol.**
  N/A — no model changes. The Flask side calls the already-
  certified FastAPI endpoint.
- **Flask never imports model code (Rules §5.1).** The
  `FastAPIClient` imports only `requests`, `pydantic`, and the
  Pydantic schema modules from `api.schemas` — no `ml.*`,
  `models.*`, or `api.services.*` imports. The locality lookup
  reads the parquet directly via `pandas` (no model code), pinned
  by `test_predict_post_does_not_import_ml_or_models`.
- **Graceful degradation on FastAPI failure (Rules §5.2).** Any
  exception from the FastAPI client (`HTTPError`, `Timeout`,
  `ConnectionError`, `ValidationError`) is caught by the route
  and translated to `FastAPIUnavailable` → `predict_result.html`
  with `unavailable=True`. The user sees a friendly "Prediction
  service is temporarily unavailable, please try again" message,
  never a 500 page or a stack trace. Timeout is short (2.5s)
  so a hung service can't freeze the UI.
- **`transact_type` is a routing key (Rules §10.3, TRD §U-TRD-4).**
  The form's `transact_type` radio forwards the exact string
  (`"Sale"` or `"Rent"`) to FastAPI; the Flask side does not
  branch on it. The routing happens server-side at FastAPI.
- **`luxury_category` is server-derived (Rules §10.2).** The form
  has no `luxury_category` dropdown. The `luxury_finish`
  checkboxes are forwarded to FastAPI as part of the `amenities`
  list (or a separate `luxury_finish_flags` field — see
  "Files to change" for the resolved contract); the server resolves
  the category from those plus `amenities`. The Flask side never
  assigns a luxury category itself. *(ponytail: simplest contract —
  the guided checklist emits 3 boolean flags; the existing
  FastAPI service already resolves `luxury_category` from
  `amenities` count, so the spec wires the checklist into the
  `amenities` list at the Flask layer and lets the server do the
  resolution unchanged.)*
- **Form enums match the cleaned dataset verbatim.** The dropdown
  `<option value="...">` strings come from `api.schemas.predict_v3`'s
  enums — never hardcoded duplicates. The route pulls them via
  `[e.value for e in PropertyType]` etc.
- **`url_for()` for every internal link.** The `<nav>` in
  `base.html`, the module-card links in `landing.html`, and the
  form `action` in `predict.html` all use `url_for()` — never
  hardcoded path strings.
- **HTTP status code policy.** Flask `GET /predict` → 200 always.
  Flask `POST /predict` with valid form → 200 (rendered result).
  Flask `POST /predict` with invalid form (Pydantic catches it
  before the FastAPI call) → 400 with a flash message + redirect
  to `GET /predict`. Flask `POST /predict` with FastAPI down →
  200 + `unavailable=True` in the template (the page itself is
  fine; the service isn't). No 500s reach the user.
- **Logging uses stdlib `logging` only.** One module-level logger
  per new file. INFO for a successful predict submission;
  WARNING for FastAPI unavailability (with the cause chained via
  `exc_info=True`); no DEBUG in normal flow.
- **No `api/services/` imports.** The Flask side imports only
  from `api.schemas` (for the Pydantic models + enums) — never
  from `api.services`, `api.routers`, `api.main`, `ml.*`, or
  `models.*`. Pinned by
  `test_predict_post_does_not_import_ml_or_models`.
- **All randomness is seeded.** N/A — no randomness in the hot
  path. The locality list is loaded once at first request and
  cached; nothing random.
- **No notebook-only steps.** Everything is reproducible via
  `python app/app.py` from repo root. No Jupyter cell populates
  the locality list or formats currency.
- **The locality dropdown is a v1 simplification.** Reading the
  parquet once and caching the list is fine for ~182k rows across
  4 cities — fits comfortably in memory. A real `/api/localities`
  FastAPI endpoint is a follow-on spec, called out as such in the
  `FastAPIClient.get_localities` docstring.

## Definition of done

1. `python -m pytest tests/test_predict_route.py -v` from repo
   root runs and passes. Tests required (exact names):
   - `test_predict_get_renders_form`
   - `test_predict_get_injects_localities_by_city`
   - `test_predict_post_forwards_to_fastapi_and_renders_result`
   - `test_predict_post_returns_unavailable_state_when_fastapi_down`
   - `test_predict_post_returns_400_on_missing_field`
   - `test_predict_post_returns_400_on_invalid_bedroom`
   - `test_predict_post_returns_400_on_bedroom_bathroom_violation`
   - `test_predict_post_renders_outlier_banner_when_flagged`
   - `test_predict_post_passes_transact_type_to_fastapi`
   - `test_predict_post_does_not_import_ml_or_models`
2. `python -m pytest -m "not realdata"` from repo root still
   passes — no real-data dependency introduced (the locality
   loader is monkeypatched in tests; the FastAPI client is
   monkeypatched in tests).
3. `ruff check app/app.py app/services/fastapi_client.py
   app/templates/predict.html app/templates/predict_result.html
   tests/test_predict_route.py` reports zero issues. (Templates
   are linted via the `jinja` rule if available; otherwise the
   Python files only.)
4. `python -c "from app.app import create_app; from
   app.services.fastapi_client import FastAPIClient,
   FastAPIUnavailable; app = create_app(); print('ok')"` from
   repo root prints `ok` — public API imports cleanly.
5. With Flask running (`python app/app.py`) and FastAPI running
   (`uvicorn api.main:app --port 8000`), a `curl -L
   http://localhost:5000/predict` returns 200 with HTML
   containing the form's 5 `<fieldset class="form-step">`
   blocks, the `<select name="city">` with 4 cities, and the
   `<button type="submit" class="predict-form__submit">Predict
   Price</button>`. Manual smoke test of the GET path.
6. With both servers still running, a `curl -L -X POST
   http://localhost:5000/predict -d "city=Gurgaon
   &sector=sector 84&property_type=flat&transact_type=Sale
   &bedRoom=3&bathroom=3&balcony=2&agePossession=Relatively New
   &built_up_area=1450&servant_room=1&furnishing_type=Semifurnished
   &floor_category=Mid Floor&facing=North&amenities=Swimming Pool
   &amenities=Club house"` returns 200 with HTML containing the
   formatted price hero (`₹ 1.42 Cr`-style or whatever the canned
   response says) and the `model_version` string from the FastAPI
   response. Manual smoke test of the POST path.
7. With Flask running but FastAPI stopped, the same POST as in
   step 6 returns 200 with HTML containing the "Prediction service
   is temporarily unavailable" copy. Manual smoke test of the
   degradation path.
8. `git status` after committing shows only the new files listed
   above, the modified `app/app.py`, the modified
   `app/templates/base.html`, the modified `app/templates/landing.html`,
   and the modified `app/static/css/style.css`. No accidental
   additions to `api/`, `data/`, `models/`, `migrations/`, or
   `requirements.txt`.
9. `CLAUDE.md`'s "Implemented vs stub routes" table is updated
   to flip **both** Flask rows for `/predict`:
   - `GET /predict` → **Implemented** (form renders, all 16
     fields visible minus the server-derived `luxury_category`).
   - `POST /predict` (Flask → FastAPI) → **Implemented**
     (forwards to FastAPI, renders the result, degrades gracefully).
   The FastAPI `POST /predict` row stays **Implemented** (Spec 17).
10. `07-TRACKER.md` is updated via `/update-tracker` to mark Day
    38 ("Price Prediction form + result page + SHAP chart") as
    **Partially done** with the actual date and a one-line summary
    noting that the form + result shell + FastAPI wiring are
    landed, but the SHAP chart, inline verdict/tier widgets, and
    inline insight cards remain follow-on work. The Decision Log
    gets one new entry: "Wired Flask `/predict` form to forward
    directly to FastAPI's `PredictRequestV3` — the Flask route
    does no model code, no separate validation, and no DB writes
    (per Rules §5.1 / §5.2 / §10.2)."
