# Plan: Price Prediction Flask Form Page (Spec 18)

Order: data/schema (none) → backend service (FastAPI client) →
backend routes (Flask) → frontend (templates + CSS) → nav wiring
(landing + base) → tokens (style.css). Each step lists the file,
the change, the spec section it satisfies, and the test that
guards it.

## Step 1 — Tokens (style.css)
- **File:** `app/static/css/style.css`
- **Change:** add `--color-warning-bg`, `--color-warning-fg`,
  `--color-cta-bg`, `--color-cta-fg` HSL tokens for the confidence
  banner + sticky submit button. Extend the existing `:root` block
  only — no other edits.
- **Spec §:** "CSS" → "Modify" bullet for `style.css`.
- **Test:** no new test; covered by visual smoke + the existing
  no-hardcoded-hex grep.

## Step 2 — FastAPI HTTP client
- **File:** `app/services/fastapi_client.py` (new)
- **Change:** `FastAPIClient` class with `post_predict(request)`
  + `get_localities(city)` + module-level `FastAPIUnavailable`
  exception. Caches the locality list once at first call
  (read from `data/processed/clean_listings.parquet` via pandas,
  in-process). Default timeout 2.5s.
- **Spec §:** "Files to change / Files to create" → FastAPIClient
  section.
- **Test:** covered transitively by route tests in step 5
  (the client is monkeypatched there). Add a tiny
  `tests/test_fastapi_client.py` with one happy-path test
  (`post_predict` returns `PredictResponseV3`) using
  `requests_mock` if available, else a tiny in-process stub.

## Step 3 — `app/services/__init__.py`
- **File:** `app/services/__init__.py` (new package marker)
- **Change:** one-line re-export: `from .fastapi_client import
  FastAPIClient, FastAPIUnavailable`. Empty if it already exists.
- **Spec §:** "Files to change / Files to create" → package marker.
- **Test:** importability covered by step 4's `python -c` check.

## Step 4 — `inr_format` helper
- **File:** `app/services/inr_format.py` (new) — kept separate
  from `fastapi_client.py` so each module is single-purpose.
  *(ponytail: one tiny module per service helper, not a kitchen
  sink.)*
- **Change:** `inr_format(value_inr: float, *, transact_type:
  str = "Sale") -> str` — converts `14200000.0` →
  `"1.42 Cr"` (Sale) or `"1,42,000 / month"` (Rent). Pure function,
  stdlib only (`math.log10`, `re`).
- **Spec §:** `predict_result.html` block overview → "Jinja
  `inr_format` filter".
- **Test:** `tests/test_inr_format.py` — pinned cases
  (`1_42_00_000 → "1.42 Cr"`, `4_20_000 → "4.20 Lakh"`, rent
  variants, edge `value=0`).

## Step 5 — Flask routes (`app/app.py`)
- **File:** `app/app.py` (modify)
- **Change:** add inside `create_app()`:
  - Lazy `_get_client() -> FastAPIClient` module-level singleton.
  - `predict_get()` — renders `predict.html` with `cities`,
    `localities_by_city`, `enum_options`. Uses
    `client.get_localities(city)` to bootstrap the JSON blob.
    Returns 200.
  - `predict_post()` — reads `request.form`, constructs
    `PredictRequestV3`, calls `client.post_predict(...)`, renders
    `predict_result.html`. Catches `FastAPIUnavailable` → render
    with `unavailable=True`. Catches `pydantic.ValidationError`
    → `flash()` + redirect to `predict_get`.
  - Registers the Jinja filter
    `app.jinja_env.filters["inr_format"] = inr_format`.
  - Updated module docstring cites Spec 11/17 + Rules §5.1/§5.2/§10.2.
- **Spec §:** "Files to change / Files to create" → "Modify"
  bullet for `app/app.py`.
- **Test:** `tests/test_predict_route.py` — the 10 required tests
  from the spec's "Definition of done" §1. Monkeypatches
  `FastAPIClient` to return canned `PredictResponseV3` /
  raise `FastAPIUnavailable`. Uses the existing `app_client`
  fixture from `conftest.py`.

## Step 6 — `predict.html` template
- **File:** `app/templates/predict.html` (new, extends `base.html`)
- **Change:** five `<fieldset class="form-step">` blocks per
  UI/UX §U-UX-6. Inline `<script>` populates the locality
  `<select>` from the `localities_by_city` JSON blob. Amenities
  step is a `<details>` collapsed by default. Form action uses
  `url_for('predict')`.
- **Spec §:** "Templates / UI" → "Create" bullet for
  `predict.html`. Field overview block.
- **Test:** `test_predict_get_renders_form` (asserts all 16 input
  field names minus `luxury_category` appear) +
  `test_predict_get_injects_localities_by_city`.

## Step 7 — `predict_result.html` template
- **File:** `app/templates/predict_result.html` (new, extends
  `base.html`)
- **Change:** `{% if unavailable %}` branch → friendly empty
  state. `{% else %}` branch → optional confidence banner +
  price hero + range + summary + model_version + SHAP placeholder
  div + "See similar properties →" CTA via `url_for('recommend')`.
  Uses `{{ x | inr_format }}` for currency.
- **Spec §:** "Templates / UI" → "Create" bullet for
  `predict_result.html`.
- **Test:** `test_predict_post_forwards_to_fastapi_and_renders_result`
  + `test_predict_post_returns_unavailable_state_when_fastapi_down`
  + `test_predict_post_renders_outlier_banner_when_flagged`.

## Step 8 — `predict.css`
- **File:** `app/static/css/predict.css` (new)
- **Change:** layout-only rules: 2-col grid on desktop, 1-col
  mobile (CSS `:where(...)` selectors per UI/UX §9), step
  dividers, sticky submit button. Uses only `var(--space-*)`,
  `var(--radius)`, `var(--color-cta-bg)`, `var(--color-cta-fg)`
  tokens. First line cites `style.css` for shared tokens.
- **Spec §:** "CSS" → "Create" bullet.
- **Test:** no dedicated test; ruff grep for hardcoded hex/pixel
  in `predict.css` is the existing CI gate.

## Step 9 — Base nav (`base.html`)
- **File:** `app/templates/base.html` (modify)
- **Change:** add `<nav class="site-nav">` block under `.brand`
  with five `url_for()` links: `/predict`, `/analytics`,
  `/recommend`, `/insights`, `/map`. Each links to the existing
  placeholder page (no new routes here).
- **Spec §:** "Templates / UI" → "Modify" bullet for `base.html`.
- **Test:** no dedicated test; smoke-verified in the GET smoke
  test (the nav appears in `predict.html` rendered HTML).

## Step 10 — Landing module-card links (`landing.html`)
- **File:** `app/templates/landing.html` (modify)
- **Change:** wrap each `.module-card` in
  `<a href="{{ url_for(endpoint) }}">` (one line per card, no
  copy change). Keeps the existing `module-card` class.
- **Spec §:** "Templates / UI" → "Modify" bullet for
  `landing.html`.
- **Test:** no dedicated test; smoke-verified by curl on `/`.

## Step 11 — CLAUDE.md + tracker updates
- **File:** `CLAUDE.md` (modify "Implemented vs stub routes"
  table) — flip both Flask `/predict` rows from Stub to
  Implemented.
- **File:** `07-TRACKER.md` (modify Day 38 → "Partially done")
  + Decision Log entry for the Flask-side forward pattern.
- **Spec §:** "Definition of done" §9 + §10.
- **Test:** covered by visual smoke; no test for the docs.

## Risks to confirm before starting

1. **Luxury-category checklist wiring.** The spec resolves this
   as "append the 3 `luxury_finish` checkbox values to the
   `amenities` list and let FastAPI resolve luxury_category
   unchanged." Confirm with the user before implementation — the
   alternative is to introduce a new field on the Pydantic
   schema (which would touch Spec 11/17 and require a fast-follow
   patch on the FastAPI side). The spec's recommendation is the
   lazy, no-touch approach; if the user disagrees, escalate before
   editing Pydantic models.

2. **Locality source.** The spec reads `clean_listings.parquet`
   once and caches the locality list in process. If the parquet
   isn't populated on the dev checkout (per Tracker §1, Day 7
   is "Not Started"), the loader falls back to a hardcoded
   stub list per city. Confirm the user is fine with the
   stub fallback before implementation — otherwise we should
   either skip the locality feature on first load or generate
   a tiny fixture parquet in `tests/fixtures/`.

3. **Sale/Rent rent pipeline.** Spec 17 notes the Rent pipeline
   may be missing on some checkouts (Rent was skipped at
   training). The form still lets users pick "Rent" — the
   FastAPI side returns a 503 in that case. Confirm this is
   the intended UX (vs. disabling the Rent radio when the
   pipeline is absent).
