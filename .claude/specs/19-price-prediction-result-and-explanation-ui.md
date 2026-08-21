# Spec: Price Prediction Result & Explanation UI

## Overview
Replace the SHAP placeholder on the Flask `predict_result.html`
(Spec 18) with the real per-prediction SHAP bar chart and add the
missing model-metadata + direction summary the UI/UX design calls
for. The result page becomes the actual answer to the user's
question — *why* this price, not just *what* price — by rendering
the `shap_contributions` already returned by FastAPI's `POST
/predict` (Spec 17) as a horizontal bar chart with `+`/`−`
labels, plus a one-line "X factors pushed the price up, Y pushed
it down" summary above the bars. The Classification module's
`VerdictBadge` + `AffordabilityChip` (UI/UX §U-UX-9) and inline
insight cards (§4.2) remain follow-on specs — this spec
closes only the SHAP chart, the model-version strip, and the
SHAP direction summary. Module: **price-prediction**.

## Depends on
- **Spec 17** (`17-price-prediction-fastapi-endpoint.md`) —
  FastAPI `POST /predict` already returns
  `shap_contributions: list[ShapContribution]` (shape `{feature,
  impact}`, `top_n ≤ SHAP_TOP_N = 7`) and a `model_version`
  string in the response body. This spec renders that data.
- **Spec 16** (`16-shap-explainability-price-model.md`) —
  `ml/explainability/explain_one` returns
  `ShapContribution(feature, label, impact, direction)` where
  `label` is the human-readable post-preprocessor name and
  `direction ∈ {"up","down"}`. The FastAPI service in Spec 17
  currently strips `label` + `direction` to wire only `{feature,
  impact}` (per Backend Schema §7). This spec adds the same
  mapping back on the Flask side (one helper, no schema drift).
- **Spec 18** (`18-price-prediction-flask-form-page.md`) —
  `predict_result.html` template, `predict_post` route,
  `inr_format` Jinja filter, `FastAPIClient.post_predict`. All
  reused; only the template + one small Jinja helper change.
- **`api/schemas/predict_v3.py`** — `ShapContribution` shape.
  Not modified.
- **`ml/explainability/labels.py`** — `FEATURE_LABEL_MAP_V2()` +
  `load_label_map_from_disk()` already supply the
  `feature → "Built-up Area (sqft)"` mapping.
- **`docs/04-UI-UX-DESIGN.md` §4.2** — final result screen layout.
- **`docs/04-UI-UX-DESIGN.md` §U-UX-2** — graceful degradation for
  missing Classification widgets (VerdictBadge / AffordabilityChip
  slot is left empty until `/classify` ships; not this spec's
  concern).
- **`docs/08-RULES.md` §6.2** — every prediction must be paired
  with an explanation element.
- **`docs/08-RULES.md` §6.4** — SHAP positive/negative bars always
  paired with +/− text (never colour-only).
- **`docs/08-RULES.md` §6.1** — async ops have loading + failure
  states (already handled by Spec 18; reused).
- **`css-design-tokens-and-card-system` skill** — colors/spacing
  via CSS variables; layout-only rules in `predict.css`.
- **`frontend-design` skill** — restraint, card language,
  progressive disclosure.
- **`chartjs-plotly-charting` skill** — Chart.js chosen for the
  SHAP bar (matches UI/UX §4.2's "horizontal SHAP bar chart").
- **`accessibility-review` skill** — colour-independent encoding
  for direction; accessible text summary for the chart.

## Routes / Endpoints
No new routes/endpoints. The Flask `POST /predict` route already
returns the data; the FastAPI `POST /predict` contract is
unchanged (Spec 17 already returns `shap_contributions`).

## Data / Schema changes
No new tables, no new columns, no new model artifacts, no new
parquet writes. The `shap_contributions` field on
`PredictResponseV3` already exists. This spec consumes it.

## Templates / UI

**Modify:**
- `app/templates/predict_result.html` — replace the SHAP
  `<div class="chart-placeholder">…</div>` with:
  1. A direction-summary line: `"<N> factors pushed the price up,
     <M> pushed it down."` — driven by the SHAP `impact` sign
     count.
  2. A horizontal SHAP bar chart (Chart.js) inside a
     `<section class="why-this-price">` block, with
     `+` / `−` text labels paired with the green/red bars
     (Rules §6.4).
  3. An accessible `<caption>` / hidden text summary for screen
     readers (Rules §6.4 + accessibility-review skill).
  4. A model-version strip (small, below the hero): `"model:
     {model_version} · luxury: {luxury_category}"` — already
     rendered today as `price-hero__meta`; this spec leaves it
     as-is (no copy change) but verifies it's still above the
     fold with the new chart.

**No CSS file rename or new CSS file.** The new SHAP chart
styles and direction-summary styles are appended to
`app/static/css/predict.css` (same file Spec 18 created). One
file, same token system.

## Files to change / Files to create

**Create:**
- `app/services/shap_format.py` — small Flask-side formatter that
  turns the API's `{feature, impact}` pairs into the
  `{feature, label, impact, direction, pct}` shape the template
  iterates over. Public API:
  - `def format_shap_for_template(contributions: list[dict],
    *, top_n: int = 7) -> list[dict]:` — pure function. Steps:
    1. Iterate the input list (already sorted by `|impact|` desc
       from Spec 17's service).
    2. For each item, look up `label` via the in-process
       `FEATURE_LABEL_MAP_V2()` merged with the on-disk overlay
       (`load_label_map_from_disk("models")`). The map load is
       cached behind an `lru_cache(maxsize=1)` so we don't read
       the JSON on every request.
    3. Compute `direction = "up"` if `impact > 0` else `"down"`
       (zero is rendered as `"neutral"` to avoid an empty bar).
    4. Compute `pct = float(impact) / max_abs_impact` (a −1..1
       ratio the bar chart uses for bar length — keeps the bar
       scale stable regardless of currency). *(ponytail: chart
       uses the magnitude only, not the raw SHAP value, so the
       longest bar is always length 1.0 — simplest possible
       normalisation, no per-currency scaling needed.)*
    5. Fallback for unknown feature names: `label = feature` (the
       raw code is at least visible; better than an empty label).
  - `def summarize_direction(contributions: list[dict]) ->
    dict[str, int]:` — counts `{"up": n, "down": m}`. Pure,
    pinned by tests.
  - Module docstring cites Spec 17 (source response shape),
    Spec 16 (label map), Rules §6.4 (colour + text pairing),
    and the `chartjs-plotly-charting` skill (Chart.js horizontal
    bar pattern).
  - Zero imports from `ml.*`, `models.*`, `api.services.*`, or
    `api.routers.*` — only stdlib + the existing label-map
    functions. Pinned by a test that introspects
    `sys.modules`.

- `tests/test_shap_format.py` — pytest tests for the formatter.
  Required tests (exact names):
  - `test_format_shap_for_template_assigns_label_from_map` —
    `feature="num__built_up_area"` → `label="Built-up Area
    (sqft)"`.
  - `test_format_shap_for_template_assigns_label_from_disk_overlay`
    — stub the on-disk JSON via `monkeypatch` (write a tmp
    overlay file, point the helper at `tmp_path`); assert the
    overlay's label wins over the static map.
  - `test_format_shap_for_template_falls_back_to_raw_feature_name`
    — an unknown `feature="num__exotic_thing"` gets
    `label="num__exotic_thing"` (no crash, no empty label).
  - `test_format_shap_for_template_marks_direction_up_for_positive`
  - `test_format_shap_for_template_marks_direction_down_for_negative`
  - `test_format_shap_for_template_marks_direction_neutral_for_zero`
  - `test_format_shap_for_template_caps_at_top_n` — input has 12
    entries; `top_n=7` returns 7.
  - `test_format_shap_for_template_pct_normalises_to_max_abs` —
    largest impact has `|pct| == 1.0`; the rest scale linearly.
  - `test_format_shap_for_template_preserves_input_order` — the
    list order is unchanged (input already pre-sorted by the
    service).
  - `test_format_shap_for_template_does_not_import_ml_or_models`
    — `sys.modules` check.
  - `test_summarize_direction_counts_up_and_down` — 3 up, 2
    down → `{"up": 3, "down": 2}`.
  - `test_summarize_direction_handles_empty_list` — `[]` →
    `{"up": 0, "down": 0}`.
  - `test_summarize_direction_ignores_zero_impact` — three
    items, one with `impact=0.0`; the count reflects only the
    non-zero entries (matches `direction = "neutral"`).

- `tests/test_predict_route.py` — add new tests for the
  SHAP chart rendering. Required tests (exact names; existing
  tests in the file remain unchanged):
  - `test_predict_post_renders_shap_chart_for_each_contribution`
    — monkeypatches `post_predict` to return a response with
    3 SHAP entries; asserts the rendered HTML contains:
    - the `chart-placeholder` text is gone (replaced by a
      `<canvas id="shap-chart">`),
    - the chart's `data-*` attributes (or a JSON
      `<script>` blob) include the `label` strings,
    - a visible `+` or `−` character for each entry.
  - `test_predict_post_renders_direction_summary_line` — same
    canned response; asserts the HTML contains
    `"<N> factors pushed the price up, <M> pushed it down"`
    with the right counts.
  - `test_predict_post_renders_chart_even_when_shap_empty` —
    canned response with `shap_contributions=[]`; page still
    renders (no 500), shows a friendly empty state inside the
    chart section ("No contribution breakdown available for
    this prediction"). Pinned per Rules §6.1's spirit.
  - `test_predict_post_renders_accessible_text_summary` — the
    rendered HTML contains an `aria-label` or visually hidden
    text describing the top SHAP feature, so a screen reader
    gets the same information as the chart.
  - `test_predict_post_does_not_expose_raw_feature_codes_to_user`
    — the raw `num__built_up_area` code does **not** appear in
    the visible page text (only the human label does); the
    raw code may appear in a `<script>` JSON blob (machine
    data is fine; visible text must be human-friendly).

**Modify:**
- `app/templates/predict_result.html` — replace the
  `<div class="chart-placeholder">…</div>` with the new
  block:
  ```
  <section class="why-this-price">
    <h2>Why this price?</h2>
    {% if shap_rows %}
      <p class="why-this-price__summary">
        <strong>{{ shap_summary.up }}</strong> factors pushed the price up,
        <strong>{{ shap_summary.down }}</strong> pushed it down.
      </p>
      <canvas id="shap-chart"
              role="img"
              aria-label="Top {{ shap_rows|length }} SHAP feature contributions for this prediction: {{ shap_rows|map(attribute='label')|join(', ') }}."
              data-rows="{{ shap_rows|tojson }}"></canvas>
      <ul class="shap-text-list" hidden>
        {% for row in shap_rows %}
          <li>
            {% if row.direction == 'up' %}+{% elif row.direction == 'down' %}−{% else %}±{% endif %}
            {{ row.label }}
            {% if row.direction != 'neutral' %}
              ({{ (row.pct * 100)|round(1) }}%)
            {% endif %}
          </li>
        {% endfor %}
      </ul>
    {% else %}
      <p class="empty-state__inline">No contribution breakdown available for this prediction.</p>
    {% endif %}
  </section>
  ```
  A tiny inline `<script>` at the bottom of the template (same
  pattern Spec 18 used for the dependent dropdown) reads the
  `data-rows` attribute and renders the Chart.js horizontal bar.
  The `<ul class="shap-text-list" hidden>` is the accessible
  text fallback (the `aria-label` gives the summary; the hidden
  list gives the per-row detail). *(ponytail: Chart.js is the
  cheapest option that already lives in the CDN choice from the
  UI/UX doc; no new dependency. No React, no build step.)*

- `app/app.py` — extend the `predict_post` view to pass the
  formatted SHAP rows to the template. Concrete additions:
  - Inside the success branch (after `response = client.post_predict(...)`):
    1. `shap_rows = format_shap_for_template([c.model_dump() for c in response.shap_contributions])`
    2. `shap_summary = summarize_direction(shap_rows)`
  - Add `shap_rows=shap_rows` and `shap_summary=shap_summary` to
    the `render_template("predict_result.html", ...)` kwargs.
  - The `unavailable=True` branch is unchanged (no SHAP data
    possible when FastAPI is down).
  - New import: `from app.services.shap_format import
    format_shap_for_template, summarize_direction`.
  - Module docstring updated to cite Spec 19 (the renderer)
    alongside Spec 17 (the source) and Rules §6.4.

- `app/services/__init__.py` — re-export
  `format_shap_for_template`, `summarize_direction` from
  `shap_format` so `app/app.py` can import them via
  `app.services`. Same re-export pattern Spec 18 established
  for `FastAPIClient` / `FastAPIUnavailable` / `inr_format`.

- `app/static/css/predict.css` — append three new layout rules
  + one token. No hardcoded hex; only `var(--color-*)` /
  `var(--space-*)` references:
  - `.why-this-price__summary` — centred, muted colour,
    one line, sits above the chart.
  - `#shap-chart` — `max-width: 100%`, `height: 320px`,
    padding `var(--space-2)`.
  - `.shap-text-list[hidden]` — leave as default `display:
    none` (the `<ul>` is screen-reader-only).
  - The empty-state copy uses the existing `.empty-state__inline`
    class (already a single-line muted text block); if it
    doesn't exist yet, append it here — `color: var(--color-muted);
    text-align: center; padding: var(--space-3);`.
  - Update the file's header comment to cite Spec 19 (SHAP
    chart styles) alongside Spec 18 (form styles).

- `app/static/css/style.css` — add three CSS variables to the
  `:root` block (the chart's positive / negative bars +
  the empty-state text colour). All HSL, matching the
  existing token style. *(ponytail: extend the token system
  once, here — never inline per component.)*
  - `--color-pos: hsl(140, 60%, 40%)` (green for "up" bars).
  - `--color-neg: hsl(0, 70%, 50%)` (red for "down" bars).
  - `--color-empty: var(--color-muted)` (alias for the inline
    empty-state — keeps the chart section visually consistent
    with the rest of the page).

**No changes** to:
- `requirements.txt` — Chart.js is loaded via CDN at template
  render time, not via npm/pip. (CDN choice matches UI/UX
  §2's "no build step" rule.)
- `api/`, `ml/`, `models/`, `data/`, `migrations/`,
  `notebooks/` — strictly a Flask-side rendering change.
- `tests/conftest.py` — existing fixtures reused.
- `CLAUDE.md`'s "Implemented vs stub routes" table — no route
  status changes (Flask `POST /predict` was already implemented
  by Spec 18).

## New dependencies
**No new dependencies.** Chart.js is a CDN script tag in the
template (no npm install). The helper uses stdlib only +
`functools.lru_cache` (already used by `app.services.fastapi_client`).
`pydantic`'s `BaseModel.model_dump` is already used elsewhere
in the same Flask layer (Spec 18).

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no DB writes from this spec.
- **No dealer/contact/media-URL fields ever reach the UI or an
  export.** The SHAP feature names come from the preprocessor's
  transformed-feature column names (e.g. `num__built_up_area`,
  `num__sector_smoothed_price`); none of them match the
  `(contact|dealer|phone|email|photo|url|spid)` regex. Pinned
  by the existing
  `test_predict_service_predict_does_not_log_pii_fields` from
  Spec 17 (no new PII surface introduced).
- **CSS variables only, never hardcoded hex values.** All new
  styles live in `predict.css` and reference `var(--color-*)`
  / `var(--space-*)` tokens; the three new tokens added to
  `style.css` are HSL. Pinned by a `grep` check for hex
  literals in `predict.css` (existing convention).
- **All templates extend `base.html`.** `predict_result.html`
  already extends `base.html` (Spec 18); no change.
- **Model changes must reference the fixed evaluation protocol.**
  N/A — no model changes. The SHAP values come from the
  already-certified v2 pipeline (Spec 15's gate).
- **Flask never imports model code (Rules §5.1).**
  `app/services/shap_format.py` imports only stdlib +
  `ml.explainability.labels` (a *display* helper, not model
  code — Spec 16 already established this import path). It
  does **not** import from `ml.explainability.contributions`,
  `ml.explainability.explainer`, `ml.training.*`, or any
  model `.pkl` loader. Pinned by
  `test_format_shap_for_template_does_not_import_ml_or_models`.
- **Graceful degradation on FastAPI failure (Rules §5.2).**
  Already handled by Spec 18's `unavailable=True` branch; this
  spec does not introduce any new failure mode. Empty SHAP
  list (FastAPI returned `[]` for some reason) renders the
  inline "No contribution breakdown available" state — Rules
  §6.1 spirit.
- **Colour never the sole carrier of meaning (Rules §6.4).**
  Each SHAP bar carries a `+` / `−` / `±` text label in the
  hidden `<ul>` AND in the chart's tooltip config (Chart.js
  `plugins.tooltip.callbacks.label` returns the text). The
  direction-summary line above the chart is plain English.
- **Accessibility (accessibility-review skill).** The
  `<canvas>` carries a meaningful `role="img"` + `aria-label`
  listing the top feature labels; the `<ul class="shap-text-
  list" hidden>` gives screen readers the per-row detail.
  Colour contrast for `--color-pos` and `--color-neg` on
  `--color-card-bg` is verified against WCAG AA in the
  stylesheet's header comment (manual note — no automated
  contrast check in CI per the existing project convention).
- **No `api/services/` imports.** The Flask side imports only
  the `predict_v3` schema module for type hints / model
  validation; the new helper doesn't touch the schema
  directly (it accepts a plain `list[dict]` so it stays
  decoupled from `ShapContribution`).
- **All randomness is seeded.** N/A — no randomness in the
  formatter.
- **CDN integrity.** Chart.js is loaded from the same CDN the
  UI/UX doc already references (`https://cdn.jsdelivr.net/npm/
  chart.js@4`). No new external origin.
- **HTML escaping.** All template variables in
  `predict_result.html` use Jinja's default auto-escape; the
  `data-rows` attribute is populated with `|tojson`, which
  Jinja emits as a safe-quoted JSON string (not user input,
  so no XSS surface). The `<ul>` items use `{{ row.label }}`,
  which is auto-escaped. Pinned by a manual code-review pass.
- **No notebook-only steps.** Everything is reproducible via
  `python app/app.py` + `uvicorn api.main:app --reload` from
  repo root.

## Definition of done

1. `python -m pytest tests/test_shap_format.py -v` from repo
   root runs and passes. Tests required (exact names):
   - `test_format_shap_for_template_assigns_label_from_map`
   - `test_format_shap_for_template_assigns_label_from_disk_overlay`
   - `test_format_shap_for_template_falls_back_to_raw_feature_name`
   - `test_format_shap_for_template_marks_direction_up_for_positive`
   - `test_format_shap_for_template_marks_direction_down_for_negative`
   - `test_format_shap_for_template_marks_direction_neutral_for_zero`
   - `test_format_shap_for_template_caps_at_top_n`
   - `test_format_shap_for_template_pct_normalises_to_max_abs`
   - `test_format_shap_for_template_preserves_input_order`
   - `test_format_shap_for_template_does_not_import_ml_or_models`
   - `test_summarize_direction_counts_up_and_down`
   - `test_summarize_direction_handles_empty_list`
   - `test_summarize_direction_ignores_zero_impact`
2. `python -m pytest tests/test_predict_route.py -v` from repo
   root runs and passes — **both** the Spec 18 tests (all
   unchanged) and the new Spec 19 tests:
   - `test_predict_post_renders_shap_chart_for_each_contribution`
   - `test_predict_post_renders_direction_summary_line`
   - `test_predict_post_renders_chart_even_when_shap_empty`
   - `test_predict_post_renders_accessible_text_summary`
   - `test_predict_post_does_not_expose_raw_feature_codes_to_user`
3. `python -m pytest -m "not realdata"` from repo root still
   passes — no real-data dependency introduced.
4. `ruff check app/app.py app/services/shap_format.py
   app/templates/predict_result.html tests/test_shap_format.py
   tests/test_predict_route.py` reports zero issues.
5. `python -c "from app.services.shap_format import
   format_shap_for_template, summarize_direction; print('ok')"`
   from repo root prints `ok` — public API imports cleanly.
6. With Flask running (`python app/app.py`) and FastAPI running
   (`uvicorn api.main:app --port 8000`), a `curl -L -X POST
   http://localhost:5000/predict -d "city=Gurgaon
   &sector=sector 84&property_type=flat&transact_type=Sale
   &bedRoom=3&bathroom=3&balcony=2&agePossession=Relatively New
   &built_up_area=1450&servant_room=1&furnishing_type=Semifurnished
   &floor_category=Mid Floor&facing=North&amenities=Swimming Pool
   &amenities=Club house"` returns 200 with HTML containing:
   - the price hero + range (Spec 18's contract — still
     present),
   - the line `"<N> factors pushed the price up, <M> pushed
     it down."` with the right counts,
   - a `<canvas id="shap-chart" …>` element with a non-empty
     `data-rows` JSON blob,
   - a hidden `<ul class="shap-text-list">` with one `<li>`
     per SHAP contribution, each carrying a `+`/`−`/`±`
     character and a human-readable label (no raw
     `num__*` codes visible in the `<li>` text).
   Manual smoke test of the rendered SHAP UI.
7. With Flask running but FastAPI stopped, the same POST as
   in step 6 returns 200 with HTML containing the existing
   "Prediction service is temporarily unavailable" copy
   (Spec 18's behaviour) — the SHAP chart code is not
   reached, no JS errors. Manual smoke test of the
   degradation path (still passes after Spec 19).
8. `git status` after committing shows only the new files
   listed above, the modified `app/app.py`, the modified
   `app/templates/predict_result.html`, the modified
   `app/services/__init__.py`, the modified
   `app/static/css/predict.css`, and the modified
   `app/static/css/style.css`. No accidental additions to
   `api/`, `data/`, `models/`, `migrations/`, or
   `requirements.txt`.
9. `CLAUDE.md`'s "Implemented vs stub routes" table is
   **unchanged** — no new routes added; both `/predict`
   rows stay **Implemented**.
10. `07-TRACKER.md` is updated via `/update-tracker` to mark
    Day 38 ("Price Prediction form + result page + SHAP
    chart") as **Done** with the actual date and a one-line
    summary: "SHAP bar chart + direction summary + accessible
    text fallback wired into `predict_result.html`; inline
    VerdictBadge / AffordabilityChip and inline insight cards
    remain follow-on specs (require `/classify` and `/insights`
    FastAPI routes, still stubs)." The Decision Log gets
    one new entry: "Reused `FEATURE_LABEL_MAP_V2` from
    `ml/explainability/labels` on the Flask side rather than
    extending the FastAPI `ShapContribution` schema to add
    `label`/`direction` — keeps the wire format minimal
    (Backend Schema §7 unchanged) and the rendering logic
    in one Flask-side helper."
