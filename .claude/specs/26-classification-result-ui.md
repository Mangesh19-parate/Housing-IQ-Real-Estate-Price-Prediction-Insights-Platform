# Spec: Classification Result UI

## Overview
Build the Flask UI layer for the Classification module — a standalone `/classify` form page (reusing the shared `PropertyForm` component) and a result page that displays the `good_deal_verdict` as the primary headline (`VerdictBadge`), the `price_tier` as a secondary `AffordabilityChip`, a horizontal 4-bar probability chart for all price tiers, and a SHAP contribution breakdown. This implements the "Good Deal first" visual hierarchy from UI/UX Update v4 (§U-UX-9/10) and wires the existing FastAPI `POST /classify` endpoint (Spec 25) into the Flask web app.

Module: **classification**.

## Depends on
- **Spec 18** (`18-price-prediction-flask-form-page.md`) — provides the shared `PropertyForm` component and `app/services/property_form.py` helper used by both `/predict` and `/classify`.
- **Spec 19** (`19-price-prediction-result-and-explanation-ui.md`) — provides the `ShapBarChart` component and `app/services/shap_format.py` helpers reused here.
- **Spec 25** (`25-classification-fastapi-endpoint.md`) — provides the FastAPI `POST /classify` route returning `ClassifyResponseV1` with `good_deal_verdict`, `verdict_probabilities`, `price_tier`, `tier_probabilities`, `shap_contributions`, `model_version`, `luxury_category`.
- **Spec 11** (`11-price-prediction-input-schema-v3.md`) — the 16-field input contract (same for both modules).
- **UI/UX Design Update v4** (`docs/04-UI-UX-DESIGN.md` §U-UX-9/10/11) — `VerdictBadge`, `AffordabilityChip`, page heading "Is this a good deal?", probability chart, copy changes.
- **`docs/05-BACKEND-SCHEMA.md` §U-SCHEMA-10** — API response contract (verdict leads).
- **`docs/08-RULES.md` §5.1/5.2** — Flask never loads models; FastAPI degrades gracefully; classification failure must not block price prediction.

## Routes / Endpoints
- **Flask:** `GET /classify` — renders the classification form page (reuses `PropertyForm`, button label "Check Price Tier" → updated to "Is this a good deal?"). Access: public.
- **Flask:** `POST /classify` — receives form data, calls FastAPI `POST /classify`, renders `classify_result.html` with verdict badge, tier chip, probability chart, SHAP chart. Access: public.
- **FastAPI:** `POST /classify` — already implemented (Spec 25). Called by Flask over internal HTTP.

## Data / Schema changes
- **No new application DB tables.** `classification_log` already exists (Backend Schema §U-SCHEMA-11).
- **No new model artifacts.** Consumes `good_deal_classifier_v1.pkl`, `price_tier_classifier_v1.pkl`, `classifier_preprocessor_v1.pkl` from Spec 23.
- **No `data/raw/` or `data/processed/` writes.**
- **No new SQL migrations.**
- **No `analytics_cache/*.json` additions.**

## Templates / UI
- **Create:** `app/templates/classify.html` — classification form page. Extends `base.html`. Reuses `PropertyForm` macro/component from `predict.html` (or a shared partial). Heading: "Is this a good deal?". Primary button: "Check Price Tier". Form submits to `POST /classify`.
- **Create:** `app/templates/classify_result.html` — classification result page. Extends `base.html`. Layout per UI/UX §U-UX-10:
  - Hero section: large `VerdictBadge` (Good Deal ✅ / Fair Price ⚖️ / Overpriced ⚠️) with color + icon + text (never color alone). Price tier shown as smaller `AffordabilityChip` (Budget/Mid-Range/Premium/Luxury) below or beside it.
  - "Why this verdict?" — horizontal SHAP bar chart (reuse `ShapBarChart` component, top-N contributions from `shap_contributions`).
  - "Price Tier Probabilities" — horizontal grouped bar chart (4 bars: Budget, Mid-Range, Premium, Luxury) showing `tier_probabilities` values. All four tiers shown, not just the winner (trust signal: 52% Premium should visibly show it was close to Mid-Range).
  - Graceful degradation: if FastAPI `/classify` fails (timeout, 503, 500), the page still renders but omits the `VerdictBadge`/`AffordabilityChip`/probability chart/SHAP section — shows a friendly "Classification temporarily unavailable" inline notice instead. The page never shows a raw error.
  - CTA: "See similar properties →" linking to `/recommend` with the same input features pre-filled (stretch).
- **Modify:** `app/templates/predict_result.html` — add `AffordabilityChip` below the price hero (already has `TierBadge` from Spec 19, now split into `VerdictBadge` + `AffordabilityChip` per UI/UX Update v4). If classification service unavailable, the chip slot disappears silently (no error banner).

## Files to change / Files to create

**Create:**
- `app/templates/classify.html` — classification form page.
- `app/templates/classify_result.html` — classification result page.
- `app/static/css/classify.css` — page-specific layout styles only (CSS variables from `style.css` for colors/spacing).
- `app/static/js/classify.js` — vanilla JS for: form validation (reuses `main.js` helpers), probability chart rendering (Chart.js horizontal bar), graceful degradation handling.
- `app/services/classify_client.py` — thin wrapper around `httpx`/`requests` to call FastAPI `POST /classify` with timeout, retry, and error normalization. Returns parsed `ClassifyResponseV1`-like dict or raises a typed exception the Flask route catches for degradation.

**Modify:**
- `app/app.py` — add `GET /classify` and `POST /classify` route handlers. `POST` calls `classify_client.classify()`, on success renders `classify_result.html`, on failure renders `classify_result.html` with `classification_error=True` context flag.
- `app/templates/base.html` — add `/classify` link to the module card grid (landing page) and nav if not already present.
- `app/templates/predict_result.html` — replace `TierBadge` with `VerdictBadge` + `AffordabilityChip` pair (partial macro in `app/templates/_badges.html` or inline).
- `app/static/css/style.css` — add CSS variable definitions for `VerdictBadge` colors (Good Deal: green `#16A34A`, Fair Price: blue `#2563EB`, Overpriced: amber `#F59E0B` / red `#DC2626`) and `AffordabilityChip` colors (Budget: `#16A34A`, Mid-Range: `#2563EB`, Premium: `#7C3AED`, Luxury: `#D97706`) per UI/UX §U-UX-1. No hardcoded hex values in templates.
- `app/services/shap_format.py` — ensure `format_shap_for_template` works for classification SHAP (same shape as price SHAP: list of `{feature, impact}`).

**No changes to:**
- `api/`, `models/`, `ml/`, `data/`, `migrations/`, `requirements.txt`.

## New dependencies
**No new dependencies.** `httpx` or `requests` already in `requirements.txt` (from Spec 17/18). Chart.js via CDN (already used by analytics/prediction). No `pip install` required.

## Rules for implementation
- **No SQLAlchemy/ORM.** All DB access via `app/database/db.py` parameterized queries (not needed here — classification log write happens in FastAPI, not Flask).
- **No dealer/contact/media-URL fields ever reach the UI or an export.** The form and result page only handle the 16 input fields + classification output.
- **CSS variables only, never hardcoded hex values.** All badge/chart colors defined in `style.css` as `--verdict-good-deal`, `--verdict-fair-price`, `--verdict-overpriced`, `--tier-budget`, `--tier-mid-range`, `--tier-premium`, `--tier-luxury`.
- **All templates extend `base.html`.** No standalone HTML.
- **Model changes must reference the fixed evaluation protocol.** The UI only displays what FastAPI returns; no model logic in Flask.
- **Flask never imports model code.** `app/services/classify_client.py` only does HTTP to FastAPI.
- **Graceful degradation (Rules §5.2).** FastAPI failure → friendly inline notice, not a 500 page. Price prediction result page must still render if classification fails (parallel call pattern from UML Sequence Diagram §3).
- **Accessibility (UI/UX §10).** Badges always carry text + icon, never color alone. Charts have text-summary fallback (hidden `<ul>` with screen-reader labels). Form fields have associated `<label>`s.
- **Reuse over duplication.** `PropertyForm` component shared with `/predict`. `ShapBarChart` component shared with `/predict/result`. Probability chart uses same Chart.js config pattern as analytics tiles.
- **URLs via `url_for()` only.** Never hardcode `/classify` or `/predict` paths in templates or JS.

## Definition of done
1. `python -m pytest tests/test_classify_route.py -v` from repo root runs and passes. Required tests (exact names):
   - `test_classify_get_renders_form_page` — `GET /classify` returns 200, template extends `base.html`, contains "Is this a good deal?" heading, form with all 16 fields, button labeled "Check Price Tier".
   - `test_classify_post_valid_payload_renders_result` — `POST /classify` with valid 16-field payload → 200, renders `classify_result.html` with `VerdictBadge`, `AffordabilityChip`, probability chart canvas, SHAP chart canvas.
   - `test_classify_post_verdict_badge_matches_response` — the rendered `VerdictBadge` text/class matches `good_deal_verdict` from the mocked FastAPI response (Good Deal → green/check, Fair Price → blue/scale, Overpriced → amber/warning).
   - `test_classify_post_affordability_chip_matches_response` — the rendered `AffordabilityChip` text/class matches `price_tier` from the mocked response.
   - `test_classify_post_probability_chart_shows_all_four_tiers` — the probability chart data includes all 4 tiers from `tier_probabilities`, not just the argmax.
   - `test_classify_post_shap_chart_renders_top_n` — SHAP chart renders `SHAP_TOP_N` (7) bars from `shap_contributions`.
   - `test_classify_post_graceful_degradation_on_fastapi_503` — when `classify_client` raises `FastAPIUnavailable`, the result page renders with "Classification temporarily unavailable" notice and **no** `VerdictBadge`/`AffordabilityChip`/charts; no 500 error.
   - `test_classify_post_graceful_degradation_on_timeout` — same as above for `httpx.TimeoutException`.
   - `test_classify_post_no_pii_in_template_context` — grep the rendered HTML for `(contact|dealer|phone|email|photo|url|spid)` — must be absent.
   - `test_classify_post_uses_url_for_for_cta_links` — any internal links in the result page use `url_for()` (checked by grepping for hardcoded `/predict`, `/recommend`, etc. in the template).
2. `python -m pytest tests/test_predict_result_integration.py -v` passes — verifies `predict_result.html` now shows `VerdictBadge` + `AffordabilityChip` (not the old single `TierBadge`) when classification succeeds, and omits the chip silently when classification fails.
3. `ruff check app/templates/classify.html app/templates/classify_result.html app/app.py app/services/classify_client.py app/static/css/classify.css app/static/js/classify.js app/static/css/style.css` reports zero issues.
4. Manual smoke test: `python app/app.py` starts Flask; navigate to `http://localhost:5000/classify` — form loads, submit a valid 3BHK Gurgaon payload → result page shows verdict badge, tier chip, 4-bar probability chart, SHAP chart. Stop FastAPI → submit again → page renders with degradation notice, no crash.
5. `git status` after committing shows only the new files listed above, the modified `app/app.py`, `app/templates/base.html`, `app/templates/predict_result.html`, `app/static/css/style.css`, and `app/services/shap_format.py`. No accidental additions to `api/`, `models/`, `data/`, `migrations/`, or `requirements.txt`.
6. `CLAUDE.md`'s "Implemented vs stub routes" table is updated: `GET /classify` and `POST /classify` (Flask) rows move from **Stub** to **Implemented**.
7. `docs/07-TRACKER.md` is updated via `/update-tracker` to mark Day 55 ("Flask UI: VerdictBadge + AffordabilityChip, /classify page, Recommender tier filter, Analytics tile 14") as **Done** with the actual date and a one-line summary. The Decision Log gets one new entry: "Split TierBadge into VerdictBadge (primary) + AffordabilityChip (secondary) per UI/UX Update v4 good-deal-first hierarchy. Classification page heading changed from 'Check Price Tier' to 'Is this a good deal?'."