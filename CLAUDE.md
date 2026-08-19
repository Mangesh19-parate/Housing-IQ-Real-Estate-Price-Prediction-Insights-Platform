# CLAUDE.md

## Project overview

HousingIQ is an India real estate price prediction & insights platform
covering 4 cities (Gurgaon, Hyderabad, Kolkata, Mumbai), built on a scraped
listings dataset (~182k rows across 4 city CSVs + 15 facet lookup tables).
It ships **5 modules**: Price Prediction, Classification (Affordability &
Investment-Tier Filter), Analytics, Recommender, and Insights — served
through a Flask web app backed by a FastAPI inference microservice.

This is a v1, portfolio-scale build: **no user accounts, no authentication,
no login/session state, no real-time scraping, no payment/lead-gen flows.**
Every page is public. Don't add auth scaffolding unless a future spec
explicitly introduces it.

---

## Architecture

```
housingiq/
├── data/
│   ├── raw/                        # gurgaon_10k.csv, hyderabad.csv, kolkata.csv,
│   │                                # mumbai.csv, facets/*.csv — immutable, never written to
│   ├── processed/
│   │   ├── clean_listings.parquet  # canonical cleaned dataset, all 4 cities
│   │   ├── analytics_cache/*.json  # precomputed chart data, keyed by city
│   │   └── feature_selection_report.md
│   └── stats/                      # locality_stats, amenity_uplift,
│                                    # age_price_trend, bhk_price_trend (Insights source)
├── notebooks/                      # EDA, ydata-profiling reports (offline artifacts only)
├── ml/
│   ├── cleaning/                   # parse_price(), parse_area(), parse_map_details(), facet decoding
│   ├── features/                   # canonical schema mapping, feature engineering
│   ├── training/                   # price regression + classification training scripts
│   ├── evaluation/                 # metrics computation, SHAP explainer generation
│   └── recommender/                # TF-IDF fit + NearestNeighbors index build
├── models/                         # versioned artifacts — see "Model artifacts" below
├── api/                            # FastAPI — model serving ONLY, never page rendering
│   ├── main.py
│   ├── routers/
│   │   ├── predict.py              # POST /predict
│   │   ├── classify.py             # POST /classify
│   │   ├── analytics.py            # GET /analytics/*
│   │   ├── recommend.py            # POST /recommend
│   │   └── insights.py             # GET /insights
│   ├── schemas/                    # Pydantic request/response models
│   └── services/                   # model-loading + inference glue, called by routers
├── app/                            # Flask — pages, forms, rendering ONLY
│   ├── app.py
│   ├── database/
│   │   └── db.py                   # SQLite/Postgres helpers: get_db(), init_db()
│   ├── templates/
│   │   ├── base.html               # shared layout — all templates must extend this
│   │   ├── landing.html
│   │   ├── predict.html / predict_result.html
│   │   ├── classify.html
│   │   ├── analytics.html
│   │   ├── recommend.html
│   │   ├── insights.html
│   │   └── map_explorer.html
│   └── static/
│       ├── css/                    # style.css (global tokens) + one file per page's layout-only rules
│       └── js/                     # main.js, charts.js, map.js — vanilla JS, no build step
├── tests/
│   ├── test_price_prediction.py
│   ├── test_classification.py
│   ├── test_analytics.py
│   ├── test_recommender.py
│   ├── test_insights.py
│   └── conftest.py
└── requirements.txt
```

**Where things belong:**
- Model inference logic → `api/` only, called by Flask over internal HTTP — Flask **never** imports model code or touches `.pkl` files directly (this is a binding architectural rule, not a preference; see TRD).
- New FastAPI routes → `api/routers/<module>.py`, with a matching Pydantic schema in `api/schemas/`
- New Flask pages → `app/app.py` route + new template extending `base.html`
- DB logic → `app/database/db.py` only, never inline in a Flask route
- Page-specific styles → new `.css` file for layout only; shared colors/spacing are CSS variables in `style.css`, never hardcoded hex/pixel values inline
- Data cleaning/parsing → `ml/cleaning/`, never inline in a notebook that isn't reproducible

---

## Data model (canonical fields)

The Price Prediction and Classification modules take these **12 authoritative
user-facing fields** (see `10-FINALIZED-INPUT-SCHEMA.md` for the full v3
contract, 16 fields total including derived ones): Property Type, City +
Sector/Locality, Bedrooms, Bathrooms, Balconies, Built-up Area (sqft),
Property Age/Possession, Furnishing Type, Facing Direction, Floor Category,
Servant Room (Y/N), Store Room (Y/N), plus a derived Luxury Category. Any
code referencing a listing attribute must use these exact canonical names —
no ad hoc renaming per module or per layer.

`sector`/`locality` is always paired with `city` — never treated as globally
unique across cities.

---

## Code style

- Python: PEP 8, snake_case for all variables and functions
- FastAPI: every endpoint has an explicit Pydantic request model and response model — no bare dicts, no untyped `dict` params
- Flask/Jinja2: `url_for()` for every internal link — never hardcode a path string
- Flask route functions: one responsibility only — fetch/call, render template, done; no inline business logic
- SQL: always parameterized (`?` placeholders) — never f-strings or `.format()` into a query, in either `app/database/db.py` or any logging path
- Model code (`ml/`): every trained model is evaluated with the fixed protocol (see below) before it's allowed into `models/`
- Error handling: Flask uses `abort()` for HTTP errors; FastAPI relies on Pydantic validation errors and explicit `HTTPException`s, never a bare string response

---

## Tech constraints

- **Serving split is fixed:** FastAPI (Uvicorn) for model inference, Flask (Jinja2) for pages/forms/rendering. Don't blur this — no model loading in Flask, no HTML rendering in FastAPI.
- **Frontend:** HTML5/CSS3/vanilla JS (fetch API) only — no React, no build step. Chart.js/Plotly.js for charts, Leaflet.js for the map/spatial view, both via CDN.
- **Persistence:** SQLite in dev, Postgres-compatible schema for the prod path. The application DB holds `prediction_log`, `recommendation_log`, `classification_log`, and a lightweight `model_registry` — operational/logging tables only, not user data (there are no user accounts).
- **Model artifacts:** joblib/pickle, versioned filenames (`price_model_v{n}.pkl`, never overwritten in place).
- **ML stack:** pandas, numpy, scikit-learn, XGBoost/LightGBM, SHAP (TreeExplainer), ydata-profiling (offline only, not served live).
- **No new pip/npm packages** without checking `requirements.txt` first and flagging the addition explicitly.
- Python 3.10+ assumed.

---

## Fixed model evaluation protocol (non-negotiable)

- Split: 70/15/15 train/val/test, `random_state=42` — regenerated deterministically from `clean_listings.parquet`, never an ad hoc sample.
- Regression (Price Prediction): R², MAE, RMSE, MAPE, reported on the **original ₹ price scale** even when the model trains on `log1p(price)`.
- Classification: accuracy, per-class precision/recall/F1, confusion matrix — accuracy alone is not sufficient given expected class imbalance.
- Target thresholds from the PRD: R² ≥ 0.80 (stretch 0.85), MAE within ±15% of actual price for 70% of test listings, `/predict` p95 latency < 300ms.
- No model is "done" until scored against this protocol — see the `housingiq-ml-evaluator` agent.

---

## Data & privacy rules (binding, from `08-RULES.md`)

- Raw CSVs under `data/raw/` are immutable — no code path ever writes to them.
- Dealer/agent contact fields, phone-like fields, and raw photo/media URLs are dropped at cleaning and must **never** reach a template, API response, or export — including via a later join that re-introduces them.
- Outlier rows are flagged (`is_outlier`), never deleted — excluded from training, retained in the analytics store.
- Every derived table or cache file states its computation date and source dataset version.
- No collaborative filtering / personalized recommendations — the Recommender is content-based only (no user interaction history exists in the dataset); cold-start fallback ranks by locality popularity + recency and is labeled "Popular in this area" in the UI, never mislabeled as "Most similar."

---

## Subagent policy

- Always use the `housingiq-test-writer` agent to add test coverage from a spec, before or immediately after implementation — not by reverse-engineering tests from the code.
- Always use the `housingiq-test-runner` agent to verify test results after any implementation.
- Always use the `housingiq-quality-reviewer` and `housingiq-security-reviewer` agents before merging a feature branch.
- Always use the `housingiq-ml-evaluator` agent before any trained model is referenced by the API or copied into `models/` for real use.
- Always research the codebase (relevant specs, skills, and existing patterns) before implementing any new feature — don't guess at conventions that are already documented.

---

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the Flask web app (dev)
python app/app.py

# Run the FastAPI inference service (dev)
uvicorn api.main:app --reload --port 8000

# Run all tests
pytest

# Run a specific test file
pytest tests/test_price_prediction.py

# Run a specific test by name
pytest -k "test_name"

# Run tests with output visible
pytest -s

# Rebuild the analytics cache
# (see .claude/commands/build-analytics-cache.md)

# Train the price model
# (see .claude/commands/train-price-model.md)
```

---

## Implemented vs stub routes

| Route | Status |
|---|---|
| `GET /` | Implemented (Step 01) — landing page with city quick-filter and module cards |
| `GET /predict` | Stub — Price Prediction form |
| `POST /predict` (Flask → calls FastAPI `POST /predict`) | Stub |
| `GET /classify` | Stub — Classification form (Affordability & Investment-Tier Filter) |
| `POST /classify` (Flask → calls FastAPI `POST /classify`) | Stub |
| `GET /analytics` | Stub — dashboard shell, 13 chart tiles (5 originally specified + 8 added) |
| `GET /recommend` | Stub — Recommender form + results grid |
| `POST /recommend` (Flask → calls FastAPI `POST /recommend`) | Stub |
| `GET /insights` | Stub — standalone Insights page (also surfaced inline after a prediction) |
| `GET /map` | Stub — map-based property explorer (Leaflet) |
| `POST /predict` (FastAPI) | Implemented (Spec 17) — v2 `_SerializableV2Pipeline` + precomputed SHAP explainer, server-derived luxury category, parameterized `prediction_log` insert |
| `POST /classify` (FastAPI) | Stub |
| `POST /recommend` (FastAPI) | Stub |
| `GET /insights` (FastAPI) | Stub |
| `GET /health` (FastAPI) | Implemented (Step 01) — returns `{"status": "ok"}`; no DB ping, no model load |

Update this table as each spec in `.claude/specs/` is implemented — it should
always reflect what's actually true in the code, not the plan.

---

## Roadmap reference

Full week-by-week plan lives in `06-IMPLEMENTATION-PLAN.md` (7 weeks: data
cleaning → EDA/outliers → feature engineering/selection → model
training/productionization → recommender + insights → analytics
precompute/Flask wiring → testing/polish/deployment). Actual progress is
tracked in `07-TRACKER.md` — keep it honest (real dates, real results,
noted deviations) via `/update-tracker`, not aspirational.