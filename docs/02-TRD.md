# Technical Requirements Document (TRD)
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

---

## 1. Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| ML training / data pipeline | Python, pandas, numpy, scikit-learn, XGBoost/LightGBM, SHAP, matplotlib/seaborn, ydata-profiling (pandas-profiling) | Standard, well-documented, matches reference projects researched |
| Model serving (inference API) | **FastAPI** + Uvicorn | Async, Pydantic validation, auto OpenAPI docs, best-practice for ML model serving per current literature (faster/safer input validation than Flask for JSON prediction payloads) |
| Web application (UI, pages, sessions, analytics rendering) | **Flask** + Jinja2 templates | Lightweight, simple routing, well suited for server-rendered dashboard pages and form-based prediction UI |
| Frontend | HTML5, CSS3, vanilla JavaScript (fetch API), Chart.js/Plotly.js for charts, Leaflet.js for map/spatial view | No build step needed; keeps stack aligned to what was specified |
| Persistence | SQLite (dev) → PostgreSQL (prod-ready path), plus flat Parquet/CSV caches for precomputed analytics | Simple for a portfolio-scale project; schema documented in Backend Schema doc |
| Model artifacts | joblib/pickle (`.pkl`) for sklearn-compatible models, versioned filenames | Matches reference "Bangalore home price" Flask project pattern |
| Explainability | SHAP (TreeExplainer for tree models) | Confirmed best practice from research: SHAP > raw feature_importances_ for per-prediction, human-facing explanations |
| Recommender | scikit-learn `TfidfVectorizer` + `NearestNeighbors`/cosine similarity | Matches content-based real-estate recommender precedent (arturlunardi/real-estate-recommender-system pattern) |

## 2. System Architecture (textual — see App Flow doc for diagram)

```
[ Browser ]
    │  HTTP (HTML pages, form submits, fetch/JSON calls)
    ▼
[ Flask App ]  ── serves templates, static assets, session/state, analytics pages
    │  internal HTTP call (localhost or service-to-service)
    ▼
[ FastAPI Inference Service ]
    ├── /predict            → loads price model, returns price + SHAP explanation
    ├── /recommend           → loads similarity index, returns top-N properties
    ├── /insights            → returns templated insight sentences from precomputed stats
    └── /health               → liveness/readiness
    │
    ▼
[ Model Artifacts / Precomputed Store ]
    ├── models/price_model_v{n}.pkl
    ├── models/preprocessor_v{n}.pkl (encoders, scalers)
    ├── models/tfidf_vectorizer.pkl + feature_matrix.npz (recommender)
    ├── data/processed/clean_listings.parquet
    ├── data/processed/analytics_cache/*.json (precomputed chart data)
    └── data/processed/locality_stats.parquet (for insights templates)
```

Flask never touches raw model files directly — it always calls FastAPI over HTTP. This keeps model-serving isolated, restartable, and independently scalable (per NFR2 in PRD).

## 3. Data Sources (as inspected)

| File | Rows (approx.) | Columns | Notes |
|---|---|---|---|
| `gurgaon_10k.csv` | ~44,890 | 67 | Richest schema (photo URLs, map details, project/building metadata, JSON-like nested strings) |
| `hyderabad.csv` | ~73,859 | 55 | Includes REGISTER_DATE, POSTING_DATE, UPDATE_DATE, VALUE_LABEL (ownership) |
| `kolkata.csv` | ~32,722 | 35 | Smallest schema |
| `mumbai.csv` | ~30,853 | 55 | Same schema family as Hyderabad |
| `facets/*.csv` (15 files) | small | 2–4 | Lookup/decode tables for coded fields |

Key raw columns common across (or mappable to) all 4 cities: `PROP_ID`, `PROPERTY_TYPE`, `CITY`, `LOCALITY`/`location`, `TRANSACT_TYPE`, `OWNTYPE`, `BEDROOM_NUM`, `BATHROOM_NUM`, `BALCONY_NUM`, `FURNISH`, `FACING`, `AGE`, `FLOOR_NUM`, `TOTAL_FLOOR`, `FEATURES`, `AMENITIES`, `AREA`, `PRICE`, `PRICE_SQFT`, `MIN_PRICE`, `MAX_PRICE`, `MAP_DETAILS` (contains LAT/LONG as a stringified dict), `BUILDING_NAME`/`SOCIETY_NAME`, `DESCRIPTION`.

## 4. Data Cleaning Requirements

1. **Parsing PRICE**: values like `"3.5 Cr"`, `"69.25 L"`, `"2.63 Cr"` → convert to numeric ₹ using a unit map (`Cr` → ×1e7, `L`/`Lac` → ×1e5). Fallback to `MIN_PRICE`/`MAX_PRICE` (already numeric in some files) when `PRICE` is unparsable.
2. **Parsing AREA**: values like `"1215 sq.ft."`, `"3434 sq.ft."` → strip unit, cast to float. Cross-check against `MIN_AREA_SQFT`/`MAX_AREA_SQFT`/`SUPER_SQFT`/`CARPET_SQFT`/`BUILTUP_SQFT` where available; prefer the most complete/consistent numeric column per city.
3. **`MAP_DETAILS` parsing**: this is a Python-dict-formatted string (e.g. `{'LATITUDE': '28.4065341', 'LONGITUDE': '76.9627918', ...}`) — parse safely with `ast.literal_eval`, extract `LATITUDE`/`LONGITUDE` as floats, drop rows/flag rows with missing coordinates for the spatial analytics view only (not dropped from the modeling set).
4. **Facet decoding**: columns such as `FURNISH`, `FACING`, `AGE`, `PROPERTY_TYPE`, `OWNTYPE` are numeric codes — join against the matching `facets/*.csv` to get human-readable labels for display; keep the numeric code for modeling (or one-hot/ordinal encode from it directly).
5. **Multi-value fields**: `FEATURES` and `AMENITIES` are comma-separated ID lists (e.g. `"33,23,12,46,25,..."`) — explode/decode via `facets/FEATURES.csv` and `facets/AMENITIES.csv`; engineer as (a) a count feature `n_amenities`, (b) top-k one-hot flags for the most common amenities (e.g., has_clubhouse, has_swimming_pool, has_gym).
6. **Drop columns not useful/unsafe for modeling or display**: photo/media URLs, dealer contact name/company/phone-like fields, internal IDs (`SPID`, `PD_URL`, `PROP_DETAILS_URL`), raw nested JSON strings once parsed into flat features.
7. **Column harmonization across cities**: build one canonical schema (documented in Backend Schema doc §2) and map each city file's columns into it; missing columns per city are filled with NaN and handled by the imputation step.
8. **Text cleanup**: lowercase, strip HTML/special characters from `DESCRIPTION` before TF-IDF / word cloud use.
9. **Deduplication**: drop duplicate `PROP_ID` rows (listings can repeat across scrape batches).

## 5. Missing Value Imputation Strategy

| Missingness level | Strategy |
|---|---|
| < 5% | Median (numeric) / mode (categorical) imputation |
| 5–40% | Group-wise median/mode imputation (by City + Locality + Property Type) to preserve local market signal; add a `was_missing` indicator flag for that column |
| 40–70% | Evaluate feature usefulness first; if kept, use model-based imputation (e.g., `IterativeImputer`) or a dedicated "Unknown" category for categoricals |
| > 70% (e.g., QUALITY_SCORE, FURNISHING_ATTRIBUTES in Gurgaon sample) | Drop column entirely — documented explicitly, not silently |

## 6. Outlier Detection & Removal

1. Compute `PRICE_SQFT` per row; flag rows where `PRICE_SQFT` falls outside [1st, 99th] percentile per city (sample inspection showed values up to ₹8.5 crore/sqft — clear scraping/data errors).
2. IQR method (`Q1 − 1.5×IQR`, `Q3 + 1.5×IQR`) applied to `PRICE`, `AREA`, and `PRICE_SQFT` per city as a second pass.
3. Domain-rule caps: `BEDROOM_NUM`/`BATHROOM_NUM` > 15 flagged as likely data entry errors unless `PROPERTY_TYPE` is a large villa/farmhouse.
4. Log-transform the target (`log1p(PRICE)`) before training to reduce the influence of remaining right-skew/heavy tail — standard practice confirmed by reference Kaggle-style pipelines.
5. Outlier rows are not deleted outright from the raw store — they're excluded from the **training set** only, and retained (flagged) in the analytics store so extreme listings can still be explored/visualized if desired.

## 7. Exploratory Data Analysis (EDA) Requirements

- **Univariate**: distribution plots (histogram/KDE) for `PRICE`, `AREA`, `PRICE_SQFT`, `BEDROOM_NUM`; bar counts for `PROPERTY_TYPE`, `FURNISH`, `FACING`, `CITY`.
- **Bivariate**: `PRICE` vs `AREA` scatter; `PRICE` vs `BEDROOM_NUM` boxplot; `PRICE_SQFT` vs `LOCALITY` boxplot (top-N localities by volume); `PRICE` vs `AGE` boxplot.
- **Multivariate**: correlation heatmap across numeric features; pair plots for a reduced feature set; `PRICE` vs `AREA` colored by `BEDROOM_NUM` or `CITY`; group-by pivot tables (City × Property Type → mean price).
- **Automated profiling**: generate a `ydata-profiling` (pandas-profiling) HTML report per city during the offline pipeline stage for a fast, comprehensive first-pass view (missingness, cardinality, correlations, alerts) — this is a one-time offline artifact, not served live in the app.

## 8. Feature Engineering

- `price_per_sqft` (derived, also used as an outlier signal and analytics dimension)
- `n_amenities`, `n_features` (counts from decoded multi-value fields)
- `has_<amenity>` binary flags for top-15 most frequent amenities/features
- `floor_ratio` = `FLOOR_NUM / TOTAL_FLOOR` (relative height in building)
- `age_bucket` (from facet AGE labels: new/1-5/5-10/10+)
- `bath_bed_ratio` = `BATHROOM_NUM / BEDROOM_NUM`
- `locality_avg_price_sqft`, `locality_listing_count` (target-aware but computed via out-of-fold/leave-one-out to avoid leakage, or computed on train-only and merged into val/test)
- `description_length`, `description_sentiment/keyword_flags` (optional, for insights/word cloud, not necessarily for price model)
- One-hot / ordinal encoding for `PROPERTY_TYPE`, `FURNISH`, `FACING`, `OWNTYPE`, `CITY`

## 9. Feature Selection (methods to apply and compare)

Per the researched comparative literature (RFE vs Permutation Importance; SHAP vs classical importance; Boruta/shap-hypetune context):

1. **Correlation-based filtering**: drop one of any feature pair with |correlation| > 0.9 (multicollinearity).
2. **Random Forest feature importance** (Gini/impurity-based) — fast baseline ranking.
3. **Gradient Boosting feature importance** (XGBoost/LightGBM gain-based) — usually more reliable than RF impurity importance for mixed-type features.
4. **Permutation importance** (sklearn `permutation_importance`) — model-agnostic, accounts for feature interactions better than impurity importance; more computationally expensive, run on a validation subsample.
5. **Lasso (L1) regression coefficients** — shrinks irrelevant linear feature weights to zero; useful sanity check against tree-based rankings.
6. **Recursive Feature Elimination (RFE / RFECV)** — iteratively removes weakest features per a chosen estimator (use with GradientBoosting/XGBoost as core estimator per research: RFE suits smaller/medium feature sets and benefits from cross-validation).
7. **Linear Regression standardized weights** — after scaling, compare raw linear coefficients as an interpretable baseline against nonlinear model importances.
8. **SHAP (TreeExplainer)** — used both as a feature-selection signal (mean |SHAP value| ranking) and, critically, as the **explainability layer for the Price Prediction module's per-prediction breakdown** (this is the same tool serving two purposes: selection at training time, explanation at serving time).

**Decision rule**: a feature is kept if it ranks in the top-N (e.g., top 20) by at least 2 of the above 4 model-based methods (RF importance, GB importance, Permutation, SHAP), or if it has a clear, non-zero Lasso coefficient. Final feature list and rationale must be logged in `data/processed/feature_selection_report.md` (or `.json`) for reproducibility.

## 10. Model Selection & Productionization

### Candidate models (to train & compare)
- Linear Regression (baseline)
- Ridge / Lasso Regression (regularized baselines)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (and/or LightGBM)

### Evaluation protocol
- Train/validation/test split: 70/15/15, stratified by City where feasible, with a fixed random_state (42) for reproducibility.
- K-fold (k=5) cross-validation on the training set for model comparison before touching the test set.
- Metrics: R², MAE, RMSE, MAPE — computed on the **original price scale** (inverse-transform from log) as well as log scale.
- Select final model per city OR one global model with `CITY` as a categorical feature — decision made empirically by comparing single-global-model vs per-city-model performance (document result in Tracker).

### Productionization steps
1. Wrap the winning model + its exact preprocessing (encoders/scaler) in a single `sklearn.Pipeline` to avoid train/serve skew.
2. Serialize with `joblib` → `models/price_model_v{n}.pkl`.
3. Write a `predict(input_dict) -> {price, range, shap_contributions}` function used directly by the FastAPI `/predict` route.
4. Add input schema validation via Pydantic models mirroring the training feature schema.
5. Add a smoke-test script that loads the pickle and runs one sample prediction as part of CI/deploy checks.
6. Log every prediction request (features + output + timestamp, no PII) to a lightweight table for future drift monitoring.

## 11. Recommender System — Technical Approach

- Feature vector per listing: numeric (scaled: price, area, bedrooms, bathrooms, floor_ratio, age) + categorical (one-hot: property type, furnish, facing) + text (TF-IDF on cleaned description/amenities, dimensionality-reduced via TruncatedSVD if needed).
- Similarity: cosine similarity via `sklearn.neighbors.NearestNeighbors(metric='cosine')` for top-N retrieval (scales fine at this dataset size; no need for ANN libraries like FAISS at v1 scale, but structured so FAISS could be swapped in later).
- Scoping: default search restricted to same City (+ optional same Locality) before computing similarity, per PRD FR11.
- Cold-start fallback (per RE-RecSys pattern, arXiv:2404.16553): when fewer than 5 neighbors clear a minimum similarity threshold, fall back to a rule-based rank by (locality popularity = listing count) + recency, matching the "cold-start users" branch of the referenced production system, adapted here to "cold-start listings" (new/rare property profiles).

## 12. Insights Module — Technical Approach

- Precompute aggregate stat tables at training time: `locality_stats` (avg price, avg price/sqft, avg area, listing count per City+Locality), `amenity_uplift` (avg price delta for listings with vs without each top amenity, controlling for locality), `age_price_trend`, `bhk_price_trend`.
- Insight generation is **template-filling**, not generative text: e.g. `"This {bhk}BHK in {locality} is priced {pct}% {above/below} the {locality} average of ₹{avg_price}."` populated from the precomputed tables — auditable, fast, no hallucination risk.

## 13. Non-Functional / Engineering Requirements

- Reproducible pipeline: a single `make pipeline` (or `python run_pipeline.py`) entry point runs clean → EDA (offline profiling only) → feature engineer → select → train → evaluate → export artifacts, in that order.
- All randomness seeded (`random_state=42`) throughout.
- Config-driven (a `config.yaml`/`.env`) for paths, hyperparameters, and feature list — not hardcoded across scripts.
- Logging (Python `logging` module) at each pipeline stage; no silent failures.
- Basic automated tests: parsing functions (PRICE/AREA/MAP_DETAILS), API contract tests (FastAPI TestClient), and one end-to-end smoke test (Flask page → FastAPI predict → response renders).

---

## UPDATE v2 — Classification Module & Literature-Backed Improvement Levers

*(See `09_LITERATURE_REVIEW_AND_IMPROVEMENT_PLAN.md` for full paper citations and match scoring.)*

### U-TRD-1. Classification Module — Technical Spec

**Label construction:**
- `price_tier` = quantile-bin `price_per_sqft` **within each city separately** (quartiles → Budget/Mid-Range/Premium/Luxury). Computed on the training set only; val/test rows are assigned using the training set's per-city quantile boundaries (no leakage).
- `good_deal_flag` (stretch) = 1 if `actual_price < 0.9 × regression_predicted_price`, else 0. Requires the price regression model to already be trained (this classifier is trained *after* the price model, using its out-of-fold predictions to avoid leakage from a model seeing its own training rows).

**Feature set:** identical to the final selected regression feature set (TRD §9) **minus** any price-derived fields (`price_per_sqft`, `price`) — the tier label is derived from `price_per_sqft`, so `price_per_sqft` itself cannot also be an input feature (direct leakage). All other engineered features (area, bedrooms, amenities, locality encodings, geospatial features) remain valid inputs.

**Candidate models:** Logistic Regression (multinomial, baseline), Random Forest Classifier, Gradient Boosting / XGBoost Classifier.

**Evaluation protocol:** same 70/15/15 split and `random_state=42` as the regression pipeline (TRD §10) for consistency. Metrics: accuracy, macro-precision/recall/F1 (macro because tiers are imbalanced — luxury is rare), one-vs-rest multi-class ROC-AUC, and a per-city confusion matrix (tier thresholds are city-relative, so cross-city confusion patterns need separate review).

**Explainability:** SHAP `TreeExplainer` applied to the winning classifier, exposed the same way as the regression model's explanation (top contributing features per prediction), so the UI can show "Why Premium?" alongside "Why this price?".

**Serving:** new FastAPI route `POST /classify` (or folded into `/predict`'s response as an additional field — implementation choice, but the schema below assumes a dedicated route for separability):
```json
// Request: same feature shape as /predict
// Response
{
  "price_tier": "Premium",
  "tier_probabilities": {"Budget": 0.02, "Mid-Range": 0.18, "Premium": 0.63, "Luxury": 0.17},
  "good_deal_flag": null,
  "shap_contributions": [{"feature": "locality_avg_price_sqft", "impact": 0.31}, ...],
  "model_version": "tier_classifier_v1"
}
```

**Artifact:** `models/tier_classifier_v{n}.pkl`, `models/tier_classifier_metrics_v{n}.json` (accuracy/F1/ROC-AUC/confusion matrix), versioned identically to the regression model per TRD §10.

### U-TRD-2. Improvement Levers to Reach the ~30–35% MAE/RMSE Reduction Target

Each lever below is additive engineering work on top of the existing TRD pipeline (§4–§11), not a replacement:

1. **Ensemble stacking** — train a `StackingRegressor` (base learners: Ridge, Random Forest, Gradient Boosting, XGBoost; meta-learner: Ridge on out-of-fold base predictions) instead of picking one winning model outright. Literature: base papers B3/B4 (literature doc §2.1).
2. **Hyperparameter optimization** — replace `RandomizedSearchCV` with Bayesian optimization (Optuna) for the boosting models' key params (`max_depth`, `learning_rate`, `n_estimators`, `subsample`). Literature: S8.
3. **Geospatial features** — compute `distance_to_metro_km` / `distance_to_cbd_km` from `latitude`/`longitude` (haversine distance to a small reference table of metro stations / city-center coordinates per city, built once as a static lookup). Literature: B2.
4. **Smoothed target encoding for `locality`** — replace plain ordinal/one-hot locality encoding with a target-encoded (mean price, Bayesian-smoothed toward the city mean for low-count localities) numeric feature, computed train-only. Literature: S12.
5. **SHAP-guided feature refinement loop** — after first model fit, inspect SHAP summary + dependence plots; drop near-zero-impact features, add flagged interaction terms (e.g., `area_sqft × locality_avg_price_sqft`), refit. Literature: B3, B4, S14, S15.
6. **Text-derived signal into the regressor** — add `n_amenities`, top-keyword TF-IDF-derived flags, and/or a compact sentence-embedding of `description_clean` as regression inputs (not just recommender inputs). Literature: S14 (SmartPrice), reporting 10–14% RMSE reduction from structured+text fusion.
7. **Outlier-robust target handling** — already log1p-transformed (TRD §6); add a `HuberRegressor`/quantile-loss option for the range-band estimate to make the ±range robust to residual heavy tails.

Each version's actual MAE/RMSE/R² must be logged in `metrics_v{n}.json` and compared against `metrics_v1` (the un-optimized baseline) in the Tracker — the 30–35% figure is the target to validate, not to assume.

---

## UPDATE v3 — Finalized Input Schema Wiring & UML Cross-Reference

### U-TRD-3. Preprocessing pipeline update (finalized 16 fields)
The `ColumnTransformer` (per the reference project's `model-selection.ipynb` pattern) now maps as:

```python
numeric_features = ['bedRoom', 'bathroom', 'built_up_area', 'servant_room', 'store_room', 'n_amenities']
ordinal_features = ['luxury_category', 'floor_category', 'furnishing_type']   # Low<Med<High etc. — OrdinalEncoder
onehot_features  = ['city', 'sector', 'property_type', 'balcony', 'agePossession', 'facing', 'transact_type']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('ord', OrdinalEncoder(categories=[...]), ordinal_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), onehot_features),
])
```
Note vs. the reference project: `sector` moves from a plain `OrdinalEncoder` (reference project's `model-selection.ipynb`) to **one-hot or smoothed target encoding** here (per Improvement Lever 4, `09_LITERATURE_REVIEW...md` §5) because this project spans many more sectors/localities across 4 cities than the single-city reference project, making a naive ordinal encoding of `sector` risk implying a false ordering.

### U-TRD-4. `transact_type` routing rule
Because Sale and Rent prices live on different scales, `transact_type` is **not** simply one-hot encoded into a single shared regression — it **routes the request to one of two trained pipelines** (`price_model_sale_v{n}.pkl` / `price_model_rent_v{n}.pkl`) at the FastAPI layer, before any feature transformation happens. This must be implemented as a routing `if` in the `/predict` handler, not as a model feature.

### U-TRD-5. UML diagrams as the pipeline's authoritative reference
`11_UML_DIAGRAMS.md` §5 (Activity Diagram) is now the canonical visual sequence for the offline ML pipeline described in TRD §4–§11 — any change to the pipeline order (e.g., reordering feature selection vs. outlier treatment) must be reflected in both places. §2 (Class Diagram) formalizes the `PricePredictionService`/`ClassificationService`/`RecommenderService`/`InsightsService` split that FastAPI's route handlers (TRD §2) implement.

---

## UPDATE v4 — Classification Module Technical Revision (Affordability + Good-Deal Framing)

### U-TRD-6. `good_deal_flag` becomes 3-class, and becomes the primary classifier
Was: binary stretch goal. Now: **primary deliverable**, 3-class — `Good Deal` / `Fair Price` / `Overpriced` — derived from the residual between actual price and the regression model's out-of-fold prediction:
```
residual_pct = (actual_price - predicted_price) / predicted_price
Good Deal:   residual_pct <= -0.10   (actual at least 10% below predicted)
Fair Price:  -0.10 < residual_pct < 0.10
Overpriced:  residual_pct >= 0.10
```
Thresholds (±10%) are a starting point — tune per city once residual distributions are known (Kolkata's smaller sample may need a wider band). Still requires the regression model's **out-of-fold** predictions for training data (not in-sample), per the original leakage rule — unchanged.

### U-TRD-7. `price_tier` (Affordability Tier) — now explicitly a filter/UX input, not just an analytics label
Technical construction is unchanged from v2 (per-city quantile-binned `price_per_sqft`, train-set boundaries). What changes is *why* it's computed: it now exists specifically to power the Recommender's tier filter (App Flow) and a "fits your budget" match indicator, not merely to exist as a generalized classifier. No pipeline code changes — this is a purpose/consumption change, not an architecture change.

### U-TRD-8. Evaluation priority reordered
Because `good_deal_flag` is now primary: evaluate it first (macro-F1, per-city confusion matrix, and specifically **recall on the "Good Deal" class** — a missed good deal is a worse UX failure than a missed "Overpriced" flag, since users acting on a false "Good Deal" signal lose real money). `price_tier` classifier evaluation (unchanged from v2: macro-F1 ≥0.75, ROC-AUC ≥0.85) remains secondary but still required.

### U-TRD-9. No MLOps tooling added
Per your explicit choice, this update does **not** introduce MLflow, DVC, GitHub Actions, or drift detection. The existing reproducibility mechanisms already specified — seeded randomness, versioned `.pkl`/`metrics.json` artifacts, config-driven paths, a single pipeline entry point (original TRD §13) — remain the full extent of "reproducibility engineering" for this project. If you want to add MLOps tooling later, that's a new scope decision, not an implicit one.