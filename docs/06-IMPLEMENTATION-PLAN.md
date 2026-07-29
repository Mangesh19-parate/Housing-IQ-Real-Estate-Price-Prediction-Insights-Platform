# Implementation Plan (Daily & Weekly)
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

Total duration: **7 weeks** (adjustable — compress by combining days if working full-time). Each week ends with a checkpoint deliverable that should be logged in the Tracker doc.

---

## Week 1 — Data Understanding & Cleaning
**Goal:** One canonical, clean dataset (`clean_listings.parquet`) covering all 4 cities.

- **Day 1:** Set up repo structure (`/data/raw`, `/data/processed`, `/notebooks`, `/src`, `/models`, `/app`, `/api`), virtual env, install pandas/numpy/scikit-learn/matplotlib/seaborn/ydata-profiling/shap/xgboost/flask/fastapi/uvicorn. Load all 4 CSVs + 15 facet files; print shapes, dtypes, `.info()`, null counts.
- **Day 2:** Write parsing functions: `parse_price()`, `parse_area()`, `parse_map_details()`. Unit-test on sample strings (`"3.5 Cr"`, `"69.25 L"`, `"2700 sq.ft."`, malformed values).
- **Day 3:** Build facet-decoding joins (FURNISH, FACING, AGE, PROPERTY_TYPE, OWNTYPE, FLOOR_NUM, TOTAL_FLOOR, CITY, LOCALITY_ID). Decode multi-value FEATURES/AMENITIES ID lists.
- **Day 4:** Define and implement the canonical schema mapping (per Backend Schema doc §2) — one function per city that maps raw columns → canonical columns; concatenate into one DataFrame with a `city` field.
- **Day 5:** Drop unusable columns (media URLs, dealer contact fields, near-100%-missing columns). Deduplicate by `PROP_ID`.
- **Day 6:** Missing value strategy implementation (per TRD §5): simple imputation for <5%, group-wise for 5–40%, "Unknown"/model-based for 40–70%, drop >70%. Add `was_missing_*` flags.
- **Day 7 (checkpoint):** Save `clean_listings.parquet`. Write a short data-cleaning report (rows before/after, columns dropped and why, missingness handled). **Weekly deliverable: clean dataset + cleaning report.**

## Week 2 — EDA & Outlier Handling
**Goal:** Full exploratory understanding + an outlier-free training-ready dataset.

- **Day 8:** Run `ydata-profiling` reports per city (offline HTML artifacts) — skim for missingness, cardinality, correlation alerts.
- **Day 9:** Univariate EDA — distributions of PRICE, AREA, PRICE_SQFT, BEDROOM_NUM; bar counts for PROPERTY_TYPE, FURNISH, FACING, CITY.
- **Day 10:** Bivariate EDA — Price vs Area scatter, Price vs Bedrooms boxplot, Price/sqft vs Locality boxplot (top-N localities), Price vs Age boxplot.
- **Day 11:** Multivariate EDA — correlation heatmap, pair plots on reduced feature set, City × Property Type pivot table of mean price.
- **Day 12:** Outlier detection — percentile capping (1st/99th) on PRICE_SQFT per city, IQR method on PRICE/AREA/PRICE_SQFT, domain-rule caps on BEDROOM_NUM/BATHROOM_NUM.
- **Day 13:** Apply log1p transform to target; flag (not delete) outlier rows via `is_outlier`; finalize the training subset (`is_outlier == False`).
- **Day 14 (checkpoint):** Compile an EDA summary doc/notebook with key charts and 5–8 written takeaways. **Weekly deliverable: EDA report + outlier-flagged dataset.**

## Week 3 — Feature Engineering & Feature Selection
**Goal:** Final, justified feature set for modeling.

- **Day 15:** Engineer `price_per_sqft`, `n_amenities`, `n_features`, `floor_ratio`, `age_bucket`, `bath_bed_ratio`.
- **Day 16:** Engineer top-15 `has_<amenity>` binary flags; encode categoricals (one-hot/ordinal) for PROPERTY_TYPE, FURNISH, FACING, OWNTYPE, CITY.
- **Day 17:** Engineer `locality_avg_price_sqft` / `locality_listing_count` using train-only (leakage-safe) aggregation.
- **Day 18:** Feature selection round 1 — correlation filtering (drop |corr| > 0.9 pairs), Lasso coefficients, Linear Regression standardized weights.
- **Day 19:** Feature selection round 2 — Random Forest importance, Gradient Boosting (XGBoost) importance, Permutation Importance (on validation subsample).
- **Day 20:** Feature selection round 3 — SHAP mean |value| ranking (TreeExplainer on a GB/XGBoost model); apply RFE/RFECV with GB as core estimator; cross-check convergence across all methods (decision rule: keep if top-N in ≥2 of the 4 model-based methods).
- **Day 21 (checkpoint):** Finalize feature list + write `feature_selection_report.md` documenting each method's ranking and the final decision. **Weekly deliverable: final feature set + selection report.**

## Week 4 — Model Selection & Productionization
**Goal:** A trained, evaluated, serialized price-prediction pipeline.

- **Day 22:** Train baseline models (Linear, Ridge, Lasso) with 5-fold CV; record R²/MAE/RMSE/MAPE.
- **Day 23:** Train Random Forest and Gradient Boosting regressors; tune key hyperparameters (n_estimators, max_depth, learning_rate) via `RandomizedSearchCV`.
- **Day 24:** Train XGBoost/LightGBM; compare all models on validation set; decide global-model vs per-city-model (empirical test both ways).
- **Day 25:** Final model selection on held-out test set; compute SHAP explainer for the winning model; validate a handful of individual SHAP explanations make intuitive sense.
- **Day 26:** Wrap preprocessing + model into one `sklearn.Pipeline`; serialize (`price_model_v1.pkl`); write `predict()` and `explain()` helper functions.
- **Day 27:** Write FastAPI `/predict` route calling the helper functions; add Pydantic request/response schemas; add `/health` route; smoke-test with sample payloads.
- **Day 28 (checkpoint):** **Weekly deliverable: working FastAPI `/predict` endpoint + metrics report (`metrics_v1.json`).**

## Week 5 — Recommender System & Insights Module
**Goal:** Working recommender + insights logic, servable via FastAPI.

- **Day 29:** Build text feature pipeline — clean DESCRIPTION, fit TF-IDF vectorizer (per city or global with city as a filter).
- **Day 30:** Build combined numeric+categorical+text feature matrix per listing; scale numeric features.
- **Day 31:** Implement `NearestNeighbors` (cosine) retrieval; test top-N retrieval on a handful of manually chosen seed listings, sanity-check relevance.
- **Day 32:** Implement cold-start fallback (locality popularity + recency ranking) per RE-RecSys-inspired pattern; wire the "used_fallback" flag.
- **Day 33:** Build FastAPI `/recommend` route + Pydantic schemas; smoke-test.
- **Day 34:** Precompute `locality_stats`, `amenity_uplift`, `age_price_trend`, `bhk_price_trend`, `furnish_price_trend`, `floor_price_trend` tables; write the insight sentence templates + templater function.
- **Day 35 (checkpoint):** Build FastAPI `/insights` route. **Weekly deliverable: working `/recommend` + `/insights` endpoints.**

## Week 6 — Analytics Precompute, Flask App & Frontend
**Goal:** Full Flask UI wired to all FastAPI endpoints, all 13 analytics tiles live.

- **Day 36:** Precompute all 13 analytics_cache JSON files (per Backend Schema §4), one script per chart or one batch script producing all.
- **Day 37:** Scaffold Flask app (routes, templates, static folder); build Landing page + nav + city quick-filter.
- **Day 38:** Build Price Prediction form + result page (calls FastAPI `/predict`); implement SHAP bar chart rendering (Chart.js).
- **Day 39:** Build Analytics dashboard shell + tile component; wire 6–7 chart tiles (spatial map via Leaflet + 5–6 chart-based tiles).
- **Day 40:** Wire remaining 6–7 chart tiles; implement City filter re-fetch behavior (AJAX, no reload).
- **Day 41:** Build Recommender form + results grid page (calls FastAPI `/recommend`); build Insights standalone page (calls FastAPI `/insights`).
- **Day 42 (checkpoint):** Full click-through of all 4 modules end-to-end on desktop + mobile viewport. **Weekly deliverable: fully wired app, feature-complete.**

## Week 7 — Testing, Polish, Documentation, Deployment Prep
**Goal:** Portfolio-ready, tested, documented, deployable app.

- **Day 43:** Write automated tests — parsing functions, FastAPI TestClient contract tests, one Flask→FastAPI end-to-end smoke test.
- **Day 44:** Responsive/UI polish pass (per UI/UX doc — component consistency, loading/error states, accessibility labels).
- **Day 45:** Performance pass — check FastAPI `/predict` p95 latency target (<300ms), optimize analytics cache loading if needed.
- **Day 46:** Error-handling pass — verify all edge cases in App Flow §4 (FastAPI down, locality not found, outlier input, empty recommender results) actually behave as specified.
- **Day 47:** Write README (setup instructions, architecture diagram, how to retrain the model), finalize this doc set (PRD/TRD/App Flow/UI-UX/Backend Schema/Rules) with any drift corrections.
- **Day 48:** Prepare deployment config (Dockerfile(s) for Flask + FastAPI, or a single container running both via a process manager; environment variable config).
- **Day 49 (checkpoint):** Final walkthrough demo, Tracker doc fully updated, retro notes (what worked, what to improve in v2 — e.g., collaborative filtering once real user interaction data exists). **Weekly deliverable: deployable, documented, tested v1.**

---

## Suggested Weekly Cadence Summary

| Week | Theme | Deliverable |
|---|---|---|
| 1 | Data cleaning | `clean_listings.parquet` + cleaning report |
| 2 | EDA + outliers | EDA report + outlier-flagged dataset |
| 3 | Feature engineering + selection | Final feature set + selection report |
| 4 | Modeling + productionization | `/predict` API + metrics report |
| 5 | Recommender + Insights | `/recommend` + `/insights` APIs |
| 6 | Flask app + analytics UI | Fully wired app (4 modules) |
| 7 | Testing + polish + deploy prep | Deployable, documented v1 |

---

## UPDATE v2 — Week 8: Classification Module + Improvement-Lever Execution

Inserted after the original Week 4 (Model Selection & Productionization) conceptually, but scheduled as **Week 8** so the original 7-week plan for the regression/recommender/insights/UI stack stays intact and this is additive work. Total project duration becomes **8 weeks**.

- **Day 50:** Construct `price_tier` labels (per-city quantile binning on `price_per_sqft`, train-set boundaries saved to `tier_quantile_boundaries_v1.json`); confirm class balance per city.
- **Day 51:** Build the classification feature set (regression feature set minus `price_per_sqft`/`price`); train Logistic Regression baseline; record accuracy/macro-F1.
- **Day 52:** Train Random Forest Classifier and XGBoost Classifier; compare against baseline; select winner via macro-F1 + multi-class ROC-AUC.
- **Day 53:** Compute SHAP explanations for the winning classifier; sanity-check a handful of tier explanations make intuitive sense; build per-city confusion matrices.
- **Day 54:** Serialize `tier_classifier_v1.pkl`; build FastAPI `/classify` route + Pydantic schemas; smoke-test.
- **Day 55:** Wire Flask: add `TierBadge` component, update `/predict/result` to parallel-call `/classify`, build standalone `/classify` page, add Tier filter to Recommender, add Analytics Tile 14.
- **Day 56:** Execute Improvement Levers 1–4 from the literature doc (stacking ensemble, Optuna tuning, geospatial distance features, smoothed target encoding for locality) on the **price regression** model; retrain; log `metrics_v2.json`; compare against `metrics_v1` baseline.
- **Day 57 (checkpoint):** Execute Improvement Levers 5–7 (SHAP-guided refinement, text-derived signal, outlier-robust loss); retrain final `metrics_v3.json`; compute actual % MAE/RMSE reduction vs. `metrics_v1` and record it honestly in the Tracker (target: ~30–35%, but log the real number). **Weekly deliverable: working Classification module + a validated, quantified improvement percentage over the original baseline model.**

### Revised Weekly Cadence Summary

| Week | Theme | Deliverable |
|---|---|---|
| 1–7 | (unchanged from original plan) | |
| **8** | **Classification module + improvement-lever execution** | **`/classify` API + UI integration + quantified MAE/RMSE improvement report** |

---

## UPDATE v3 — Input Schema Lock-In & UML Diagram Production Added to Schedule

### U-PLAN-1. Day 4 (Week 1) revised
Day 4's "canonical schema mapping across 4 cities" task now explicitly means: implement the finalized 16-field schema from `10_FINALIZED_INPUT_SCHEMA.md` (12 reference-matched fields + City, Facing, Amenities, Transaction Type), not a placeholder schema — this was previously left generic and is now locked before any modeling work begins.

### U-PLAN-2. New Day 58 — UML Diagram Production & Review
- **Day 58:** Produce/finalize the full UML set (`11_UML_DIAGRAMS.md`): Use Case, Class, Sequence (×2), Activity, Component, Deployment, ER, State diagrams. Cross-check each diagram against the current TRD/Backend Schema/App Flow content (they must describe the *same* system, not diverge) and correct any drift found. This sits after Week 8 (Classification module) since the diagrams reflect the final, classification-inclusive architecture.

### Revised Weekly Cadence Summary

| Week | Theme | Deliverable |
|---|---|---|
| 1–7 | (unchanged) | |
| 8 | Classification module + improvement levers | `/classify` API + UI + quantified improvement report |
| **8 (Day 58, same week)** | **UML documentation pass** | **Full UML diagram set, cross-checked against all docs** |

---

## UPDATE v4 — Week 8 Tasks Revised for Good-Deal-First Classification

### Revised Week 8 (replaces v2's Day 50–57 detail where noted)
- **Day 50 (unchanged):** Construct `price_tier` labels (per-city quantile binning) — now explicitly framed as "Affordability Tier," built for the Recommender filter, not as a standalone deliverable.
- **Day 51 (revised):** Train the price regression model's out-of-fold predictions across the full training set (needed as the *input* to Day 52's primary classifier — this must happen before Day 52, not after).
- **Day 52 (revised, now the priority day):** Construct `good_deal_verdict` 3-class labels from residual thresholds (±10%, tune per city); train Logistic Regression baseline + Random Forest/XGBoost classifiers; select winner prioritizing **recall on the "Good Deal" class** (per TRD Update v4 §U-TRD-8), not just overall accuracy.
- **Day 53 (unchanged):** SHAP explanations + per-city confusion matrices — now run first for `good_deal_verdict` (primary), then for `price_tier` (secondary) if time allows.
- **Day 54 (unchanged):** Serialize both classifiers; FastAPI `/classify` route returns both, verdict-first (per Backend Schema Update v4).
- **Day 55 (revised):** Flask UI: build `VerdictBadge` (primary) + `AffordabilityChip` (secondary), update `/predict/result`, rename `/classify` page to "Is this a good deal?", update Recommender filter copy to "Fits my budget."
- **Day 56–57 (unchanged):** Improvement levers on the price regression model, as in v2.
- **No new day added for MLOps tooling** — per your explicit choice, Week 8 stays within its original 8-day scope; no MLflow/DVC/CI-CD tasks are scheduled.