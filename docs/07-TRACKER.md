# Project Tracker
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

Use this as a living checklist. Update status honestly (Not Started / In Progress / Blocked / Done) and log actual dates/results next to each planned date — the plan dates are targets, not guarantees.

---

## 1. Master Task Tracker

### Week 1 — Data Understanding & Cleaning
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 1 | Repo setup, env, load raw data | Not Started | | |
| 2 | Parsing functions (price/area/map_details) + unit tests | Not Started | | |
| 3 | Facet decoding joins + multi-value field decoding | Not Started | | |
| 4 | Canonical schema mapping across 4 cities | Not Started | | |
| 5 | Drop unusable columns, deduplicate | Not Started | | |
| 6 | Missing value imputation implementation | Not Started | | |
| 7 | **Checkpoint:** clean_listings.parquet + cleaning report | Not Started | | |

### Week 2 — EDA & Outlier Handling
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 8 | ydata-profiling reports per city | Not Started | | |
| 9 | Univariate EDA | Not Started | | |
| 10 | Bivariate EDA | Not Started | | |
| 11 | Multivariate EDA | Not Started | | |
| 12 | Outlier detection (percentile + IQR + domain rules) | Done | 2026-08-06 | `ml/cleaning/dedup.py` + `outliers.py` + `assemble.py` (Step 06, 41 tests). Per-city 1st/99th percentile + Tukey fence (1.5×IQR) on `price_inr`/`area_sqft`/`price_per_sqft`; bedRoom/bathroom > 15 cap with villa/farmhouse/independent-house exemptions. 3-level dedup tiebreaker (nonnull count → register_date → row_order). Real data: 38,487 rows post-dedup, 5,036 (≈13%) flagged outlier. Public symbols re-exported from `ml.cleaning`. Raw immutability gate (`assert_raw_readonly`) wired into assembler. |
| 13 | Log-transform target, flag outliers, finalize training subset | Not Started | | |
| 14 | **Checkpoint:** EDA report + outlier-flagged dataset | Not Started | | |

### Week 3 — Feature Engineering & Selection
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 15 | Core engineered features (price/sqft, amenity counts, floor_ratio, age_bucket, bath_bed_ratio) | Not Started | | |
| 16 | Top-15 amenity flags + categorical encoding | Not Started | | |
| 17 | Leakage-safe locality aggregate features | Not Started | | |
| 18 | Feature selection round 1: correlation, Lasso, Linear weights | Not Started | | |
| 19 | Feature selection round 2: RF importance, GB importance, Permutation importance | Not Started | | |
| 20 | Feature selection round 3: SHAP ranking, RFE/RFECV, final decision | Not Started | | |
| 21 | **Checkpoint:** final feature set + feature_selection_report.md | Not Started | | |

### Week 4 — Model Selection & Productionization
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 22 | Baseline models (Linear/Ridge/Lasso) + CV metrics | Not Started | | |
| 23 | Random Forest + Gradient Boosting tuning | Not Started | | |
| 24 | XGBoost/LightGBM, global vs per-city decision | Not Started | | |
| 25 | Final model selection + SHAP explainer validation | Not Started | | |
| 26 | Pipeline wrap + serialize (price_model_v1.pkl) | Not Started | | |
| 27 | FastAPI /predict route + schemas + smoke test | Not Started | | |
| 28 | **Checkpoint:** working /predict endpoint + metrics_v1.json | Not Started | | |

### Week 5 — Recommender & Insights
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 29 | TF-IDF text pipeline | Not Started | | |
| 30 | Combined feature matrix (numeric+categorical+text) | Not Started | | |
| 31 | NearestNeighbors retrieval + relevance sanity check | Not Started | | |
| 32 | Cold-start fallback logic | Not Started | | |
| 33 | FastAPI /recommend route + smoke test | Not Started | | |
| 34 | Precompute stats tables + insight templater | Not Started | | |
| 35 | **Checkpoint:** /recommend + /insights endpoints working | Not Started | | |

### Week 6 — Flask App & Analytics UI
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 36 | Precompute all 13 analytics_cache JSON files | Not Started | | |
| 37 | Flask scaffold + Landing page | Not Started | | |
| 38 | Price Prediction form + result page + SHAP chart | Not Started | | |
| 39 | Analytics dashboard shell + tiles 1–7 | Not Started | | |
| 40 | Analytics tiles 8–13 + city filter AJAX | Not Started | | |
| 41 | Recommender pages + Insights page | Not Started | | |
| 42 | **Checkpoint:** full end-to-end click-through, desktop + mobile | Not Started | | |

### Week 7 — Testing, Polish, Deploy Prep
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 43 | Automated tests (parsing, API contracts, e2e smoke) | Not Started | | |
| 44 | Responsive/accessibility polish pass | Not Started | | |
| 45 | Performance pass (latency targets) | Not Started | | |
| 46 | Error-handling edge case verification | Not Started | | |
| 47 | README + docs finalization | Not Started | | |
| 48 | Deployment config (Docker etc.) | Not Started | | |
| 49 | **Checkpoint:** final demo + retro | Not Started | | |

## 2. KPI / Success Metric Tracker

| Metric (from PRD §3) | Target | Current | Last measured |
|---|---|---|---|
| Test R² (price model) | ≥ 0.80 | — | — |
| MAE within ±15% for 70% of listings | 70% | — | — |
| Analytics chart tiles live | 13 | — | — |
| Recommender relevance (precision@5, manual) | ≥ 0.6 | — | — |
| Insight cards per prediction | ≥ 3 | — | — |
| FastAPI /predict p95 latency | < 300ms | — | — |

## 3. Decision Log
Record every non-trivial decision here so future-you (or a teammate) knows *why*, not just *what*.

| Date | Decision | Rationale | Alternatives considered |
|---|---|---|---|
| | e.g. "Global model with CITY as feature, not 4 per-city models" | | |
| | e.g. "Drop QUALITY_SCORE / FURNISHING_ATTRIBUTES columns" | ~100% missing in sample | Model-based imputation (rejected — no signal to impute from) |

## 4. Risk & Blocker Log

| Date raised | Risk/Blocker | Severity | Owner | Resolution / Status |
|---|---|---|---|---|
| | | | | |

## 5. Bug Tracker (post-integration)

| ID | Description | Module | Severity | Status | Fixed in |
|---|---|---|---|---|---|
| | | | | | |

## 6. Weekly Retro Prompts
At the end of each week, answer briefly:
1. What did we ship this week vs. what was planned?
2. What surprised us in the data or the model behavior?
3. What's the single biggest risk to next week's plan?
4. Any scope change needed to PRD/TRD as a result?

---

## UPDATE v2 — Week 8 Tasks, New KPIs, Literature Log

### Week 8 — Classification Module + Improvement Levers
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 50 | Construct price_tier labels (per-city quantile binning) | Not Started | | |
| 51 | Classification feature set + Logistic Regression baseline | Not Started | | |
| 52 | Random Forest / XGBoost classifiers + selection | Not Started | | |
| 53 | SHAP explanations + per-city confusion matrices | Not Started | | |
| 54 | Serialize model + FastAPI /classify route + smoke test | Not Started | | |
| 55 | Flask UI: TierBadge, /classify page, Recommender tier filter, Analytics tile 14 | Not Started | | |
| 56 | Improvement levers 1–4 (stacking, Optuna, geospatial, target encoding) on regression model | Not Started | | |
| 57 | **Checkpoint:** Levers 5–7 + final metrics_v3.json + quantified improvement % | Not Started | | |

### Updated KPI / Success Metric Tracker (additions)

| Metric | Target | Current | Last measured |
|---|---|---|---|
| MAE/RMSE reduction vs. metrics_v1 baseline | ~30–35% | — | — |
| Price-tier classifier macro-F1 | ≥ 0.75 | — | — |
| Price-tier classifier multi-class ROC-AUC (OvR) | ≥ 0.85 | — | — |
| FastAPI /classify p95 latency | < 300ms | — | — |

### Literature Review Log (reference for Decision Log entries)
22 papers reviewed (4 base papers at 75–90% match, 18 supporting papers at 45–70% match) — full list with citations in `09_LITERATURE_REVIEW_AND_IMPROVEMENT_PLAN.md`. Any modeling decision justified by a specific paper (e.g., "used Optuna per S8") should reference the paper ID (B1–B4, S1–S18) in the Decision Log below, not just "we read a paper somewhere."

| Date | Decision | Paper ID referenced | Rationale |
|---|---|---|---|
| | e.g. "Added distance-to-metro feature" | B2 | Roy 2026 found this improved Delhi NCR valuation |
| | e.g. "Chose StackingRegressor over single XGBoost" | B3, B4 | Both base papers report stacking/ensemble gains over single models |

---

## UPDATE v3 — Input Schema Lock & UML Task

| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 4 (revised) | Implement finalized 16-field canonical schema (not placeholder) | Not Started | | |
| 58 | Produce + cross-check full UML diagram set against TRD/Backend Schema/App Flow | Not Started | | |

### Decision Log addition
| Date | Decision | Rationale |
|---|---|---|
| | Added City, Facing, Amenities, Transaction Type to the reference project's 12-field input set | 4-city scope + available-but-unused raw fields + Sale/Rent price-scale mismatch, per `10_FINALIZED_INPUT_SCHEMA.md` §2 |
| | Routed `transact_type` to two separate model pipelines instead of one-hot encoding it | Sale and Rent prices are on incomparable scales; encoding as a plain feature risks the model learning a spurious linear offset instead of two genuinely different price distributions |

---

## UPDATE v4 — Good-Deal-First Classification Tasks & KPIs

### Week 8 task revisions
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 51 | Generate out-of-fold regression predictions (input to Day 52) | Not Started | | |
| 52 | Train good_deal_verdict 3-class classifier (priority: Good Deal recall) | Not Started | | |
| 55 | Build VerdictBadge + AffordabilityChip, rename /classify page copy | Not Started | | |

### KPI Tracker — revised/added
| Metric | Target | Current | Last measured |
|---|---|---|---|
| `good_deal_verdict` — recall on "Good Deal" class | ≥ 0.80 (missed good deals cost users money — prioritize over overall accuracy) | — | — |
| `good_deal_verdict` — macro-F1 (3-class) | ≥ 0.70 | — | — |
| `price_tier` (secondary) macro-F1 | ≥ 0.75 (unchanged from v2) | — | — |

### Decision Log addition
| Date | Decision | Rationale |
|---|---|---|
| | Promoted `good_deal_verdict` (3-class) from optional stretch to primary classifier; demoted `price_tier` to a supporting/filter role | External review correctly flagged that "classify price tier" alone lacked a stated business reason; "is this priced fairly" is a real decision-support question a regression number alone doesn't answer |
| | Declined to add MLflow/DVC/CI-CD/drift detection | Explicit scope choice — prioritize finishing a coherent 8-week solo build over expanding into MLOps tooling |
| | Declined to replace Flask with React | Flask+FastAPI was the originally specified stack; external review's suggestion was a stylistic preference, not a technical requirement |