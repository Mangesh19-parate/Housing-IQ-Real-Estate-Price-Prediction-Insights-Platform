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
| 15 | Core engineered features (price/sqft, amenity counts, floor_ratio, age_bucket, bath_bed_ratio) | Done | 2026-08-14 | `ml/features/feature_frame.py` (Step 12). `derive_row_features` + `build_feature_frame` add 8 non-locality columns (price_per_sqft, n_amenities, n_features, floor_ratio, age_bucket_ord, bath_bed_ratio, area_per_bedroom, top_amenities_count) with NaN guards for bedrooms==0. 17 tests in `tests/test_feature_frame.py`. |
| 16 | Top-15 amenity flags + categorical encoding | Done | 2026-08-14 | `select_top_amenities` (K=10) emits `has_<slugified>` flags resolved at fit time; ColumnTransformer pinned to 15 numeric + 4 ordinal + 4 one-hot. OrdinalEncoder/OneHotEncoder category orderings pinned by `tests/test_preprocessor.py`. |
| 17 | Leakage-safe locality aggregate features | Done | 2026-08-14 | `LocalityAggregator` (Step 12 Phase 2): fit on `is_outlier == False` rows only; per-row leave-one-out via `(group_sum - own_value)/(group_count - 1)`. Bayesian smoother `locality_smoothed_price` with prior weight 20, city-prior fallback for unseen (city, locality). 9 tests in `tests/test_locality_aggregator.py`. |
| 18 | Feature selection round 1: correlation, Lasso, Linear weights | Done | 2026-08-14 | Round 1 report (`scripts/build_feature_selection_report.py`) emits 6 sections to `data/processed/feature_selection_report.md`. Round 1 covers the correlation filter + base/engineered column rationale only; Round 2/3 deferred to training spec (SHAP/RF/GB/permutation require a fitted model). |
| 19 | Feature selection round 2: RF importance, GB importance, Permutation importance | Done | 2026-08-15 | `append_round_2_3` (Spec 13). RF/GB/XGB impurity tables + permutation importance on validation slice. `tests/test_report.py` pins file contract. |
| 20 | Feature selection round 3: SHAP ranking, RFE/RFECV, final decision | Done | 2026-08-15 | SHAP TreeExplainer on the tree-model winner; consensus cross-method ranking rendered into `data/processed/feature_selection_report.md`. |
| 21 | **Checkpoint:** final feature set + feature_selection_report.md | Not Started | | Awaiting Day 26 (final model selection) to fully close. |

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

### Spec 14 — v2 Boosted-Tree Improvement Levers (overlap with Week 4)
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 14-S1 | Lever sub-package: geo, target encoding, stacking, Optuna | Done | 2026-08-15 | `ml/training/levers/{__init__,geospatial,target_encoding,stacking,optuna_search}.py`. 21 tests in `tests/test_{geospatial,target_encoding,stacking,optuna_search}.py` — all green. |
| 14-S2 | Extend v1 modules for v2 metrics (eval / persistence / report / `__init__`) | Done | 2026-08-15 | `vs_v1_metrics`, `improvement_target_met`, `IMPROVEMENT_TARGET_PCT=32.5`, `load_metrics`, `write_v2_lever_section`, `PRICE_MODEL_VERSION_V2`, `make_v2_estimator` re-exported from `ml.training`. Full v1 suite (379 tests) still passes. |
| 14-S3 | v2 candidate factory (5 names) | Done | 2026-08-15 | `V2_CANDIDATE_MODELS = {xgb_v1_defaults, xgb_optuna, lgbm_v1_defaults, lgbm_optuna, stacking}`; `make_v2_estimator(name, params=...)` for Optuna injection. 7 tests in `tests/test_candidates_v2.py`. |
| 14-S4 | `scripts/train_price_model_v2.py` + pipeline wiring | Done | 2026-08-15 | End-to-end script; 12-step structure mirrors Spec 13. Preprocessor NEVER refit (Rules §2.4) — geo + sector columns appended post-transform via `_V2PreprocAdapter`. Optuna 40 trials × 2 (XGB+LGBM), 10-min cap. Honest shortfall logging (Rules §9.2). Registered in `scripts/run_pipeline.py`. |
| 14-S5 | Lint clean + tests green | Done | 2026-08-15 | `ruff check` clean across all v2 files; 30/30 v2 tests pass; 379/379 full suite passes. |
| 14-S6 | Run end-to-end against real data, compare to v1 | Not Started | | Requires populated `clean_listings.parquet` + Step 12 artifacts on a non-empty checkout. |

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

### Spec 26 — Classification Result UI (implemented 2026-09-06)
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 26-S1 | CSS variables for VerdictBadge + AffordabilityChip in style.css | Done | 2026-09-06 | 7 new CSS vars: --verdict-good-deal, --verdict-fair-price, --verdict-overpriced, --tier-budget, --tier-mid-range, --tier-premium, --tier-luxury. |
| 26-S2 | ClassifyClient HTTP wrapper (app/services/classify_client.py) | Done | 2026-09-06 | Mirrors FastAPIClient pattern: 2.5s timeout, error collapse to ClassifyUnavailable, Pydantic response validation. |
| 26-S3 | Flask GET /classify route + classify.html template | Done | 2026-09-06 | Reuses 16-field form from predict.html; heading "Is this a good deal?"; button "Check Price Tier". endpoint="classify_get" for url_for() compatibility. |
| 26-S4 | Flask POST /classify route + classify_result.html template | Done | 2026-09-06 | VerdictBadge (primary, large, icon+text), AffordabilityChip (secondary), 4-bar horizontal tier probability chart (Chart.js), SHAP bar chart (reusing shap_format helper). Graceful degradation when FastAPI unavailable. |
| 26-S5 | predict_result.html updated with VerdictBadge + AffordabilityChip | Done | 2026-09-06 | Inline classification badges below price hero when classification succeeds; omits silently on failure. |
| 26-S6 | Tests: test_classify_route.py (16 tests) | Done | 2026-09-06 | GET form, POST happy path, validation, verdict variants, tier variants, SHAP chart, tier probability chart, layering rule, graceful degradation. All pass. |

### Week 6 — Flask App & Analytics UI
| Day | Task | Status | Actual date | Notes / Result |
|---|---|---|---|---|
| 36 | Precompute all 13 analytics_cache JSON files | Not Started | | |
| 37 | Flask scaffold + Landing page | Done | 2026-08-21 | `app/app.py` (create_app factory, landing route, stub routes for classify/analytics/recommend/insights/map). Templates extend `base.html`; `@url_for()` for every link (Rules §6.3). 9 tests in `tests/test_scaffolding.py`. Landing module-grid uses endpoint names; post-Spec-19 fix: `predict_get` (not the old `predict`) so `url_for()` resolves. |
| 38 | Price Prediction form + result page + SHAP chart | Done | 2026-08-21 | Spec 19. Flask form (Spec 18, `tests/test_predict_route.py` 16 tests) + per-prediction SHAP bar chart on `predict_result.html`. New `app/services/shap_format.py` (label map + `format_shap_for_template` + `summarize_direction`, `@lru_cache(maxsize=1)` for the on-disk label overlay). Chart.js via CDN, horizontal bar with `+`/`−`/`±` glyphs (colour is never the sole carrier of meaning, Rules §6.4). Hidden `<ul>` screen-reader fallback. Empty-state copy when FastAPI returns no SHAP. 16 helper tests (`tests/test_shap_format.py`) + 5 route-level tests green; ruff clean. |
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
| 2026-08-21 | SHAP wire format stayed minimal (`{feature, impact}`); label, direction, and normalised magnitude are re-derived on the Flask side via `app/services/shap_format.py` | Keeps the FastAPI↔Flask contract aligned with `Backend Schema §7` (and Spec 17); lets the Flask template change its display (top-N, label-source, % normalisation) without an API change. Helpers live in `app/services/`, importing only the display-only `ml.explainability.labels` label map (Rules §5.1). | Including `direction`/`label`/`pct` in the FastAPI response would duplicate derivation logic and lock the wire format to one renderer's needs. |
| 2026-08-21 | Reroute landing module-grid to `predict_get` (function-name endpoint) instead of the previous `endpoint="predict"` alias | Spec 18's `app.py` set `endpoint="predict"` for backwards-compat; `predict_result.html` + `base.html` already call `url_for("predict_get")`. After Spec 19, the canonical endpoint is the function name (Flask default), so the landing modules list was updated to match. Keeps every `url_for()` call site in lockstep — no alias left to drift. | Leaving the alias (`endpoint="predict"`) keeps one template happy but adds a long-term footgun: future contributors would have to know which endpoint is "real". |

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
| 50 | Construct price_tier labels (per-city quantile binning) | Done | 2026-08-28 | Spec 21. `ml/classification/{tiers,verdicts,thresholds,report}.py` + `scripts/build_classification_labels.py` + 27 pinned tests across `tests/test_{tiers,verdicts,thresholds,label_construction_report,build_classification_labels_cli}.py`. `price_tier` uses (0.25, 0.50, 0.75) per-(city, transact) quantile cutpoints computed on `is_outlier == False AND split == "train"` (random_state=42, reused `split_train_val_test`). `good_deal_verdict` uses two-pass: pass 1 computes residuals (`(actual - pred)/pred`); thresholds calibrated per-city `(low, high) = (median ± 0.5·IQR)`, clamped `[default_low − 0.10, default_high + 0.10]`; pass 2 assigns verdicts. Default thresholds ±10% (Rules §12.2). Null verdict rows carry `reason` in {`missing_oof_prediction`, `non_positive_predicted_price`}. Artifacts: `data/processed/classifier/{price_tier_labels,good_deal_labels}.parquet`, `price_tier_quantile_boundaries.json`, `good_deal_thresholds.json`, append-only `label_construction_report.md`. `ruff check` clean. Full suite: 564 passed, 0 failed (15 realdata deselected). CLI non-fatal in `scripts/run_pipeline.py`. Per-row `model_version` column. No contact/PII/URL fields in any input or output. |
| 51 | Generate out-of-fold regression predictions (input to Day 52) ← needs Spec 21 labels | Not Started | | Day 50 ships the label-construction half; the OOF predictions themselves arrive from Spec 15 `ml/training/evaluation.py::write_oof_predictions` (added to `__all__` in this spec) and are written to `data/processed/oof_predictions_{model_version}.parquet` for the `good_deal_verdict` residual joins. |
| 51 | Classification feature set + Logistic Regression baseline — **feature-set part** | Done | 2026-08-29 | Spec 22. `ml/classification/features.py` (`build_classifier_feature_frame`), `ml/classification/feature_report.py`, `scripts/build_classifier_features.py`, 18 tests across `tests/test_{classifier_features,classifier_feature_report,build_classifier_features_cli}.py`. Reuses price-model pipeline (`build_feature_frame` + `LocalityAggregator` + `split_train_val_test`), drops 4 price-derived columns per Rules §8.1+§12.4, retains `locality_listing_count` (count, not price-derived) and `luxury_category` (Rules §10.2 server-derived applies to API surface, not offline). Artifacts: `data/processed/classifier/feature_frame.parquet`, append-only `feature_frame_report.md`. Per-city non-outlier row counts in report. `ruff check` clean. Full suite: 582 passed, 0 failed (15 realdata deselected). CLI non-fatal in `scripts/run_pipeline.py`. |
| 51 | Classification feature set + Logistic Regression baseline — **baseline training** | Not Started | | ← needs Spec 22 feature frame |
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
| 2026-08-28 | Per-city `good_deal_verdict` threshold formula: `(low, high) = (median_residual ± 0.5 × iqr_residual)`, clamped to `[default_low − 0.10, default_high + 0.10]`, with defaults ±10% (Rules §12.2) | IQR is the most data-driven summary of the regression's natural noise floor; `±0.5 × IQR` declares a "good deal" as anything beating half a band of within-city noise. The clamp prevents pathological cities from producing bands wider than ±20%. Per-city calibration is mandatory because v2 model residuals scale with locality price level. Spec 22 may empirically tune the `0.5` multiplier. | (a) Single global `±10%` thresholds (rejected — ignores per-city noise scale); (b) `±σ` (rejected — outlier-sensitive); (c) `±1.5 × IQR` (rejected — too wide a "good deal" band; would label too many listings as such and dilute the Precision–Recall trade-off Day 52 is optimizing) |
| 2026-08-28 | Two-pass `build_good_deal_labels` design (pass 1 → residuals → calibrate thresholds → pass 2 → verdicts) | Keeps each builder pure and side-effect-free, lets Spec 22 re-call `build_good_deal_labels(thresholds=...)` with different per-city overrides in tune experiments, and audit-trail-separates "raw residuals" from "verdict assignment". One extra ~2-second left-join on the 180k-row frame. | (a) Single-pass with two-stage return tuple (rejected — couples the threshold calibrator to the builder and makes Spec 22's re-tuning awkward); (b) Persist residuals to parquet between passes (rejected — adds a new artifact class for a 2-second saving) |
| 2026-08-29 | Keep `locality_listing_count` in classifier feature frame (Spec 22) | A count describes locality popularity, not price level — it does not leak the regression target. Rules §8.1/§12.4 ban price-derived columns; a count is not price-derived. | (a) Drop all locality aggregates (rejected — loses useful popularity signal); (b) Keep `locality_avg_price_sqft` (rejected — price-derived, leaks target) |
| 2026-08-29 | Keep `luxury_category` in offline classifier feature frame (Spec 22) | Rules §10.2's "server-derived" rule applies to the **API surface** (client never sends it); offline consumption of the already-resolved value is allowed. The classifier runs offline with full historical data. | (a) Drop `luxury_category` from offline frame (rejected — loses a strong categorical signal the model already computed); (b) Compute luxury_category at serve-time only (rejected — duplicates logic; offline and online must share the same canonical value) |