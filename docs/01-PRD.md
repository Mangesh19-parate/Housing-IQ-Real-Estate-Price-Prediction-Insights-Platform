# Product Requirements Document (PRD)
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

**Version:** 1.0
**Owner:** L99
**Status:** Draft for build
**Related docs:** TRD, App Flow, UI/UX Design, Backend Schema, Implementation Plan, Tracker, Rules

---

## 1. Background & Research Basis

This PRD is based on:
1. An inspection of the actual dataset supplied (`Housing predictor.zip`) — 4 city-level scraped listing files (Gurgaon ~44.8k rows, Hyderabad ~73.8k rows, Kolkata ~32.7k rows, Mumbai ~30.8k rows) plus 15 "facet" lookup CSVs (AGE, AMENITIES, BATHROOM_NUM, BEDROOM_NUM, BUILDING_ID, CITY, FACING_DIRECTION, FEATURES, FLOOR_NUM, FURNISH, LOCALITY_ID, OWNERSHIP_TYPE, PROPERTY_TYPE, SUB_AVAILABILITY, TOTAL_FLOOR) that decode numeric/coded columns into human labels.
2. Review of 9+ public reference projects and papers on house price prediction, real-estate recommender systems, and ML model serving, including:
   - Bengaluru home price predictor (sklearn + Flask + HTML/CSS/JS) — the closest structural precedent to this project's stack.
   - Kaggle "House Prices: Advanced Regression Techniques" style pipelines (Linear/Ridge/Lasso, GradientBoosting, XGBoost).
   - MLOps-flavored house-price repos combining FastAPI + Docker + MLflow + monitoring.
   - Content-based real-estate recommender repos (TF-IDF + cosine similarity on property attributes).
   - `RE-RecSys` (ACM CODS-COMAD 2024, arXiv:2404.16553) — production real-estate recommender pattern: rule-based for cold-start users, content filtering for short-term users, content+collaborative hybrid for long-term users.
   - Comparative literature on feature selection: RFE vs Permutation Importance, SHAP-based selection vs classical importance methods.
   - Flask vs FastAPI serving-pattern articles — informs the decision to split the system into a Flask web app (UI/session/pages) and a FastAPI inference microservice (model serving), which matches the tech stack given.

## 2. Problem Statement

Buyers, sellers, brokers, and analysts in the Indian residential real-estate market (Gurgaon, Hyderabad, Kolkata, Mumbai) lack a single, data-driven tool to:
- Get a fair, explainable estimate of a property's price from its attributes.
- Understand market patterns (price by locality, city, size, amenities) visually.
- Discover comparable/alternative properties matching their needs.
- See plain-language, auto-generated insights instead of raw tables and charts.

## 3. Goals & Success Metrics

| Goal | Metric | Target |
|---|---|---|
| Accurate price prediction | R² on hold-out test set | ≥ 0.80 (city-wise), stretch ≥ 0.85 |
| Accurate price prediction | MAE (in ₹) on log-price, expressed back in ₹ | Within ±15% of actual price for 70% of test listings |
| Useful analytics | Number of distinct chart views in Analytics module | ≥ 12 (5 given + 8 new, see App Flow) |
| Useful recommendations | Relevance (manual sampling / offline precision@5 on held-out similar listings) | ≥ 0.6 |
| Usable insights | Auto-generated natural-language insight cards per prediction | ≥ 3 per prediction |
| Performance | Prediction API (FastAPI) response time | < 300ms p95 |
| Adoption (post-launch) | Weekly active sessions (if deployed) | Baseline + track |

## 4. Users / Personas

1. **Home Buyer** — wants to check if an asking price is fair, and see similar/cheaper alternatives.
2. **Seller / Broker** — wants to price a listing competitively using market comps.
3. **Analyst / Investor** — wants city/locality-level trends, not a single prediction.
4. **Project Owner (you)** — wants a portfolio-quality, end-to-end ML web app demonstrating the full data science lifecycle.

## 5. Scope — In / Out

### In scope (v1)
- 4 cities present in the data: Gurgaon, Hyderabad, Kolkata, Mumbai.
- 4 modules: **Price Prediction**, **Analytics**, **Recommender System**, **Insights**.
- Full offline ML pipeline: cleaning, EDA, outlier handling, missing-value imputation, feature engineering, feature selection, model selection, and productionization.
- Web UI: Flask-rendered pages (HTML/CSS/JS) for the app shell + forms + analytics dashboards; FastAPI as the internal model-serving/inference microservice.
- Static/batch analytics (precomputed at build/train time and served from cache/DB, not recomputed per request).

### Out of scope (v1)
- Real-time scraping / live data refresh from 99acres or any listing site.
- User accounts, authentication, saved searches, or personalized collaborative-filtering recommendations (no user interaction history exists in the dataset — see Rules doc on cold-start handling).
- Payment/lead-gen/contact-broker workflows.
- Mobile native app (web-responsive only).
- Multi-language support.

## 6. Functional Requirements by Module

### 6.1 Price Prediction Module
- FR1: User selects City → Locality → Property Type, and enters Area (sqft), Bedrooms, Bathrooms, Balconies, Furnishing, Facing, Age, Floor/Total Floor, and (optionally) Amenities checklist.
- FR2: System returns a predicted price (point estimate) and a price range (e.g., ±1 std of residuals or quantile bounds).
- FR3: System shows the top contributing features for that specific prediction (SHAP local explanation) as a simple bar/waterfall visualization.
- FR4: System validates inputs (e.g., area > 0, bedrooms ≤ bathrooms+3 as a sanity flag) and shows friendly errors rather than raw stack traces.
- FR5: Prediction requests are logged (anonymized) for later drift/monitoring analysis.

### 6.2 Analytics Module
- FR6: Dashboard with the following views (5 originally specified + 8 newly proposed — full list and rationale in **App Flow doc, Section 4**):
  1. Spatial analysis (map/scatter of listings using MAP_DETAILS lat/long)
  2. Price distribution across sectors/localities
  3. Price vs. Area analysis
  4. Number-of-rooms (BHK) pie/donut chart
  5. Top-feature word cloud (from DESCRIPTION / FEATURES / TOP_USPS)
  6. Correlation heatmap of numeric features
  7. City-wise price comparison (box plot across 4 cities)
  8. Price-per-sqft by locality heatmap
  9. Furnishing status vs. price (box plot)
  10. Property age vs. price trend
  11. Amenity count vs. price (scatter + trend line)
  12. Floor number vs. price premium (line chart)
  13. Top builders/societies by average price (bar/treemap)
- FR7: All charts filterable by City (minimum); Locality/Property Type filters as stretch.
- FR8: Charts are precomputed server-side (pandas) and rendered via a JS charting library, not recalculated live per pageview.

### 6.3 Recommender System Module
- FR9: Given a seed property (either a listing ID or a user-entered spec), return top-N (default 5) similar properties.
- FR10: Similarity is content-based: numeric features (price, area, bedrooms, bathrooms, floor, age) + categorical (furnish, facing, property type) + text features (TF-IDF on description/amenities) combined into one feature vector, ranked by cosine similarity.
- FR11: Recommendations are city/locality-scoped by default (don't recommend a Mumbai flat for a Gurgaon search) unless user opts into "expand search."
- FR12: Because there is no click/interaction history, this module is Content-Based only in v1 (see Rules doc — no fabricated collaborative-filtering signal). RE-RecSys-style cold-start rule-based fallback (rank by locality popularity + recency) is used when the seed has very few neighbors.

### 6.4 Insights Module
- FR13: After a prediction or a recommender query, auto-generate 3–5 plain-English insight sentences, e.g., "This price is 12% below the average for 3BHK flats in Sector 84, Gurgaon" or "Properties with a Clubhouse amenity sell for ~8% more in this locality on average."
- FR14: Insights are template-based, populated from precomputed aggregate stats (locality averages, feature-uplift stats) — not free-form LLM generation, to keep them auditable and fast.
- FR15: Insight cards are shown on the Price Prediction result screen and as a standalone "Market Insights" page summarizing city-level takeaways.

## 7. Non-Functional Requirements
- NFR1: Works on desktop and mobile-responsive web (Flask templates with responsive CSS).
- NFR2: FastAPI inference service must be independently restartable without taking down the Flask app (graceful degradation: show a "predictions temporarily unavailable" state).
- NFR3: All model artifacts versioned (filename includes date/version) and loaded from a fixed `/models` path.
- NFR4: No PII is stored; dataset contact fields (CONTACT_NAME, phone-like fields, dealer info) are dropped in the cleaning stage and never surfaced in the UI.
- NFR5: Codebase organized so ML pipeline (offline, notebooks/scripts) is cleanly separated from serving code (Flask/FastAPI).

## 8. Assumptions
- The 4 provided city CSVs are the full source of truth for v1; no additional scraping is done.
- "Ground truth" price is derived from `PRICE` / `MIN_PRICE` / `MAX_PRICE` / `PRICE_SQFT` fields (which need heavy cleaning — see TRD).
- Facet CSVs are the authoritative decode tables for coded columns (e.g., `FURNISH`, `FACING`, `AGE`, `PROPERTY_TYPE`).

## 9. Risks
| Risk | Impact | Mitigation |
|---|---|---|
| Price/Area fields are free-text strings ("3.5 Cr", "2700 sq.ft.") with inconsistent units | Parsing errors, wrong scale | Dedicated parsing functions + unit tests (see TRD §5) |
| Heavy missingness in some columns (e.g., QUALITY_SCORE, FURNISHING_ATTRIBUTES ~100% null in Gurgaon sample) | Wasted features | Drop columns with >60–70% missingness; impute the rest |
| Extreme outliers in PRICE_SQFT (seen up to ₹8.5 crore/sqft in a 5k-row sample — clearly data errors) | Model skew | IQR/percentile capping + log-transform target (see TRD §6) |
| No user interaction data → cannot build true collaborative filtering | Recommender limited to content-based | Explicitly scoped as content-based in PRD/Rules; documented as a known v2 opportunity |
| 4 separate CSVs with different schemas (Gurgaon has more columns than Kolkata) | Schema merge complexity | Build one canonical schema; map/rename per-city columns into it (see Backend Schema doc) |

## 10. Milestones (high level — see Implementation Plan for day-by-day)
1. Data understanding & cleaning
2. EDA & feature engineering
3. Feature selection & modeling
4. Recommender + Insights logic
5. Flask + FastAPI integration
6. UI polish, analytics dashboards
7. Testing, documentation, deployment prep

---

## UPDATE v2 — Classification Module, Reference-Project Benchmarking & 35% Improvement Target

*(Added after reviewing 22 research papers and the supplied reference project `dsmp-capstone-project-master.zip`. Full detail in `09_LITERATURE_REVIEW_AND_IMPROVEMENT_PLAN.md` — this section only states what changes in the PRD itself.)*

### U1. Reference project confirmed & positioned
The supplied reference project is a single-city (Gurgaon-only) capstone doing regression + a manual luxury-score bin + a pure content-based recommender, with no serving layer and no formal explainability. This project explicitly extends it to 4 cities, a real Flask+FastAPI serving architecture, SHAP-based explainability, a cold-start-aware recommender, and (new) a proper Classification module. See literature doc §1 and §4 for the full gap analysis.

### U2. New Module — Price-Tier & "Good Deal" Classification
Added as a 5th module, reusing the same cleaned/engineered feature set as Price Prediction (no new data collection):
- **FR16:** Classify each listing into `Budget` / `Mid-Range` / `Premium` / `Luxury`, based on city-relative `price_per_sqft` quantiles.
- **FR17 (stretch):** Classify a listing as `Good Deal` vs `Overpriced` by comparing its actual price against the regression model's predicted price.
- **FR18:** Show the tier as a badge on the Price Prediction result screen; add it as an Analytics tile (14th) and a Recommender filter facet.
- Full technical spec: TRD Update §U-TRD-1.

### U3. Revised Success Metrics (Goal table addition)

| Goal | Metric | Target |
|---|---|---|
| Improved price accuracy vs. single-model baseline | MAE/RMSE reduction | **~30–35%** reduction vs. an unoptimized single-model (Linear Regression) baseline, validated empirically per model version (see literature doc §5) |
| Price-tier classification quality | Macro-F1 | ≥ 0.75 across 4 tiers |
| Price-tier classification quality | Multi-class ROC-AUC (OvR) | ≥ 0.85 |

### U4. New Risk
| Risk | Impact | Mitigation |
|---|---|---|
| Price-tier label leaking into the price regression as a feature (it's derived from price) | Artificially inflated regression accuracy, invalid model | Hard rule in Rules doc: tier label is display/analytics-only, never a regression input feature |

### U5. Scope note
Tech stack confirmed unchanged: Flask + HTML/CSS/JS (frontend), FastAPI (inference/classification serving), ML models (scikit-learn/XGBoost). No new stack components required for the Classification module — it is served via a new FastAPI route, same as `/predict` and `/recommend`.

---

## UPDATE v3 — Finalized 16-Field Input Schema & UML Diagrams

*(Full field-by-field detail in `10_FINALIZED_INPUT_SCHEMA.md`; full diagram set in `11_UML_DIAGRAMS.md`.)*

### U3.1 Price Prediction input contract — now locked
The Price Prediction (and Classification) modules take these **12 user-specified fields**: Property Type, Sector, No. of Bedrooms, No. of Bathrooms, Balconies, Property Age, Built-up Area, Servant Room, Store Room, Furnishing Type, Luxury Category, Floor Category — matched field-for-field against the reference project's actual final model file (`gurgaon_properties_post_feature_selection_v2.csv`).

**Plus 4 additions** (rationale in schema doc §2), because this project is broader in scope than the single-city reference project:
- **City** — required since 4 cities are in scope, not 1; disambiguates `sector`.
- **Facing Direction** — available in the raw data, unused in the reference project, worth testing.
- **Amenities (multi-select)** — replaces the reference project's single opaque `luxury_score` with a transparent, SHAP-explainable feature set.
- **Transaction Type (Sale/Rent)** — needed because this dataset mixes both, unlike the reference project.

### U3.2 Functional Requirement update
FR1 (Price Prediction form) is revised to: *"User selects City → Sector/Locality → Property Type → Transaction Type, and enters Bedrooms, Bathrooms, Balconies, Built-up Area, Property Age, Furnishing Type, Facing Direction, Floor Category, Servant Room (Y/N), Store Room (Y/N), and either self-reports or is guided through deriving a Luxury Category, plus an optional Amenities multi-select."*

### U3.3 UML Diagrams added
This project now includes a full UML diagram set (`11_UML_DIAGRAMS.md`): Use Case, Class, 2× Sequence (Prediction+Classification parallel call, Recommender with fallback), Activity (offline ML pipeline), Component, Deployment, ER/Data Model, and a State diagram for a prediction request's lifecycle. These are the authoritative structural/behavioral reference going forward — if this PRD's functional requirements and the UML diagrams ever disagree, the diagrams should be treated as needing a documentation fix, not silently trusted over the PRD.

---

## UPDATE v4 — Classification Module Reframed (Real Purpose, Not Pattern-Completeness)

Following external review and explicit decision: **Keep Flask. Skip added MLOps tooling. Reframe Classification.**

### U4.1 Classification module's purpose, restated
Previously justified as "generalizes the reference project's ad-hoc luxury_score bin into a trained classifier" — a documentation-integrity reason, not a business one. **New stated purpose:** the Classification module is an **Affordability & Investment-Tier Filter** that answers two real user questions the regression model alone cannot:
1. *Buyer:* "Which listings can I actually afford, without me manually comparing every predicted price to my budget?" → `price_tier` becomes a **search/recommender filter facet** (already wired in App Flow — now it has a stated reason to exist there).
2. *Investor/Seller:* "Is this listing under- or over-valued relative to its market segment?" → the `good_deal_flag` (previously "stretch") is now the module's **primary** deliverable, not secondary — it's the investment-decision signal, while `price_tier` is the supporting affordability-segmentation signal.

### U4.2 FR13 (Classification) — revised
- **FR16 (was tier-only):** Classify each listing into an Affordability Tier (`Budget`/`Mid-Range`/`Premium`/`Luxury`, city-relative), used as a **filter in the Recommender module** and a **budget-matching signal** on the Price Prediction result screen ("This fits your Mid-Range search").
- **FR17 (promoted from stretch to primary):** Classify a listing as `Good Deal` / `Fair Price` / `Overpriced` (3-class, not binary — see TRD Update v4) by comparing actual price to the regression model's predicted price. This is now the module's headline feature for interview/demo purposes: *"the app tells you not just what a home should cost, but whether this specific listing is a good deal."*

### U4.3 Interview-facing framing (for your own use, not a system requirement)
If asked "why classify when regression already predicts price?" — answer: *"Regression gives a number. Classification turns that number into a decision — which affordability segment it's in, and whether it's priced fairly relative to that segment. That's the difference between a price estimator and a decision-support tool."*

### U4.4 Explicitly not changed
- Frontend stack: **Flask remains** (HTML/CSS/JS), unchanged from original spec.
- No MLflow/DVC/CI-CD/drift-detection additions — scope stays as previously documented (7–8 week solo build), per your explicit choice to tighten rather than expand.