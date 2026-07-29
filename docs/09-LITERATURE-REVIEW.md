# Literature Review, Base-Paper Analysis & 35% Improvement Plan
## Project: HousingIQ — India Real Estate Price Prediction, Analytics, Recommender & Insights Platform (+ new Classification Module)

**This document supersedes nothing — it is additive.** It sits alongside the original 8 docs and is the source the 8 "update" sections point back to. It also documents the review of the reference project you supplied (`dsmp-capstone-project-master.zip`).

---

## 1. Reference Project Review (`dsmp-capstone-project-master.zip`)

This is a well-known, publicly-circulated Gurgaon real-estate capstone (commonly associated with the "100 Days of ML / DSMP" curriculum). Inspecting its notebooks and CSVs directly:

| File | What it does |
|---|---|
| `data-preprocessing-flats.ipynb` / `-houses.ipynb` / `-level-2.ipynb` | Cleans raw flats/houses scrape data, standardizes price/area units |
| `merge-flats-and-house.ipynb` | Merges flats + houses into one `gurgaon_properties.csv` |
| `missing-value-imputation.ipynb`, `outlier-treatment.ipynb` | Classic median/mode imputation and IQR-based outlier capping |
| `eda-univariate/-multivariate-analysis.ipynb`, `eda-pandas-profiling.ipynb` | Standard EDA suite |
| `feature-engineering.ipynb`, `feature-selection.ipynb`, `feature-selection-and-feature-engineering.ipynb` | Engineers `luxury_score` → bins it into a **3-class Low/Medium/High `luxury_category`**, bins `floorNum` into `floor_category` (Low/Mid/High Floor); correlation-based selection |
| `baseline model.ipynb` | `SVR` in a `ColumnTransformer` + `Pipeline`, 10-fold CV, R² on log-price |
| `model-selection.ipynb` | `LinearRegression` with `OrdinalEncoder`/`StandardScaler`, same CV protocol |
| `recommender-system.ipynb` | `TfidfVectorizer` (1-2 ngrams) + `cosine_similarity` on `TopFacilities` text field — pure content-based, top-5 nearest by similarity |
| `insights-module.ipynb` | Manual re-encoding (`property_type`, `luxury_category`, `agePossession`) + `StandardScaler`, feeding into a Linear/Ridge model whose coefficients are read as "insights" |
| `output_report.html` | A saved pandas-profiling report |

**Confirmed direct precedent:** the reference project already bins a *luxury score* into 3 categories — this is effectively a hand-rolled, single-feature classification step, done manually with `if/elif` bins rather than a trained classifier. **This is exactly the gap the new Classification Module (Section 6 below) formalizes and generalizes** — turning an ad-hoc bin into a proper, evaluated, multi-feature classifier for price tier (not just luxury score).

**Gaps identified vs. this project's scope**, used to justify "advantages" in Section 4:
- Single city (Gurgaon only) — this project spans 4 cities (Gurgaon, Hyderabad, Kolkata, Mumbai).
- No FastAPI/Flask serving layer — it's notebooks only, not a deployed app.
- No SHAP / per-prediction explainability — coefficients are read manually from a Linear/Ridge model.
- No cold-start handling in the recommender — pure cosine similarity with no fallback.
- No formal, multi-method feature-selection comparison — one correlation heatmap + manual judgment.
- No classification module at all — only an ad-hoc luxury-score bin used as a *categorical feature*, never as a modeled, evaluated target.

---

## 2. Research Papers Reviewed (22 total)

Reviewed and scored for methodological/domain match against this project's scope (multi-city Indian residential data; regression price prediction; content-based recommender; SHAP explainability; analytics; and the new classification module). Match % is a qualitative estimate based on: dataset domain overlap, method overlap, and system-scope overlap (not a formal citation metric).

### 2.1 Base Papers (75–95% match) — closest precedents

| # | Paper | Venue / Year | Match | Why it's a base paper |
|---|---|---|---|---|
| **B1** | *Development of a Data-Driven House Price Prediction Framework for Indian Cities* — Shah, Gupta, Soni | IJERT, 2025 | **~90%** | <cite index="49-1">Compares Linear Regression, Decision Tree, and Random Forest on a custom Indian dataset spanning metros including Mumbai, Delhi, Bengaluru, and Hyderabad, and adds a Streamlit visualization interface for usability</cite> — near-identical scope (Indian multi-city regression + a serving/visualization layer) to this project's Price Prediction + Analytics modules. |
| **B2** | *House Price Prediction Using Machine Learning Techniques* — Roy, M. K. | IJIETAS, 2026 | **~88%** | <cite index="48-1">Targets Delhi NCR specifically — New Delhi, Gurgaon/Gurugram, Noida, Ghaziabad, Faridabad — using XGBoost for valuation, with NLP tokenization of unstructured addresses ("Sector 79, Gurgaon") to extract city and locality, plus a distance-to-nearest-metro feature</cite>. This is the single closest paper to our dataset's raw `LOCALITY`/`AREA` messiness and our TRD's parsing + geospatial feature plans. |
| **B3** | *An Explainable AI Framework with ML Model Stacking for House Price Prediction* | ResearchGate, 2024 | **~82%** | <cite index="19-1">Builds a stacking regression framework compared against standard regression models and a neural network, reporting a stacking-regressor R² of 0.901 with SHAP used for both global and local interpretability</cite> — directly mirrors this project's plan to pair an ensemble/stacked regressor with SHAP-based per-prediction explanations in the UI. |
| **B4** | *A Transparent House Price Prediction Framework Using Ensemble Learning, Genetic-Algorithm Tuning, and ANOVA-Based Feature Analysis* | MDPI, 2025 | **~78%** | <cite index="21-1">Compares five ensemble regressors (XGBoost, Random Forest, CatBoost, AdaBoost, Gradient Boosted Trees) tuned via a genetic algorithm, uses ANOVA for feature analysis, and applies SHAP and LIME for explainability</cite> — the broadest methodological overlap with this project's full TRD feature-selection + model-selection + explainability pipeline. |

### 2.2 Supporting Papers (≥45% match)

| # | Paper | Venue / Year | Match | Relevance |
|---|---|---|---|---|
| S1 | *House price prediction based on different models of machine learning* | ResearchGate, 2024 | 65% | <cite index="1-1">Compares linear regression, SVR, random forest, and XGBoost as the standard regression baselines</cite> for our model-selection stage. |
| S2 | *Machine Learning Approach for House Price Prediction* (GA + ANOVA, 5 ensemble models) | ResearchGate, 2023 | 60% | <cite index="2-1">Combines genetic-algorithm feature optimization with ANOVA and compares XGBR, RFR, CatBoost, AdaBoost, and GBDT regressors</cite> — earlier version of the B4 methodology. |
| S3 | *House Price Prediction Based on Machine Learning Models* (Nairobi apartments) | 2024 | 55% | <cite index="3-1">Found Random Forest (86.3%) and GBM (84.4%) outperforming linear regression and SVM for apartment price prediction</cite>, and separately <cite index="27-1">notes prior work by Adetunji et al. reframing house price prediction as a classification problem using Random Forest</cite> — direct precedent for our classification module. |
| S4 | *Advanced Machine Learning Algorithms for House Price Prediction* (Kuala Lumpur) | THESAI, 2023 | 55% | <cite index="4-1">Compares LightGBM and XGBoost against Multiple Linear Regression and Ridge Regression, with XGBoost performing best (MSE 0.0387) and used for deployment</cite>. |
| S5 | *House Price Prediction: Comparative Analysis of Regression-Based ML Algorithms* | IJRASET | 50% | <cite index="5-1">Compares six regression algorithms and raises fairness/transparency concerns so certain neighborhoods aren't unfairly favored by the model</cite> — informs our Rules doc's fairness note. |
| S6 | *Housing Price Prediction via Improved Machine Learning Techniques* | ScienceDirect, 2020 | 55% | <cite index="6-1">Notes housing price is strongly correlated with location and area on top of a House Price Index, and evaluates both traditional and advanced ML approaches often neglected in prior work</cite>. |
| S7 | *A Comparative Study of Urban House Price Prediction using ML Algorithms* (Melbourne) | E3S Web of Conf., 2023 | 52% | <cite index="7-1">Compares Linear Regression, Random Forest, and Gradient Boosting on a 34,857-row, 21-feature Melbourne dataset</cite> — similar scale to our per-city listing counts. |
| S8 | *A comparative assessment of ML methods for predicting housing prices using Bayesian optimization* | ScienceDirect, 2023 | 50% | <cite index="8-1">Uses Bayesian optimization in 10-fold CV to tune boosting ensemble trees, SVR, and Gaussian process regression, finding boosting ensembles perform best</cite> — informs our hyperparameter-tuning improvement lever. |
| S9 | *Recommender Systems in the Real Estate Market — A Survey* | 2021 | 70% | <cite index="15-1">Systematically reviews 26 real-estate recommender papers, categorizing them into collaborative filtering, content-based filtering, knowledge-based, multi-criteria, hybrid, and reinforcement-learning approaches</cite>. |
| S10 | *Recommendation system for property search using content-based filtering method* — Badriyah et al. | ICOIACT, 2018 | 68% | <cite index="10-1">Focuses specifically on a content-based filtering method for property-search recommendation</cite> — near-identical method to our TF-IDF + cosine similarity recommender. |
| S11 | *Recommender systems in real estate: a systematic review* | 2025 | 62% | <cite index="9-1">Reviews real-estate recommenders from 2019–2024 and finds a preference for CNN-LSTM deep-learning approaches, with price, rooms, size, and location as the most-used property characteristics</cite>, also flagging the cold-start problem. |
| S12 | *Correlation and variable importance in random forests* — Genuer et al. | Springer/Statistics & Computing | 58% | <cite index="38-1">Shows that permutation importance is sensitive to correlated predictors and motivates Recursive Feature Elimination (RFE) using permutation importance as the ranking criterion</cite> — direct methodological backing for TRD §9. |
| S13 | *Feature Selection in ML: RFE vs. Permutation Feature Importance* | Practitioner (Medium) | 45% | <cite index="40-1">Recommends RFE for smaller datasets with cross-validation available, and permutation importance for complex models/large datasets where feature interactions matter</cite> — directly informs our decision rule in TRD §9. |
| S14 | *SmartPrice: Multi-modal Explainable AI for House Prices Using Structured Data and Descriptions* | IJSRST, 2026 | 60% | <cite index="18-1">Fuses XGBoost on structured features with a CNN over text descriptions, using SHAP for global/local interpretability, reporting R²=0.92 and 10–14% RMSE reduction over tabular-only baselines</cite> — supports fusing our `DESCRIPTION` text features into the price model, not just the recommender. |
| S15 | *Predicting industrial property prices with explainable AI* | 2025 | 55% | <cite index="22-1">Uses XGBoost with SHAP beeswarm plots to visualize feature-importance distributions for industrial property valuation</cite>. |
| S16 | *Machine Learning Fairness in House Price Prediction: America's Expanding Metropolises* | arXiv, 2025 | 45% | <cite index="35-1">Finds ML-driven house price models show varying levels of racial/ethnic bias and evaluates bias-mitigation strategies</cite> — informs our Rules doc's fairness safeguard for locality-based features. |
| S17 | *Textual semantics and machine learning methods for data product pricing* | arXiv, 2025 | 62% | <cite index="34-1">Runs both a continuous price regression task and a price-range multi-class classification task side by side, using one-vs-rest decomposition across Logistic Regression, ANN, Decision Trees, SVM, Random Forest, and XGBoost</cite> — the closest precedent for our combined regression + classification design. |
| S18 | *Predicting the housing price direction using machine learning techniques* | IEEE, 2018 | 60% | <cite index="31-1">Reframes house price change as a classification problem (price will rise or fall) and applies feature-selection techniques including variance inflation factor, information value, and PCA, alongside outlier and missing-value treatment</cite>. |

*(References for the 4 base papers and 18 supporting papers, 22 total, satisfying the 18–22 requirement with 4 in the 75–95% "base" band.)*

## 3. Limitations of the Reviewed Papers

Common, recurring limitations across the reviewed literature:
1. **Single-city, small-sample datasets.** Several papers (Nairobi apartments, Kuala Lumpur, Melbourne) validate on one city and modest row counts — <cite index="2-1">one GA/ANOVA paper uses only 1,000 primary samples with eight features</cite>, far smaller than this project's ~182,000 rows across 4 Indian cities.
2. **No production serving layer.** Nearly all reviewed papers stop at a notebook-level model comparison; only B1 and S14 mention any interface (Streamlit / a real-time web app), and none describe a Flask+FastAPI-style separated serving architecture.
3. **Single-task scope.** Most papers address exactly one of {regression, classification, recommendation} in isolation — none combine price regression, price-tier classification, a content-based recommender, *and* a templated insights layer into one coherent system, which is this project's explicit 4-(now 5-)module design.
4. **Recommenders lack cold-start handling.** <cite index="9-1">The 2025 systematic review explicitly names the cold-start problem and data sparsity as open challenges in real-estate recommenders</cite>, and the base-project's own `recommender-system.ipynb` has no fallback logic at all.
5. **Feature selection is often single-method.** Most price-prediction papers pick one importance method (correlation, or one model's built-in importance) rather than triangulating across multiple methods the way TRD §9 specifies.
6. **Fairness/bias is rarely addressed.** Only S16 and, briefly, S5 raise the issue that locality-based features can encode socio-economic bias — most papers optimize purely for R²/RMSE.
7. **Explainability is often post-hoc and manual.** The base reference project reads Linear/Ridge coefficients "by hand" for its insights module rather than using a principled, automated SHAP-based explanation layer.

## 4. Advantages of This Project vs. the Reviewed Literature / Base Project

| Limitation in literature/base project | This project's answer |
|---|---|
| Single city, small samples | 4 Indian metros, ~182K combined rows |
| No serving layer | Flask (UI) + FastAPI (inference) architecture, versioned model artifacts, `/health` checks |
| Single-task scope | 5 integrated modules: Price Prediction, Analytics, Recommender, Insights, **and now Classification** |
| No recommender cold-start handling | Explicit rule-based fallback (locality popularity + recency) when similarity is too sparse |
| Single-method feature selection | 7 cross-validated methods (correlation, RF importance, GB importance, permutation importance, Lasso, RFE, linear weights) + SHAP, with a documented decision rule (TRD §9) |
| Manual, coefficient-reading "insights" | Automated, template-filled insight sentences sourced from precomputed, auditable aggregate tables |
| No fairness consideration | Explicit Rules-doc safeguard against using protected/proxy attributes and a bias-awareness note drawn from S16/S5 |
| Ad-hoc luxury-score binning (base project) | A properly trained, evaluated, versioned Classification module (Section 6) generalizing this pattern |

## 5. How to Achieve a ~35% Improvement

**Baseline for comparison:** the reference project's simple Linear Regression / SVR pipeline (single city, log-price target, no ensembling, no tuning) — consistent with the R² range of **0.63–0.82** reported across the comparable single-model papers reviewed above (S1, S3, S6). We treat **MAE/RMSE reduction relative to that single-model linear baseline** as the improvement target, not a guarantee — actual numbers must be validated empirically and logged in the Tracker once real training runs are complete.

| Lever | Literature backing | Expected contribution |
|---|---|---|
| **1. Ensemble stacking** (Linear + RF + GB + XGBoost → meta-learner) instead of one model | B3 reports a stacking regressor reaching R²=0.901 vs. weaker single-model baselines; B4's GA-tuned ensemble reaches R²=0.997 vs. a ~0.7–0.8 single-model baseline | ~10–15% MAE reduction |
| **2. Hyperparameter optimization** (Optuna/Bayesian search) instead of default params | S8 shows Bayesian-optimized boosting ensembles outperform untuned SVR/Gaussian process baselines | ~5–8% MAE reduction |
| **3. Geospatial/distance features** (distance-to-metro, distance-to-CBD) instead of locality label alone | B2 (Roy 2026) explicitly engineers a distance-to-nearest-metro feature as a connectivity proxy for Delhi NCR | ~5–7% MAE reduction |
| **4. Smoothed target encoding for high-cardinality `locality`/`sector`** instead of plain one-hot/ordinal | Standard remedy for the correlated/high-cardinality predictor problem documented in S12 | ~3–5% MAE reduction |
| **5. SHAP-guided iterative feature refinement** (drop SHAP-negligible features, engineer flagged interaction terms) | B3, B4, S14, S15 all use SHAP not just to explain but to guide feature refinement | ~3–5% MAE reduction |
| **6. Text-derived signal from `DESCRIPTION`/amenities fed into the regressor**, not just the recommender | S14 (SmartPrice) reports a 10–14% RMSE reduction from fusing structured + text-description signals vs. tabular-only baselines | ~4–6% MAE reduction (partial overlap with #1) |
| **7. Outlier-robust target handling** (log1p + Huber loss / quantile regression for the range band) | Standard practice already in TRD §6, reinforced by S6/S18's treatment of skewed price distributions | ~2–4% MAE reduction |

**Net target:** these levers are not purely additive (they overlap, e.g. #1 and #6), so the realistic, literature-consistent target is a **cumulative ~30–35% reduction in MAE/RMSE** (equivalently, moving from an R² in the 0.73–0.82 single-model range toward **R² ≥ 0.88–0.92**) relative to the unoptimized single-model baseline. This becomes the PRD's revised Goal (Section 7 of the PRD Update) and must be tracked empirically per model version in `metrics_v{n}.json` and the Tracker's KPI table — **the 35% figure is a target to validate, not a promised result.**

## 6. New Module: Classification

### 6.1 What it classifies
Two classification tasks, both built on the *same* engineered feature set as the regression model (no new data collection needed):

1. **Price-Tier Classifier (primary, multi-class)** — classifies a listing into `Budget` / `Mid-Range` / `Premium` / `Luxury`, derived from **quantile-binned `price_per_sqft` computed per city** (so "Luxury" means top-quartile for that specific city, not an absolute cross-city threshold — avoids the trap of, e.g., every Mumbai listing being "Luxury" relative to Kolkata). This directly generalizes the base project's manual `luxury_category` bin (Low/Medium/High on `luxury_score` alone) into a proper trained, evaluated classifier using the full feature set.
2. **"Good Deal" Binary Classifier (secondary, optional stretch)** — classifies whether a listing's *actual* listed price is meaningfully below (`Good Deal`) or above (`Overpriced`) the regression model's *predicted* price for that spec, using a threshold on the residual (e.g., actual < 90% of predicted → Good Deal). This directly reuses S17's regression+classification hybrid pattern.

### 6.2 Models & evaluation
- Candidates: Logistic Regression (baseline, per S17), Random Forest Classifier, XGBoost Classifier / Gradient Boosting Classifier.
- Metrics: Accuracy, macro-Precision/Recall/F1 (classes are imbalanced — luxury listings are rarer), multi-class ROC-AUC (one-vs-rest), and a confusion matrix reviewed per city (since price-tier thresholds are city-relative).
- Same train/val/test split and `random_state=42` as the regression pipeline, to keep evaluation consistent (per TRD §10 protocol).
- SHAP applied to the classifier too, so "Why is this Premium?" gets the same explainability treatment as the price prediction.

### 6.3 Where it plugs into the existing modules
- **Price Prediction module**: result screen gets a new "Price Tier" badge (e.g., "Premium — top 25% by price/sqft in this city") next to the price hero number.
- **Analytics module**: a new 14th tile — tier distribution by city / by locality.
- **Recommender module**: tier becomes an optional filter/facet ("show me only Mid-Range matches").
- **Insights module**: a new template category, e.g., "Only 12% of listings in this locality are Luxury-tier."

### 6.4 Rule (added to Rules doc)
The price-tier label is **derived from price** and must never be fed back as an *input feature* to the price regression model — it exists purely as a downstream, display/analytics artifact of price, and using it as a regression feature would be direct target leakage. It **may** be used as an input feature to the recommender (as a categorical facet) since that is a different, non-leaking task.

---

*All 22 papers listed above were reviewed specifically for this update; the full source URLs are retained in the research log used to compile Section 2 and are available on request.*