# Backend Schema Document
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

---

## 1. Storage Overview

| Store | Technology | Purpose |
|---|---|---|
| Raw data | Original CSVs (`gurgaon_10k.csv`, `hyderabad.csv`, `kolkata.csv`, `mumbai.csv`, `facets/*.csv`) | Immutable source of truth, never modified |
| Processed/clean data | Parquet (`data/processed/clean_listings.parquet`) | Canonical, cleaned, feature-engineered dataset used for training and serving lookups |
| Precomputed analytics | JSON files (`data/processed/analytics_cache/*.json`) | Fast-loading chart data for the Analytics module (no live recompute) |
| Aggregate stats for Insights | Parquet/SQLite tables (`locality_stats`, `amenity_uplift`, `age_price_trend`, `bhk_price_trend`) | Source for the Insights templater |
| Model artifacts | joblib/pickle files under `models/` | Trained price model, preprocessing pipeline, TF-IDF vectorizer, similarity index |
| Application DB | SQLite (dev) / PostgreSQL (prod path) | Prediction request logs, locality/city reference tables served via API, optional saved-search stretch feature |

## 2. Canonical Listing Schema (post-cleaning)

This is the unified schema all 4 raw city files are mapped into (raw column names shown for traceability).

| Canonical field | Type | Source (raw) | Notes |
|---|---|---|---|
| `listing_id` | string (PK) | `PROP_ID` | Deduplicated |
| `city` | string (FK → `city_ref.city_id`) | `CITY` | Decoded via `facets/CITY.csv` |
| `locality` | string (FK → `locality_ref.locality_id`) | `LOCALITY` / `location.LOCALITY_NAME` | Decoded via `facets/LOCALITY_ID.csv` where applicable |
| `property_type` | string (FK → `property_type_ref`) | `PROPERTY_TYPE` | Decoded via `facets/PROPERTY_TYPE.csv` |
| `transact_type` | string | `TRANSACT_TYPE` | Sale/rent flag if present |
| `ownership_type` | string (FK → `ownership_ref`) | `OWNTYPE` | Decoded via `facets/OWNERSHIP_TYPE.csv` |
| `bedrooms` | int | `BEDROOM_NUM` | |
| `bathrooms` | int | `BATHROOM_NUM` | |
| `balconies` | int | `BALCONY_NUM` | |
| `furnish` | string (FK → `furnish_ref`) | `FURNISH` | Decoded via `facets/FURNISH.csv` |
| `facing` | string (FK → `facing_ref`) | `FACING` | Decoded via `facets/FACING_DIRECTION.csv` |
| `age_bucket` | string (FK → `age_ref`) | `AGE` | Decoded via `facets/AGE.csv` |
| `floor_num` | string/int | `FLOOR_NUM` | Non-numeric values (B=Basement, G=Ground, L=Lower Ground, M=Multi-storied) mapped via `facets/FLOOR_NUM.csv` |
| `total_floor` | int | `TOTAL_FLOOR` | |
| `floor_ratio` | float (engineered) | `floor_num / total_floor` | |
| `area_sqft` | float | `AREA` (parsed) / `SUPER_SQFT` / `BUILTUP_SQFT` / `CARPET_SQFT` | Cleaned per TRD §4.2 |
| `price_inr` | float | `PRICE` (parsed) / `MIN_PRICE` / `MAX_PRICE` | Cleaned per TRD §4.1 |
| `price_per_sqft` | float (engineered) | derived | Also cross-checked vs raw `PRICE_SQFT` |
| `features_list` | list[string] | `FEATURES` (decoded) | Via `facets/FEATURES.csv` |
| `amenities_list` | list[string] | `AMENITIES` (decoded) | Via `facets/AMENITIES.csv` |
| `n_amenities` | int (engineered) | derived | |
| `n_features` | int (engineered) | derived | |
| `has_<amenity>` | bool (engineered, ~15 cols) | derived | One per top-15 amenity |
| `building_name` | string | `BUILDING_NAME` / `SOCIETY_NAME` | |
| `building_id` | string (FK → `building_ref`) | `BUILDING_ID` | Decoded via `facets/BUILDING_ID.csv` |
| `latitude` | float | `MAP_DETAILS.LATITUDE` | Parsed from stringified dict |
| `longitude` | float | `MAP_DETAILS.LONGITUDE` | Parsed from stringified dict |
| `description_clean` | text | `DESCRIPTION` | Cleaned for TF-IDF/word cloud |
| `register_date` | date | `REGISTER_DATE` | Present in Gurgaon/Hyderabad/Mumbai only; null for Kolkata |
| `is_outlier` | bool (engineered) | derived | Flag from TRD §6, excluded from training only |
| `was_missing_<field>` | bool (engineered, per imputed field) | derived | Imputation flags per TRD §5 |

## 3. Reference (Lookup) Tables — from `facets/`

| Table | Columns | Source file |
|---|---|---|
| `city_ref` | `city_id`, `city_label` | `CITY.csv` |
| `locality_ref` | `locality_id`, `locality_label` | `LOCALITY_ID.csv` |
| `property_type_ref` | `type_id`, `type_label` | `PROPERTY_TYPE.csv` |
| `ownership_ref` | `own_id`, `own_label` | `OWNERSHIP_TYPE.csv` |
| `furnish_ref` | `furnish_id`, `furnish_label` | `FURNISH.csv` |
| `facing_ref` | `facing_id`, `facing_label` | `FACING_DIRECTION.csv` |
| `age_ref` | `age_id`, `age_label` | `AGE.csv` |
| `sub_availability_ref` | `sub_avail_id`, `sub_avail_label` | `SUB_AVAILABILITY.csv` |
| `floor_ref` | `floor_code`, `floor_label` | `FLOOR_NUM.csv` |
| `total_floor_ref` | `floor_id`, `floor_label` | `TOTAL_FLOOR.csv` |
| `bedroom_ref` | `bed_id`, `bed_label` | `BEDROOM_NUM.csv` |
| `bathroom_ref` | `bath_id`, `bath_label` | `BATHROOM_NUM.csv` |
| `building_ref` | `building_id`, `building_label` | `BUILDING_ID.csv` |
| `features_ref` | `feature_id`, `feature_label` | `FEATURES.csv` |
| `amenities_ref` | `amenity_id`, `category`, `type`, `amenity_label` | `AMENITIES.csv` |

## 4. Aggregate / Precomputed Tables (for Analytics & Insights)

**`locality_stats`**
| Column | Type |
|---|---|
| `city` | string |
| `locality` | string |
| `avg_price` | float |
| `median_price` | float |
| `avg_price_per_sqft` | float |
| `avg_area` | float |
| `listing_count` | int |
| `avg_bedrooms` | float |

**`amenity_uplift`**
| Column | Type |
|---|---|
| `city` | string |
| `locality` | string (nullable → city-level fallback row when null) |
| `amenity` | string |
| `avg_price_with` | float |
| `avg_price_without` | float |
| `pct_uplift` | float |

**`age_price_trend`** / **`bhk_price_trend`** / **`furnish_price_trend`** / **`floor_price_trend`**
| Column | Type |
|---|---|
| `city` | string |
| `bucket` (age_bucket / bedrooms / furnish / floor_bucket) | string/int |
| `avg_price` | float |
| `avg_price_per_sqft` | float |
| `listing_count` | int |

**`analytics_cache/*.json`** — one JSON file per chart id (e.g., `chart_06_correlation_heatmap.json`, `chart_09_furnish_vs_price.json`) containing pre-shaped `{labels: [...], series: [...]}` structures the frontend charting library consumes directly, keyed by city (`"ALL"`, `"Gurgaon"`, `"Hyderabad"`, `"Kolkata"`, `"Mumbai"`).

## 5. Application DB Tables (operational, SQLite/Postgres)

**`prediction_log`**
| Column | Type | Notes |
|---|---|---|
| `id` | int (PK, autoincrement) | |
| `timestamp` | datetime | |
| `city` | string | |
| `locality` | string | |
| `input_features_json` | text/JSON | full request payload (no PII — form has none) |
| `predicted_price` | float | |
| `predicted_range_low` / `predicted_range_high` | float | |
| `model_version` | string | e.g. `price_model_v3` |
| `is_outlier_input` | bool | flag from distance-to-distribution check |
| `latency_ms` | int | wall-clock time for the /predict call — basic observability, not full monitoring (added per implementation-readiness review; ~1 line of code at the FastAPI route, not new infrastructure) |

**`recommendation_log`** (optional, for later analysis of match quality)
| Column | Type |
|---|---|
| `id` | int (PK) |
| `timestamp` | datetime |
| `seed_features_json` | text/JSON |
| `returned_listing_ids` | text/JSON (array) |
| `used_fallback` | bool |

## 6. Model Artifact Files

| File | Contents |
|---|---|
| `models/price_model_v{n}.pkl` | Full `sklearn.Pipeline` (preprocessing + final estimator) |
| `models/shap_explainer_v{n}.pkl` | Fitted `shap.TreeExplainer` (or recomputed at load time from the model) |
| `models/tfidf_vectorizer_v{n}.pkl` | Fitted TF-IDF vectorizer for recommender text features |
| `models/recommender_index_{city}.pkl` or `.npz` | Precomputed feature matrix / `NearestNeighbors` index, one per city |
| `models/feature_list_v{n}.json` | Final selected feature list + selection method scores (traceability, per TRD §9) |
| `models/metrics_v{n}.json` | Train/val/test R², MAE, RMSE, MAPE per model version |

## 7. API Contracts (FastAPI, internal)

**`POST /predict`**
```json
// Request
{
  "city": "Gurgaon", "locality": "Sector 84 Gurgaon", "property_type": "Residential Apartment",
  "area_sqft": 1450, "bedrooms": 3, "bathrooms": 3, "balconies": 2,
  "furnish": "Semifurnished", "facing": "North", "age_bucket": "1-5 Year Old Property",
  "floor_num": 7, "total_floor": 14, "amenities": ["Swimming Pool", "Club house"]
}
// Response
{
  "predicted_price": 14200000, "range_low": 12800000, "range_high": 15600000,
  "shap_contributions": [{"feature": "area_sqft", "impact": 0.18}, {"feature": "locality", "impact": 0.09}, ...],
  "is_outlier_input": false, "model_version": "price_model_v3"
}
```

**`POST /recommend`**
```json
// Request: same feature shape as /predict, plus "top_n": 5, "expand_search": false
// Response
{
  "results": [
    {"listing_id": "T71585466", "price_inr": 36000000, "area_sqft": 2870, "bedrooms": 4,
     "locality": "Sector 81 Gurgaon", "similarity": 0.92, "matched_on": ["locality", "price_per_sqft"]}
  ],
  "used_fallback": false
}
```

**`GET /insights?city=Gurgaon&locality=Sector%2084%20Gurgaon`**
```json
{
  "insights": [
    "3BHKs in Sector 84 Gurgaon average ₹1.5 Cr, 8% above the Gurgaon city average.",
    "Listings with a Clubhouse amenity in this locality sell for ~6% more on average.",
    "Prices in this locality have shown a stable trend across property ages."
  ]
}
```

**`GET /health`** → `{"status": "ok", "model_version": "price_model_v3"}`

---

## UPDATE v2 — Classification Module Schema

### U-SCHEMA-1. New Canonical Field
| Canonical field | Type | Source | Notes |
|---|---|---|---|
| `price_tier` | string (categorical: Budget/Mid-Range/Premium/Luxury) | derived (engineered) | Quantile-binned from `price_per_sqft`, **computed per city**, train-set quantile boundaries applied to val/test. **Never used as a regression input feature** (leakage rule, see Rules doc update). |
| `good_deal_flag` | bool (nullable, stretch) | derived (engineered, post-regression) | 1 if `actual_price < 0.9 × predicted_price` |

### U-SCHEMA-2. New Model Artifacts
| File | Contents |
|---|---|
| `models/tier_classifier_v{n}.pkl` | Full `sklearn.Pipeline` (preprocessing + classifier) for `price_tier` |
| `models/tier_classifier_metrics_v{n}.json` | Accuracy, macro-P/R/F1, multi-class ROC-AUC, per-city confusion matrix |
| `models/tier_quantile_boundaries_v{n}.json` | Per-city `price_per_sqft` quantile cut points used to construct/re-apply the `price_tier` label consistently |
| `models/good_deal_classifier_v{n}.pkl` (stretch) | Binary classifier for `good_deal_flag` |

### U-SCHEMA-3. New API Contract

**`POST /classify`**
```json
// Request: identical shape to /predict
{ "city": "Gurgaon", "locality": "Sector 84 Gurgaon", "area_sqft": 1450, "bedrooms": 3, ... }

// Response
{
  "price_tier": "Premium",
  "tier_probabilities": {"Budget": 0.02, "Mid-Range": 0.18, "Premium": 0.63, "Luxury": 0.17},
  "shap_contributions": [{"feature": "locality_avg_price_sqft", "impact": 0.31}, {"feature": "area_sqft", "impact": 0.19}],
  "model_version": "tier_classifier_v1"
}
```

### U-SCHEMA-4. Operational Log Table Addition
**`classification_log`**
| Column | Type |
|---|---|
| `id` | int (PK) |
| `timestamp` | datetime |
| `city` | string |
| `input_features_json` | text/JSON |
| `predicted_tier` | string |
| `tier_probabilities_json` | text/JSON |
| `model_version` | string |

---

## UPDATE v3 — Finalized Canonical Schema (16-Field Input Contract)

### U-SCHEMA-5. Revised canonical `Listing` fields tied to the finalized input schema
Supersedes the illustrative version in the original Backend Schema §2 — this is now the exact field set the model pipelines consume, matching `10_FINALIZED_INPUT_SCHEMA.md`:

| Canonical field | Type | Reference-project match | Notes |
|---|---|---|---|
| `city` | string (FK) | **added** (reference project is Gurgaon-only) | Required to scope `sector` across 4 cities |
| `sector` | string (FK) | matches reference `sector` | City-scoped |
| `property_type` | string (FK) | matches reference `property_type` | flat / house (extendable) |
| `transact_type` | string | **added** | Sale / Rent — routes to one of two model pipelines, not a plain feature |
| `bedRoom` | int | matches reference `bedRoom` | |
| `bathroom` | int | matches reference `bathroom` | |
| `balcony` | string/ordinal | matches reference `balcony` | 0/1/2/3/3+ |
| `agePossession` | string | matches reference `agePossession` | New/Relatively New/Moderately Old/Old/Under Construction |
| `built_up_area` | float | matches reference `built_up_area` | sqft |
| `servant_room` | bool | matches reference `servant room` | |
| `store_room` | bool | matches reference `store room` | |
| `furnishing_type` | int (0/1/2) | matches reference `furnishing_type` | Unfurnished/Semi/Furnished |
| `luxury_category` | string | matches reference `luxury_category` | Low/Medium/High — now UI-derived from a checklist, not self-reported |
| `floor_category` | string | matches reference `floor_category` | Low/Mid/High Floor |
| `facing` | string (FK) | **added** | Available in raw data, unused in reference project |
| `amenities_list` / `n_amenities` | list / int | **added** | Replaces reference project's single opaque `luxury_score` for amenity signal |
| `price` | float | matches reference `price` | Target, scale depends on `transact_type` |
| `price_per_sqft` | float | engineered | |
| `price_tier` | string | new (Classification module) | Never fed back as a price regression input |

### U-SCHEMA-6. API contract update — `transact_type` routing
```json
// POST /predict — transact_type now determines internal routing, request shape otherwise unchanged
{
  "city": "Gurgaon", "sector": "sector 84", "property_type": "flat", "transact_type": "Sale",
  "bedRoom": 3, "bathroom": 3, "balcony": "2", "agePossession": "Relatively New",
  "built_up_area": 1450, "servant_room": true, "store_room": false,
  "furnishing_type": 1, "luxury_category": "High", "floor_category": "Mid Floor",
  "facing": "North", "amenities": ["Clubhouse", "Swimming Pool"]
}
```
FastAPI's `/predict` handler dispatches to `price_model_sale_v{n}.pkl` or `price_model_rent_v{n}.pkl` based on `transact_type` **before** any preprocessing — this is a routing decision, not a model input.

### U-SCHEMA-7. See also
Full ER diagram reflecting this exact schema is in `11_UML_DIAGRAMS.md` §8.

---

## UPDATE v4 — Schema Changes for Good-Deal-First Classification

### U-SCHEMA-8. `good_deal_flag` becomes a 3-class field, promoted to primary
| Canonical field | Type | Notes |
|---|---|---|
| `good_deal_verdict` (renamed from `good_deal_flag`) | string (categorical: `Good Deal` / `Fair Price` / `Overpriced`) | **Primary classification output.** Derived from `(actual_price - predicted_price) / predicted_price`, thresholds ±10% (tunable per city). Requires out-of-fold regression predictions at training time — unchanged leakage rule from v2. |
| `price_tier` | string (categorical: Budget/Mid-Range/Premium/Luxury) | **Now explicitly a secondary/supporting field** — powers the Recommender's "Fits my budget" filter and the Affordability chip. Still never a regression input feature (unchanged rule). |

### U-SCHEMA-9. Model artifacts — renamed/reprioritized
| File | Contents |
|---|---|
| `models/good_deal_classifier_v{n}.pkl` | **Primary** — 3-class classifier (was optional/binary in v2, now core) |
| `models/tier_classifier_v{n}.pkl` | Secondary — affordability tier classifier (unchanged from v2) |
| `models/good_deal_classifier_metrics_v{n}.json` | Accuracy, macro-F1, **per-class recall (esp. "Good Deal" recall — see TRD Update v4 §U-TRD-8)**, per-city confusion matrix |

### U-SCHEMA-10. API contract update

**`POST /classify`** (response shape updated — verdict now leads)
```json
{
  "good_deal_verdict": "Good Deal",
  "verdict_probabilities": {"Good Deal": 0.71, "Fair Price": 0.24, "Overpriced": 0.05},
  "price_tier": "Premium",
  "tier_probabilities": {"Budget": 0.02, "Mid-Range": 0.18, "Premium": 0.63, "Luxury": 0.17},
  "shap_contributions": [{"feature": "built_up_area", "impact": 0.22}, ...],
  "model_version": "good_deal_classifier_v1"
}
```

### U-SCHEMA-11. Operational log table renamed
`classification_log` → columns updated: `predicted_verdict` (was `predicted_tier` as the primary field), `predicted_tier` (now secondary column, still logged).

### U-SCHEMA-12. No new infrastructure added
Per your choice to skip MLOps tooling: no new experiment-tracking tables, no feature-store schema, no drift-detection tables are introduced in this update. The existing `prediction_log` / `classification_log` / `recommendation_log` tables (v2) remain the full extent of operational logging.

---

## UPDATE v5 (final, pre-freeze) — Lightweight Model Registry

Documentation is now frozen per an explicit decision (see Rules Update v5). This is the only addition made in response to the second external review — everything else in that review (MLflow, DVC, CI/CD, monitoring, geospatial clustering) is logged as a deliberate deferral, not implemented.

### U-SCHEMA-13. `model_registry` table (lightweight, no MLflow required)
A single flat table/CSV, updated manually or via one line in the training script — not a new service:

| Column | Type | Notes |
|---|---|---|
| `model_name` | string | e.g. `price_model`, `good_deal_classifier`, `tier_classifier` |
| `version` | string | e.g. `v3` |
| `training_dataset_version` | string | e.g. `clean_listings_2026-07-01.parquet` filename/hash |
| `git_commit` | string | commit hash at training time (`git rev-parse HEAD`) |
| `training_date` | datetime | |
| `rmse` / `mae` / `r2` (or classifier equivalents) | float | pulled straight from the existing `metrics_v{n}.json` — this table doesn't compute anything new, it just indexes what already exists |
| `hyperparameters` | JSON text | |
| `feature_hash` | string | hash of the sorted final feature list, so two versions trained on the same features are identifiable at a glance |

This gives the "can another engineer reproduce this six months later" traceability the reviews asked for, without adopting MLflow/DVC as new infrastructure — it's one CSV row appended per training run.