# App Flow Document
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

---

## 1. High-Level System Flow

```
                        ┌───────────────────────────┐
                        │        Landing Page        │
                        │  (city selector, module    │
                        │   nav: Predict/Analytics/  │
                        │   Recommend/Insights)       │
                        └──────────────┬─────────────┘
                                       │
         ┌────────────────┬───────────┼───────────────┬────────────────┐
         ▼                ▼           ▼                ▼                
 ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 │ Price          │ │ Analytics     │ │ Recommender    │ │ Insights       │
 │ Prediction     │ │ Dashboard     │ │ System         │ │ Module         │
 │ Module         │ │ Module        │ │ Module         │ │                │
 └───────┬────────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
         │                  │                 │                 │
         ▼                  ▼                 ▼                 ▼
   Flask route         Flask route       Flask route        Flask route
   /predict (GET       /analytics        /recommend          /insights
   form, POST submit)  (renders cached   (GET form, POST      (renders
                        chart JSON)       submit)              templated
         │                                    │                sentences)
         ▼                                    ▼                     │
  FastAPI /predict                     FastAPI /recommend            │
  (loads model,                        (loads TF-IDF +               │
   returns price +                      similarity index,            │
   SHAP breakdown)                      returns top-N)                │
         │                                    │                       │
         └──────────────► Insight templater ◄─┘◄──────────────────────┘
                          (reads locality_stats,
                           amenity_uplift tables)
```

## 2. Detailed Page-Level Flow

### 2.1 Landing / Home
1. User lands on `/`.
2. Sees: hero section, 4 module cards (Predict / Analytics / Recommend / Insights), city quick-filter (Gurgaon, Hyderabad, Kolkata, Mumbai — default: All).
3. Clicking a module card routes to that module's page, carrying the selected city as a query param.

### 2.2 Price Prediction Module Flow
1. `/predict` (GET) → renders form: City, Locality (dependent dropdown populated via `/api/localities?city=`), Property Type, Area (sqft), Bedrooms, Bathrooms, Balconies, Furnishing, Facing, Age, Floor, Total Floor, Amenities (multi-select checklist).
2. User submits form → Flask POST handler validates basic input (non-empty, ranges) → calls FastAPI `POST /predict` with the JSON payload.
3. FastAPI: loads pipeline (`price_model_v{n}.pkl`), transforms input, predicts `log_price`, inverse-transforms → price + range (from residual std or quantile model), computes SHAP values for this single instance.
4. Flask renders `/predict/result`: 
   - Predicted price (large, prominent) + range band.
   - "Why this price?" SHAP contribution bar chart (top 6 features, pushing price up/down).
   - 3–5 auto-generated Insight cards (calls insight templater with the same input + predicted price).
   - "See similar properties" CTA → routes to Recommender module pre-filled with this input.
5. Error path: if FastAPI is unreachable/times out, Flask shows a friendly "Prediction service is temporarily unavailable, please try again" state (per TRD NFR2) instead of crashing.

### 2.3 Analytics Module Flow
1. `/analytics` (GET) → renders dashboard shell with a City filter (default: All cities combined).
2. Page loads precomputed chart JSON from `data/processed/analytics_cache/*.json` (built offline by the pipeline, not recomputed on each request) via `/api/analytics/<chart_id>?city=`.
3. JS charting layer (Chart.js/Plotly) renders each chart into its card/tile.
4. Changing the City filter re-fetches the cached JSON for that city slice and re-renders in place (no full page reload).

**Analytics tiles rendered (13 total — 5 originally specified + 8 newly added this session):**

| # | Chart | Type | Source columns | Status |
|---|---|---|---|---|
| 1 | Spatial analysis — listing map | Map (Leaflet, lat/long clustered markers, colored by price/sqft) | `MAP_DETAILS` (LATITUDE/LONGITUDE), `PRICE_SQFT` | Originally specified |
| 2 | Price distribution across sectors/localities | Box plot, top-N localities by volume | `LOCALITY`, `PRICE` | Originally specified |
| 3 | Price vs. Area analysis | Scatter plot, trend line | `AREA`, `PRICE` | Originally specified |
| 4 | Number of rooms (BHK) distribution | Pie/donut chart | `BEDROOM_NUM` | Originally specified |
| 5 | Top-feature word cloud | Word cloud (image/HTML canvas) | `DESCRIPTION`, `TOP_USPS`, decoded `FEATURES` | Originally specified |
| 6 | Correlation heatmap | Heatmap | All numeric engineered features | **New** |
| 7 | City-wise price comparison | Box/violin plot, 4 cities side by side | `CITY`, `PRICE` | **New** |
| 8 | Price-per-sqft by locality | Choropleth-style heatmap or ranked bar | `PRICE_SQFT`, `LOCALITY` | **New** |
| 9 | Furnishing status vs. price | Box plot (Furnished/Semi/Unfurnished) | `FURNISH`, `PRICE` | **New** |
| 10 | Property age vs. price trend | Line/box plot across age buckets | `AGE` (facet-decoded), `PRICE` | **New** |
| 11 | Amenity count vs. price | Scatter + regression trend line | `n_amenities` (engineered), `PRICE` | **New** |
| 12 | Floor number vs. price premium | Line chart (avg price by floor bucket) | `FLOOR_NUM`, `TOTAL_FLOOR`, `PRICE` | **New** |
| 13 | Top builders/societies by avg price | Horizontal bar / treemap | `BUILDING_NAME`/`SOCIETY_NAME`, `PRICE` | **New** |

*(Rationale for the 8 new charts: #6–8 surface the structural drivers of price at a macro level — useful for the Insights module's uplift statistics; #9–13 each map directly to a feature engineered in the TRD's Feature Engineering section, so the analytics view doubles as a visual justification for why that feature was included in the model.)*

### 2.4 Recommender Module Flow
1. `/recommend` (GET) → two entry paths:
   - **From a listing/seed**: user picks/enters a reference property's attributes (or arrives here from "See similar properties" in Price Prediction, pre-filled).
   - **From scratch**: user fills a lightweight preference form (city, budget range, bedrooms, locality).
2. POST → Flask calls FastAPI `POST /recommend` with the feature vector.
3. FastAPI: builds/loads the TF-IDF + numeric feature matrix for the relevant city, computes cosine similarity, returns top-N (default 5) with a similarity score.
4. If fewer than 5 results clear the similarity threshold → cold-start fallback triggers (rank by locality popularity + recency, per TRD §11) and the UI labels these as "Popular in this area" rather than "Most similar," so the user isn't misled about why they're shown.
5. Flask renders `/recommend/result`: card grid of recommended properties (price, area, BHK, locality, top 2–3 matching attributes highlighted, e.g. "Same 3BHK, similar price/sqft").

### 2.5 Insights Module Flow
1. `/insights` (GET) → standalone page: City selector → renders a set of templated market-insight sentences and 2–3 supporting mini-charts (reuses Analytics cache).
2. Also invoked inline (not as a separate page navigation) from the Price Prediction result screen (see 2.2 step 4) and optionally from Recommender results ("Why these matches?").
3. Insight templater logic: pulls from `locality_stats`, `amenity_uplift`, `age_price_trend` precomputed tables and fills sentence templates (TRD §12) — no live recomputation, no generative text.

## 3. Navigation Map (site structure)

```
/                      → Landing
/predict               → Price Prediction form
/predict/result        → Prediction result + SHAP + insights + CTA
/analytics             → Analytics dashboard (13 tiles)
/recommend             → Recommender form (seed or preference-based)
/recommend/result       → Recommended properties grid
/insights              → Standalone market insights page
/api/localities         → (internal) dependent-dropdown data
/api/analytics/<id>     → (internal) cached chart JSON
--- FastAPI (internal only, not user-facing routes) ---
/predict  (POST)
/recommend (POST)
/insights  (GET, params)
/health
```

## 4. Error / Edge-Case Flows
- FastAPI down → Flask degrades gracefully (see 2.2.5).
- Locality not found in facet table → fallback to City-level aggregates for insights.
- User inputs an area/price combination far outside training distribution → prediction still returned, but UI flags "Low confidence — outside typical range for this locality" using a distance-to-training-distribution check.
- Recommender seed has no close matches even after fallback → show "No strong matches found — try widening your city/locality filter" empty state, never a blank page.

---

## UPDATE v2 — Classification Module Flow

### U-FLOW-1. Updated Navigation Map
```
/                      → Landing (now 5 module cards)
/predict               → Price Prediction form
/predict/result        → Prediction result + SHAP + Price-Tier badge + insights + CTA   ← updated
/analytics             → Analytics dashboard (14 tiles now)                              ← updated
/recommend             → Recommender form (now with optional Tier filter)                ← updated
/recommend/result       → Recommended properties grid (tier badge per card)              ← updated
/insights              → Standalone market insights page (tier-mix insight added)         ← updated
/classify              → NEW — standalone "Check property tier" form + result page
--- FastAPI (internal) ---
/predict (POST), /recommend (POST), /insights (GET), /classify (POST)  ← new route
/health
```

### U-FLOW-2. Price Prediction Result Screen (updated)
After Flask receives the `/predict` response, it now **also** calls FastAPI `/classify` with the same payload (parallel call, not sequential — both requests fire together to avoid adding latency) and renders:
```
₹ 1.42 Cr (range)   |   🏷 Premium (63% confidence)      ← new tier badge, top-right of hero
3 BHK · 1450 sqft · Sector 84, Gurgaon
Why this price?  [SHAP bars]        Why this tier?  [SHAP bars, collapsed by default]
Market Insights (now includes: "Only 18% of Sector 84 listings are Premium-tier or above.")
```
If `/classify` fails independently of `/predict` (per the same graceful-degradation rule as before), the tier badge is simply omitted — the price prediction itself must not be blocked by a classification-service failure.

### U-FLOW-3. Analytics — 14th Tile
| # | Chart | Type | Source |
|---|---|---|---|
| 14 | Price-tier distribution by city/locality | Stacked bar (Budget/Mid/Premium/Luxury per city) | `price_tier` (new engineered column) |

### U-FLOW-4. Recommender — Tier Filter
Form gets an optional "Tier" multi-select (Budget/Mid-Range/Premium/Luxury) alongside City/Locality/Budget-range; result cards show a small tier chip next to the existing similarity/"Popular in this area" badge.

### U-FLOW-5. Standalone `/classify` Page (new)
Mirrors the `/predict` form (can literally reuse the same form component), but the primary CTA is "Check Price Tier" instead of "Predict Price." Result page shows: tier badge (large), tier-probability bar (all 4 tiers with their probabilities, not just the winner), and SHAP "why this tier" breakdown. A secondary link offers "Also predict the exact price →" routing to `/predict` with the same inputs carried over.

---

## UPDATE v3 — Finalized Form Flow (16 Fields) & UML Sequence Cross-Reference

### U-FLOW-6. `/predict` and `/classify` form flow, finalized field order
```
Step 1 (always visible): City → Sector/Locality (dependent dropdown) → Property Type → Transaction Type (Sale/Rent)
Step 2 (core structure): Bedrooms, Bathrooms, Balconies, Built-up Area (sqft)
Step 3 (secondary, still required): Property Age/Possession, Furnishing Type, Facing Direction, Floor Category
Step 4 (guided, not raw numeric entry): Servant Room (Y/N), Store Room (Y/N), Luxury Category (derived from a short amenities/finish mini-checklist, mirroring how the reference project derives luxury_category from a luxury_score rather than asking the user to self-report "Low/Medium/High")
Step 5 (optional, collapsed): Amenities multi-select
```
This exact order is now the source of truth for the Flask template field ordering in `/predict` and `/classify` (§2.2, §2.5 of the original App Flow doc).

### U-FLOW-7. Sequence diagrams now formalize the module interactions
`11_UML_DIAGRAMS.md` §3 and §4 (Sequence Diagrams) make explicit two behaviors this App Flow doc described only in prose:
- The `/predict` and `/classify` FastAPI calls fire **in parallel** (not sequentially) from Flask, per §3's `par` block — this was implicit in App Flow §2.2 but is now an explicit, diagrammed requirement, since a sequential call would double the perceived latency for no reason.
- The Recommender's cold-start fallback (App Flow §2.4) is now an explicit `alt/else` branch in §4's sequence diagram, making the "used_fallback" flag's trigger condition (fewer than 5 candidates above the similarity threshold) unambiguous for implementation.

---

## UPDATE v4 — Classification Flow Reframed Around Affordability + Good-Deal

### U-FLOW-8. Price Prediction result screen — updated purpose for the tier badge
The `TierBadge` next to the price hero now reads as a **Good Deal / Fair Price / Overpriced** signal (primary), with the Affordability Tier (Budget/Mid/Premium/Luxury) shown as a secondary, smaller chip:
```
₹ 1.42 Cr (range)     ✅ Good Deal        🏷 Premium tier
3 BHK · 1450 sqft · Sector 84, Gurgaon
```
"Why this call?" (SHAP) now explains the **Good Deal/Overpriced** classification first, since that's the module's headline signal; the affordability-tier explanation remains available but collapsed by default (same collapsible pattern as before).

### U-FLOW-9. Recommender module — tier filter now has a stated reason
The Tier filter added in v2 update was previously undermotivated ("why would a recommender need this?"). Now explicit: it's an **affordability filter** — "only show me Mid-Range or below" — directly serving the buyer persona's real constraint (budget), not just a generic facet for completeness.

### U-FLOW-10. `/classify` standalone page — renamed in user-facing copy
Page title changes from generic "Check property tier" to **"Is this a good deal?"** — reflecting the module's real purpose. Form is unchanged (same 16-field input); result now leads with the Good Deal/Fair/Overpriced verdict, with the affordability tier shown underneath as supporting context ("...and it's in the Premium tier for Sector 84").