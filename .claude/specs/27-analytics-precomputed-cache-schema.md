---
# Spec: Analytics Precomputed Cache Schema

## Overview
This spec defines the schema and structure for the precomputed analytics cache JSON files that power the Analytics module's 13 chart tiles. The cache is built offline during the ML pipeline (Day 36 of the Implementation Plan) and served via FastAPI `GET /api/analytics/<chart_id>?city=` to the Flask frontend, which renders charts using Chart.js/Leaflet. Each JSON file contains pre-shaped `{labels: [...], series: [...]}` structures keyed by city (`"ALL"`, `"Gurgaon"`, `"Hyderabad"`, `"Kolkata"`, `"Mumbai"`), plus metadata stating the source Parquet version and build timestamp (per Rules §1.3).

## Depends on
- Spec 07: `clean_listings.parquet` must exist (canonical cleaned dataset)
- Spec 12: Feature engineering complete (engineered columns like `price_per_sqft`, `n_amenities`, `floor_ratio`, `age_bucket`, `has_<amenity>` flags available)
- Spec 17: FastAPI service running (to serve the cache via `/api/analytics/<chart_id>`)
- Backend Schema §4: Defines the `analytics_cache/*.json` structure and the 4 aggregate tables (`locality_stats`, `amenity_uplift`, `age_price_trend`, `bhk_price_trend`, `furnish_price_trend`, `floor_price_trend`)

## Routes / Endpoints
- FastAPI: `GET /api/analytics/<chart_id>?city=<city>` — returns the cached JSON for a specific chart ID and city filter
- Flask: `GET /analytics` — renders dashboard shell; JS fetches chart JSON via the above endpoint
- Flask: `GET /api/analytics/<chart_id>?city=<city>` — internal proxy to FastAPI (or direct static file serving if configured)

## Data / Schema Changes

### New Cache Files (13 files in `data/processed/analytics_cache/`)
Each file follows this schema:
```json
{
  "chart_id": "chart_01_spatial_map",
  "title": "Listing Map by Price per Sqft",
  "chart_type": "map",
  "city_slices": {
    "ALL": { "markers": [...], "bounds": [[lat, lng], [lat, lng]] },
    "Gurgaon": { "markers": [...], "bounds": [[lat, lng], [lat, lng]] },
    "Hyderabad": { ... },
    "Kolkata": { ... },
    "Mumbai": { ... }
  },
  "metadata": {
    "source_parquet": "clean_listings_2026-07-01.parquet",
    "source_parquet_hash": "sha256:...",
    "built_at": "2026-07-15T10:30:00Z",
    "row_count": 182000,
    "outlier_count": 5036
  }
}
```

### Chart ID ↔ Chart Mapping (13 charts per App Flow §2.3)
| Chart ID | Chart Title | Chart Type | Source Columns / Tables |
|----------|-------------|------------|------------------------|
| `chart_01_spatial_map` | Spatial analysis — listing map | Leaflet map (clustered markers, colored by price/sqft) | `latitude`, `longitude`, `price_per_sqft` from `clean_listings.parquet` |
| `chart_02_price_distribution` | Price distribution across sectors/localities | Box plot (top-N localities by volume) | `locality`, `price_inr` |
| `chart_03_price_vs_area` | Price vs. Area analysis | Scatter plot + trend line | `area_sqft`, `price_inr` |
| `chart_04_bhk_distribution` | Number of rooms (BHK) distribution | Pie/donut chart | `bedrooms` |
| `chart_05_word_cloud` | Top-feature word cloud | Word cloud (image/HTML canvas) | `description_clean`, decoded `features_list`, `amenities_list` |
| `chart_06_correlation_heatmap` | Correlation heatmap | Heatmap | All numeric engineered features |
| `chart_07_city_price_comparison` | City-wise price comparison | Box/violin plot (4 cities) | `city`, `price_inr` |
| `chart_08_price_sqft_locality_heatmap` | Price-per-sqft by locality heatmap | Choropleth-style heatmap or ranked bar | `locality`, `price_per_sqft` |
| `chart_09_furnish_vs_price` | Furnishing status vs. price | Box plot (Furnished/Semi/Unfurnished) | `furnish`, `price_inr` |
| `chart_10_age_price_trend` | Property age vs. price trend | Line/box plot across age buckets | `age_bucket`, `price_inr` |
| `chart_11_amenity_count_vs_price` | Amenity count vs. price | Scatter + regression trend line | `n_amenities`, `price_inr` |
| `chart_12_floor_price_premium` | Floor number vs. price premium | Line chart (avg price by floor bucket) | `floor_num`, `total_floor`, `price_inr` |
| `chart_13_top_builders_price` | Top builders/societies by avg price | Horizontal bar / treemap | `building_name`, `price_inr` |

### City Slices
Each chart must provide data for 5 city slices: `"ALL"` (all 4 cities combined), `"Gurgaon"`, `"Hyderabad"`, `"Kolkata"`, `"Mumbai"`. The `"ALL"` slice is used as the default when no city filter is selected.

### Metadata Requirements (per Rules §1.3)
Every cache file must include a `metadata` object with:
- `source_parquet`: filename of the source `clean_listings.parquet`
- `source_parquet_hash`: SHA256 hash of the source file for cache invalidation
- `built_at`: ISO 8601 timestamp of cache build
- `row_count`: total rows in source (post-cleaning, pre-outlier-filter)
- `outlier_count`: rows flagged `is_outlier == True`

## Templates / UI
- **Modify:** `app/templates/analytics.html` — JS charting layer consumes these JSON files; ensure the `chart_id` mapping matches
- **Create:** None (cache files are data, not templates)

## Files to Change / Files to Create

### New Files
1. `scripts/build_analytics_cache.py` — Main script to generate all 13 cache JSON files
2. `ml/analytics/cache_builders.py` — Reusable builder functions per chart (one function per chart_id)
3. `ml/analytics/__init__.py` — Exports builder functions
4. `data/processed/analytics_cache/chart_01_spatial_map.json` through `chart_13_top_builders_price.json` — Generated output (gitignored, built at runtime)

### Modified Files
1. `api/routers/analytics.py` — Add `GET /api/analytics/<chart_id>` route to serve cache files
2. `api/schemas/analytics.py` — Pydantic response model for the cache JSON structure
3. `app/templates/analytics.html` — Ensure JS fetches use correct chart_id values

## New Dependencies
- No new pip packages — uses existing: pandas, numpy, scikit-learn, joblib
- Chart.js and Leaflet already loaded via CDN in templates

## Rules for Implementation
- No SQLAlchemy/ORM unless already in use; parameterized queries only
- No dealer/contact/media-URL fields ever reach the UI or an export
- CSS variables only, never hardcoded hex values
- All templates extend `base.html`
- Model changes must reference the fixed evaluation protocol
- Cache files must include metadata header with source Parquet version and build timestamp (Rules §1.3)
- Outlier rows (`is_outlier == True`) are excluded from chart computations but retained in source — document this in metadata
- The `build_analytics_cache.py` script must be idempotent and safe to re-run
- Each chart builder function must be pure (no side effects) and testable in isolation

## Definition of Done
1. `scripts/build_analytics_cache.py` runs without errors and produces 13 JSON files in `data/processed/analytics_cache/`
2. Each JSON file validates against the schema above (has `chart_id`, `title`, `chart_type`, `city_slices` with 5 keys, `metadata`)
3. FastAPI `GET /api/analytics/<chart_id>?city=<city>` returns the correct slice for all 13 chart_ids and 5 city values
4. Flask `/analytics` page loads and renders all 13 chart tiles (visual verification)
5. City filter AJAX re-fetch works: changing city selector updates all tiles without page reload
6. Unit tests exist for at least 3 chart builders (testing output structure and city slicing)
7. `ruff check` passes on all new/modified files