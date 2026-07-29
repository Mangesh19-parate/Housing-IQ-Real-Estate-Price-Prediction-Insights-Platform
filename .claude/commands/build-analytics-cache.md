You are regenerating the precomputed analytics cache.

## Step 1 - Confirm source freshness
Check `data/processed/clean_listings.parquet`'s version/timestamp. If it's
older than the last cache build, warn the user the cache may already be
current and confirm they still want to rebuild.

## Step 2 - Rebuild aggregates
Regenerate `locality_stats`, `amenity_uplift`, `age_price_trend`, and
`bhk_price_trend` per spec 27-31.

## Step 3 - Rebuild cache files
Write `data/processed/analytics_cache/*.json`, each with a metadata header
stating the source Parquet version and build timestamp (per Rules doc §1.3).

## Step 4 - Sanity check
Diff row counts and headline numbers against the previous cache version;
flag any change larger than the user's configured threshold.

## Step 5 - Report
Print what changed and confirm the cache is ready for the Analytics/Insights
pages to consume.
