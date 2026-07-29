You are seeding the dev database/Parquet store with a sample of cleaned listings.

User input: $ARGUMENTS (city name, or "all"; optional row count, default 500)

## Step 1 - Locate source
Read from `data/processed/clean_listings.parquet`. If it doesn't exist yet,
tell the user to run the cleaning pipeline first (spec 07).

## Step 2 - Sample
Take a stratified sample across property types and price bands for the
requested city/cities, up to the requested row count.

## Step 3 - Load
Insert into the dev SQLite DB / dev cache used by the Flask app, respecting
the `PRAGMA foreign_keys = ON` requirement.

## Step 4 - Report
Print row counts loaded per city and any rows skipped due to FK violations.
