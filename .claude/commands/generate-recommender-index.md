You are rebuilding the recommender's TF-IDF vectorizer and similarity index.

## Step 1 - Load corpus
Load cleaned listing text/amenity fields from
`data/processed/clean_listings.parquet`.

## Step 2 - Fit vectorizer
Fit (or refit) the TF-IDF vectorizer per the `tfidf-text-features` skill,
including any custom stopword list.

## Step 3 - Build similarity index
Blend TF-IDF output with numeric/categorical similarity features per the
`recommender-similarity-search` skill; build the nearest-neighbor index.

## Step 4 - Save
Persist both the vectorizer and the index together, versioned, under
`models/`, so they are never mismatched.

## Step 5 - Spot-check
Run a handful of known-similar property pairs through the new index and
confirm the results look sane before reporting done.
