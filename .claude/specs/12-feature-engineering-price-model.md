# Spec: Feature Engineering for the Price Model

## Overview
Add the model-ready feature-engineering layer that consumes the canonical
cleaned DataFrame produced by Step 07 (`data/processed/clean_listings.parquet`)
and emits a deterministic, single-source-of-truth feature DataFrame that the
price regression training script (Week 4, future spec) can consume directly.
This is the implementation of `02-TRD.md` §8 ("Feature Engineering") plus the
encoding block from §U-TRD-3 (`ColumnTransformer` shape) — concretely: derive
the engineered columns (`price_per_sqft`, `n_amenities`, `n_features`,
`floor_ratio`, `age_bucket_ord`, `bath_bed_ratio`, `area_per_bedroom`, and the
top-K `has_<amenity>` flags), produce the leakage-safe locality aggregates
(`locality_avg_price_sqft`, `locality_listing_count`, `locality_smoothed_price`
— train-only fit, applied to val/test/inference), build the unified
`ColumnTransformer` (numeric scale + ordinal encode + one-hot), and serialize
the fitted preprocessor + the final feature list as
`models/feature_pipeline_v1.pkl` and `models/feature_list_v1.json`. The output
is **two reusable callables** — `build_feature_frame(clean_df)` returns a
deterministic DataFrame in the locked column order; `make_preprocessor()`
returns the unfitted `ColumnTransformer` — so the training script and the
serving path both import the same code, eliminating train/serve skew
(Rules §2.4). Module: **price-prediction**.

This spec is deliberately scoped to the **price model** only. The
Classification module (Week 8) reuses the same feature frame minus
`price_per_sqft` and `price` (Rules §8.1 / TRD §U-TRD-1); that reuse is wired
later when the classifier lands, not here. The Recommender module also
reuses a subset but with TF-IDF added — that is its own spec.

## Depends on
- **Step 07** — `07-clean-listings-parquet-pipeline` — produces the canonical
  `clean_listings.parquet` this spec consumes. Column order matches
  `CANONICAL_COLUMNS` (Step 05), with `is_outlier` (Step 06) and
  `was_missing_*` flags (Step 07) appended.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` — provides
  `is_outlier` so this spec's leakage-safe locality aggregation can fit only
  on `is_outlier == False` rows (Rules §2.3: outlier-flagged rows must not
  influence learned aggregates).
- **Step 05** — `05-canonical-schema-mapping-per-city` — provides
  `CANONICAL_COLUMNS` and the per-city mappers; the field names referenced
  here (`bedRoom`, `built_up_area`, `agePossession`, `luxury_category`, etc.)
  are locked to the reference project (`10-FINALIZED-INPUT-SCHEMA.md` §1 + §2).
- **Step 11** — `11-price-prediction-input-schema-v3` — provides
  `INPUT_FIELDS_V3` from `api/schemas/predict_v3.py`, which is the
  authoritative 16-field tuple the serving path validates against. The
  feature frame produced here **must contain every column that tuple
  names**, plus the engineered ones — so the Pydantic-validated request body
  can be losslessly turned into a feature row.
- **`02-TRD.md` §8 + §U-TRD-3** — engineered column list + `ColumnTransformer`
  shape (numeric / ordinal / one-hot split).
- **`05-BACKEND-SCHEMA.md` §U-SCHEMA-5** — the 16-field canonical schema
  this feature frame mirrors.
- **`10-FINALIZED-INPUT-SCHEMA.md`** — the 16-field input contract; the
  feature frame's "raw" columns match this contract verbatim.
- **`08-RULES.md` §2.3** — leakage rule: locality aggregates computed
  train-only, applied (not refit) on val/test/serving.

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python feature
transforms + serialized `.pkl`/`.json` artifacts. The serving path (FastAPI
`/predict`, future spec) will load `feature_pipeline_v1.pkl` and call
`build_feature_frame(request_dict)` — but wiring that path is **out of
scope here**.

## Data / Schema changes
- **Read** `data/processed/clean_listings.parquet` (Step 07's output).
- **Write** `models/feature_pipeline_v1.pkl` — the fitted `ColumnTransformer`
  (only the fitted preprocessor; the model itself is a later spec).
  Train-only fit (Rules §2.3). Filename pinned by Rules §2.5 (versioned,
  never overwritten in place).
- **Write** `models/feature_list_v1.json` — the final ordered feature list
  (column names output by `build_feature_frame()` after the preprocessor's
  one-hot expansion), plus the engineered-feature recipe (which base columns
  produced which derived columns). This is the traceability artifact TRD §9
  asks for ("Final feature list and rationale must be logged").
- **Write** `data/processed/feature_selection_report.md` — the §TRD 9
  selection round-1 report (correlation filtering + base-column summary).
  This spec only writes Round 1; Round 2/3 (tree-based importance,
  permutation, SHAP) require a fitted model and land with the training
  spec. Documented in "Rules for implementation" below.
- **No writes to `data/raw/`** — Rules §1.2.
- **No writes to `data/processed/clean_listings.parquet`** — Step 07 owns
  it; this spec is read-only on the cleaning artifact.
- **No writes to `data/processed/analytics_cache/`** — that lives in Week 6.
- **No application DB changes** — `data/app.db` is untouched.
- **No SQL migration.**

## Templates / UI
None. This spec is offline feature engineering — no Flask templates, no
static assets, no HTML.

## Files to change / Files to create

**Create:**
- `ml/features/__init__.py` — empty; re-exports below.
- `ml/features/feature_frame.py` — the deterministic feature-DataFrame
  builder. Public API:
  - `ENGINEERED_COLUMNS: tuple[str, ...]` — the ordered tuple of columns
    `build_feature_frame()` adds beyond the 16 contract fields. Locks to:
    `("price_per_sqft", "n_amenities", "n_features", "floor_ratio",
    "age_bucket_ord", "bath_bed_ratio", "area_per_bedroom",
    "locality_avg_price_sqft", "locality_listing_count",
    "locality_smoothed_price", "top_amenities_count")` — 11 columns. The
    last three (`locality_*`) come from `LocalityAggregator`; the rest are
    pure row-level math.
  - `TOP_AMENITY_FEATURES: tuple[str, ...]` — the canonical top-K amenities
    (K=10) the `has_<amenity>` flags are built for. The exact list is
    determined at runtime by `select_top_amenities(df, k=10)` against the
    cleaned data (the top-10 most-frequent amenity labels in the corpus),
    but this constant pins the K and the column-naming convention
    (`has_<amenity_slug>`). Documented in the module docstring.
  - `select_top_amenities(df: pd.DataFrame, k: int = 10) -> list[str]` —
    returns the top-K most-frequent amenity **labels** (decoded, from
    `amenities_list`) across the input frame. Pure function.
  - `slugify_amenity(label: str) -> str` — converts `"Swimming Pool"` to
    `"swimming_pool"` for column names. Lowercase, snake_case, strips
    punctuation, collapses whitespace.
  - `derive_row_features(df: pd.DataFrame) -> pd.DataFrame` — adds the 8
    non-locality engineered columns (`price_per_sqft`, `n_amenities`,
    `n_features`, `floor_ratio`, `age_bucket_ord`, `bath_bed_ratio`,
    `area_per_bedroom`, and the 10 `has_<amenity>` flags) to a copy of
    `df`. Pure function. **Leakage note**: `price_per_sqft` here is a
    per-row `price / built_up_area` derivation on the row itself — that's
    target-leakage-free because both `price` and `built_up_area` are
    already observed per row (it's a within-row ratio, not a
    group-average). The classifier spec removes this column before
    training (Rules §8.1).
  - `class AgeBucket(str, Enum):` — `NEW = "New Property"`,
    `RELATIVELY_NEW = "Relatively New"`, `MODERATELY_OLD = "Moderately
    Old"`, `OLD = "Old Property"`, `UNDER_CONSTRUCTION = "Under
    Construction"`. Ordinal mapping (0–4) hardcoded below. Mirrors the
    Step 11 Pydantic enum exactly — both layers use the same string
    values.
  - `AGE_BUCKET_ORDINAL: dict[str, int]` — `{"New Property": 0,
    "Under Construction": 1, "Relatively New": 2, "Moderately Old": 3,
    "Old Property": 4}`. The ordering reflects property desirability
    for resale ("newer is more valuable"), not raw age; documented in
    the module docstring with that rationale. *(ponytail: ordinal mapping
    is the one place a "domain judgment" sneaks in; pin it to the
    docstring + this dict so the rationale survives.)*
  - `class FloorCategory(str, Enum):` — `LOW = "Low Floor"`,
    `MID = "Mid Floor"`, `HIGH = "High Floor"`. Ordinal mapping
    (0–2) hardcoded below.
  - `FLOOR_CATEGORY_ORDINAL: dict[str, int]` — `{"Low Floor": 0,
    "Mid Floor": 1, "High Floor": 2}`.
  - `class LuxuryCategory(str, Enum):` — `LOW = "Low"`,
    `MEDIUM = "Medium"`, `HIGH = "High"`. Ordinal (0–2).
  - `LUXURY_CATEGORY_ORDINAL: dict[str, int]` — `{"Low": 0, "Medium": 1,
    "High": 2}`.
  - `class FurnishingType(str, Enum):` — `UNFURNISHED = "Unfurnished"`,
    `SEMIFURNISHED = "Semifurnished"`, `FURNISHED = "Furnished"`.
    Ordinal (0–2).
  - `FURNISHING_TYPE_ORDINAL: dict[str, int]` — `{"Unfurnished": 0,
    "Semifurnished": 1, "Furnished": 2}`. Matches the reference project's
    `0/1/2` encoding.
  - `class Balcony(str, Enum):` — `ZERO = "0"`, `ONE = "1"`, `TWO = "2"`,
    `THREE = "3"`, `THREE_PLUS = "3+"`. Ordinal (0–4).
  - `BALCONY_ORDINAL: dict[str, int]` — `{"0": 0, "1": 1, "2": 2, "3": 3,
    "3+": 4}`.
  - `build_feature_frame(df: pd.DataFrame) -> pd.DataFrame` — top-level
    helper. Steps:
    1. Validate input has every column in `INPUT_FIELDS_V3` (Step 11)
       plus `is_outlier` (Step 06) plus `was_missing_<col>` for any
       imputed column. Raises `ValueError` listing missing columns.
    2. Apply `derive_row_features(df)`.
    3. Drop `is_outlier` and `was_missing_*` from the output — those
       are diagnostic flags, not model inputs. (Outlier rows are still
       **present** in the output frame, because the training script does
       the `is_outlier == False` filter at training time, not here.)
    4. Reorder columns so the 16 contract fields come first (per the
       `INPUT_FIELDS_V3` order), then `ENGINEERED_COLUMNS`. Final
       column order is deterministic and pinned by a test.
    5. Returns the new frame. Pure function; no I/O.

- `ml/features/locality_aggregator.py` — leakage-safe locality aggregates.
  Public API:
  - `SMOOTHING_PRIOR_WEIGHT: float = 20.0` — Bayesian smoothing weight
    toward the city mean. Tunable constant; default 20 matches the
    literature (S12) — at ~20 listings per locality the smoothed
    estimate leans ~50/50 toward the locality mean vs. the city mean.
    *(ponytail: pin the default; if a tuning step needs a different
    number it lives in `feature_selection_report.md`.)*
  - `class LocalityAggregator:` — sklearn-style fit/transform API (not a
    full `TransformerMixin` — just `fit` + `transform`, no
    `fit_transform` shenanigans).
    - `fit(train_df: pd.DataFrame) -> "LocalityAggregator"` — computes
      per-(city, locality) aggregates from the training frame:
      `locality_avg_price_sqft` (mean of `price_per_sqft`),
      `locality_listing_count` (row count), `locality_smoothed_price`
      (Bayesian-smoothed mean of `price_inr`, prior = city mean,
      weight = `SMOOTHING_PRIOR_WEIGHT`). **Filters to
      `is_outlier == False` first** (Rules §2.3 + §8.4: outliers
      must not influence learned aggregates). Also computes the
      per-city priors for the smoother. Stores the per-(city,
      locality) frame internally as a sorted `DataFrame`. Idempotent:
      re-calling `fit` overwrites the prior state. Deterministic.
    - `transform(df: pd.DataFrame) -> pd.DataFrame` — left-joins the
      learned aggregates onto the input frame by `(city, locality)`,
      filling missing combinations (e.g. a test-set locality not
      seen in training) with the city-level fallback (mean across
      all localities in the city). Returns a new frame with the
      three `locality_*` columns added. Does **not** refit; this is
      the leakage-safe "apply, don't recompute" rule from Rules §8.2.
    - `fit_transform(train_df: pd.DataFrame) -> pd.DataFrame` —
      convenience wrapper that calls `fit` then `transform` on the
      same frame.
    - `fitted_aggregates_: pd.DataFrame` — attribute exposed after
      `fit`, used by tests to assert what was learned (pin the
      expected row count + a sample of expected aggregates against a
      fixture).
    - `city_priors_: dict[str, float]` — attribute exposed after
      `fit`, the per-city price-per-sqft means used as the
      smoother's prior.
  - `LocalityAggregator` is **not** itself a `ColumnTransformer` step —
  it produces the locality columns before the preprocessor runs, so the
  preprocessor can treat them as plain numeric features. Documented in
  the module docstring.

- `ml/features/preprocessor.py` — the `ColumnTransformer` factory. Public
  API:
  - `NUMERIC_FEATURES: tuple[str, ...]` — `("bedRoom", "bathroom",
    "built_up_area", "servant_room", "store_room", "n_amenities",
    "n_features", "floor_ratio", "age_bucket_ord", "bath_bed_ratio",
    "area_per_bedroom", "locality_avg_price_sqft",
    "locality_listing_count", "locality_smoothed_price",
    "top_amenities_count")` + the 10 `has_<amenity>` columns (resolved
    at fit time from `TOP_AMENITY_FEATURES`). 14 fixed + 10 amenity
    flags = 24 numeric features.
  - `ORDINAL_FEATURES: tuple[str, ...]` — `("luxury_category",
    "floor_category", "furnishing_type", "balcony")`. 4 ordinal
    features. Each has an explicit category ordering passed to
    `OrdinalEncoder` so train/serve is unambiguous.
  - `ONEHOT_FEATURES: tuple[str, ...]` — `("city", "property_type",
    "agePossession", "facing")`. 4 one-hot groups. **`sector` is NOT
    one-hot here** — TRD §U-TRD-3 specifies one-hot or target
    encoding; this spec picks **target encoding** (smoothed, added
    in `LocalityAggregator` via `locality_smoothed_price` /
    `locality_avg_price_sqft`), so `sector` does not appear in
    `ONEHOT_FEATURES` either. `transact_type` is also absent here
    — it is a routing key per TRD §U-TRD-4 and is handled at the
    FastAPI layer, not the feature pipeline.
  - `ORDINAL_CATEGORY_ORDERINGS: dict[str, list[str]]` — the exact
    category ordering for each ordinal feature. Pinned because
    `OrdinalEncoder` is order-sensitive: train/serve skew if this
    ever drifts.
  - `make_preprocessor() -> ColumnTransformer` — returns an **unfitted**
    `ColumnTransformer` with the three transformer tuples above. Pure
    factory. Deterministic output (sklearn's `ColumnTransformer` is
    deterministic given fixed input tuples).
  - `fit_preprocessor(train_feature_frame: pd.DataFrame,
    locality_aggregator: LocalityAggregator) -> ColumnTransformer`
    — convenience: applies the locality aggregator, fits the
    preprocessor, returns the fitted transformer. Pure with respect
    to the input frame (no I/O).
  - `transform_with_preprocessor(fitted: ColumnTransformer,
    df: pd.DataFrame) -> pd.DataFrame` — applies a fitted
    preprocessor, returns the resulting DataFrame with the expanded
    column set. Used by both training (val/test) and serving paths.

- `ml/features/persistence.py` — write the versioned artifacts. Public
  API:
  - `save_feature_artifacts(fitted_preprocessor: ColumnTransformer,
    locality_aggregator: LocalityAggregator,
    feature_list: list[str], version: str = "v1") -> dict[str, Path]`
    — writes:
      - `models/feature_pipeline_{version}.pkl` — `joblib.dump` of a
        tuple `(fitted_preprocessor, locality_aggregator)` so the
        serving path can load both at once. Tuple, not dict, because
        the order is meaningful: preprocessor first, aggregator
        second.
      - `models/feature_list_{version}.json` — `json.dump` of the
        final ordered feature list + the engineered-feature recipe
        (which base column produced which derived column). This is
        the `02-TRD.md` §6 traceability artifact.
      Returns a dict of `{artifact_name: Path}` for tests to assert
      the right files landed.
  - `load_feature_artifacts(version: str = "v1") -> tuple[ColumnTransformer,
    LocalityAggregator, list[str]]` — the inverse, used by future
    specs (training script + FastAPI serving). Asserts the files
    exist before reading; raises `FileNotFoundError` with the
    expected path so missing-artifact errors are actionable.
  - `FEATURE_ARTIFACT_VERSION: str = "v1"` — pinned module constant.
    Future spec (a training retrain) bumps to `"v2"`.

- `scripts/build_features.py` — the one entry point that reads the
  cleaned Parquet, produces the feature frame, fits the locality
  aggregator + preprocessor on the **training subset only** (the
  `is_outlier == False` rows from the 70% train split per the fixed
  protocol), and serializes the artifacts. Invoked as
  `python scripts/build_features.py` from repo root. Idempotent.
  Logs (stdlib `logging`) at INFO: rows in / rows after outlier filter
  / locality aggregator row count / preprocessor fit time /
  artifacts written.

- `scripts/build_feature_selection_report.py` — writes
  `data/processed/feature_selection_report.md` (the TRD §9 Round 1
  correlation report). Uses only base + engineered columns, no
  fitted model required. Output sections:
    - "Engineered column summary" — for each engineered column:
      definition, dtype, missingness %, sample statistics (mean,
      median, std) per city.
    - "Numeric correlation matrix" — Pearson correlations on the
      numeric features (Round 1's |corr| > 0.9 multicollinearity
      filter). Identifies pairs to drop with rationale (e.g.
      `built_up_area` vs `area_per_bedroom`).
    - "Categorical cardinality table" — per categorical feature, the
      number of unique values and the top-5 most-frequent values per
      city.
    - "Locality-aggregate preview" — per city, the top-5
      highest-priced localities and the 5 lowest-priced (by
      `locality_avg_price_sqft`), the listing-count distribution,
      and the smoother's effective prior weight (informational).
    - "Top amenities selected" — the K=10 amenities the `has_*`
      flags were built for, with their corpus frequency.
    - "Round 1 selection decisions" — explicit keep/drop list with
      rationale for every drop. Anything kept without an explicit
      justification is a defect.
  - Round 2 (tree-based importance) and Round 3 (permutation + SHAP)
  reports are produced by the training spec (Week 4) once a model
  exists. Documented here as out-of-scope, with a one-line
  handoff: "Round 2 + 3 reports will be appended by
  `scripts/train_price_model.py` (future spec) using the saved
  fitted model."

- `tests/test_feature_frame.py` — pytest unit tests for the row-feature
  builders. Pure-Python; uses small in-memory DataFrames built from
  `pd.DataFrame(...)`, no Parquet / DB / network dependency.
- `tests/test_locality_aggregator.py` — pytest unit tests for the
  leakage-safe locality aggregator. Includes an explicit
  `test_locality_aggregator_does_not_leak_target` that fits on a
  training frame containing row N, then asserts that row N's
  transformed `locality_avg_price_sqft` excludes row N's own
  contribution — i.e. `LocalityAggregator` uses a leave-one-out
  semantics internally, OR the test sets up two disjoint (city,
  locality) groups so the test's claim is well-defined. **Decision**:
  implement the aggregator as "compute the group mean on the **other**
  rows in the same group" (leave-one-out by row), not "compute the
  group mean on all rows including self". This is stricter than the
  Rules §2.3 minimum ("train-only fit") but is the standard fix for
  the "locality aggregate includes the target row's own price"
  leakage class. Documented in the module docstring with a citation
  to the Rules doc + TRD §8.
- `tests/test_preprocessor.py` — pytest unit tests for the
  `ColumnTransformer` factory + fit/transform round-trip.
- `tests/test_persistence.py` — pytest unit tests for the artifact
  save/load round-trip; uses `tmp_path` fixture.
- `tests/test_build_features_script.py` — pytest unit test that runs
  the script against a tiny synthetic Parquet fixture (built in
  `tmp_path` by the test) and asserts the four expected artifacts
  land in `models/` / `data/processed/`.

**Modify:**
- `requirements.txt` — add `joblib` if not already present (it is used
  by both this spec and the future training spec for serializing
  sklearn pipelines; the standard pattern). Verify against the
  pinned versions in Step 01; flag the addition explicitly per
  CLAUDE.md "no new packages without checking first." *(ponytail:
  `joblib` ships as a scikit-learn dependency and is already
  importable in the venv; if `pip freeze | grep -i joblib` returns
  empty, add it with a pinned minor version.)*
- `scripts/run_pipeline.py` — append one line:
  `subprocess.run([sys.executable, "scripts/build_features.py"],
  check=True)` after the Step 07 imputation script line, so the
  feature build is reproducible end-to-end via `make pipeline`
  (TRD §13). No logic change; just sequence registration.
- `ml/cleaning/__init__.py` — no change (cleaning doesn't import from
  features; layering rule preserved).
- `ml/__init__.py` — add `from ml import features  # noqa: F401` so
  the new submodule is importable as `ml.features.build_feature_frame`
  consistently with `ml.cleaning`.

**No changes** to:
- `app/`, `api/`, `data/raw/`, `data/processed/clean_listings.parquet`,
  `notebooks/`, `migrations/`, `tests/conftest.py` (existing fixtures
  remain; this spec adds new fixtures in the new test files only).
- `CLAUDE.md`'s "Implemented vs stub routes" table — this spec adds
  **no routes**.

## New dependencies
- `joblib` — for serializing the `(preprocessor, aggregator)` tuple.
  Standard scikit-learn dep; verify it's already transitively
  installed before adding. If not, add with a pinned minor version
  matching scikit-learn's compatibility window.
- **No** new npm packages, **no** new FastAPI/Flask routes, **no**
  new database drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL.
- **No dealer/contact/media-URL fields ever reach the UI or an export.**
  The feature frame is offline-only and never touches an HTTP layer;
  the script does not log any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. The
  `test_feature_frame_excludes_contact_fields` test pins this.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** The
  `LocalityAggregator` and preprocessor are fit on the **same 70%
  train split** as the eventual price model will use — but the split
  is **defined here** (the spec writes the split helper that both
  this spec's artifact build AND the future training spec call),
  so train/val/test boundaries are identical across both. The
  split helper lives in `ml/features/split.py` (new file, small):
    - `FIXED_RANDOM_STATE: int = 42` — pinned constant (Rules §5.4,
      TRD §10).
    - `split_train_val_test(df: pd.DataFrame,
      target: str = "price") -> tuple[pd.DataFrame, pd.DataFrame,
      pd.DataFrame]` — returns `(train_df, val_df, test_df)` using
      sklearn's `train_test_split` with `test_size=0.30` first
      (yielding 70% train + 30% temp), then `test_size=0.50` on the
      temp (yielding 15% val + 15% test). Both calls use
      `random_state=FIXED_RANDOM_STATE`. The split is **stratified
      on `city`** to preserve per-city proportions. The classifier
      spec (Week 8) reuses the exact same split helper.
- **Leakage rule is hard, not nice-to-have (Rules §2.3, §8.4).**
    - `LocalityAggregator.fit()` runs on `train_df[is_outlier ==
      False]` only. Outlier rows are still in the input frame but
      excluded from the aggregate calculation.
    - `LocalityAggregator.transform()` is a pure join; it never
      refits. This is the "apply, don't recompute" rule (Rules
      §8.2).
    - `price_per_sqft` is a within-row derivation (allowed), but
      it is **never** a regression input for the classifier (Rules
      §8.1). The classifier spec removes it; the price model keeps
      it. Pinned by `test_classifier_feature_excludes_price_per_sqft`
      in the future classifier spec, not here.
- **No `price` as a regression input.** The feature frame includes
  `price` only as a training target (the training spec drops it
  before `fit`); it is not in `NUMERIC_FEATURES`,
  `ORDINAL_FEATURES`, or `ONEHOT_FEATURES`. The
  `test_feature_frame_excludes_price_from_inputs` test pins this.
- **Ordinal categoricals use fixed category orderings.**
  `OrdinalEncoder` is order-sensitive; the `ORDINAL_CATEGORY_ORDERINGS`
  dict is the single source of truth, and the preprocessor passes it
  to `OrdinalEncoder(categories=[...])` so train and serve agree.
  `test_ordinal_category_orderings_are_pinned` asserts the dict
  contents.
- **One-hot `handle_unknown="ignore"`.** Per Step 11 spec, the
  `ColumnTransformer`'s one-hot branch uses
  `OneHotEncoder(handle_unknown="ignore", drop="first")` (matching
  TRD §U-TRD-3). The `drop="first"` avoids the dummy-variable trap;
  the `handle_unknown="ignore"` ensures a serving-time locality
  unseen in training produces a row of zeros in the one-hot
  columns, not a `ValueError`.
- **`sector` is target-encoded, not one-hot.** TRD §U-TRD-3 leaves
  this as an open choice; this spec picks **smoothed target
  encoding** via `locality_smoothed_price` / `locality_avg_price_sqft`
  (literature lever 4). The `sector` column itself does not appear
  in the preprocessor's input lists. Pinned by
  `test_sector_not_in_column_transformer_inputs`. *(ponytail: one
  encoding choice, not two — the one-hot path would explode the
  feature count for 100+ localities × 4 cities, and target encoding
  carries the locality-mean signal more compactly.)*
- **`transact_type` is a routing key, not a feature (TRD §U-TRD-4,
  Rules §10.3).** It is absent from `NUMERIC_FEATURES`,
  `ORDINAL_FEATURES`, `ONEHOT_FEATURES`. The FastAPI `/predict`
  handler dispatches on it before calling `build_feature_frame()`.
  Documented in the module docstring + pinned by
  `test_transact_type_not_in_column_transformer_inputs`.
- **All randomness is seeded (Rules §5.4).** The split helper uses
  `random_state=42` (pinned). The amenity top-K selection is
  deterministic (frequency sort, no sampling).
- **Config values live in code constants, not environment variables.**
  `SMOOTHING_PRIOR_WEIGHT`, `FIXED_RANDOM_STATE`, the top-K value
  `K=10`, and the version string `v1` are module-level constants.
  Future tuning (different K, different smoothing weight) edits
  the constant and logs the change in
  `feature_selection_report.md`'s "Decisions" section.
- **`luxury_category` is server-derived at inference time, never a
  client-supplied feature (Rules §10.2).** The feature frame
  includes it (it's a real cleaned-data column), but the FastAPI
  request body drops it on parse (Step 11's
  `Field(default=None, exclude=True)`). At inference, the serving
  path resolves it from the amenity checklist before calling
  `build_feature_frame()`. That resolution is a future serving
  spec, not this one — but the feature frame is set up to accept
  it as an already-resolved value.
- **Logging uses stdlib `logging` only.** One module-level logger
  per file (`logger = logging.getLogger(__name__)`); no logger
  reconfiguration. INFO-level for stage boundaries; WARNING for
  unexpected-but-non-fatal conditions (e.g. a locality in val/test
  not seen in training); ERROR for hard failures.
- **No imports from `app/`, `api/`, `models/` (consumers, not
  peers).** The feature pipeline is pure `ml/`. The persistence
  layer writes to `models/` (it's the artifact directory, not the
  module namespace). The `test_feature_pipeline_does_not_import_app_or_api`
  test pins this.
- **No writes to `data/raw/`.** The script re-invokes
  `assert_raw_readonly()` (Step 02) even though it doesn't touch
  raw CSVs — symmetry gate.
- **No notebook-only steps (Rules §5.3).** Everything in this spec
  is reproducible via `python scripts/build_features.py`. No
  Jupyter cell mutates a column or fits a transform that the
  script can't reproduce.

## Definition of done

1. `python -m pytest tests/test_feature_frame.py
   tests/test_locality_aggregator.py tests/test_preprocessor.py
   tests/test_persistence.py tests/test_build_features_script.py -v`
   from repo root runs and passes. Tests required (exact names):
   - **Feature frame** (`test_feature_frame.py`):
     - `test_engineered_columns_constant_has_expected_entries` — pins
       `len(ENGINEERED_COLUMNS) == 11` and the names.
     - `test_derive_row_features_adds_expected_columns` — confirms
       every column in `ENGINEERED_COLUMNS` (minus the locality ones,
       which `LocalityAggregator` adds) appears in the output.
     - `test_price_per_sqft_is_within_row_ratio` — synthetic row
       with `price=10_000_000` and `built_up_area=1000` yields
       `price_per_sqft=10_000.0` exactly.
     - `test_n_amenities_counts_list_length` — `amenities_list` of
       length 3 yields `n_amenities=3`.
     - `test_floor_ratio_is_floor_over_total` — `floor_num=7,
       total_floor=14` yields `floor_ratio=0.5`.
     - `test_bath_bed_ratio_handles_zero_bedrooms` — yields
       NaN without raising (pin the divide-by-zero guard).
     - `test_area_per_bedroom_handles_zero_bedrooms` — yields NaN
       without raising.
     - `test_age_bucket_ordinal_mapping_is_pinned` — exact equality
       against the dict documented above.
     - `test_top_amenities_count_equals_10_when_data_has_at_least_10`
       — confirms K=10 default.
     - `test_slugify_amenity_normalizes_punctuation` — `"Swimming
       Pool"` → `"swimming_pool"`; `"Club House / Lounge"` →
       `"club_house_lounge"`.
     - `test_build_feature_frame_raises_on_missing_input_field` —
       omitting `built_up_area` raises `ValueError`.
     - `test_build_feature_frame_column_order_is_deterministic` —
       exact column order match against the pinned tuple.
     - `test_feature_frame_excludes_contact_fields` — output
       columns contain no name matching `(contact|dealer|phone|
       email|photo|url|spid)`, case-insensitive.
     - `test_feature_frame_excludes_price_from_inputs` —
       `"price"` not in `NUMERIC_FEATURES +
       ORDINAL_FEATURES + ONEHOT_FEATURES`.
   - **Locality aggregator** (`test_locality_aggregator.py`):
     - `test_locality_aggregator_fit_computes_group_means` —
       synthetic 3-city, 5-locality fixture; asserts the
       `fitted_aggregates_` row count + a sample of expected
       values within tolerance.
     - `test_locality_aggregator_transform_joins_by_city_locality`
       — input frame with a new (city, locality) combo falls back
       to the city mean (not zero, not NaN).
     - `test_locality_aggregator_transform_does_not_refit` — fit on
       frame A; transform on frame B with a new (city, locality)
       inserted; assert the inserted row's locality column equals
       the city mean, not a recomputed mean from B's rows.
     - `test_locality_aggregator_excludes_outlier_rows_from_fit` —
       frame contains rows flagged `is_outlier=True`; assert their
       prices do not appear in the group means.
     - `test_locality_aggregator_leave_one_out_semantics` — fits
       on a (city, locality) group with N=10 rows having prices
       `[1, 2, ..., 10]`; transforms row #5 (price=6) and asserts
       `locality_avg_price_sqft == mean([1,2,3,4,5,7,8,9,10]) =
       5.444...` (the row's own contribution is excluded). Locks
       the LOO semantic.
     - `test_locality_aggregator_smoothed_price_blends_with_prior`
       — at the city-mean level the smoother's output equals the
       city mean exactly (LOO prior weight = `n /
       (n+SMOOTHING_PRIOR_WEIGHT)` → limit at n→∞ is the
       locality mean, limit at n→0 is the city mean; pinned
       algebraically for one row).
     - `test_locality_aggregator_idempotent_fit` — calling `fit`
       twice overwrites prior state without leaking rows.
   - **Preprocessor** (`test_preprocessor.py`):
     - `test_numeric_features_constant_matches_trd_section_utrd3`
       — exact equality against the pinned tuple.
     - `test_ordinal_features_constant_has_four_entries` —
       `len(ORDINAL_FEATURES) == 4` and names match the pinned
       tuple.
     - `test_onehot_features_constant_has_four_entries` —
       `len(ONEHOT_FEATURES) == 4` and names match.
     - `test_sector_not_in_column_transformer_inputs` — `"sector"`
       not in any of the three input tuples (target-encoded
       instead).
     - `test_transact_type_not_in_column_transformer_inputs` —
       `"transact_type"` not in any of the three input tuples
       (routing key, not feature).
     - `test_ordinal_category_orderings_are_pinned` — exact
       equality against the pinned dict.
     - `test_make_preprocessor_returns_unfitted_column_transformer`
       — `hasattr(preprocessor, "transformers_") == False`
       until `fit` is called.
     - `test_preprocessor_fit_transform_round_trip` — fits on a
       synthetic frame, transforms a separate frame; output has
       expected shape (numeric count + ordinal count + one-hot
       expanded column count).
     - `test_preprocessor_handle_unknown_ignore_for_onehot` — a
       serving-time frame containing an unseen `facing` value
       does not raise; the unknown-category one-hot row is all
       zeros.
   - **Persistence** (`test_persistence.py`):
     - `test_save_feature_artifacts_writes_three_files` — saving
       to `tmp_path` produces `feature_pipeline_v1.pkl` and
       `feature_list_v1.json` (and the script-callable test also
       produces `data/processed/feature_selection_report.md`).
     - `test_load_feature_artifacts_round_trip` — save then load;
       loaded preprocessor transforms a sample identically to
       the pre-save preprocessor.
     - `test_load_feature_artifacts_raises_on_missing_version` —
       loading `"v999"` raises `FileNotFoundError` with the
       expected path in the message.
     - `test_feature_list_json_contains_recipe_section` — the
       JSON has both the `feature_names` list and the
       `engineered_feature_recipe` dict (which base column
       produced which derived column).
   - **Script** (`test_build_features_script.py`):
     - `test_build_features_script_runs_end_to_end_on_synthetic_parquet`
       — write a tiny synthetic `clean_listings.parquet` to
       `tmp_path`, set the env var `HOUSINGIQ_PROCESSED_DIR` (or
       a CLI flag) to point the script at it, run the script via
       `subprocess.run`, assert the four expected artifacts land
       and the report markdown contains the "Engineered column
       summary" section.
2. `python -m pytest -m "not realdata"` from repo root still passes
   (no real-data dependency introduced by this spec).
3. `ruff check ml/features/ scripts/build_features.py
   scripts/build_feature_selection_report.py tests/test_feature_frame.py
   tests/test_locality_aggregator.py tests/test_preprocessor.py
   tests/test_persistence.py tests/test_build_features_script.py`
   reports zero issues.
4. `python -c "from ml.features import build_feature_frame,
   LocalityAggregator, make_preprocessor, INPUT_NUMERIC_FEATURES;
   from ml.features.feature_frame import ENGINEERED_COLUMNS;
   print(len(ENGINEERED_COLUMNS))"` from repo root prints `11`
   without error — public API imports cleanly.
5. `python scripts/build_features.py` from repo root exits 0 and
   prints the four INFO log lines (rows in / rows after outlier
   filter / aggregator row count / artifacts written) — manual
   smoke test of the script entry point.
6. `python scripts/build_feature_selection_report.py` from repo
   root exits 0 and writes `data/processed/feature_selection_report.md`
   with all six required sections — manual smoke test of the
   report builder.
7. `git status` after committing shows only the new files
   listed above, the modified `requirements.txt`, the modified
   `scripts/run_pipeline.py`, and the modified `ml/__init__.py`.
   No accidental additions to `app/`, `api/`,
   `data/processed/clean_listings.parquet`, `data/raw/`, or
   `notebooks/`.
8. `CLAUDE.md`'s "Implemented vs stub routes" table is unchanged
   — this spec adds **no routes**.
9. `07-TRACKER.md` is updated via `/update-tracker` to mark Days
   15, 16, and 17 (Week 3 — feature engineering: core engineered
   features, amenity flags + categorical encoding, leakage-safe
   locality aggregates) as **Done** with the actual date and the
   artifacts produced. Days 18–21 (Round 1/2/3 selection + the
   Week 3 checkpoint) remain Not Started because Round 2/3
   require a fitted model and land with the training spec.
