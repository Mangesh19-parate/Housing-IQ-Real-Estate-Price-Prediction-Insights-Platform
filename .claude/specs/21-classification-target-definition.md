# Spec: Classification Target Definition

## Overview
Define the two target labels that drive the HousingIQ Classification module
and write the pure, leakage-safe builder that produces them on top of the
cleaned dataset. Per the v4 framing in `01-PRD.md` §U4.1–§U4.2, the
Classification module is the **Affordability & Investment-Tier Filter**,
not a generic pattern-completeness exercise, so the two targets have two
different business meanings:

- `price_tier` (`Budget` / `Mid-Range` / `Premium` / `Luxury`) — an
  **affordability segmentation** label, quantile-binned from
  `price_per_sqft` per city. Powers the Recommender's "Fits my budget"
  filter and the Affordability chip on the Predict result screen.
  **Supporting / secondary** — never the module's headline output.
- `good_deal_verdict` (`Good Deal` / `Fair Price` / `Overpriced`) — an
  **investment-decision** label, derived from the residual between a
  listing's actual price and the **out-of-fold** price regression
  prediction, with ±10% thresholds (tunable per city). This is the
  **primary** Classification deliverable.

This spec ships the deterministic, leakage-safe construction of both
labels **plus** the per-city residual-threshold calibration that the
classifier training spec (Spec 22) will read from disk. No classifier
is trained here — only the targets are defined, so the Classification
module's modeling spec (Spec 22) has a clean, reproducible input.

Module: **classification**.

## Depends on
- **Step 07** — `07-clean-listings-parquet-pipeline` — supplies
  `data/processed/clean_listings.parquet`, the canonical input. The
  builder reads this file and **never** writes back to it.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` — supplies
  the `is_outlier` flag. The builder excludes outlier rows from the
  training subset per Rules §1.4, but **retains** them in the labeled
  output (with the flag preserved) so the classifier training spec can
  decide for itself whether to use the full set or the filtered subset
  (mirrors Step 12's `LocalityAggregator` precedent).
- **Step 12** — `12-feature-engineering-price-model` — supplies
  `build_feature_frame` (price_per_sqft is one of the 11 engineered
  columns) and `ml/features/split.py` (70/15/15 with
  `random_state=42`). The builder reuses `split_train_val_test`
  unchanged so the train-set quantile boundaries this spec computes
  are reproducible on any checkout that has the same dataset version.
- **Step 13 / 14** — `13-baseline-regression-model-training` and
  `14-xgboost-lightgbm-price-model-training` — supply the trained
  `price_model_{sale,rent}_v{n}.pkl` artifacts and the per-row
  out-of-fold regression predictions written by the training scripts
  to `data/processed/oof_predictions_v{n}.parquet` (per
  TRD U-TRD-1 / U-TRD-6, the classifier requires **out-of-fold**
  predictions, never in-sample). If the OOF prediction file is
  missing, the spec's `build_good_deal_labels()` returns a labelled
  frame with `good_deal_verdict = null` and a clear `reason` column
  for every row, so the caller can fall back to a single
  train-only-derived proxy (see Rules for implementation) without
  silently shipping a leaking target.
- **`02-TRD.md` §U-TRD-1 + §U-TRD-6** — the original technical spec
  for label construction (per-city quantile binning; ±10% residual
  thresholds tunable per city).
- **`01-PRD.md` §U4.1–§U4.4** — the v4 reframing that promotes
  `good_deal_verdict` to the primary deliverable.
- **`05-BACKEND-SCHEMA.md` §U-SCHEMA-8** — final 3-class verdict field
  name (`good_deal_verdict`) and the renamed/repurposed `price_tier`.
- **`08-RULES.md` §8** — Classification-module rules: leakage ban on
  `price_tier` as a regression input, train-set-only quantile
  boundaries, per-city tier definition, per-city confusion-matrix
  review.
- **`08-RULES.md` §2.1 + §5.4** — fixed 70/15/15 split,
  `random_state=42`, original-₹-scale metrics.
- **`08-RULES.md` §2.3** — leakage-safe aggregation; the OOF-prediction
  rule for `good_deal_verdict` is a direct application.

## Routes / Endpoints
No new routes/endpoints. This spec is offline tooling only — a
pure-Python label builder, a calibration helper, a CLI, and tests. The
FastAPI `/classify` route is a follow-on spec (the
`api/routers/classify.py` schema will read the labels this spec
produces from the `data/processed/classifier/` directory).

## Data / Schema changes

**Read:**
- `data/processed/clean_listings.parquet` (Step 07).
- `data/processed/oof_predictions_v{n}.parquet` (Step 13 / 14 — per-row
  out-of-fold price regression prediction, with columns
  `listing_id`, `transact_type`, `oof_predicted_price`).

**Write (new directory `data/processed/classifier/`):**
- `data/processed/classifier/price_tier_labels.parquet` — one row per
  listing in `clean_listings.parquet` (incl. outliers — `is_outlier`
  preserved). Columns:
  - `listing_id` (str, PK; matches Step 07's `listing_id`).
  - `city` (str).
  - `locality` (str).
  - `transact_type` (str — `Sale` / `Rent`).
  - `price_per_sqft` (float; copied from the cleaned frame).
  - `price_tier` (str; one of `Budget` / `Mid-Range` / `Premium` /
    `Luxury`).
  - `tier_quantile_q1` (float; the per-city `price_per_sqft` 25th
    percentile from the training subset — null on outlier rows).
  - `tier_quantile_q2` (float; 50th percentile).
  - `tier_quantile_q3` (float; 75th percentile).
  - `is_outlier` (bool; copied from Step 06).
  - `dataset_version` (str; the parquet's filename + sha1 of the
    first 1 MB, per Rules §1.3).
- `data/processed/classifier/price_tier_quantile_boundaries.json` —
  per-(city × transact_type) 25th/50th/75th `price_per_sqft`
  cut-points, computed on the **training subset only**
  (`is_outlier == False` AND `split == "train"`, per Rules §8.2).
  Schema:
  ```json
  {
    "Gurgaon": {"Sale": [3500.0, 5200.0, 7800.0], "Rent": [18.0, 27.0, 42.0]},
    "Hyderabad": {"Sale": [...], "Rent": [...]},
    "Kolkata":   {"Sale": [...], "Rent": [...]},
    "Mumbai":    {"Sale": [...], "Rent": [...]}
  }
  ```
  Rent is omitted (key absent) when the per-(city × transact_type)
  training subset has fewer than 100 rows (small-sample guard, same
  shape as the v1/v2 `rent_min_rows=500` precedent but tighter
  because quartile estimation needs more than 50% of a city-sample
  on each side of the median to be meaningful).
- `data/processed/classifier/good_deal_labels.parquet` — one row per
  listing that has a matching OOF prediction (joined on
  `listing_id` + `transact_type`). Columns:
  - `listing_id` (str).
  - `city` (str).
  - `transact_type` (str).
  - `actual_price` (float; from `clean_listings.parquet`).
  - `oof_predicted_price` (float).
  - `residual_pct` (float;
    `(actual_price - oof_predicted_price) / oof_predicted_price`).
  - `good_deal_verdict` (str; one of `Good Deal` / `Fair Price` /
    `Overpriced`; null when the OOF prediction is missing for the
    row).
  - `verdict_threshold_low` (float; the city-specific low threshold;
    null when OOF is missing).
  - `verdict_threshold_high` (float; the city-specific high
    threshold; null when OOF is missing).
  - `is_outlier` (bool; copied from Step 06).
  - `reason` (str; null when the verdict is set; populated with a
    short human-readable explanation when verdict is null, e.g.
    `"missing_oof_prediction"`, `"below_min_rows"`,
    `"non_positive_predicted_price"`).
  - `model_version` (str; the price model version the OOF
    predictions came from, e.g. `"v2"`).
- `data/processed/classifier/good_deal_thresholds.json` — the
  calibrated per-city residual thresholds used to assign verdicts.
  Schema:
  ```json
  {
    "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 12345,
                "median_residual": 0.012, "iqr_residual": 0.18},
    "Hyderabad": {...}, "Kolkata": {...}, "Mumbai": {...},
    "_default": {"low": -0.10, "high": 0.10, "rationale":
                  "Used when a city has < 100 train rows."}
  }
  ```
  The `_default` block is the Rules-§12.2 ±10% starting point
  applied when per-city calibration is impossible.

**No writes** to:
- `data/raw/` (immutable, Rules §1.2).
- `data/processed/clean_listings.parquet` (immutable input, Rules
  §1.2 + Step 07).
- `data/processed/feature_selection_report.md` (this spec doesn't
  amend the feature-selection report — it writes its own
  `data/processed/classifier/label_construction_report.md`, see
  below).
- `data/processed/analytics_cache/`, `data/model_registry.csv`,
  `data/app.db`.
- `models/` (no new model artifacts in this spec — only label files;
  the trained classifier `*.pkl` lands in `models/` in Spec 22).

**Report:**
- `data/processed/classifier/label_construction_report.md` — append-
  only, one section per run, with the per-city tier row counts
  (so a reviewer can confirm the quartile split is roughly even),
  the per-city verdict row counts and `median_residual` /
  `iqr_residual`, the `dataset_version` and `git_commit` of the
  run, and a "Leakage audit" line confirming the quantile
  boundaries were computed on the **train subset only** (not on
  the union of train+val+test). Pre-populated with a header
  section; the first run appends the first "Run YYYY-MM-DD" block.
  Mirrors the style of `data/processed/feature_selection_report.md`.

## Templates / UI
None. This spec is offline tooling — no Flask templates, no static
assets, no HTML. The UI that surfaces `price_tier` and
`good_deal_verdict` (VerdictBadge + AffordabilityChip on the Predict
result screen, the tier filter in the Recommender form) is a
follow-on spec.

## Files to change / Files to create

**Create:**
- `ml/classification/` — new sub-package, one file per concern, so
  the label-construction logic is testable in isolation from the
  classifier-training logic that will land in Spec 22. Re-exports
  from `ml/classification/__init__.py`.
  - `__init__.py` — empty; re-exports the five public symbols:
    `build_price_tier_labels`, `build_good_deal_labels`,
    `calibrate_good_deal_thresholds`, `TIER_LABELS`,
    `VERDICT_LABELS`.
  - `tiers.py` — the `price_tier` label builder. Public API:
    - `TIER_LABELS: tuple[str, ...] = ("Budget", "Mid-Range",
      "Premium", "Luxury")` — pinned literal; classifier training
      will one-hot / ordinal-encode from this order.
    - `QUANTILE_CUTPOINTS: tuple[float, ...] = (0.25, 0.50, 0.75)`
      — the quartile boundaries. Pinned (mirrors the
      `protocol.py` precedent in Spec 15).
    - `compute_per_city_quantile_boundaries(train_df: pd.DataFrame,
      price_per_sqft_col: str = "price_per_sqft",
      group_keys: tuple[str, ...] = ("city", "transact_type"),
      min_rows: int = 100) -> dict[str, dict[str, list[float]]]` —
      pure function; returns the per-(city × transact_type)
      25/50/75 percentile cut-points. Skips any group with
      `n_train < min_rows` and emits a logged WARNING.
    - `build_price_tier_labels(clean_listings: pd.DataFrame,
      boundaries: dict[str, dict[str, list[float]]] | None = None,
      processed_dir: Path | str | None = None,
      split_col: str = "split",
      min_rows: int = 100) -> tuple[pd.DataFrame, dict[str,
      dict[str, list[float]]]]` — the public entry point. Steps:
      1. Read `clean_listings.parquet` from `processed_dir` (or
         accept a pre-loaded `clean_listings` DataFrame for
         testability).
      2. Reuse `ml.features.split.split_train_val_test(df,
         target="price", random_state=42)` to tag rows with
         a `split` column (`train` / `val` / `test`) — the
         exact same helper Step 12 already uses, so the
         boundaries this spec produces are byte-equivalent to
         whatever Spec 22 will see.
      3. Filter to `is_outlier == False` for boundary
         computation (Rules §1.4 + §8.2: train-set boundaries
         on the non-outlier training subset).
      4. Call `compute_per_city_quantile_boundaries(...)` on
         the filtered training subset. If `boundaries=None`,
         the function computes them; if `boundaries` is passed
         (e.g., from a previous run's
         `price_tier_quantile_boundaries.json`), the function
         **reuses** them — never recomputes (Rules §8.2:
         "re-applied, not recomputed" on val/test).
      5. For every row in the **full** `clean_listings`
         (including outliers), look up the matching
         `(city, transact_type)` boundary, and assign:
         - `price_tier = "Budget"` if
           `price_per_sqft < q1`.
         - `price_tier = "Mid-Range"` if
           `q1 <= price_per_sqft < q2`.
         - `price_tier = "Premium"` if
           `q2 <= price_per_sqft < q3`.
         - `price_tier = "Luxury"` if
           `price_per_sqft >= q3`.
         - `price_tier = null` if the row's
           `(city, transact_type)` group was skipped
           (< min_rows). Outlier rows get a tier label
           (so the analytics store stays complete) but the
           `tier_quantile_q{1,2,3}` columns are null on
           outlier rows (the boundaries weren't computed
           using them).
         - `price_tier = null` if `price_per_sqft` is
           null/missing for the row.
      6. Return the labeled DataFrame + the boundaries dict.
         The function is **pure** with respect to its inputs;
         persistence is the caller's job (the CLI below).
  - `verdicts.py` — the `good_deal_verdict` label builder. Public
    API:
    - `VERDICT_LABELS: tuple[str, ...] = ("Good Deal",
      "Fair Price", "Overpriced")` — pinned literal.
    - `DEFAULT_THRESHOLD_LOW: float = -0.10` — the Rules §12.2
      starting point; the calibration step (`thresholds.py`) may
      override this per city.
    - `DEFAULT_THRESHOLD_HIGH: float = 0.10`.
    - `build_good_deal_labels(clean_listings: pd.DataFrame,
      oof_predictions: pd.DataFrame | None = None,
      thresholds: dict[str, dict[str, float]] | None = None,
      model_version: str = "v2",
      processed_dir: Path | str | None = None) -> tuple[pd.DataFrame,
      dict[str, dict[str, float]]]` — the public entry point.
      Steps:
      1. Read `clean_listings.parquet` from `processed_dir` (or
         accept a pre-loaded DataFrame).
      2. If `oof_predictions` is None, attempt to load
         `data/processed/oof_predictions_{model_version}.parquet`;
         if that file is also missing, log a WARNING and
         continue with `oof_predictions = None` (caller
         surfaces the all-nulls `good_deal_verdict` frame).
      3. Left-join `clean_listings[[listing_id,
         transact_type, city, actual_price (=price), is_outlier]]`
         against `oof_predictions[[listing_id, transact_type,
         oof_predicted_price]]` on `(listing_id,
         transact_type)`. Rows without a matching OOF
         prediction get `oof_predicted_price = null` and
         `reason = "missing_oof_prediction"`.
      4. Compute `residual_pct = (actual_price -
         oof_predicted_price) / oof_predicted_price` for
         every row. Rows with `oof_predicted_price <= 0`
         or null get `residual_pct = null` and `reason =
         "non_positive_predicted_price"`.
      5. Look up the per-city threshold: if the row's `city`
         key exists in `thresholds`, use
         `(thresholds[city]["low"],
         thresholds[city]["high"])`; else fall back to
         `(DEFAULT_THRESHOLD_LOW, DEFAULT_THRESHOLD_HIGH)`
         and log INFO that the default was used.
      6. Assign `good_deal_verdict`:
         - `"Good Deal"` if `residual_pct <=
           threshold_low`.
         - `"Fair Price"` if `threshold_low <
           residual_pct < threshold_high`.
         - `"Overpriced"` if `residual_pct >=
           threshold_high`.
         - `null` if `residual_pct` is null (with `reason`
           set in step 3 / 4).
      7. Return the labeled DataFrame + the thresholds dict
         (the caller can persist thresholds separately).
         The function is pure with respect to its inputs.
  - `thresholds.py` — the per-city residual-threshold
    calibrator. Public API:
    - `MIN_CITY_TRAIN_ROWS: int = 100` — the minimum
      non-outlier training rows for per-city calibration;
      below this, the city uses the default ±10% thresholds
      and the calibrator emits a logged WARNING.
    - `calibrate_good_deal_thresholds(good_deal_labels:
      pd.DataFrame, is_train_mask: pd.Series, default_low: float
      = DEFAULT_THRESHOLD_LOW, default_high: float =
      DEFAULT_THRESHOLD_HIGH) -> dict[str, dict[str, float]]`
      — pure function. Steps:
      1. Filter `good_deal_labels` to
         `is_outlier == False` AND `is_train_mask == True`
         AND `residual_pct` not null.
      2. Per city: compute `n`, `median_residual`,
         `iqr_residual` (75th − 25th percentile of
         `residual_pct`).
      3. If `n < MIN_CITY_TRAIN_ROWS`, fall back to
         `(default_low, default_high)` and emit a logged
         WARNING that the city is using the default.
      4. Otherwise: emit
         `(low=median_residual - 0.5*iqr_residual,
         high=median_residual + 0.5*iqr_residual)` as the
         per-city band, **clamped** to
         `[default_low - 0.10, default_high + 0.10]` so a
         pathological city can't produce a band wider than
         ±20%. The clamp + the band formula are the
         Rule-§12.2 starting-point + the "tune per city"
         allowance; the Decision Log entry for this spec
         documents the formula.
         **Why this formula:** the per-city
         `iqr_residual` is the most data-driven summary of
         the natural noise floor in the regression model's
         residual distribution; setting the band to
         `median ± 0.5 × IQR` declares a "good deal" as
         anything that beats the typical within-city
         regression error by half a band. This is one line
         of math and is auditable in the
         `label_construction_report.md` per-city section;
         Spec 22 may replace it with an empirically tuned
         formula, but the formula is pinned here for v1.
      5. Return the per-city thresholds dict + a
         `"_default"` block carrying `(default_low,
         default_high)` and a `rationale` string
         ("Used when a city has < 100 train rows.").
  - `report.py` — the report writer. Public API:
    - `write_label_construction_report(*, output_dir: Path,
      price_tier_df: pd.DataFrame, good_deal_df: pd.DataFrame,
      quantile_boundaries: dict, good_deal_thresholds: dict,
      dataset_version: str, git_commit: str, min_rows: int) ->
      Path` — appends one "Run YYYY-MM-DD HH:MM:SS" section
      to `data/processed/classifier/label_construction_report.md`
      (or writes the header on first call). Includes:
      - Per-city `price_tier` row counts (so the reviewer
        can see if a tier is suspiciously empty — e.g.,
        Kolkata Luxury might have < 5 rows).
      - Per-city `good_deal_verdict` row counts +
        `median_residual` + `iqr_residual` + the chosen
        `low` / `high` thresholds.
      - A "Leakage audit" line stating "Quantile
        boundaries computed on `is_outlier==False` AND
        `split=='train'` rows only; size = N. Verified
        `is_outlier==False` AND `split=='train'` sum
        equals N." (catches a future bug where someone
        recomputes on the full frame).
      - `dataset_version` and `git_commit`.
    - `write_price_tier_artifacts(df: pd.DataFrame,
      boundaries: dict, output_dir: Path) -> tuple[Path, Path]`
      — writes the two price-tier files described in the
      "Data / Schema changes" section.
    - `write_good_deal_artifacts(df: pd.DataFrame, thresholds:
      dict, output_dir: Path) -> tuple[Path, Path]` — writes
      the two good-deal files.
  - `__init__.py` — re-exports the five public symbols.

- `scripts/build_classification_labels.py` — the CLI entry point.
  Invoked as
  `python scripts/build_classification_labels.py [--processed-dir
  data/processed] [--output-dir data/processed/classifier]
  [--model-version v2] [--min-city-rows 100]`. Steps:
  1. Parse args (one argparse group).
  2. Call `build_price_tier_labels(...)` to get
     `(price_tier_df, quantile_boundaries)`.
  3. Call `build_good_deal_labels(..., thresholds=None)` to
     get `(good_deal_df_with_null_verdicts,
     _unused_thresholds)`. (Threshold assignment happens
     after calibration, so the labels come back with
     `good_deal_verdict = null` at this point; the next
     step assigns them.)
  4. Call `calibrate_good_deal_thresholds(good_deal_df,
     is_train_mask=...)` to get the per-city thresholds.
  5. Call `build_good_deal_labels(..., thresholds=thresholds)`
     **a second time** to assign `good_deal_verdict` using the
     calibrated thresholds. (Two-pass design: the first
     pass needs `residual_pct` to calibrate, the second pass
     uses the calibrated thresholds to label. Both calls are
     pure functions; the cost is one extra left-join on a
     ~180k-row frame, ~2 seconds.)
  6. Write the four artifact files via
     `write_price_tier_artifacts` + `write_good_deal_artifacts`.
  7. Append the "Run" section to
     `data/processed/classifier/label_construction_report.md`.
  8. Print a one-line stdout summary: `"[OK] price_tier rows=N1
     good_deal rows=N2 (with verdict), N3 rows without OOF
     prediction; per-city thresholds: {city: (low, high),
     ...}; written to <output_dir>"`.
  9. Exit 0. (The script is idempotent — a re-run with the
     same inputs overwrites the four artifact files; a re-run
     with a different `model_version` writes a new set with
     the version embedded in the per-row `model_version`
     column, not in the filename — filenames stay canonical
     so Spec 22 has one fixed path to read from.)

- `tests/test_tiers.py` — pytest tests for
  `compute_per_city_quantile_boundaries` + `build_price_tier_labels`.
  Required tests (exact names):
  - `test_tier_labels_are_pinned` — `TIER_LABELS == ("Budget",
    "Mid-Range", "Premium", "Luxury")`.
  - `test_quantile_cutpoints_are_pinned` — `QUANTILE_CUTPOINTS
    == (0.25, 0.50, 0.75)`.
  - `test_compute_per_city_quantile_boundaries_returns_three_keys_per_city`
    — synthetic 4-city DataFrame; assert every city has a
    3-element list.
  - `test_compute_per_city_quantile_boundaries_skips_small_groups`
    — synthetic city with 50 rows; assert that city is
    missing from the returned dict + a logged WARNING
    occurred.
  - `test_build_price_tier_labels_assigns_four_tiers`
    — synthetic 4-city × 4-tier-perfect DataFrame; assert
    exactly the expected tier labels in the right cells.
  - `test_build_price_tier_labels_reuses_passed_boundaries`
    — pass `boundaries={"Gurgaon": {"Sale": [1.0, 2.0,
    3.0]}, ...}`; assert tier assignment uses these
    literals and does not call `quantile()` on the input
    frame.
  - `test_build_price_tier_labels_preserves_is_outlier_flag`
    — synthetic frame with 10% outliers; assert the
    output frame's `is_outlier` column is bit-for-bit
    identical.
  - `test_build_price_tier_labels_assigns_null_for_missing_city_group`
    — synthetic row whose city is not in the boundaries
    dict; assert `price_tier = null` and `reason = null`
    (this is the "unseen city" edge case, not the
    "missing OOF" edge case — different columns).

- `tests/test_verdicts.py` — pytest tests for
  `build_good_deal_labels`. Required tests (exact names):
  - `test_verdict_labels_are_pinned` — `VERDICT_LABELS ==
    ("Good Deal", "Fair Price", "Overpriced")`.
  - `test_default_thresholds_match_rules_section_12_2` —
    `DEFAULT_THRESHOLD_LOW == -0.10` and
    `DEFAULT_THRESHOLD_HIGH == 0.10`.
  - `test_build_good_deal_labels_assigns_three_verdicts`
    — synthetic frame with three rows (residual = −0.20,
    0.0, +0.20); assert `Good Deal`, `Fair Price`,
    `Overpriced` respectively, with the default ±10%
    thresholds.
  - `test_build_good_deal_labels_uses_per_city_thresholds`
    — synthetic frame with two rows in the same city,
    one with `residual_pct = -0.07` (Good Deal under
    `low=-0.05`, Fair Price under `low=-0.10`); assert
    the per-city override is honored.
  - `test_build_good_deal_labels_assigns_null_when_oof_missing`
    — synthetic frame with `oof_predictions=None`; assert
    every row's `good_deal_verdict = null` and
    `reason = "missing_oof_prediction"`.
  - `test_build_good_deal_labels_assigns_null_when_predicted_non_positive`
    — synthetic frame with one row's
    `oof_predicted_price = 0`; assert that row's
    `good_deal_verdict = null` and
    `reason = "non_positive_predicted_price"`.
  - `test_build_good_deal_labels_does_not_silently_recompute_oof`
    — pass an `oof_predictions` DataFrame with only 3 of
    10 listing_ids; assert exactly those 3 rows get
    non-null verdicts + the other 7 have
    `reason = "missing_oof_prediction"`.

- `tests/test_thresholds.py` — pytest tests for
  `calibrate_good_deal_thresholds`. Required tests (exact names):
  - `test_min_city_train_rows_is_pinned` — `MIN_CITY_TRAIN_ROWS
    == 100`.
  - `test_calibrate_good_deal_thresholds_returns_per_city_dict`
    — synthetic 4-city frame with 100+ rows each; assert
    the returned dict has 4 city keys.
  - `test_calibrate_good_deal_thresholds_falls_back_to_default_for_small_city`
    — synthetic frame with one city at 50 rows; assert
    that city uses `(DEFAULT_THRESHOLD_LOW,
    DEFAULT_THRESHOLD_HIGH)` and a logged WARNING.
  - `test_calibrate_good_deal_thresholds_clamps_pathological_cities`
    — synthetic frame with one city whose
    `iqr_residual = 0.80` (huge); assert the returned
    `(low, high)` is clamped within
    `[default_low - 0.10, default_high + 0.10]`.
  - `test_calibrate_good_deal_thresholds_includes_default_block`
    — assert the returned dict has a `"_default"` key
    with the default thresholds and a `rationale` string.

- `tests/test_label_construction_report.py` — pytest tests for
  the report writer. Required tests (exact names):
  - `test_write_label_construction_report_writes_header_on_first_run`
    — empty `output_dir`; assert the report file exists
    with a header.
  - `test_write_label_construction_report_appends_runs_not_overwrites`
    — call `write_label_construction_report` twice with
    different `git_commit` values; assert both "Run"
    sections are present in the file (append-only).
  - `test_write_label_construction_report_includes_leakage_audit_line`
    — assert the written "Run" section contains the
    string "Leakage audit".
  - `test_write_label_construction_report_includes_per_city_tier_counts`
    — synthetic 4-city frame; assert each city name
    appears in the written report with row counts.

- `tests/test_build_classification_labels_cli.py` — pytest test
  for the CLI. Required tests (exact names):
  - `test_cli_runs_end_to_end_on_synthetic_artifacts` —
    build a tiny synthetic `clean_listings.parquet` +
    tiny synthetic `oof_predictions_v2.parquet` in
    `tmp_path`, invoke the CLI via `subprocess.run`,
    assert exit code 0 + the four artifact files exist
    in `tmp_path/output/`.
  - `test_cli_writes_deterministic_artifacts_on_rerun` —
    run the CLI twice with the same inputs; assert
    `price_tier_labels.parquet` and
    `good_deal_labels.parquet` are byte-identical
    (modulo the `evaluated_at` row column if it
    exists — but it doesn't; the row schema is
    deterministic, so byte-equality is achievable).
  - `test_cli_does_not_log_contact_fields` — grep the
    captured stdout for any column name matching the
    regex `(contact|dealer|phone|email|photo|url|spid)`
    — must be absent.

**Modify:**
- `ml/__init__.py` — add `from ml import classification  #
  noqa: F401` so the new submodule is importable consistently
  with `ml.training` / `ml.features` / `ml.cleaning` /
  `ml.evaluation`.
- `ml/training/evaluation.py` — add a new function
  `write_oof_predictions(predictions: pd.DataFrame,
  model_version: str, processed_dir: Path | str = ...) -> Path`
  so Spec 22 / the existing Step 14 script can emit the OOF
  prediction file this spec consumes. The function is a
  ~10-line wrapper around `pd.DataFrame.to_parquet` that:
  - Asserts the input has columns `listing_id`,
    `transact_type`, `oof_predicted_price`.
  - Writes to
    `processed_dir / f"oof_predictions_{model_version}.parquet"`.
  - Idempotent: overwrites the same file on a re-run with
    the same `model_version` (per the spec's design — only
    one OOF file per model version is on disk at a time,
    so Spec 22 reads from one fixed path).
  This is a small additive helper; the existing
  `evaluation.py` surface is unchanged.
- `scripts/run_pipeline.py` — append one CLI line after the
  current evaluation-gate line (Spec 15's
  `scripts/evaluate_price_model.py` invocation):
  ```python
  subprocess.run(
      [sys.executable, "scripts/build_classification_labels.py",
       "--model-version", "v2"],
      check=False,
  )
  ```
  The pipeline prints the label-builder's stdout summary
  regardless and proceeds to the next stage. Exit code is
  surfaced in the pipeline's final summary. The
  label-builder's `model_version` defaults to `"v2"` (the
  current best price model); an arg override lets a future
  spec re-run labels against a newer model version.
- `requirements.txt` — no new packages; this spec only uses
  pandas / numpy (already pinned) + stdlib. Verify with
  `pip freeze | grep -E "(numpy|pandas)"` and flag
  explicitly per CLAUDE.md "no new packages without
  checking first."

**No changes** to:
- `app/`, `api/` (no Flask/FastAPI code in this spec; routes
  land in follow-on specs).
- `data/raw/`, `data/processed/clean_listings.parquet`,
  `data/processed/feature_selection_report.md`,
  `data/processed/analytics_cache/`, `data/app.db`,
  `data/model_registry.csv` (this spec writes its own
  artifacts under `data/processed/classifier/` and never
  touches the others).
- `notebooks/`, `migrations/`, `tests/conftest.py`
  (existing fixtures remain; new fixtures live in the new
  test files).
- `ml/training/`, `ml/evaluation/` — the existing
  evaluation-gate surface (Spec 15) is read by this spec's
  pipeline, not edited. The new `write_oof_predictions`
  helper is **additive** in `ml/training/evaluation.py`
  and does not change any existing function's behavior.
- `CLAUDE.md`'s "Implemented vs stub routes" table — this
  spec adds **no routes**. The `POST /classify` FastAPI
  route stays a Stub until a follow-on spec wires the
  classifier trained from the labels this spec produces.

## New dependencies
None. The spec uses pandas / numpy (already pinned) + stdlib
(`argparse`, `json`, `logging`, `pathlib`, `subprocess`,
`datetime`). **No** new pip/npm packages, **no**
Flask/FastAPI route additions, **no** new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The spec reads +
  writes Parquet + JSON files only.
- **No dealer/contact/media-URL fields ever reach the
  UI or an export.** The label-construction code does not
  log any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. Pinned by
  a test that greps the CLI's captured stdout for the
  regex — must be absent.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no
  templates.
- **Model evaluation must reference the fixed
  evaluation protocol (Rules §2.1).** The quantile
  boundaries + the residual thresholds are both computed
  on the same `is_outlier == False` training subset that
  the price model's fixed 70/15/15 protocol uses; the
  boundaries helper reuses `ml.features.split.
  split_train_val_test(..., random_state=42)` unchanged,
  so a reviewer auditing "what train set did the labels
  see?" gets the same answer as "what train set did the
  price model see?" — bit-equivalent.
- **Single source of truth for the pinned labels.** The
  `TIER_LABELS` and `VERDICT_LABELS` tuples in
  `ml/classification/tiers.py` and `verdicts.py` are the
  only place these literal label sets live. Spec 22's
  classifier training will ordinal-encode from
  `TIER_LABELS` order and one-hot-encode from
  `VERDICT_LABELS` order; the order is the encoding.
- **`random_state=42` everywhere (Rules §5.4).** The
  quantile helper does not call any random function, but
  the split helper it wraps is pinned to
  `random_state=42`. The threshold calibrator also does
  not randomize; if a future iteration uses a randomized
  search, the same constant is used.
- **Outliers excluded from boundary computation (Rules
  §1.4 + §8.2).** The `build_price_tier_labels` boundary
  computation step filters to `is_outlier == False` AND
  `split == "train"`. The output frame **keeps** the
  outlier rows (with the `is_outlier` flag preserved and
  the quantile columns nulled) so the analytics store
  stays complete and the classifier training can decide
  for itself whether to use them.
- **`transact_type` is a routing key (Rules §10.3).** The
  quantile boundaries are computed per
  `(city, transact_type)` — Sale and Rent never share a
  cut-point set, just as the price regression never
  scores them in the same model pass.
- **`price_tier` is never a regression input feature
  (Rules §8.1).** The label builder **writes** the tier
  column to `price_tier_labels.parquet` for downstream
  consumption; it does **not** write it back to
  `clean_listings.parquet` and does **not** add it to
  any feature DataFrame the price regression reads. The
  classifier training (Spec 22) is the only consumer.
- **Out-of-fold predictions for `good_deal_verdict`
  (Rules §8.4).** The verdict builder's input is the
  OOF prediction parquet, not the in-sample training
  predictions. If the OOF parquet is missing, every
  verdict comes back null with
  `reason = "missing_oof_prediction"` — never silently
  recomputed from in-sample. The
  `test_build_good_deal_labels_does_not_silently_recompute_oof`
  test pins this behavior.
- **Per-city tier definition (Rules §8.3).** The
  quantile boundaries and the verdict thresholds are
  both per-city. The city-relative framing is preserved
  in the report's per-city section so a reviewer can
  see "Luxury" in Mumbai and "Luxury" in Kolkata are
  different price ranges.
- **Per-city confusion-matrix review (Rules §8.5,
  adapted for the target itself).** The label report
  includes per-city row counts for both `price_tier`
  and `good_deal_verdict`; if any cell has < 10 rows
  (Kolkata-Luxury is the obvious candidate), a logged
  WARNING names the cell so Spec 22 can decide whether
  to drop the city from its evaluation slice.
- **Versioned artifacts, never overwritten (Rules
  §2.5).** The four artifact files
  (`price_tier_labels.parquet`,
  `price_tier_quantile_boundaries.json`,
  `good_deal_labels.parquet`,
  `good_deal_thresholds.json`) have fixed filenames —
  a re-run overwrites them. The per-row `model_version`
  column on `good_deal_labels.parquet` records which
  price model the OOF predictions came from; a future
  re-run with `--model-version v3` overwrites the same
  file with v3-tagged rows. The
  `label_construction_report.md` is append-only (one
  "Run" block per CLI invocation), so the v2-run and
  v3-run sections coexist.
- **No MLflow/DVC/etc. (Rules §13 + §15).** No new
  infrastructure is introduced; the
  `label_construction_report.md` is the audit trail.
- **Outlier rows are not deleted (Rules §1.4).** The
  output frames preserve every row from the input
  `clean_listings.parquet`, including outliers. The
  `is_outlier` column is bit-identical between input
  and output (pinned by
  `test_build_price_tier_labels_preserves_is_outlier_flag`).
- **Honest logging of shortfalls (Rules §9.2).** The
  report's per-city section flags any city with < 100
  train rows (falls back to default thresholds), any
  city × tier cell with < 10 rows, and any city with
  0 OOF predictions. No "we probably have enough data"
  rationalization — the WARNING names the cell.
- **No notebook-only steps (Rules §5.3).** Everything
  in this spec is reproducible via
  `python scripts/build_classification_labels.py
  --model-version v2`. No Jupyter cell computes a
  label or a threshold the script can't reproduce.
- **No FastAPI imports in `ml/classification/`.** The
  label builder is pure-Python; no FastAPI / httpx
  client imports. Keeps the label builder runnable in
  offline CI without a live inference service.
- **Logging uses stdlib `logging` only.** One module-
  level logger per file (`logger =
  logging.getLogger(__name__)`). INFO for stage
  boundaries; WARNING for expected-but-noteworthy
  conditions (city skipped, OOF missing, per-cell row
  count < 10); ERROR for hard failures (no
  `clean_listings.parquet` at the configured path,
  bad schema on the OOF parquet).
- **CLI never aborts the pipeline on non-zero exit.**
  The pipeline calls the CLI with `check=False`. The
  label-builder's stdout summary is printed regardless;
  the pipeline proceeds. (This spec's CLI exits 0 on
  success and 1 only on a hard failure — missing input
  parquet, schema mismatch. A null-OOF scenario is a
  WARNING, not a hard failure.)
- **Per-row `reason` column on `good_deal_labels.parquet`.**
  Every row where `good_deal_verdict` is null has a
  non-null `reason` so a reviewer can audit why —
  missing OOF, non-positive prediction, below-min-rows
  city. Pinned by
  `test_build_good_deal_labels_assigns_null_when_oof_missing`
  + the `non_positive_predicted_price` test.

## Definition of done

1. `python -m pytest tests/test_tiers.py
   tests/test_verdicts.py tests/test_thresholds.py
   tests/test_label_construction_report.py
   tests/test_build_classification_labels_cli.py -v` from repo
   root runs and passes. Tests required (exact names):
   - **Tiers** (`test_tiers.py`):
     - `test_tier_labels_are_pinned`
     - `test_quantile_cutpoints_are_pinned`
     - `test_compute_per_city_quantile_boundaries_returns_three_keys_per_city`
     - `test_compute_per_city_quantile_boundaries_skips_small_groups`
     - `test_build_price_tier_labels_assigns_four_tiers`
     - `test_build_price_tier_labels_reuses_passed_boundaries`
     - `test_build_price_tier_labels_preserves_is_outlier_flag`
     - `test_build_price_tier_labels_assigns_null_for_missing_city_group`
   - **Verdicts** (`test_verdicts.py`):
     - `test_verdict_labels_are_pinned`
     - `test_default_thresholds_match_rules_section_12_2`
     - `test_build_good_deal_labels_assigns_three_verdicts`
     - `test_build_good_deal_labels_uses_per_city_thresholds`
     - `test_build_good_deal_labels_assigns_null_when_oof_missing`
     - `test_build_good_deal_labels_assigns_null_when_predicted_non_positive`
     - `test_build_good_deal_labels_does_not_silently_recompute_oof`
   - **Thresholds** (`test_thresholds.py`):
     - `test_min_city_train_rows_is_pinned`
     - `test_calibrate_good_deal_thresholds_returns_per_city_dict`
     - `test_calibrate_good_deal_thresholds_falls_back_to_default_for_small_city`
     - `test_calibrate_good_deal_thresholds_clamps_pathological_cities`
     - `test_calibrate_good_deal_thresholds_includes_default_block`
   - **Report writer**
     (`test_label_construction_report.py`):
     - `test_write_label_construction_report_writes_header_on_first_run`
     - `test_write_label_construction_report_appends_runs_not_overwrites`
     - `test_write_label_construction_report_includes_leakage_audit_line`
     - `test_write_label_construction_report_includes_per_city_tier_counts`
   - **CLI**
     (`test_build_classification_labels_cli.py`):
     - `test_cli_runs_end_to_end_on_synthetic_artifacts`
     - `test_cli_writes_deterministic_artifacts_on_rerun`
     - `test_cli_does_not_log_contact_fields` — grep
       the captured stdout for any column name matching
       `(contact|dealer|phone|email|photo|url|spid)`
       — must be absent.
2. `python -m pytest -m "not realdata"` from repo root
   still passes (no real-data dependency introduced
   by this spec).
3. `ruff check ml/classification/
   scripts/build_classification_labels.py tests/test_tiers.py
   tests/test_verdicts.py tests/test_thresholds.py
   tests/test_label_construction_report.py
   tests/test_build_classification_labels_cli.py` reports
   zero issues.
4. `python -c "from ml.classification import
   build_price_tier_labels, build_good_deal_labels,
   calibrate_good_deal_thresholds, TIER_LABELS,
   VERDICT_LABELS; print(TIER_LABELS, VERDICT_LABELS)"` from
   repo root prints
   `('Budget', 'Mid-Range', 'Premium', 'Luxury')
    ('Good Deal', 'Fair Price', 'Overpriced')` without
   error — public API imports cleanly.
5. `python scripts/build_classification_labels.py
   --model-version v2` from repo root exits 0 and
   prints the `[OK]` summary line — manual smoke test
   of the CLI.
6. After running step 5, the four artifact files
   exist at the canonical paths:
   - `data/processed/classifier/price_tier_labels.parquet`
   - `data/processed/classifier/price_tier_quantile_boundaries.json`
   - `data/processed/classifier/good_deal_labels.parquet`
   - `data/processed/classifier/good_deal_thresholds.json`
   And `data/processed/classifier/label_construction_report.md`
   contains a "Run YYYY-MM-DD HH:MM:SS" section with
   the per-city `price_tier` row counts, the per-city
   `good_deal_verdict` row counts + `median_residual` +
   `iqr_residual` + the chosen `(low, high)` thresholds,
   the "Leakage audit" line, `dataset_version`, and
   `git_commit`.
7. The four artifact files **do not** contain any
   column whose name matches
   `(contact|dealer|phone|email|photo|url|spid)` —
   spot-check by reading the parquet schemas with
   `pyarrow` or pandas and grepping the column names.
8. The `git_commit` recorded in the run's report
   section matches `git rev-parse HEAD` at the time
   the CLI was invoked.
9. `git status` after committing shows only the new
   files listed above, the modified
   `scripts/run_pipeline.py`, the modified
   `ml/__init__.py`, and the modified
   `ml/training/evaluation.py` (additive
   `write_oof_predictions` helper only). No accidental
   additions to `app/`, `api/`,
   `data/processed/clean_listings.parquet`,
   `data/processed/feature_selection_report.md`,
   `data/raw/`, `notebooks/`, or
   `data/processed/analytics_cache/`.
10. `CLAUDE.md`'s "Implemented vs stub routes" table
    is unchanged — this spec adds **no routes**. The
    `POST /classify` FastAPI route stays a Stub until
    a follow-on spec (Spec 22: classifier training +
    serving) wires the classifier trained from the
    labels this spec produces.
11. `07-TRACKER.md` is updated via `/update-tracker` to
    mark Day 50 (Construct price_tier labels) as
    **Done** with the actual measured per-city tier
    row counts from this spec's report. Day 51
    (Classification feature set + Logistic Regression
    baseline) is moved from "Not Started" to
    "Not Started" with a `← needs Spec 21 labels` note
    so the next session picks up from a clean handoff.
12. The Decision Log in `07-TRACKER.md` gains one
    new entry dated today with the rationale for:
    (a) the `median_residual ± 0.5 × iqr_residual`
    per-city verdict-threshold formula (and its
    `±0.10` clamp), and (b) the two-pass
    `build_good_deal_labels` design (first pass for
    residual computation, second pass for threshold
    assignment). Both decisions are auditable from
    the report file in step 6 without rerunning any
    code.
