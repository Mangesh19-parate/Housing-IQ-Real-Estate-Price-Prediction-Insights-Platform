# Implementation Plan: Step 07 — Clean Listings Parquet Pipeline

## Context
Step 07 consumes the deduped + outlier-flagged canonical DataFrame that Step 06's `assemble_cleaned_frame` emits, applies TRD §5's 4-tier missing-value strategy (low/medium/high/drop with constants pinned), and writes the canonical `data/processed/clean_listings.parquet` artifact with a `.meta.json` sidecar (Rules §1.5). It is the **only** spec allowed to write this file, and every downstream consumer (feature engineering, regression, classification, recommender, analytics, insights) reads it. Net effect: a single training-and-serving artifact is now reproducible from raw with one call to `run_clean_listings_pipeline(raw_dir, facet_dir)`.

Ponytail shortcuts (intentional):
- `IMPUTATION_HIGH_TIER_COLUMNS` is left as `Final[tuple[...]] = ()` — the high tier is dynamic per-frame, comes out of `classify_missingness_tiers`. Don't enumerate a constant.
- No new `tests/fixtures/imputation_fixtures.py` — `tests/fixtures/dedup_outlier_fixtures.py::make_frame` covers all 42 test needs.
- No new pip package (`pandas` + `pyarrow` already pinned). If `pyarrow` is absent at test time, the writer test uses `pytest.importorskip("pyarrow")` rather than adding a dep.
- No `tests/fixtures/build_runner.py` for end-to-end — pipeline tests `monkeypatch` `assemble_cleaned_frame` to return a small literal frame.

---

## 1. Spec deltas / open issues

| # | Spec says | Code reality | Resolution |
|---|---|---|---|
| 1 | `IMPUTATION_NUMERIC_LOW = ("balconies", "floor_num_int", "total_floor", "area_sqft")` | `CANONICAL_COLUMNS` has `floor_num` (string-shaped, decoded via Step 04 `decode_floor_num`), not `floor_num_int`. | Implementer filters the tuple against `df.columns` and silently skips absent names. Tests anchor on `balconies` / `total_floor` / `area_sqft`. `ponytail:` comment in `imputation.py` documents the typo. |
| 2 | Reclassify missingness after dropping >70% columns | OK as written — reclassify on the slimmed frame so dropped columns don't distort survivors' percentages. | Document the dual-call pattern in `impute_missing_values` docstring + `_log_imputation_summary`. |
| 3 | Drop >70% columns THEN add `was_missing_*` flags on `(medium+high)` | Right order — dropped columns must not get a flag column (otherwise the Parquet round-trip carries a useless flag). | Confirm with `test_add_was_missing_flags_does_not_create_flag_for_column_without_nans`. |
| 4 | Sidecar `<output_path>.meta.json` — literally `with_suffix(path.suffix + ".meta.json")` | `Path("clean_listings.parquet").with_suffix(".parquet" + ".meta.json")` evaluates to `clean_listings.parquet.meta.json` (two-dot extension, NOT replacement). | **AMBIGUITY (resolved):** spec matches Python reality. The sidecar name is `<file>.parquet.meta.json`. Document with a comment in `writers.py`. |
| 5 | `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER` from canonical + `is_outlier` + `was_missing_*` (alpha) + `outlier_reasons` | `CANONICAL_COLUMNS` already contains `is_outlier` as its last entry. Set-dedup would lose the outlier-flag boundary. | **Resolution:** Pin `CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER` (import-time constant) to the columns knowable at import time — `CANONICAL_COLUMNS + ("outlier_reasons",)`. Add `build_clean_listings_columns_order(df) -> tuple[str, ...]` helper that performs insertion-order union with the actual `df.filter(like="was_missing_")` sorted alphabetically. The writer calls the helper. |
| 6 | `outlier_reasons` round-trip with non-empty list | pyarrow ≥ 8 round-trips Python lists in object columns natively. | Test constructs a frame with `outlier_reasons = ["percentile_price_inr", "iqr_area_sqft"]` and asserts read-back equality via `pd.testing.assert_frame_equal(check_dtype=False)`. |
| 7 | `run_clean_listings_pipeline(persist=False)` pure path | OK — skip writer entirely; rest is unchanged. | `if persist: write_clean_listings_parquet(...)`. `assert_raw_readonly` is still invoked (symmetry gate). |
| 8 | `verify_clean_listings_parquet` returns dict even on missing file | OK — `exists=False` is the trivial case; on read failure, dict has `exists=False, error=str(e)`. | Short-circuit on `not path.exists()`; otherwise attempt `read_clean_listings_parquet` inside try/except. |
| 9 | Test count pinned at 21 / 12 / 9 = 42 | All names verbatim from spec DoD §1. | No drift. |

---

## 2. Module layout

### 2.1 `ml/cleaning/imputation.py` — new

```python
_LOG = logging.getLogger("ml.cleaning.imputation")

# Tier boundaries (TRD §5, single source of truth)
MISSINGNESS_LOW_THRESHOLD:    Final[float] = 0.05
MISSINGNESS_MEDIUM_THRESHOLD: Final[float] = 0.40
MISSINGNESS_HIGH_THRESHOLD:   Final[float] = 0.70
IMPUTATION_DROP_THRESHOLD:    Final[float] = 0.70

# ponytail: "floor_num_int" is a historical typo kept for parity with TRD
# examples; CANONICAL_COLUMNS only ships floor_num (string-shaped from the
# Step 04 decoder) — the missing name is skipped at runtime.
IMPUTATION_NUMERIC_LOW: Final[tuple[str, ...]] = (
    "balconies", "floor_num_int", "total_floor", "area_sqft",
)
IMPUTATION_CATEGORICAL_LOW: Final[tuple[str, ...]] = (
    "furnish", "facing", "ownership_type", "age_bucket", "property_type",
)
IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS: Final[tuple[str, ...]] = (
    "price_inr", "price_per_sqft", "bedrooms", "bathrooms",
)
# High tier is dynamic (whatever reclassify returns); pinned empty.
IMPUTATION_HIGH_TIER_COLUMNS: Final[tuple[str, ...]] = ()
```

Signatures:

- `classify_missingness_tiers(df) -> dict[str, list[str]]` — Iterates `df.columns`, computes `df[col].isna().mean()`, buckets into `low` (<0.05), `medium` (0.05–0.40), `high` (0.40–0.70), `drop` (>0.70). Always returns 4 keys, all lists. Pure.
- `add_was_missing_flags(df, columns) -> pd.DataFrame` — For each col in `columns` ∩ `df.columns` AND having at least one NaN, adds `was_missing_<col>` bool. Columns without NaNs get no flag. Returns a copy.
- `impute_low_tier(df) -> pd.DataFrame` — Re-runs `classify_missingness_tiers` internally; filters `IMPUTATION_NUMERIC_LOW` ∩ `df.columns` ∩ `low`, fills NaN with column median (non-null). Categorical: most-common value. One INFO line.
- `impute_medium_tier(df) -> pd.DataFrame` — Restricts to `IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS` ∩ `df.columns` ∩ `medium`. Group-wise median by `(city, locality, property_type)` via `df.groupby([...])[col].transform(lambda s: s.fillna(s.median()))`. Fallback: where transform still NaN (group has zero non-nulls), replace with global median. One INFO line; fallback count logged once.
- `impute_high_tier(df) -> pd.DataFrame` — For each `high`-tier column: object/string → fill with literal `"Unknown"`; numeric → leave NaN (flag carries signal). No per-row logs.
- `drop_high_missing_columns(df) -> tuple[pd.DataFrame, list[str]]` — Returns `(df_after, dropped_names)`. `_log_imputation_summary` aggregates drops in one INFO line with name + pct.
- `impute_missing_values(df) -> pd.DataFrame` — Pure orchestrator: classify → drop >70% → reclassify on slim → `add_was_missing_flags` on `(medium+high)` → low → medium → high → return. One INFO `impute.summary`.
- `_log_imputation_summary(df_before, df_after, dropped, tiers) -> None` — Aggregated INFO line: tier counts, dropped names with pct, `was_missing_*` count, NaN counts before/after.

### 2.2 `ml/cleaning/writers.py` — new

```python
CLEAN_LISTINGS_PARQUET_PATH:     Final[Path]  = Path("data/processed/clean_listings.parquet")
CLEAN_LISTINGS_DATASET_VERSION:  Final[str]  = "v1"
# Import-time constant covers the columns discoverable without the frame.
# was_missing_* columns are dynamic — the writer rebuilds the full tuple via
# build_clean_listings_columns_order(df).
CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER: Final[tuple[str, ...]] = (
    *CANONICAL_COLUMNS,         # already includes "is_outlier" as the last entry
    "outlier_reasons",
)
```

Functions:

- `build_clean_listings_columns_order(df) -> tuple[str, ...]` — `dict.fromkeys((*CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER, *sorted(df.filter(like="was_missing_").columns)))` → tuple. Used by writer + tests.
- `write_clean_listings_parquet(df, output_path=None) -> Path` — Resolve `output_path` (default `CLEAN_LISTINGS_PARQUET_PATH`), `mkdir(parents=True, exist_ok=True)`, reorder via `build_clean_listings_columns_order(df)`, `warnings.warn` + drop any column not in that order (no-op in practice), `df.to_parquet(path, index=False, engine="pyarrow")`. Sidecar `path.with_suffix(path.suffix + ".meta.json")` — resolves to `<file>.parquet.meta.json` (intentional, see Spec delta §4). Returns the path written.
- `read_clean_listings_parquet(path=None) -> pd.DataFrame` — Defaults to `CLEAN_LISTINGS_PARQUET_PATH`. `pd.read_parquet(path, engine="pyarrow")`.
- `verify_clean_listings_parquet(path=None) -> dict` — Returns the 7-key dict per spec. `exists=False` short-circuits. Otherwise attempt `read_clean_listings_parquet` inside try/except; populate counts + booleans on success; on failure, return `exists=False, error=str(e)`.

### 2.3 `ml/cleaning/pipeline.py` — new

```python
PIPELINE_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "rows_in", "rows_dropped_dedup", "rows_dropped_outlier_flag",
    "rows_dropped_high_missing_columns", "rows_in_after_imputation",
    "parquet_path", "dataset_version", "computed_at_utc",
)
```

- `run_clean_listings_pipeline(raw_dir, facet_dir, output_path=None, persist=True) -> pd.DataFrame` — Step list:
  1. `assert_raw_readonly(raw_dir.parent)` — symmetry gate.
  2. `df = assemble_cleaned_frame(raw_dir, facet_dir)` — captures `rows_in / rows_dropped_*_dedup / rows_dropped_outlier_flag` from frame-size diffs (no need to re-derive from log lines).
  3. `df = impute_missing_values(df)` — captures `rows_dropped_high_missing_columns` from a second size diff.
  4. If `persist`: `path = write_clean_listings_parquet(df, output_path)`; capture `parquet_path`, `dataset_version`, `computed_at_utc = datetime.utcnow().isoformat()`.
  5. One INFO `pipeline.summary` line containing every key in `PIPELINE_REPORT_FIELDS`.
  6. Returns `df` regardless of `persist`.

### 2.4 `ml/cleaning/__init__.py` — modify

Add a `# Step 07` block (alphabetical within the block, mirroring Step 06):

```python
from ml.cleaning.imputation import (
    IMPUTATION_CATEGORICAL_LOW,
    IMPUTATION_DROP_THRESHOLD,
    IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS,
    IMPUTATION_NUMERIC_LOW,
    IMPUTATION_HIGH_TIER_COLUMNS,
    MISSINGNESS_HIGH_THRESHOLD,
    MISSINGNESS_LOW_THRESHOLD,
    MISSINGNESS_MEDIUM_THRESHOLD,
    add_was_missing_flags,
    classify_missingness_tiers,
    drop_high_missing_columns,
    impute_missing_values,
)
from ml.cleaning.pipeline import (
    PIPELINE_REPORT_FIELDS,
    run_clean_listings_pipeline,
)
from ml.cleaning.writers import (
    CLEAN_LISTINGS_DATASET_VERSION,
    CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER,
    CLEAN_LISTINGS_PARQUET_PATH,
    build_clean_listings_columns_order,
    read_clean_listings_parquet,
    verify_clean_listings_parquet,
    write_clean_listings_parquet,
)
```

Insert matching `__all__` entries immediately after Step 06's block (alphabetical).

### 2.5 `scripts/run_pipeline.py` — modify

Add one line to the import block:

```python
from ml.cleaning import (  # noqa: E402
    assemble,  # noqa: F401  (Step 06 dedup + outlier flagging orchestrator)
    canonical_mapping,  # noqa: F401  (Step 05 per-city canonical schema mapping)
    facet_decoders,  # noqa: F401  (Step 04 decoders; Step 05 wires per-row)
    pipeline as _clean_pipeline,  # noqa: F401  (Step 07: impute + Parquet writer orchestrator)
)
```

Don't touch `main()` body — minimum diff is the alias line, mirroring the prior pattern.

---

## 3. Test plan (42 tests)

### 3.1 `tests/test_imputation.py` — 21 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_missingness_threshold_constants_match_trd` | All 4 constants equal 0.05/0.40/0.70/0.70. |
| 2 | `test_classify_missingness_tiers_returns_four_keys` | Dict has exactly `{low, medium, high, drop}`. |
| 3 | `test_classify_missingness_tiers_low_under_5pct` | 4% NaN col → `low`. |
| 4 | `test_classify_missingness_tiers_medium_between_5_and_40pct` | 20% → `medium`. |
| 5 | `test_classify_missingness_tiers_high_between_40_and_70pct` | 50% → `high`. |
| 6 | `test_classify_missingness_tiers_drop_above_70pct` | 80% → `drop`. |
| 7 | `test_classify_missingness_tiers_uses_input_frame_not_imputed` | 60% input → `high` (not zeroed by a hypothetical fillna). |
| 8 | `test_impute_low_tier_numeric_uses_global_median` | `balconies` 4% NaN → filled with median. |
| 9 | `test_impute_low_tier_categorical_uses_global_mode` | `furnish` 4% NaN → filled with mode. |
| 10 | `test_impute_low_tier_no_op_for_columns_not_in_tier` | `bedrooms` 50% NaN → untouched. |
| 11 | `test_impute_medium_tier_uses_groupwise_median` | Group median wins over global when constructed to disagree. |
| 12 | `test_impute_medium_tier_falls_back_to_global_when_group_empty` | All-null group → global median; fallback logged. |
| 13 | `test_impute_high_tier_categorical_filled_with_unknown` | 50% missing string col → `"Unknown"`. |
| 14 | `test_impute_high_tier_numeric_left_nan_with_flag` | 50% missing numeric col stays NaN; `was_missing_*` carries signal. |
| 15 | `test_drop_high_missing_columns_drops_above_70pct` | 80% col dropped; `dropped` list contains it. |
| 16 | `test_drop_high_missing_columns_logs_dropped` | `caplog` captures name + pct. |
| 17 | `test_add_was_missing_flags_creates_one_flag_per_imputed_column` | 3 NaN cols → 3 flags. |
| 18 | `test_add_was_missing_flags_are_set_before_imputation` | Flag→fill sequence: flag stays `True` post-fill. |
| 19 | `test_add_was_missing_flags_does_not_create_flag_for_column_without_nans` | Zero-NaN col → no flag sibling. |
| 20 | `test_impute_missing_values_is_idempotent` | `impute(impute(df)) == impute(df)` via `pd.testing.assert_frame_equal`. |
| 21 | `test_impute_missing_values_logs_summary` | `caplog` INFO captures tier counts + dropped names + NaNs before/after. |
| + | `test_impute_does_not_write_to_disk` (AST scan) | No `to_parquet`/`to_csv`/`to_json`/`open(` in source. |

### 3.2 `tests/test_writers.py` — 12 tests, `tmp_path`-based

| # | Name | Anchors |
|---|---|---|
| 1 | `test_clean_listings_parquet_path_constant_default` | `Path("data/processed/clean_listings.parquet")` exact. |
| 2 | `test_clean_listings_parquet_columns_order_is_deterministic` | `build_clean_listings_columns_order(df)` → canonical → `is_outlier` (already in canonical) → sorted `was_missing_*` → `outlier_reasons`. |
| 3 | `test_write_clean_listings_parquet_creates_file` | After write, `tmp_path/x.parquet` exists. |
| 4 | `test_write_clean_listings_parquet_creates_sidecar_meta_json` | Sidecar `<x>.parquet.meta.json` exists with required keys. |
| 5 | `test_write_clean_listings_parquet_writes_in_canonical_column_order` | Read-back columns equal the built order. |
| 6 | `test_write_clean_listings_parquet_round_trip_outlier_reasons_list` | `outlier_reasons = ["percentile_price_inr", "iqr_area_sqft"]` round-trips intact. |
| 7 | `test_write_clean_listings_parquet_round_trip_preserves_was_missing_flags` | `was_missing_*` bools survive with True/False distribution. |
| 8 | `test_write_clean_listings_parquet_returns_path` | Return value equals the path. |
| 9 | `test_read_clean_listings_parquet_round_trip` | Write + read → equivalent df (`check_dtype=False`). |
| 10 | `test_verify_clean_listings_parquet_passes_after_write` | exists, listing_id_unique, has_is_outlier, has_was_missing_columns, columns_match_canonical_order all True. |
| 11 | `test_verify_clean_listings_parquet_fails_for_missing_file` | Nonexistent → `exists=False`. |
| 12 | `test_writers_module_does_not_touch_data_raw` | AST scan: no `Path("data/raw"` literal; no `ml.cleaning.ingest` writer import. |

`pytest.importorskip("pyarrow")` at file top so a missing pyarrow becomes a clean skip, not an import error.

### 3.3 `tests/test_pipeline.py` — 9 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_pipeline_report_fields_constant_has_expected_keys` | `PIPELINE_REPORT_FIELDS` ⊇ `{rows_in, rows_in_after_imputation, parquet_path, dataset_version, computed_at_utc}`. |
| 2 | `test_run_clean_listings_pipeline_returns_dataframe` | Returns `pd.DataFrame`. Monkeypatch `ml.cleaning.pipeline.assemble_cleaned_frame` to return a literal canonical frame. |
| 3 | `test_run_clean_listings_pipeline_persist_false_does_not_write` | With `persist=False`, `CLEAN_LISTINGS_PARQUET_PATH` mtime is unchanged. |
| 4 | `test_run_clean_listings_pipeline_persist_true_writes_parquet` | With `persist=True` + `tmp_path` output, Parquet exists at that path with expected row count. |
| 5 | `test_run_clean_listings_pipeline_logs_report_fields` | `caplog` captures one summary line containing every `PIPELINE_REPORT_FIELDS` key. |
| 6 | `test_run_clean_listings_pipeline_asserts_raw_readonly` | Tamper raw_dir; gate raises; pipeline raises. |
| 7 | `test_run_clean_listings_pipeline_is_pure_no_io_when_persist_false` | Same args twice → equal df. |
| 8 | `test_run_clean_listings_pipeline_does_not_import_app_or_api` | AST scan: no `app.*` / `api.*` imports. |
| 9 | `test_run_clean_listings_pipeline_handles_already_imputed_input` | Run twice in sequence, both `persist=True` to `tmp_path1` / `tmp_path2` → identical row count + column set. |

Tests 2, 4, 5, 7, 9 monkeypatch `assemble_cleaned_frame` to a small literal frame so they run in <1 s without touching real CSVs.

---

## 4. Implementation order (12 steps)

1. Read `tests/fixtures/canonical_mapping_fixtures.py` for shared helpers (in case there's a fixture builder beyond `make_frame`). Reuse `make_frame` from `dedup_outlier_fixtures.py` for everything else.
2. Write `ml/cleaning/imputation.py` per §2.1. Constants → pure helpers → orchestrator → summary logger. Manually verify idempotency once before moving on.
3. Write `ml/cleaning/writers.py` per §2.2. Constants + `build_clean_listings_columns_order` first, then `write` / `read` / `verify`.
4. Write `ml/cleaning/pipeline.py` per §2.3. Import `assemble_cleaned_frame` directly (not via package re-exports) for explicit wiring.
5. Modify `ml/cleaning/__init__.py` — add the 17 re-exports + matching `__all__` entries, alphabetical within the Step 07 block.
6. Modify `scripts/run_pipeline.py` — add the `pipeline as _clean_pipeline  # noqa: F401` alias line per §2.5.
7. Write `tests/test_imputation.py` (21 tests).
8. Write `tests/test_writers.py` (12 tests, `tmp_path`).
9. Write `tests/test_pipeline.py` (9 tests, mostly `monkeypatch`ed).
10. `python -m pytest tests/test_imputation.py tests/test_writers.py tests/test_pipeline.py -v` — iterate until green.
11. `python -m pytest -m "not realdata"` — confirm no real-data dependency snuck in.
12. `ruff check …` on the 6 affected files; then the spec §5 smoke checks (constants print; real-data `persist=False` smoke); then `git status` confirms only the expected files changed.

---

## 5. Critical files

**Create (3):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\cleaning\imputation.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\cleaning\writers.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\cleaning\pipeline.py`

**Modify (2):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\cleaning\__init__.py` — 17 new re-exports + `__all__` entries.
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\scripts\run_pipeline.py` — one-line alias import.

**Tests (3):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_imputation.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_writers.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_pipeline.py`

**Reused (no modifications):**
- `ml/cleaning/assemble.py` — `assemble_cleaned_frame`, `ASSEMBLE_CITY_FILES`
- `ml/cleaning/canonical_mapping.py` — `CANONICAL_COLUMNS` (~37 names, ends with `is_outlier`)
- `ml/cleaning/ingest.py` — `assert_raw_readonly`
- `tests/fixtures/dedup_outlier_fixtures.py` — `make_frame(rows, columns)`

---

## 6. Verification

```bash
# Step 1 — unit + integration tests, per spec DoD §1
python -m pytest tests/test_imputation.py tests/test_writers.py tests/test_pipeline.py -v

# Step 2 — confirm no real-data dependency
python -m pytest -m "not realdata"

# Step 3 — ruff clean per spec DoD §3
ruff check ml/cleaning/imputation.py ml/cleaning/writers.py ml/cleaning/pipeline.py tests/test_imputation.py tests/test_writers.py tests/test_pipeline.py

# Step 4 — import constants match TRD §5 (spec DoD §4)
python -c "from ml.cleaning.imputation import MISSINGNESS_LOW_THRESHOLD, MISSINGNESS_MEDIUM_THRESHOLD, MISSINGNESS_HIGH_THRESHOLD, IMPUTATION_DROP_THRESHOLD, impute_missing_values; from ml.cleaning.writers import CLEAN_LISTINGS_PARQUET_PATH, CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER, write_clean_listings_parquet; from ml.cleaning.pipeline import run_clean_listings_pipeline; print(MISSINGNESS_LOW_THRESHOLD, MISSINGNESS_MEDIUM_THRESHOLD, MISSINGNESS_HIGH_THRESHOLD, IMPUTATION_DROP_THRESHOLD, CLEAN_LISTINGS_PARQUET_PATH, len(CLEAN_LISTINGS_PARQUET_COLUMNS_ORDER))"
# Expected: 0.05 0.4 0.7 0.7 data/processed/clean_listings.parquet 38

# Step 5 — real-data smoke (spec DoD §5, gated on the above passing)
python -c "from pathlib import Path; from ml.cleaning.pipeline import run_clean_listings_pipeline; df = run_clean_listings_pipeline(Path('data/raw'), Path('data/raw/facets'), persist=False); print(df.shape, df['is_outlier'].sum(), df['listing_id'].is_unique, df.filter(like='was_missing_').shape[1])"
# Expected: (~182000, ~40) <is_outlier_count> True <n_was_missing_flags>

# Step 6 — git hygiene
git status   # only the 5 files + spec; Parquet untracked.
```

Expected smoke output:
- Constants: `0.05 0.4 0.7 0.7 data/processed/clean_listings.parquet 38` (37 canonical + `outlier_reasons` = 38; `was_missing_*` columns are frame-derived extensions computed at write time, not part of the import-time constant).
- Real-data: a shape near `(~182000, ~40)` where the column count is `38 + N_was_missing`. Non-zero `is_outlier` sum. `True` for `listing_id.is_unique`. `N_was_missing` is the count of columns whose 40–70% NaN landed them in high tier (or 5–40% in medium tier with `was_missing_*` true).
- `git status`: spec file + 2 modified source files + 3 new test files + 3 new source files = 8 working-tree entries, Parquet untracked.
