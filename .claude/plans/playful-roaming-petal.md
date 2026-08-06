# Spec 06 — Data Deduplication & Outlier Flagging: Implementation Plan

## Context

Spec 06 (`.claude/specs/06-data-deduplication-and-outlier-flagging.md`) sits between Step 05 (`canonical_mapping.py`) and Step 07 (imputation + `clean_listings.parquet` writer). It does one job: take the four per-city canonical frames Step 05 emits, concatenate them, drop duplicate listings, and flag (not delete) outlier rows — producing a single consolidated DataFrame that Step 07 will hand to the Parquet writer.

The spec was authored before I read Step 05's actual public surface. Exploration found **5 conflicts** between the spec text and the codebase:

1. **Step 02 doesn't expose `assert_raw_readonly` or `load_raw_city_frames`** — Step 02's public API is `load_raw_listings(data_dir) -> dict[str, pd.DataFrame]` and the immutability check is inline in that function. (Refactor: ship a small Step 02 patch to expose both names.)
2. **`property_type_label` doesn't exist** in `CANONICAL_COLUMNS` — the canonical column is `property_type`. The spec prose (line 54) and DoD tests (lines 142–145) must read `property_type`.
3. **`price_per_sqft` is `pd.NA` in the canonical frame** — Step 05 sets it to `pd.NA` at line 554 of `canonical_mapping.py`. Outlier flagging on NaN flags nothing. (Fix: compute `price_per_sqft = price_inr / area_sqft` inline in `assemble.py` right before outlier flagging runs.)
4. **`register_date` is a free-text string** in the canonical frame (e.g. `"29th Sep, 2023"`), not a parsed date. Tiebreaker sorts lexicographically — acceptable, flag in docstring.
5. **`is_outlier` already exists** as `bool` with value `False` for every row from Step 05 — `flag_all_outliers` must overwrite the column in place, not append.

User already confirmed the recommended fixes for #1, #2, and #3. Items #4 and #5 are documentation/discipline points the implementation must honor.

---

## Critical files to create

- `ml/cleaning/dedup.py` (new — ~120 lines)
- `ml/cleaning/outliers.py` (new — ~180 lines)
- `ml/cleaning/assemble.py` (new — ~90 lines)
- `tests/test_dedup.py` (new, 14 tests per spec)
- `tests/test_outliers.py` (new, 16 tests per spec)
- `tests/test_assemble.py` (new, 11 tests per spec)
- `tests/fixtures/dedup_outlier_fixtures.py` (new — literal DataFrame constants + multi-city builders, mirroring `tests/fixtures/canonical_mapping_fixtures.py`)

## Critical files to modify

- `ml/cleaning/ingest.py` — extract `assert_raw_readonly()` and `load_raw_city_frames()` as public functions (small Step 02 patch).
- `ml/cleaning/__init__.py` — add the 10 spec-required re-exports (see "Step 6: Re-export" below).
- `scripts/run_pipeline.py` — one-line wiring placeholder.
- `.claude/specs/06-data-deduplication-and-outlier-flagging.md` — fix the **three prose mismatches** identified above (no DoD test changes; just the spec wording so future readers don't get confused):
  - line 8 and line 64 (Depends on / Files): `load_raw_city_frames` → confirmed in Step 02 patch below.
  - line 54 (Files to create / outliers.py): `property_type_label` → `property_type`.
  - DoD test names (lines 142–145): update four test names' wording to use `property_type` in prose form, not the column name itself (the test body uses `property_type_label` strings — switch those to `property_type`).
  - line 174 (DoD #4): update the one-liner `'listing_id 3 outlier_reasons 4'` check from the **previous spec text** — keep this DoD item, but make sure the line still prints correctly with the renamed symbol. (The rename is internal: the constant is still called `OUTLIER_REASON_COLUMN` and equals `"outlier_reasons"`; only the test bodies change. This item still prints `outlier_reasons` correctly.)

## Critical files NOT to touch

- `ml/cleaning/canonical_mapping.py` — Step 05 leaves `price_per_sqft` as NaN on purpose (per Step 05 DoD rule "feature engineering is OUT of scope"). We compute it in `assemble.py`, not by editing Step 05.
- `requirements.txt`, `.gitignore`, `pytest.ini`, `pyproject.toml` — no changes.

---

## Step 1 — Step 02 patch: expose the two missing helpers

**File:** `ml/cleaning/ingest.py`

The user's choice was to expose them so spec 06 can use them by name. Two small additions:

1. Extract a public `assert_raw_readonly(data_dir: Path) -> None` from the inline block at the end of `load_raw_listings` (current lines 343–347). It should:
   - Call `_snapshot_raw_files(data_dir)` once.
   - Compare current snapshot to a freshly taken one.
   - Raise the existing `RuntimeError("data/raw/ was modified during ingestion — raw immutability violated (Rules §1.1)")` if they differ.
   - Be safe to call repeatedly (idempotent); in practice Step 06 will call it before reads but Step 02's full ingest still does its own before/after.

2. Add a public `load_raw_city_frames(data_dir: Path) -> dict[str, pd.DataFrame]` as a thin alias of `load_raw_listings(data_dir)` so the spec's name works. Implementation = `return load_raw_listings(data_dir)`. Keeps Step 05 callers working unchanged.

Then update `ml/cleaning/__init__.py`:
- Import both new names from `ml.cleaning.ingest`.
- Add `"assert_raw_readonly"`, `"load_raw_city_frames"` to `__all__`.

Add 2–3 tests to a new `tests/test_ingest_extras.py` (or extend `tests/test_ingest.py` if it exists) that:
- `test_assert_raw_readonly_passes_on_clean_raw_dir` — synthetic `tmp_path/raw/` with one file, snapshot → assert → no raise.
- `test_assert_raw_readonly_raises_when_file_modified` — write a file, snapshot, modify its mtime via `os.utime`, assert → `RuntimeError` with the "Rules §1.1" message.
- `test_load_raw_city_frames_returns_four_cities` — against real `data/raw/`, returns `{"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}`.

Run `pytest tests/ -m "not realdata"` — must stay green.

---

## Step 2 — Fix the three spec-vs-codebase conflicts in the spec file

**File:** `.claude/specs/06-data-deduplication-and-outlier-flagging.md`

These are prose/test-name wording edits, not behavior changes — get them in **before** the implementation PR so the spec describes the code that ships:

1. **`property_type_label` → `property_type`** in three places:
   - Line 54 (Files to create / outliers.py): "`property_type_label`" → "`property_type`".
   - Lines 142–145 (DoD tests): the four `test_flag_domain_rule_outliers_*` test bodies use `property_type_label="flat"` / `"villa"` / `"farmhouse"`. Rename the column in test bodies to `property_type`. Test names themselves stay the same.
2. **Add a note** to line 60 (already-cited ref to "Step 05's contract"): explicitly call out that `price_per_sqft` arrives as NaN from Step 05 and `assemble.py` derives it before outlier flagging. One sentence in the module docstring of `outliers.py` and a one-line clarification in `assemble.py`'s docstring.
3. **Note (no edit needed):** `register_date` lex sort is acceptable for the tiebreaker. Mention in `dedup.py` module docstring.

---

## Step 3 — Build `ml/cleaning/dedup.py`

**File:** `ml/cleaning/dedup.py` (new, ~120 lines)

Public API per spec lines 32–45 + the 14 DoD tests at lines 117–133:

```python
"""Deduplication layer — TRD §4.9, 02-TRD.

Drops duplicate PROP_ID/listing_id rows with a documented tiebreaker policy:
  1. Most non-null canonical fields wins.
  2. Most-recent register_date wins (sorted as string — Step 05 leaves it
     as raw text, lex sort is deterministic and good enough for tiebreaking).
  3. First-seen input row wins (stable row_order tiebreaker).

Outliers are flagged, NOT deleted — see ml.cleaning.outliers.
"""
from __future__ import annotations
import logging
import pandas as pd

from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS

_LOG = logging.getLogger("ml.cleaning.dedup")

DEDUP_KEY_COLUMN: str = "listing_id"
CONFLICT_TIEBREAKER_ORDER: tuple[str, ...] = (
    "nonnull_fields_count",  # computed column
    "register_date",          # raw text, lex sort is fine
    "row_order",              # input row position
)

def compute_nonnull_field_count(df: pd.DataFrame) -> pd.Series:
    """Per-row count of non-null values across CANONICAL_COLUMNS (int Series)."""

def _log_dedup_drop(reason: str, count: int, total: int) -> None:
    """Log structured INFO line for a dedup drop bucket.
    Mirrors ml.cleaning.parsing._log_unparseable style.
    """

def deduplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null/empty listing_id, then groupby(listing_id).first()
    using CONFLICT_TIEBREAKER_ORDER. Returns deduped DataFrame with index reset.
    """
```

Implementation notes (ponytail hints):
- `compute_nonnull_field_count` = `df[list(CANONICAL_COLUMNS)].notna().sum(axis=1)` (column-list form, vectorized, no Python-level loop).
- Inside `deduplicate_listings`: add `_nonnull = compute_nonnull_field_count(df)`, add `_row_order = np.arange(len(df))`, build a working frame, sort by `(nonnull DESC, register_date DESC, row_order ASC)`, then `result = working.groupby(DEDUP_KEY_COLUMN, as_index=False).first()`. Reset index at the end.
- Null/empty `listing_id` = `df[DEDUP_KEY_COLUMN].isna() | (df[DEDUP_KEY_COLUMN].astype(str).str.strip() == "")`.
- The working frame contains only the three tiebreaker columns + `listing_id`; `groupby.first()` on the canonical frame is what keeps the rest. Cleaner than carrying extra columns through the sort.
- The module-level `_LOG` is private, consistent with `ml.cleaning.parsing._log`.
- Do **not** `import` anything from `app.*` or `api.*` (enforced by spec DoD test).

## Step 4 — Build `ml/cleaning/outliers.py`

**File:** `ml/cleaning/outliers.py` (new, ~180 lines)

Public API per spec lines 46–60 + the 16 DoD tests at lines 135–153:

```python
"""Outlier flagging layer — TRD §6, 02-TRD.

Flags rows whose price/area is extreme by:
  1. per-city 1st/99th percentile bounds (TRD §6.1)
  2. per-city 1.5 × IQR (Tukey fence) (TRD §6.2)
  3. domain rules (bedRoom/bathroom > 15, unless villa/farmhouse) (TRD §6.3)

Adds two columns to the canonical frame, both via assignment (not append):
  - ``is_outlier`` (bool) — or-combined flag.
  - ``outlier_reasons`` (object, list[str]) — fixed-set reason codes from
    ``{"percentile_<col>", "iqr_<col>", "domain_<col>"}``.

Flagged rows are RETAINED — Rules §1.4. Training-time exclusion is the
modeling step's concern, not this one. log1p target transform (TRD §6.4)
is also out of scope here — Day 13 of the implementation plan.
"""
from __future__ import annotations
import json
import logging
import pandas as pd

_LOG = logging.getLogger("ml.cleaning.outliers")

# Fixed constants — single source of truth.
OUTLIER_NUMERIC_COLUMNS: tuple[str, ...] = ("price_inr", "area_sqft", "price_per_sqft")
PERCENTILE_LOWER: float = 0.01
PERCENTILE_UPPER: float = 0.99
IQR_MULTIPLIER: float = 1.5
OUTLIER_DOMAIN_RULES: dict[str, dict[str, int | str]] = {
    "bedRoom":   {"max": 15, "note": "unless property_type is villa/farmhouse/independent house"},
    "bathroom":  {"max": 15, "note": "unless property_type is villa/farmhouse/independent house"},
}
OUTLIER_PROPERTY_TYPE_EXEMPTIONS: frozenset[str] = frozenset(
    {"villa", "farmhouse", "independent house"}
)
OUTLIER_REASON_COLUMN: str = "outlier_reasons"

def flag_percentile_outliers(df: pd.DataFrame, column: str) -> pd.Series: ...
def flag_iqr_outliers(df: pd.DataFrame, column: str) -> pd.Series: ...
def flag_domain_rule_outliers(df: pd.DataFrame) -> pd.Series: ...
def _log_outlier_summary(df: pd.DataFrame) -> None: ...
def flag_all_outliers(df: pd.DataFrame) -> pd.DataFrame: ...
```

Implementation details:
- `flag_percentile_outliers`: `df.groupby("city")[column].transform(lambda s: (s < s.quantile(PERCENTILE_LOWER)) | (s > s.quantile(PERCENTILE_UPPER)))`. Returns bool Series aligned to `df.index`.
- `flag_iqr_outliers`: per-city `Q1 = group.quantile(0.25)`, `Q3 = group.quantile(0.75)`, `IQR = Q3 - Q1`. Boolean = `(s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)`. Same `transform` shape.
- `flag_domain_rule_outliers`: iterate `OUTLIER_DOMAIN_RULES.items()`, mask = `df[column] > rule["max"]`. For `bedRoom`/`bathroom`, override the mask where `df["property_type"].isin(OUTLIER_PROPERTY_TYPE_EXEMPTIONS)`. OR-merge all columns.
- `flag_all_outliers`:
  - For each `column in OUTLIER_NUMERIC_COLUMNS`: collect two reason strings (`f"percentile_{column}"`, `f"iqr_{column}"`).
  - Build a `pd.Series[dtype=object]` of empty lists (one per row), name it after `OUTLIER_REASON_COLUMN`, length `len(df)`. Append reason strings when the corresponding flag is True.
  - Domain-rule reasons (`f"domain_{col}"`) collected separately.
  - `is_outlier` = any flag True OR `outlier_reasons` non-empty.
  - **Assignment, not append**: `df = df.copy(); df["is_outlier"] = ...; df[OUTLIER_REASON_COLUMN] = ...` (Step 05 already added the column; we replace, not add). If the existing `is_outlier` column exists from Step 05, replace it; if not, add it. Use `df.assign(**{...})` or direct `df[col] = ...`.
  - Call `_log_outlier_summary(df)`.
  - Return the frame.

## Step 5 — Build `ml/cleaning/assemble.py`

**File:** `ml/cleaning/assemble.py` (new, ~90 lines)

Public API per spec lines 62–72 + the 11 DoD tests at lines 155–167:

```python
"""Orchestrator — Step 06 entrypoint.

Single public function ``assemble_cleaned_frame`` ties Step 02 (raw read)
+ Step 04 (facet decode) + Step 05 (canonical mapping) + dedup + outlier
flagging into one callable. Step 07 (missing-value imputation) will
import this to get its training-ready DataFrame.

No ``to_parquet`` / ``to_csv`` / ``to_json`` here — Step 07 owns all writes.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

from ml.cleaning.dedup import deduplicate_listings
from ml.cleaning.outliers import flag_all_outliers
from ml.cleaning.canonical_mapping import (
    CANONICAL_COLUMNS,
    CITY_FRAME_LOADERS,
    map_city,
)
from ml.cleaning.facet_decoders import load_facet_frames
from ml.cleaning.ingest import assert_raw_readonly

_LOG = logging.getLogger("ml.cleaning.assemble")

ASSEMBLE_CITY_FILES: dict[str, str] = {
    "Gurgaon": "gurgaon_10k.csv",
    "Hyderabad": "hyderabad.csv",
    "Kolkata": "kolkata.csv",
    "Mumbai": "mumbai.csv",
}
ASSEMBLE_REPORT_FIELDS: tuple[str, ...] = (
    "rows_in",
    "rows_dropped_no_listing_id",
    "rows_dropped_duplicate",
    "rows_in_after_dedup",
    "rows_flagged_outlier",
    "rows_in_after_outlier_flag",
    "per_city_breakdown",
)

def _derive_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_per_sqft = price_inr / area_sqft where both are non-null.

    Step 05 leaves ``price_per_sqft`` as NaN by design (price_per_sqft is a
    feature-engineering concern per Step 05's docstring). TRD §6 requires
    outlier flagging on this column, so the assembler derives it here and
    passes the result to flag_all_outliers. Overwrites the column in place.
    """
    df = df.copy()
    mask = df["price_inr"].notna() & df["area_sqft"].notna() & (df["area_sqft"] > 0)
    df.loc[mask, "price_per_sqft"] = df.loc[mask, "price_inr"] / df.loc[mask, "area_sqft"]
    return df

def assemble_cleaned_frame(raw_dir: Path, facet_dir: Path) -> pd.DataFrame:
    """Read raw + facets, run per-city mappers, concat, dedup, flag outliers.

    Returns a single DataFrame. Does NOT write to data/processed/.
    """
    assert_raw_readonly(raw_dir.parent)  # data/raw is at raw_dir.parent/raw
    facets = load_facet_frames(facet_dir)
    frames = [
        map_city(name, raw_dir / fname, facets)
        for name, fname in ASSEMBLE_CITY_FILES.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    rows_in = len(df)
    df = _derive_price_per_sqft(df)
    df = deduplicate_listings(df)
    # ... log fields, then flag:
    df = flag_all_outliers(df)
    _log_summary(df, rows_in, ...)
    return df
```

Implementation notes:
- `raw_dir` parameter is the **city CSV folder** (`data/raw`), per spec. `assert_raw_readonly` from Step 02 expects `data_dir` whose `/raw` is the raw folder — so call `assert_raw_readonly(raw_dir.parent)` won't work either. **Decision:** the simplest contract is that `raw_dir` argument equals `data/raw/` and `assert_raw_readonly(raw_dir)` after the Step 02 patch exposes a variant that takes the raw dir directly. Re-confirm at impl time; otherwise pass `raw_dir.parent`.
- Log structure: a single `INFO` line near the end with all of `ASSEMBLE_REPORT_FIELDS` plus a per-city breakdown dict.
- Order: load → concat → derive `price_per_sqft` → dedup (so dedup operates on the finalized numeric value) → flag outliers.
- No write side effects. DoD test 162 (`test_assemble_cleaned_frame_does_not_write_to_data_processed`) verifies via `os.listdir` snapshot.

## Step 6 — Re-export the 10 new symbols from `ml/cleaning/__init__.py`

**File:** `ml/cleaning/__init__.py` (modify)

Add imports + `__all__` entries for the spec-mandated 10 symbols:

```python
from ml.cleaning.dedup import (
    DEDUP_KEY_COLUMN,
    deduplicate_listings,
    CONFLICT_TIEBREAKER_ORDER,        # bonus — useful for tests/QA
    compute_nonnull_field_count,      # bonus
)
from ml.cleaning.outliers import (
    OUTLIER_NUMERIC_COLUMNS,
    OUTLIER_DOMAIN_RULES,
    OUTLIER_PROPERTY_TYPE_EXEMPTIONS, # bonus — used by tests
    PERCENTILE_LOWER,
    PERCENTILE_UPPER,
    IQR_MULTIPLIER,
    flag_all_outliers,
    OUTLIER_REASON_COLUMN,
)
from ml.cleaning.assemble import assemble_cleaned_frame, ASSEMBLE_CITY_FILES
```

10 mandatory names + a few helpful bonuses. Update `__all__` correspondingly with a comment `# Step 06`.

## Step 7 — Build the three test files

**Files:** `tests/test_dedup.py`, `tests/test_outliers.py`, `tests/test_assemble.py` (new)

**Pattern:** Copy the structure of `tests/test_canonical_mapping.py` (8 section headings A–H). All tests use literal DataFrames / `pd.DataFrame(...)` literals — no real-data dependency. The spec DoD provides exact test names; copy each name verbatim.

**Key fixture:** `tests/fixtures/dedup_outlier_fixtures.py` — literal DataFrames:
- `BUILDING_ROW = {"listing_id": "X", "city": "Gurgaon", "price_inr": 1.0e7, "area_sqft": 1500.0, "price_per_sqft": 6666.67, "bedRoom": 3, "bathroom": 3, "property_type": "flat", "register_date": "29th Sep, 2023"}`
- `make_frame(rows: list[dict]) -> pd.DataFrame` — builds a frame, fills missing canonical columns with NaN.
- `make_multi_city_frame() -> pd.DataFrame` — 4-city fixture with at least one outlier per type.

### Tests covering our 3 spec-fix points

These tests will catch any regression of the fix decisions:

- **Property_type_column name** (covers fix #2): `test_flag_domain_rule_outliers_does_not_flag_villa_with_high_bedroom` must construct a row with `property_type="villa"` (not `property_type_label`).

- **price_per_sqft derivation** (covers fix #3): `test_assemble_cleaned_frame_derives_price_per_sqft_before_outlier_flagging` (NEW) — assembler on a synthetic single-city input where `price_inr=100_000_000`, `area_sqft=100` → `price_per_sqft=1_000_000` (extreme), so it gets flagged by `flag_percentile_outliers` on `price_per_sqft`. Proves the derivation runs.

- **`is_outlier` overwrite** (covers point #5): `test_flag_all_outliers_overwrites_existing_is_outlier_column` (NEW) — start with a frame that already has `is_outlier=True` from Step 05; after `flag_all_outliers` the column reflects the new flags, not the old ones.

- **Step 02 patch**: tests in `tests/test_ingest.py` (or new `tests/test_ingest_extras.py`).

### Per-test file organization (mirroring `test_canonical_mapping.py`)
- Section A: constants + shape (3–5 tests)
- Section B: pure-function core (4–6 tests per file)
- Section C: logging via `caplog` (1–2 tests)
- Section D: idempotency + purity (1–2 tests)
- Section E: orchestration tests for `assemble.py` (10 tests)
- Section F: realdata marker (1 test, opt-in via `-m realdata`)

## Step 8 — Wire `scripts/run_pipeline.py`

**File:** `scripts/run_pipeline.py` (modify)

Add `assemble  # noqa: F401  (Step 06 dedup + outlier flagging orchestrator)` to the existing `ml.cleaning` import block at lines 24–31. Keep the alphabetical-ish ordering used by Step 04/05 (`canonical_mapping`, `assemble`, `facet_decoders` — assemble sits between, since it depends on both). Match the existing comment style.

No other change. `main()` still only calls `ingest_raw.main()` — Step 06 isn't a pipeline stage in this orchestrator yet (Step 07 will be the first stage to call `assemble_cleaned_frame`).

## Step 9 — Run gates (DoD order)

Run in this exact order; each must pass before moving to the next:

1. `python -m pytest tests/test_dedup.py tests/test_outliers.py tests/test_assemble.py -v` — all 41 tests (14 + 16 + 11) green.
2. `python -m pytest -m "not realdata"` — full suite still green (no accidental realdata dependency introduced).
3. `ruff check ml/cleaning/dedup.py ml/cleaning/outliers.py ml/cleaning/assemble.py tests/test_dedup.py tests/test_outliers.py tests/test_assemble.py` — zero issues.
4. DoD smoke test #4 (public-API import check, line 171 of spec) — must print `listing_id 3 outlier_reasons 4`.
5. DoD real-data smoke test #5 (line 172) — runs `assemble_cleaned_frame(Path('data/raw'), Path('data/raw/facets'))` against the ~182k-row real dataset, prints shape + `is_outlier.sum()` + uniqueness flag + reason set.
6. `git status` — only the expected files modified (no `data/processed/`, `data/raw/`, `models/`, `notebooks/`).
7. Tracker update — via `/update-tracker`, mark Day 12 row as Done with today's date + link to the spec. Outside this PR's commit, per spec DoD #8.

---

## Summary of decisions taken (recap)

- **Step 02 helpers** — exposed via a small Step 02 patch; spec wording stays accurate.
- **`property_type_label` → `property_type`** — spec text + 4 DoD test body refs updated; behavior unchanged.
- **`price_per_sqft` derivation** — assembled in `assemble.py` immediately before dedup/outlier flagging; spec note clarifies this is the contract.

This avoids diverging from Step 05's "feature engineering out of scope" rule and keeps Step 06 self-contained, with a single line of new code that does the math.

Total new files: 6 (`dedup.py`, `outliers.py`, `assemble.py`, 3 test files, 1 fixture file). Modified: 4 (`ingest.py`, `__init__.py`, `run_pipeline.py`, the spec prose for the wording fixes). Nothing else touches the repo.
