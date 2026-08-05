# Plan: Implement Spec 05 — Canonical Schema Mapping Per City

## Context

Step 05 is the central data-assembly step of the Week 1 cleaning chain. Steps 02/03/04 produced the building blocks (raw loader, `parse_price`/`parse_area`, 15 facet decoders); this step wires them into one canonical DataFrame per city, conforming to the 16-field input contract (`10-FINALIZED-INPUT-SCHEMA.md` §3) + the ~28-column extended schema (`05-BACKEND-SCHEMA.md` §2 + §U-SCHEMA-5). Step 06 will concatenate the four city frames, dedupe, and write `clean_listings.parquet`. This step writes no DataFrame to disk.

## Findings from codebase exploration (drive 3 spec deviations)

The spec assumed four helpers that don't exist in the actual code. Locked with the user before planning:

| Spec assumption | Actual code | Resolution |
| --- | --- | --- |
| `load_raw_city_frames()` returns dict keyed by city | `ml.cleaning.ingest.load_raw_listings(data_dir) -> dict[str, pd.DataFrame]` keyed by city | Use as-is. Pass `Path("data")` (parent of `raw/`). |
| `assert_raw_readonly()` standalone helper | `ml.cleaning.ingest._snapshot_raw_files(data_dir)` private (sha256+size+mtime per file) | Use the snapshot pattern from Step 04 tests; not `os.access`. |
| `parse_map_details(value)` from Step 03 | Doesn't exist (`parsing.py` ships `parse_price`, `parse_area` only) | Inline a `_parse_map_details(value) -> tuple[float \| None, float \| None]` helper in this module using `ast.literal_eval` (same shape as the other `parse_*` helpers — never raises on bad input). |
| `JOIN_KEY_FORMAT` constant | Not needed — `decode_*` decoders carry their own `_index` on `facet_df.attrs`. Helpers already present. | Vectorized `.map()` calling each decoder directly. |

Confirmed per-city column quirks (from actual raw-byte inspection):
- **Hyderabad & Mumbai**: `location` is a nested dict string; flat `LOCALITY` column absent. Extract via `ast.literal_eval(row["location"]).get("LOCALITY_NAME")`. Same for `BUILDING_NAME`/`BUILDING_ID` from `location`.
- **Hyderabad & Mumbai**: `VALUE_LABEL` is the human-readable ownership string; `OWNTYPE` is the code. Use `VALUE_LABEL` directly for `ownership_type`.
- **Kolkata**: no `BUILDING_ID`, no `REGISTER_DATE`, no `BATHROOM_NUM`; `BALCONY_NUM` mostly NaN; `FEATURES` is `"N"` or NaN; `AMENITIES` mostly NaN.
- **Gurgaon**: Bathrooms column present (`BATHROOM_NUM`), PII/url columns present (must drop), `LOCALITY` is a flat string.
- All four cities: `TRANSACT_TYPE` arrives as `1.0`/`2.0` (coded), not the literal `"Sale"`/`"Rent"` strings the schema names. Need a 1.0→"Sale"/2.0→"Rent" mapping (constant `_TRANSACT_TYPE_CODE_TO_LABEL = {1.0: "Sale", 2.0: "Rent"}`).

## Files to create

### `ml/cleaning/canonical_mapping.py`

**Module docstring** ~50 lines: documents the canonical ~28-column schema, references `10-FINALIZED-INPUT-SCHEMA.md` §3 + `05-BACKEND-SCHEMA.md` §2 + §U-SCHEMA-5 as the schema authority, lists the four city file paths, notes that this spec does NOT write parquet (Step 06's job), and explicitly calls out that `luxury_category` is left NaN per Rules §10.2.

**Constants:**
- `SPEC_VERSION: Final[str] = "05-canonical-schema-mapping-v1"`
- `UNSAFE_COLUMNS: Final[tuple[str, ...]]` — single grep-able tuple of PII/URL/scrape-id columns (the list from the spec §"Rules for implementation"). ~25-30 names including `PHOTO_URL`, `MEDIUM_PHOTO_URL`, `THUMBNAIL_PHOTO_URL`, `LARGE_PHOTO_URL`, `DEALER_PHOTO_URL`, `PROP_DETAILS_URL`, `PROP_URL`, `URL`, `PD_URL`, `PROPERTY_IMAGES`, `THUMBNAIL_IMAGES`, `FSL_Data`, `profile`, `xid`, `metadata`, `COMMON_FURNISHING_ATTRIBUTES`, `QUALITY_SCORE`, `FURNISHING_ATTRIBUTES`, `CONTACT_NAME`, `CONTACT_COMPANY_NAME`, `DEALER_NAME`, `DEALER_COMPANY`, `DEALER_PHONE`, `PHONE_NUMBER`, `CONTACT_PHONE`, `DEALER_EMAIL`, `CONTACT_EMAIL`, `SPID`.
- `_LOG: Final` — logger named `"ml.cleaning.canonical_mapping"`.
- `_TRANSACT_TYPE_CODE_TO_LABEL: Final[dict[float, str]] = {1.0: "Sale", 2.0: "Rent"}`.
- `_RE_HTML_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")`.
- `_RE_URL: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+")`.
- `_RE_EMAIL: Final[re.Pattern[str]] = re.compile(r"\S+@\S+")`.
- `_RE_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")`.

**`CANONICAL_COLUMNS: Final[tuple[str, ...]]`** — 28 names in the documented order. Combined from `05-BACKEND-SCHEMA.md` §2 plus the 16 input-contract fields with their exact canonical names:
```
listing_id, city, sector, locality, property_type, transact_type, ownership_type,
bedrooms, bathrooms, balconies, bedRoom, bathroom, servant_room, store_room,
furnish, furnishing_type, facing, age_bucket, agePossession, floor_num, total_floor,
floor_category, luxury_category, area_sqft, built_up_area, price_inr,
features_list, amenities_list, n_amenities, n_features, building_name, building_id,
latitude, longitude, description_clean, register_date, is_outlier
```
(notes: `bedrooms`/`bathrooms` come from extended schema, `bedRoom`/`bathroom` are the 16-input-contract camelCase alias. Both columns appear — they're separate fields per the docs. The 16-input-contract subset is preserved verbatim.)

**`CITY_COLUMN_ALIASES: Final[dict[str, dict[str, str]]]`** — per-city raw→canonical mapping using actual raw column names from the byte dump. ~25 entries per city covering all canonical columns the city actually has. City entries where the column is absent get `None`/omitted from the alias dict (the row gets NaN at the end).

**Public functions:**

1. `clean_description(series: pd.Series) -> pd.Series` (vectorized): per-row `html.unescape(s).lower()` → strip `<...>` → drop URLs → drop emails → collapse whitespace. NaN passes through unchanged.

2. `_parse_map_details(value: object) -> tuple[float | None, float | None]`: `ast.literal_eval(value)` if it's a string-shaped dict, extract `LATITUDE`/`LONGITUDE` cast to `float`. NaN/unparseable/None → `(None, None)` — never raises (matches `parse_price`/`parse_area` convention).

3. `_extract_locality_from_location(value: object) -> str | None` and `_extract_building_from_location(value: object) -> tuple[str | None, int | None]`: small `ast.literal_eval` helpers for the Hyderabad/Mumbai/Kolkata `location` nested-dict case. NaN → `(None, ...)`.

4. `normalize_columns(df: pd.DataFrame, city: str) -> pd.DataFrame`:
   - Look up `CITY_COLUMN_ALIASES[city]`
   - Drop columns in `UNSAFE_COLUMNS` that appear in `df.columns`
   - Rename raw→canonical via the alias dict (only columns present; pass others through unchanged — `test_normalize_columns_passes_through_unknown_columns_as_is`)
   - Return the frame (do not reindex to `CANONICAL_COLUMNS` — that's `map_<city>`'s job, last step)

5. `map_gurgaon(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame` / `map_hyderabad` / `map_kolkata` / `map_mumbai`:
   - `df = pd.read_csv(raw_path, ...)` (or take the already-loaded frame via `load_raw_listings` — see "Choice" below).
   - Apply `normalize_columns(df, city)`.
   - Vectorize the 13 single-value decoders: `df["FURNISH"].map(lambda v: decode_furnish(v, facet_frames["FURNISH"]))` → new column `furnish` (string labels); same for `FACING`→`facing`, `AGE`→`age_bucket`, `PROPERTY_TYPE`→`property_type`, `OWNTYPE`→`ownership_type` (only for Gurgaon/Kolkata — Hyderabad/Mumbai use `VALUE_LABEL`), `BEDROOM_NUM`→`bedrooms`, `BATHROOM_NUM`→`bathrooms`, `BALCONY_NUM`→`balconies`, `FLOOR_NUM`→`floor_num`, `TOTAL_FLOOR`→`total_floor`. Each guarded with `if "<raw_col>" in df.columns` else column = NaN.
   - `df["features_list"] = df["FEATURES"].map(lambda v: decode_features(v, facet_frames["FEATURES"]))`; same for `amenities_list`.
   - `df["price_inr"] = df["PRICE"].map(lambda v: parse_price(v))`.
   - `df["area_sqft"] = df["AREA"].map(lambda v: parse_area(v))`.
   - Per-city:
     - `lat, lon = _parse_map_details(v)` from `MAP_DETAILS`; assign `df["latitude"]`, `df["longitude"]`.
     - Hyderabad/Mumbai/Kolkata: pull `locality`, `building_name`, `building_id` from `location` nested dict via `_extract_locality_from_location`/`_extract_building_from_location`. Gurgaon uses flat `LOCALITY`.
     - Hyderabad/Mumbai: `df["ownership_type"] = df["VALUE_LABEL"]` (pass-through, no decode).
     - Kolkata: missing columns (`register_date`, `building_id`, `bathrooms`) default to NaN.
   - `df["listing_id"] = df["PROP_ID"].astype("string")` (Gurgaon/Hyderabad/Kolkata/Mumbai all have `PROP_ID` after Step 02 ingestion — Mumbai also has `SPID` but `PROP_ID` is the canonical key).
   - `df["description_clean"] = clean_description(df["DESCRIPTION"])`.
   - `df["transact_type"]` = map `df["TRANSACT_TYPE"]` via `_TRANSACT_TYPE_CODE_TO_LABEL` (raw is `1.0`/`2.0` not literal `"Sale"`/`"Rent"`).
   - `df["luxury_category"] = float("nan")` (explicit, per Rules §10.2 — server-side derivation, not self-report).
   - Engineered columns fill-NaN: `df["floor_ratio"] = float("nan")`, `df["price_per_sqft"] = float("nan")`, `df["n_amenities"] = pd.NA`, `df["n_features"] = pd.NA`, `df["sector"] = pd.NA`, `df["bedRoom"] = pd.NA` (test expects "bedRoom" exactly — but the mapper uses `bedrooms` from BEDROOM_NUM; **we'll alias both `bedrooms` and `bedRoom` to the same BEDROOM_NUM-derived column** to satisfy test `test_bedroom_canonical_name_uses_camelcase`), `df["bathroom"] = pd.NA`, `df["furnishing_type"] = pd.NA`, `df["floor_category"] = pd.NA`, `df["agePossession"] = pd.NA`, `df["built_up_area"] = pd.NA`, `df["servant_room"] = pd.NA`, `df["store_room"] = pd.NA`.
   - `df["is_outlier"] = False`.
   - Reindex: `df = df.reindex(columns=list(CANONICAL_COLUMNS), fill_value=pd.NA)` — locks the column order.
   - Return `df`. Idempotent per spec (`pd.testing.assert_frame_equal` works).
   - **Choice**: `map_<city>(raw_path, ...)` accepts `raw_path: Path` and reads the CSV directly via `pd.read_csv(raw_path)` — keeps the test signature consistent (tests build a synthetic CSV in `tmp_path`, pass the path). Step 06's orchestrator can call `load_raw_listings` itself then dispatch by city.

6. `CITY_FRAME_LOADERS: Final[dict[str, Callable]]` — `{ "Gurgaon": map_gurgaon, "Hyderabad": map_hyderabad, "Kolkata": map_kolkata, "Mumbai": map_mumbai }`.

7. `map_city(name: str, raw_path: Path, facet_frames: dict) -> pd.DataFrame` — raise `ValueError(f"Unknown city: {name}")` if not in `CITY_FRAME_LOADERS`, else dispatch.

### `tests/test_canonical_mapping.py`

**41 tests** matching the spec's Definition of Done #1, organized into 8 groups:

1. **Constants (5 tests):**
   - `test_canonical_columns_constant_matches_backend_schema`
   - `test_city_column_aliases_has_four_cities`
   - `test_city_column_aliases_canonical_to_raw_is_nonempty_per_city`
   - `test_city_frame_loaders_has_four_entries`
   - `test_bedroom_canonical_name_uses_camelcase` (locks `bedRoom` verbatim)

2. **`normalize_columns` (3 tests):**
   - `test_normalize_columns_drops_unsafe_columns` — synthetic DF with all unsafe cols + 5 legit → only legit survive
   - `test_normalize_columns_renames_via_alias_dict` — synthetic DF + alias dict, check 5 rename paths per city
   - `test_normalize_columns_passes_through_unknown_columns_as_is` — `SOME_GURGAON_SPECIFIC_FIELD` survives

3. **`clean_description` (7 tests):** lowercase, strip HTML, drop URL, drop email, collapse whitespace, NaN passthrough, vectorized (Series→Series)

4. **`map_gurgaon` (10 tests):** emits all canonical columns + row count, decodes furnish via Step 04, decodes amenities as list, parses price/area/lat-long (the lat/long test asserts the in-module `_parse_map_details`, not Step 03), cleans description, idempotent, drops unsafe cols, **doesn't write to data/raw or data/processed** (uses `_snapshot_raw_files` snapshot pattern, not `os.access`), leaves `luxury_category=NaN`, preserves `transact_type` as decoded "Sale"/"Rent" string (post code→string map), `bedRoom` camelCase

5. **`map_hyderabad` (2 tests):** uses `VALUE_LABEL` for ownership (pass-through), extracts `locality` from nested `location` dict

6. **`map_kolkata` (1 test):** `register_date=NaN` (column absent in raw)

7. **`map_mumbai` (2 tests):** emits all canonical columns, doesn't drop Mumbai rows

8. **Dispatcher + isolation (5 tests):** `map_city` dispatch correct (×4 cities), `map_city("Atlantis")` raises `ValueError`, `caplog` delegation (a row with `FURNISH=999` produces a warning containing `"furnish"` — proves the chain funnels back through `_log_unparseable` from `ml.cleaning.parsing`), `ast` scan asserts module imports nothing from `app.*` / `api.*`, source-string scan asserts no `open(`, `to_parquet`, `to_csv`, `to_json`, or `Path("data/processed"` literal.

**Test mechanics:**
- Fixture builder: `tests/fixtures/canonical_mapping_fixtures.py` mirrors `facet_decode_fixtures.py` style — literal `list[dict[object]]` + `_DF = pd.DataFrame(...)` constants. Includes:
  - `GURGAON_RAW_ROWS`, `GURGAON_RAW_DF` (3-row literal frame with all 67 Gurgaon cols populated)
  - Per-city `_RAW_ROWS/_DF` for Hyderabad/Kolkata/Mumbai
  - 15 facet `_*_DF` constants reused from `facet_decode_fixtures.py`
  - Helper `build_synthetic_raw_dir(tmp_path)` writes 4 city CSVs + `facets/` dir from the _RAW/_FACET constants; returns the synthetic `data_dir`.
- `_attach_index` runs at import time for each facet DF (same pattern as Step 04 tests).
- Immutability test pattern, AST scan pattern, source-string scan pattern copied verbatim from `test_facet_decoders.py`.
- `caplog.set_level(logging.WARNING, logger="ml.cleaning.parsing")` — same convention as Step 04 (decoders funnel through `_log_unparseable`).

## Files to modify

### `scripts/run_pipeline.py`
Append the existing pattern at line 27:
```python
from ml.cleaning import canonical_mapping  # noqa: F401  (Step 05 per-city mapper; Step 06 wires the per-row loop)
```
No change to `main()`.

### `ml/cleaning/__init__.py`
Currently empty (confirmed). Add re-exports of the public API per spec §"Modify":
```python
from ml.cleaning.canonical_mapping import (
    CANONICAL_COLUMNS, CITY_COLUMN_ALIASES, CITY_FRAME_LOADERS,
    clean_description, map_city, normalize_columns,
)
```
This is a deviation from the existing project convention (which uses full-submodule imports in test files), but the spec explicitly says to re-export.

## Verification

Run in order from repo root:

1. `python -m pytest tests/test_canonical_mapping.py -v` — all 41 tests pass.
2. `python -m pytest -m "not realdata" -q` — full cleaning-suite still clean (no realdata regression).
3. `ruff check ml/cleaning/canonical_mapping.py tests/test_canonical_mapping.py tests/fixtures/canonical_mapping_fixtures.py` — zero issues.
4. `python -c "from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS, CITY_COLUMN_ALIASES, map_city, CITY_FRAME_LOADERS, normalize_columns, clean_description; print(len(CANONICAL_COLUMNS), len(CITY_COLUMN_ALIASES), len(CITY_FRAME_LOADERS))"` — prints `37 4 4` (the 37 reflects both canonical names per duplicated field like `bedrooms`/`bedRoom`; the exact count gets locked by the `test_canonical_columns_constant_matches_backend_schema` test).
5. `python -c "import pandas as pd; from pathlib import Path; from ml.cleaning.canonical_mapping import map_city, load_facet_frames; ff = load_facet_frames(Path('data/raw/facets')); df = map_city('Gurgaon', Path('data/raw/gurgaon_10k.csv'), ff); print(df.shape, df.columns.tolist()[:6])"` — should complete without error and print the first 6 columns from `CANONICAL_COLUMNS` order.
6. `git status` — only the 4 changed/created files + this plan + the spec show in the diff.

## What is explicitly NOT in this step

- No `to_parquet` / `to_csv` / `to_json` (Step 06 owns writes).
- No feature engineering (`price_per_sqft`, `floor_ratio`, `n_amenities`, locality aggregates) — Step 08.
- No outlier flagging (`is_outlier = False` placeholder only) — Step 07.
- No missing-value imputation (`was_missing_*` flags remain NaN) — Step 06.
- No split between sale/rent (single-frame, raw `transact_type` string preserved).
- No SQL I/O. No `app.*` / `api.*` imports. No writes to `data/raw` or `data/processed`.

## Critical files reference

- Spec: `.claude/specs/05-canonical-schema-mapping-per-city.md`
- Step 02 raw loader: `ml/cleaning/ingest.py` (`load_raw_listings`, `_snapshot_raw_files`, `RAW_FILE_TO_CITY`, `_DRAFT_CANONICAL_MAP`, `PII_PATTERN`)
- Step 03 parsers: `ml/cleaning/parsing.py` (`parse_price`, `parse_area`, `_log_unparseable`, `SQFT_PER_SQM`)
- Step 04 decoders: `ml/cleaning/facet_decoders.py` (`load_facet_frames`, `decode_row`, all 13+2 `decode_*`, `DEFAULT_UNKNOWN_LABEL`, `FLOOR_NUM_ABOVE_MAX_LABEL`)
- Step 04 test patterns: `tests/test_facet_decoders.py`, `tests/fixtures/facet_decode_fixtures.py`
- Pipeline entry: `scripts/run_pipeline.py`
- Schema authority: `docs/05-BACKEND-SCHEMA.md` §2 + §U-SCHEMA-5, `docs/10-FINALIZED-INPUT-SCHEMA.md` §3
- Rules: `docs/08-RULES.md` §1.1 (PII), §1.2 (raw immutable), §10.2 (no luxury self-report)
