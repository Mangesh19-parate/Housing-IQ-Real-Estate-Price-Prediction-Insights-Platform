# Spec: Facet Decoding Joins

## Overview
Implement the per-facet ID-to-label join functions that turn the 15 coded columns in the raw city CSVs (e.g., `FURNISH=4`, `FACING=1`, `AGE=2`, plus the multi-value `FEATURES`/`AMENITIES` ID lists) into human-readable labels and downstream-feature-ready lists. This is Step 04 of the foundation module, Week 1, Day 3 of the Implementation Plan. It consumes the parsed numerics from Step 03 (`parse_price`/`parse_area`) and the inventory from Step 02 (which already located the raw coded columns per city and computed the join coverage), and produces the human-readable columns that Step 05's canonical-schema mapping will assemble into `clean_listings.parquet`. Module: **foundation**. This is the third concrete data-pipeline step of the 7-week roadmap.

## Depends on
- Step 02 — `02-raw-data-ingestion-and-schema-inventory` (provides the per-facet inventory and the `facet_join_coverage.csv` that tells us which coded columns have incomplete coverage).
- Step 03 — `03-price-and-area-parsing-utilities` (provides `parse_price`/`parse_area`; not strictly required for the decoders themselves but this spec is the next link in the same cleaning chain and shares the same module + test conventions).
- Step 01 — `01-repo-scaffolding-and-environment-setup` (provides `pytest.ini`, `ruff.toml`, `tests/conftest.py`, `scripts/run_pipeline.py`).

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python utilities + pytest coverage. The decoders are imported by the cleaning pipeline that Step 05 builds.

## Data / Schema changes
**No writes to `data/raw/`** — Rule §1.1 and §1.2 (raw immutable; cleaning writes only to `/processed`) are binding. The decoders are pure functions reading from `data/raw/facets/*.csv` that are loaded into memory once per pipeline run; no writes occur.

**No writes to `data/processed/` either** — this spec produces no derived artifacts. The decoders return values in-memory; the first spec that writes `clean_listings.parquet` is Step 05.

**No application DB changes** (`data/app.db` and its four tables from Step 01 stay as-is).

**No new files under `tests/fixtures/`** — this spec reuses the synthetic-`data/raw`-tree fixture pattern introduced in Step 02 (`tests/fixtures/raw_snapshot_fixture.py`) and extends it with a small in-test facet builder. The decoders' unit tests use literal DataFrames / dictionaries, not the real ~180k-row CSVs, so the test suite stays fast and CI-friendly.

No new model artifacts (Step 05+ produces downstream artifacts).

## Templates / UI
None. No Flask templates, no CSS, no JS, no static assets.

## Files to change / Files to create

**Create:**
- `ml/cleaning/facet_decoders.py` — the per-facet decode module. Public API:
  - **Single-value decoders** (one per single-value facet, named `decode_<facet>(value, facet_df) -> str | None`):
    - `decode_furnish(value, facet_df) -> str | None` — single int ID → label string (`"Furnished"`, `"Unfurnished"`, ...).
    - `decode_facing(value, facet_df) -> str | None` — single int → label (`"North"`, `"East"`, ...).
    - `decode_age(value, facet_df) -> str | None` — single int → label (`"New Property"`, `"1-5 Year Old Property"`, ...).
    - `decode_property_type(value, facet_df) -> str | None` — single int/str → label.
    - `decode_owntype(value, facet_df) -> str | None` — single int → label.
    - `decode_locality_id(value, facet_df) -> str | None` — single int → locality label.
    - `decode_building_id(value, facet_df) -> str | None` — single int/str → building label.
    - `decode_bedroom_num(value, facet_df) -> str | None` — single int → label (e.g. `"3"`).
    - `decode_bathroom_num(value, facet_df) -> str | None` — single int → label.
    - `decode_floor_num(value, facet_df) -> str | None` — int OR string code (`"B"`, `"G"`, `"L"`, `"M"`) → label (`"Basement"`, `"Ground"`, `"Lower Ground"`, `"Multi-Storied"`).
    - `decode_total_floor(value, facet_df) -> str | None` — single int → label.
    - `decode_sub_availability(value, facet_df) -> str | None` — single int → label.
    - `decode_city(value, facet_df) -> str | None` — single int → label (note: in the dataset, `CITY` is already a string label like `"Gurgaon"`; this decoder exists for completeness / future-proofing).
  - **Multi-value decoders** (the two columns whose raw value is a comma-separated list of IDs):
    - `decode_features(value, facet_df) -> list[str]` — comma-separated string of IDs → list of labels. Empty / NaN → `[]`. Failed-to-decode IDs are dropped silently (logged via the same `_log_unparseable` helper from `ml.cleaning.parsing`) and the remaining labels are returned.
    - `decode_amenities(value, facet_df) -> list[str]` — same shape as `decode_features`.
  - **Driver:**
    - `load_facet_frames(facets_dir: Path) -> dict[str, pd.DataFrame]` — loads all 15 facet CSVs once into a `{facet_name: df}` dictionary. Normalizes each `id` column to a consistent join key (zero-padded string OR int, decided per facet; the choice is documented in the module docstring and locked in by a single `JOIN_KEY_FORMAT` per facet dict). Called once per pipeline run; subsequent calls re-use the cached dictionary.
    - `decode_row(row: pd.Series, facet_frames: dict[str, pd.DataFrame]) -> dict` — convenience function that applies every single-value + multi-value decoder to the relevant columns in row and returns a dict of decoded values, ready to be assigned to the canonical dataframe. Kept as a thin wrapper so future specs (Step 05) can call one function per row.
  - **Module-level constants:**
    - `SINGLE_VALUE_FACETS: tuple[str, ...]` — the 13 single-value facets in the order they're applied.
    - `MULTI_VALUE_FACETS: tuple[str, ...]` — `("FEATURES", "AMENITIES")`.
    - `DEFAULT_UNKNOWN_LABEL: str = "unknown"` — used for unmapped IDs in single-value decoders (per the facet-decoding skill's binding rule: "Unknown/unmapped IDs decode to `'unknown'`, not `NaN` and not a silent drop"). Single-value decoders therefore return the string label or `"unknown"`, never `None` — the `None` return is reserved for the input itself being missing/NaN; an unmapped-but-present ID returns `"unknown"`.
    - `MULTI_VALUE_DELIMITER: str = ","` — single comma, no whitespace (raw data confirms: `"33,23,12,46"` — no spaces).
  - **Module docstring** enumerates the exact source columns this module is intended to handle and the canonical output column each maps to (per `10-FINALIZED-INPUT-SCHEMA.md` §1–§2 and `05-BACKEND-SCHEMA.md` §2), so Step 05 has a one-stop reference.
  - **Reuse of `ml.cleaning.parsing._log_unparseable`.** The unparseable-ID warning log line (with field name, truncated value, optional city tag) is emitted by the same helper that Step 03 uses, so both stages produce identically-shaped log lines and Step 05's `_unmapped_ids.csv` collates them.

- `tests/test_facet_decoders.py` — pytest unit tests, all using literal DataFrames / dictionaries (no real-data dependency). Required tests listed in "Definition of done".
- `tests/fixtures/facet_decode_fixtures.py` — small builders for synthetic single-row facet DataFrames used by the pytest suite. Pure literals, no real-data dependency.

**Modify:**
- `scripts/run_pipeline.py` — already imports `ingest_raw` from Step 02. This spec adds a no-op wiring placeholder so the orchestrator still runs end-to-end: `from ml.cleaning import facet_decoders  # noqa: F401`. The actual decode-by-row step is invoked by Step 05's cleaning stage, not here. No code change beyond the import + a one-line comment that explains why.

**No changes** to:
- `requirements.txt` — `pandas` and `numpy` already pinned; the decoders use only stdlib (`logging`) plus the already-pinned `pandas` for the in-memory facet DataFrames.
- `.gitignore`, `pytest.ini`, `ruff.toml`, `app/`, `api/`, `data/`, `models/`, `notebooks/`, `scripts/ingest_raw.py`, `scripts/__init__.py`, `tests/conftest.py`, `ml/cleaning/parsing.py`, `ml/cleaning/ingest.py`, `ml/cleaning/__init__.py`.

## New dependencies
No new dependencies. `pandas` (already in `requirements.txt`) is imported for the in-memory facet DataFrame lookups. `logging` is Python stdlib.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL is written by this spec.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** N/A — the decoders operate on coded columns only. The input columns (FACING, FURNISH, etc.) are not PII; the raw data also has a `PHOTO_URL`/`MEDIUM_PHOTO_URL`/`DEALER_PHOTO_URL`/`CONTACT_NAME`/`CONTACT_COMPANY_NAME` set, but those are dropped at Step 05 (canonical schema mapping) and never reach this module's input.
- **CSS variables only.** N/A — no templates or styles.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifacts produced.
- **Raw data is immutable.** `load_facet_frames()` opens files in read-only mode and never writes to `data/raw/`. The defensive check from Step 02 (`os.access(path, os.W_OK)` returns False for raw files) is reused via the same `assert_raw_readonly()` helper imported from `ml.cleaning.ingest` — this spec does not duplicate that logic.
- **Pure functions, deterministic, no side effects beyond logging.** Each `decode_<facet>(value, facet_df)` is a pure function of its two arguments; calling it twice on the same `(value, facet_df)` returns the same value. No module-level cache, no hidden state, no randomness.
- **Unknown IDs decode to `'unknown'`, never `None` and never a silent drop.** The facet-decoding skill (`/facet-decoding`) makes this a binding rule. The single-value decoders therefore return the label string for a known ID, the literal string `"unknown"` for an unknown-but-present ID, and `None` only when the input value itself is missing/NaN. The multi-value decoders return a list and silently drop unknown IDs (logging them via `_log_unparseable`) — they never propagate the literal `"unknown"` label into the list, because "unknown" is not a real amenity and would corrupt amenity counting downstream.
- **Type normalization at the join layer.** Raw listing columns are `int64` for numeric codes; facet CSV `id` columns are zero-padded strings (`"001"`, `"B"`, etc.). `load_facet_frames()` builds a per-facet index (`pd.Series` keyed by the normalized form) at load time; the decoders ask the index for the key in the same form the raw value is already in. Two valid choices: normalize facet IDs to `int` (works for all numeric facets, fails for `FLOOR_NUM` string codes `B/G/L/M`), OR normalize both sides to string with a leading zero (works for everything, but conflicts with float NaN handling). The chosen approach: normalize each facet ID to a string form once at load time (`FLOOR_NUM` keeps strings as-is; numeric facets get zero-padded strings to match the CSV), and the raw numeric input is `str(int(value))` on lookup. This is the only safe cross-type scheme; the choice is documented in the module docstring.
- **Floor code handling.** `FLOOR_NUM` is a special case: the raw column contains both integers (`1`, `2`, ..., `95`) and string codes (`"B"`, `"G"`, `"L"`, `"M"`). The decoder must accept either form and return the matching label (`"Basement"`, `"Ground"`, `"Lower Ground"`, `"Multi-Storied"`, `"1"`, `"2"`, ...). Numbers above the facet's max get a domain-rule fallback: `"95+"` if the input is a numeric code greater than the facet's max code, otherwise `"unknown"`. This is documented inline because Indian real-estate data regularly has 40-floor buildings that exceed the facet's labeled inventory.
- **Multi-value parsing is bounded.** `decode_features(value)` and `decode_amenities(value)` split on a single comma, strip whitespace defensively (raw data is whitespace-free today but raw data drift is real), and drop empty tokens. A regex check (`re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", value)`) pre-validates the value; a malformed value is logged and returns `[]`. There is no silent truncation to a "first N" — the full list is decoded.
- **Unmapped-ID rate is reported, not just warned.** Each call to `decode_floor_num` / `decode_amenities` / etc. that returns `"unknown"` (single-value) or skips an ID (multi-value) increments a per-pipeline-run counter held in `load_facet_frames()`'s returned dict (a `decode_stats` sub-dict). Step 05's cleaning script reads this counter post-row-iteration and writes `_unmapped_ids.csv` (per the data-cleaning skill's "_parse_failures.csv" pattern). The decoders themselves do NOT write any file — they only increment an in-memory counter and emit a `logging.warning` line per occurrence (rate-limited at one per 100 to avoid log floods on 180k-row reads).
- **No SQL row I/O.** N/A.
- **Idempotency.** Two consecutive calls to `decode_furnish(4, facet_df)` with the same `facet_df` produce the same string. The decoded counter is incremented per call, not per unique input — that's the cleaning-script's job to deduplicate when writing `_unmapped_ids.csv`.
- **All randomness is seeded.** N/A — no randomness used.
- **No integration with the rest of the pipeline in this spec.** Step 05 owns the wiring (loading raw CSVs, iterating rows, applying decoders, writing `clean_listings.parquet`). This spec only ships the per-facet decoders + their unit tests. The temptation to "just wire it up to one city CSV to see it work" is a Step 05 concern; doing it here would couple two specs and make it harder to review either in isolation (same rule as Step 03).
- **Logging uses the stdlib `logging` module, not `print()`.** Reuse `ml.cleaning.parsing._log_unparseable(field, value, city=None)` so log lines are consistent across Step 03 and Step 04.

## Definition of done
A specific, testable checklist verifiable by running the test suite.

1. `python -m pytest tests/test_facet_decoders.py -v` from repo root runs and passes. Tests required (exact names):
   - `test_decode_furnish_known_id` — `decode_furnish(4, facet_df)` returns the matching label (e.g., `"Semifurnished"` based on the `FURNISH.csv` `id`→`label` mapping).
   - `test_decode_furnish_unknown_id_returns_unknown` — `decode_furnish(999, facet_df)` returns `"unknown"` (not `None`, not an exception).
   - `test_decode_furnish_nan_input_returns_none` — `decode_furnish(pd.NA, facet_df)` and `decode_furnish(float("nan"), facet_df)` return `None`.
   - `test_decode_facing_known_id` — `decode_facing(1, facet_df)` returns the matching label.
   - `test_decode_age_known_id` — `decode_age(2, facet_df)` returns the matching label.
   - `test_decode_property_type_known_id` — `decode_property_type(1, facet_df)` returns `"Residential Apartment"`.
   - `test_decode_owntype_known_id` — `decode_owntype(...)` returns the matching label.
   - `test_decode_floor_num_integer_known` — `decode_floor_num(1, facet_df)` returns `"1"`.
   - `test_decode_floor_num_string_code_basement` — `decode_floor_num("B", facet_df)` returns `"Basement"`.
   - `test_decode_floor_num_string_code_multi_storied` — `decode_floor_num("M", facet_df)` returns `"Multi-Storied"`.
   - `test_decode_floor_num_above_max_returns_above_max` — `decode_floor_num(95, facet_df)` returns `"95+"` (the documented domain-rule fallback for codes above the facet's max).
   - `test_decode_features_comma_separated_list` — `decode_features("33,23,12,46", facet_df)` returns a list of the matching labels in the same order.
   - `test_decode_features_nan_returns_empty_list` — `decode_features(pd.NA, facet_df)` returns `[]`.
   - `test_decode_features_unknown_ids_dropped_silently` — `decode_features("33,999,12", facet_df)` returns a list of length 2 (the 999 is dropped, not turned into `"unknown"`); the call logs a `WARNING` for the dropped ID.
   - `test_decode_features_whitespace_stripped` — `decode_features("33, 23 , 12", facet_df)` returns the same result as `decode_features("33,23,12", facet_df)`.
   - `test_decode_features_malformed_returns_empty` — `decode_features("not-an-id-list", facet_df)` returns `[]` and logs a warning.
   - `test_decode_amenities_known_list` — `decode_amenities("20,21,32", facet_df)` returns the matching labels in order.
   - `test_decode_amenities_drop_unknown` — same shape as the features drop-unknown test.
   - `test_load_facet_frames_returns_fifteen_entries` — `load_facet_frames(facets_dir)` returns a dict with exactly 15 keys matching the documented facet names (`AGE`, `AMENITIES`, `BATHROOM_NUM`, `BEDROOM_NUM`, `BUILDING_ID`, `CITY`, `FACING_DIRECTION`, `FEATURES`, `FLOOR_NUM`, `FURNISH`, `LOCALITY_ID`, `OWNERSHIP_TYPE`, `PROPERTY_TYPE`, `SUB_AVAILABILITY`, `TOTAL_FLOOR`). Same set as Step 02's test.
   - `test_load_facet_frames_does_not_modify_files` — snapshots `data/raw/facets/` mtimes+sizes before and after via `assert_raw_readonly()` from `ml.cleaning.ingest`, asserts identical.
   - `test_load_facet_frames_normalizes_id_keys` — every value in the resulting dict has its `id` column normalized to a string form such that lookups by both integer (`decode_furnish(4, ...)`) and string (`decode_furnish("4", ...)`) inputs work. This locks in the join-key normalization rule.
   - `test_decode_idempotent` — `decode_furnish(4, facet_df) == decode_furnish(4, facet_df)` (calling twice on the same input gives the same answer; no module-level cache leaks).
   - `test_decode_does_not_import_app_or_api` — `ml.cleaning.facet_decoders` module-level imports do not include anything from `app.*` or `api.*` (asserted via `sys.modules` introspection or a static `ast` scan). Keeps the decoder dependency-light and side-effect-free.
   - `test_decode_does_not_touch_data_raw_or_data_processed` — the decoder module's source string contains no `open(`, `Path("data/raw"`, or `Path("data/processed"` literals — guarantees no accidental I/O.
   - `test_decode_emits_log_warning_for_unknown_id` — using `caplog.set_level(logging.WARNING)`, calling `decode_furnish(999, facet_df)` produces a log line containing the field name (`"furnish"`) and the truncated value (`"999"`).
   - `test_decode_per_facet_log_field_name_used` — the log field tag is the facet name (e.g., `"furnish"`, `"features"`), not the raw column name (`"FURNISH"`), so downstream log parsers can map by canonical field.
2. `python -m pytest -m "not realdata"` from repo root still passes — confirms no real-data dependency was accidentally introduced.
3. `ruff check ml/cleaning/facet_decoders.py tests/test_facet_decoders.py tests/fixtures/facet_decode_fixtures.py` reports zero issues.
4. `python -c "from ml.cleaning.facet_decoders import decode_furnish, decode_floor_num, decode_amenities, load_facet_frames, SINGLE_VALUE_FACETS, MULTI_VALUE_FACETS, DEFAULT_UNKNOWN_LABEL; print(len(SINGLE_VALUE_FACETS), len(MULTI_VALUE_FACETS), DEFAULT_UNKNOWN_LABEL)"` from repo root prints `13 2 unknown` — confirms the public API imports cleanly and the facet counts are correct.
5. `git status` after committing the spec, decoder module, tests, and the modified `scripts/run_pipeline.py` shows only those files changed (plus this spec). No accidental additions to `data/processed/`, `data/raw/`, `models/`, or `notebooks/`.
6. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes. (Updating that table is the job of the first spec that actually wires a Flask page or FastAPI endpoint.)
7. `07-TRACKER.md` "Week 1 — Data Understanding & Cleaning" Day 3 row's status is updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.
