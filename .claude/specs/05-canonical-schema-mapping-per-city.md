# Spec: Canonical Schema Mapping Per City

## Overview
Implement the per-city canonical-schema mappers that take each of the 4 raw city CSVs (`gurgaon_10k.csv`, `hyderabad.csv`, `kolkata.csv`, `mumbai.csv`) through the cleaning utilities built in Steps 02–04 (raw inventory, `parse_price`/`parse_area`, `decode_*` facet decoders) and emit one unified DataFrame conforming to the **16-field finalized input contract** (`10-FINALIZED-INPUT-SCHEMA.md` §3 + `05-BACKEND-SCHEMA.md` §U-SCHEMA-5). This is Step 05 of the foundation module, Week 1, **Day 4** of the Implementation Plan. It is the central data-assembly step of the cleaning chain: Steps 02/03/04 produced the building blocks, this spec wires them per-city into a single canonical frame that Step 06 will concatenate, deduplicate, and write to `clean_listings.parquet`. Module: **foundation**.

## Depends on
- **Step 02** — `02-raw-data-ingestion-and-schema-inventory` (`scripts/ingest_raw.py`, `ml/cleaning/ingest.py`) — provides `load_raw_city_frames()`, per-facet inventory, and `assert_raw_readonly()`.
- **Step 03** — `03-price-and-area-parsing-utilities` (`ml/cleaning/parsing.py`) — provides `parse_price`, `parse_area`, `parse_map_details`, `_log_unparseable`.
- **Step 04** — `04-facet-decoding-joins` (`ml/cleaning/facet_decoders.py`) — provides the 13 single-value + 2 multi-value `decode_*` functions and `load_facet_frames()`.
- **Step 01** — `01-repo-scaffolding-and-environment-setup` (`pytest.ini`, `ruff.toml`, `tests/conftest.py`, `scripts/run_pipeline.py`).

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python cleaning utilities + pytest coverage. The canonical mapper is imported by Step 06 (the `clean_listings.parquet` writing step).

## Data / Schema changes
**No writes to `data/raw/`** — Rules §1.1, §1.2 are binding. `assert_raw_readonly()` (Step 02) gates every read.

**No writes to `data/processed/` either** — this spec produces no Parquet/JSON artifacts. The mappers return `pd.DataFrame` objects in-memory; the first spec that writes `clean_listings.parquet` is Step 06. The Day-4 row of `06-IMPLEMENTATION-PLAN.md` says "Day 4 — define and implement the canonical schema mapping ... concatenate into one DataFrame" — the per-city mapping is Step 05's scope; concatenation + write is Step 06's.

**No application DB changes** — `data/app.db` (Step 01) is unchanged.

**No new model artifacts.**

**No new files under `tests/fixtures/`** — the test suite uses literal DataFrames / dictionaries per the Step 02/03/04 pattern. A small in-test city-frame builder is fine, but no fixture file is added unless required by ≥3 tests.

## Templates / UI
None.

## Files to change / Files to create

**Create:**
- `ml/cleaning/canonical_mapping.py` — the per-city canonical-schema mapper. Public API:
  - **City-keyed dispatch table:**
    - `CITY_FRAME_LOADERS: dict[str, Callable[[Path], pd.DataFrame]]` — `{ "Gurgaon": map_gurgaon, "Hyderabad": map_hyderabad, "Kolkata": map_kolkata, "Mumbai": map_mumbai }`. The orchestrator (Step 06) iterates this dict.
  - **One mapper per city** — each is a thin function with the signature `map_<city>(raw_path: Path, facet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame`. Each:
    1. Loads its raw CSV via `load_raw_city_frames()[<city>]` (from Step 02) — kept DRY by delegating to the existing loader, not reimplementing CSV reading.
    2. Calls `parse_price`, `parse_area`, `parse_map_details` on the relevant raw columns → adds canonical `price_inr`, `area_sqft`, `latitude`, `longitude` columns.
    3. Applies every `decode_*` from Step 04 → adds the decoded label columns (`furnish_label`, `facing_label`, `age_label`, `property_type_label`, `owntype_label`, `city_label`, `locality_label`, `building_label`, `floor_label`, `total_floor_label`, `bedroom_label`, `bathroom_label`, `sub_availability_label`).
    4. Decodes `FEATURES`/`AMENITIES` into `features_list`, `amenities_list` (lists of strings).
    5. Constructs the **16 canonical input fields** (matching `10-FINALIZED-INPUT-SCHEMA.md` §3 and `05-BACKEND-SCHEMA.md` §U-SCHEMA-5): `city`, `sector`, `property_type`, `transact_type`, `bedRoom`, `bathroom`, `balcony`, `agePossession`, `built_up_area`, `servant_room`, `store_room`, `furnishing_type`, `luxury_category`, `floor_category`, `facing`, `amenities_list` (list-of-strings → later engineered to `n_amenities` + `has_<amenity>` by Step 06+).
    6. Preserves the **extended canonical schema** from `05-BACKEND-SCHEMA.md` §2 (full set of ~28 columns used downstream by feature engineering + analytics + insights): `listing_id`, `city`, `locality`, `property_type`, `transact_type`, `ownership_type`, `bedrooms`, `bathrooms`, `balconies`, `furnish`, `facing`, `age_bucket`, `floor_num`, `total_floor`, `floor_ratio` (filled with NaN here, computed in Step 08), `area_sqft`, `price_inr`, `price_per_sqft` (filled with NaN here, computed in Step 08), `features_list`, `amenities_list`, `n_amenities` (filled NaN, computed Step 08), `n_features` (NaN, Step 08), `building_name`, `building_id`, `latitude`, `longitude`, `description_clean` (lowercased/stripped, see Rules below), `register_date`, `is_outlier` (False default, set in Step 07), `was_missing_<field>` (filled NaN, set in Step 06). The 16 input-contract fields and the extended schema are the same DataFrame — the 16 are a subset of the extended set, kept as the canonical columns.
  - **`CANONICAL_COLUMNS: tuple[str, ...]`** — the tuple of ~28 column names in the order they must appear in the final DataFrame. This is the single source of truth for "what columns does the canonical frame have?" — Step 06's concatenation uses this exact order. Drawn from `05-BACKEND-SCHEMA.md` §2 + §U-SCHEMA-5.
  - **`CITY_COLUMN_ALIASES: dict[str, dict[str, str]]`** — per-city mapping of `{canonical_name: raw_column_name}` for the columns each city actually has. Some columns are absent per city (e.g., Kolkata has no `REGISTER_DATE`) — the mapper fills NaN. This dict is the documented "where does each canonical field come from in each city" reference that the rest of the codebase consults.
  - **`map_city(name: str, raw_path: Path, facet_frames: dict) -> pd.DataFrame`** — dispatcher: looks up `name` in `CITY_FRAME_LOADERS`, calls the per-city mapper, and returns the canonical DataFrame. Raises `ValueError` on an unknown city name (no silent fallthrough to a default mapper).
  - **`normalize_columns(df: pd.DataFrame, city: str) -> pd.DataFrame`** — shared helper that, given a per-city DataFrame, renames raw columns to canonical via `CITY_COLUMN_ALIASES[city]`, drops the documented PII/URL/contact columns (per Rules §1.1), and returns the renamed frame. Each per-city mapper calls this then layers city-specific quirks on top.
  - **`clean_description(series: pd.Series) -> pd.Series`** — vectorized helper that lowercases + strips HTML tags + collapses whitespace + drops URLs/emails (per Rules §1.1 — even though DESCRIPTION itself is not PII, the raw column can contain agent copy with embedded contact info). Returns the cleaned string series; NaN passes through.
  - **Module docstring** that explicitly enumerates which raw columns per city map to which canonical column, references `10-FINALIZED-INPUT-SCHEMA.md` §3 + `05-BACKEND-SCHEMA.md` §2 / §U-SCHEMA-5 as the schema authority, and calls out that this spec does **not** write `clean_listings.parquet` (Step 06's job).
  - **Reuse of `_log_unparseable`** from `ml.cleaning.parsing` for any field that fails parsing — keeps log lines identical across Steps 03/04/05.

- `tests/test_canonical_mapping.py` — pytest unit tests, all using literal DataFrames / dictionaries (no real-data dependency). Required tests listed in "Definition of done".

**Modify:**
- `scripts/run_pipeline.py` — add a no-op wiring placeholder so the orchestrator still runs end-to-end after this spec lands: `from ml.cleaning import canonical_mapping  # noqa: F401`. Same shape as the Step 04 wiring. No code change beyond the import + a one-line comment that explains why.
- `ml/cleaning/__init__.py` — re-export `CANONICAL_COLUMNS`, `CITY_COLUMN_ALIASES`, `map_city`, `CITY_FRAME_LOADERS`, `normalize_columns`, `clean_description` so other modules (Step 06+) import them by short name.

**No changes** to:
- `requirements.txt` — `pandas` is already pinned; the mappers use only stdlib + pandas + the Step 02/03/04 helpers.
- `.gitignore`, `pytest.ini`, `ruff.toml`, `app/`, `api/`, `data/`, `models/`, `notebooks/`, `scripts/ingest_raw.py`, `tests/conftest.py`, `ml/cleaning/parsing.py`, `ml/cleaning/facet_decoders.py`, `ml/cleaning/ingest.py`.

## New dependencies
No new dependencies. `pandas` (already in `requirements.txt`) is the only new import beyond what Steps 02/03/04 already require. `re` (HTML tag stripper) and `html` (stdlib) are stdlib.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL is written by this spec.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** This is binding per Rules §1.1. Each per-city mapper explicitly drops the documented unsafe columns. The full drop list (compiled from `02-TRD.md` §4.6 and the Step 02 inventory):
  - Photo/media URL columns: `PHOTO_URL`, `MEDIUM_PHOTO_URL`, `THUMBNAIL_PHOTO_URL`, `LARGE_PHOTO_URL`, `DEALER_PHOTO_URL`, `PROP_DETAILS_URL`, `PROP_URL`, `URL`.
  - Contact fields: `CONTACT_NAME`, `CONTACT_COMPANY_NAME`, `DEALER_NAME`, `DEALER_COMPANY`, `DEALER_PHONE`, `PHONE_NUMBER`, `CONTACT_PHONE`, `DEALER_EMAIL`, `CONTACT_EMAIL`.
  - Internal scrape IDs that have no use downstream: `SPID`, `PROP_ID` (replaced with canonical `listing_id` derived from `PROP_ID` — never carry the raw through), `SCRAPED_AT`, `RAW_JSON`.
  - This list lives in a single `UNSAFE_COLUMNS: tuple[str, ...]` constant at module top, so it's grep-able and reviewable in one place. `normalize_columns` applies the drop before the rename, so unsafe columns never enter the canonical frame in any form.
- **Raw data is immutable.** `load_raw_city_frames()` (Step 02) opens files in read-only mode and asserts `assert_raw_readonly()` before each read; this spec reuses those helpers — does NOT re-open or re-load CSVs.
- **CSS variables only.** N/A — no templates or styles.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifacts produced.
- **Pure functions, deterministic, no side effects beyond logging.** Each `map_<city>(...)` is a pure function of its arguments; calling it twice on the same `(raw_path, facet_frames)` returns equal DataFrames (column-dtypes + values + ordering). No module-level cache beyond what `load_facet_frames` already provides.
- **City routing is by string lookup, not by column sniffing.** `map_city(name, ...)` dispatches on the explicit `name` argument; it does not try to detect the city from the CSV's contents (Gurgaon's `CITY` column is already `"Gurgaon"`, but Kolkata's is `"Kolkata"`, and one of the four files has been observed with a title-case discrepancy in earlier inspection — name dispatch is the safer contract).
- **Per-city quirks are isolated to the per-city mapper, not pushed into `normalize_columns`.** `normalize_columns` handles only: alias-based rename + unsafe-column drop. Any quirk specific to one city (e.g., Kolkata has no `REGISTER_DATE`; Hyderabad uses `VALUE_LABEL` for ownership not `OWNTYPE`) lives in that city's `map_<city>` function. Cross-city shared logic lives in `clean_description` and the rename helper.
- **`luxury_category` is NOT collected from raw data and is NOT computed here.** Per Rules §10.2 (Update v3) and `10-FINALIZED-INPUT-SCHEMA.md` §3, `luxury_category` is derived server-side from the amenity checklist at prediction time, not self-reported. The canonical DataFrame therefore leaves `luxury_category` as NaN — Step 06's training-time derivation of `luxury_score`/`luxury_category` happens during feature engineering (Week 3), not during cleaning. This is explicitly documented in the module docstring so the next reader doesn't try to fill it.
- **`transact_type` is preserved as a raw string.** `Sale` / `Rent` strings pass through to the canonical column verbatim; the FastAPI `/predict` route (Step 09+) does the routing per `05-BACKEND-SCHEMA.md` §U-SCHEMA-6 / `02-TRD.md` §U-TRD-4. This spec only **maps** the column; it does **not** split the data into sale/rent frames.
- **Description cleanup is bounded.** `clean_description` does the minimum to make the column safe for downstream TF-IDF (Week 5): lowercase, strip HTML tags via `html.unescape` + a regex over `<...>` blocks, collapse runs of whitespace, drop URLs (`http://`, `https://`, `www.`), drop email-like substrings (regex `\S+@\S+`). It does NOT do stemming, lemmatization, stop-word removal — that's a TF-IDF concern at training time. NaN in → NaN out.
- **Facets are decoded via Step 04's helpers, not reimplemented.** Every `decode_*` call goes through `ml.cleaning.facet_decoders.decode_*`. The mapper does not have its own join logic.
- **`listing_id` is derived from raw `PROP_ID` (cast to string, stripped) per city.** Two cities use `PROP_ID`, one uses `PROPHEADID`, one uses `PROPERTY_ID` — the per-city mapper pulls from whichever raw column is present. The canonical `listing_id` is always a string. Empty/NaN `PROP_ID` → `listing_id = None` (Step 06 deduplication will drop these).
- **`balcony` is preserved as the raw category string** (`"0"`, `"1"`, `"2"`, `"3"`, `"3+"`) — matching the reference project's `balcony` column type (categorical/ordinal per `10-FINALIZED-INPUT-SCHEMA.md` §1). Numeric coercion is a model-pipeline concern.
- **`furnishing_type` is preserved as the decoded string label** (`"Unfurnished"`, `"Semifurnished"`, `"Furnished"`) — Step 06's `ColumnTransformer` (per `02-TRD.md` §U-TRD-3) maps to `0/1/2` via `OrdinalEncoder` at training time. The cleaning step keeps the human-readable form because that's what the UI / insights cards will display.
- **The 16 input-contract fields appear with their canonical names verbatim.** Specifically, `bedRoom` (camelCase, matches the reference project), `built_up_area`, `agePossession`, `floor_category`, `luxury_category`, `servant_room`, `store_room` — even when those conflict with the snake_case style used elsewhere in the canonical schema. This is a deliberate exception per `10-FINALIZED-INPUT-SCHEMA.md` §1's note: "Cross-checked directly against the reference project's final model-ready file." Renaming these would break the contract with the reference project and the FastAPI schema.
- **Per-city raw column quirks documented in `CITY_COLUMN_ALIASES`.** Examples the next reader needs to know:
  - Hyderabad/Mumbai use `location.LOCALITY_NAME` (a nested dict field, not a flat `LOCALITY`) — the mapper extracts the dict value via `ast.literal_eval` + `.get("LOCALITY_NAME")` before aliasing to `locality`.
  - Hyderabad uses `VALUE_LABEL` for ownership (already a string label, not a coded int) — passes through to `ownership_type` directly without decoding.
  - Mumbai's `FLOOR_NUM` is the same numeric-with-string-codes shape as Gurgaon's — same decoder call works.
  - Kolkata's `AREA` is sometimes a string like `"750 sq.ft."`, sometimes already numeric — both handled by `parse_area`.
- **Missing-value handling is OUT of scope here.** The mapper fills missing canonical fields with NaN; Step 06's imputation pass (Day 6 of the Implementation Plan) is where `was_missing_*` flags get added and the <5% / 5–40% / 40–70% / >70% strategy from `02-TRD.md` §5 gets applied. This spec only emits the raw canonical frame; imputation is a separate step.
- **Outlier flagging is OUT of scope here.** `is_outlier` defaults to `False` in this spec; Step 07 (Week 2) is where the percentile/IQR/domain-rule flags get computed per `02-TRD.md` §6.
- **Feature engineering (`price_per_sqft`, `floor_ratio`, `n_amenities`, `n_features`, `age_bucket`, `bath_bed_ratio`, locality aggregates) is OUT of scope here.** Step 08 (Week 3) is where these get engineered per `02-TRD.md` §8.
- **No DataFrame writes.** The mappers return DataFrames; they do not call `to_parquet`, `to_csv`, `to_json`, or any other writer. Step 06 owns all writes.
- **No SQL row I/O.** N/A.
- **Logging uses the stdlib `logging` module, not `print()`.** Reuse `ml.cleaning.parsing._log_unparseable(field, value, city)` so log lines are consistent with Steps 03/04.
- **Idempotency.** Two consecutive calls to `map_gurgaon(same_path, same_facet_frames)` produce two DataFrames with identical values, dtypes, and column ordering. Tested explicitly in `test_map_gurgaon_idempotent`.
- **All randomness is seeded.** N/A — no randomness used.

## Definition of done
A specific, testable checklist verifiable by running the test suite.

1. `python -m pytest tests/test_canonical_mapping.py -v` from repo root runs and passes. Tests required (exact names):
   - `test_canonical_columns_constant_matches_backend_schema` — `CANONICAL_COLUMNS` is a tuple of strings containing every column listed in `05-BACKEND-SCHEMA.md` §2 + §U-SCHEMA-5's revised canonical field set, including the 16 input-contract fields with their exact canonical names (`bedRoom`, `built_up_area`, `agePossession`, `furnishing_type`, `luxury_category`, `floor_category`, `servant_room`, `store_room`). Test reads the constant and asserts the set is a superset of the documented schema.
   - `test_city_column_aliases_has_four_cities` — `CITY_COLUMN_ALIASES` has exactly the 4 expected keys: `Gurgaon`, `Hyderabad`, `Kolkata`, `Mumbai`.
   - `test_city_column_aliases_canonical_to_raw_is_nonempty_per_city` — for each city, the alias dict has at least the 12 documented raw → canonical mappings (verified by checking each city's dict contains entries for `listing_id`, `city`, `locality`, `bedrooms`, `bathrooms`, `price_inr`, `area_sqft`, `latitude`, `longitude`, plus at least 4 of the decoded-label columns).
   - `test_city_frame_loaders_has_four_entries` — `CITY_FRAME_LOADERS` has exactly 4 keys matching the 4 city names.
   - `test_map_unknown_city_raises_value_error` — `map_city("Atlantis", raw_path, facet_frames)` raises `ValueError` (no silent default).
   - `test_normalize_columns_drops_unsafe_columns` — given a synthetic per-city frame containing `PHOTO_URL`, `DEALER_NAME`, `DEALER_PHONE`, `CONTACT_EMAIL`, `PROP_DETAILS_URL`, `SPID`, `PROP_URL`, plus legitimate columns, `normalize_columns` returns a frame that has NONE of those unsafe columns present under any name (grep-able from `df.columns.tolist()`).
   - `test_normalize_columns_renames_via_alias_dict` — given a per-city frame with raw column names and a per-city alias dict, `normalize_columns` returns a frame with the canonical names (verified for at least 5 columns per city).
   - `test_normalize_columns_passes_through_unknown_columns_as_is` — a raw column not in the alias dict (e.g., `SOME_GURGAON_SPECIFIC_FIELD`) is preserved unchanged under its raw name, not dropped silently. This catches accidental alias misconfiguration.
   - `test_clean_description_lowercases` — `"Hello World"` → `"hello world"`.
   - `test_clean_description_strips_html_tags` — `"<p>3BHK with <b>clubhouse</b></p>"` → `"3bHK with clubhouse"` (the `3BHK` lowercase is the cleaner's intentional behavior; assert exact).
   - `test_clean_description_drops_urls` — `"Visit https://example.com for details"` → `"Visit for details"` (URL dropped, surrounding whitespace collapsed).
   - `test_clean_description_drops_emails` — `"Contact agent@gmail.com today"` → `"Contact today"`.
   - `test_clean_description_collapses_whitespace` — `"  multiple    spaces  here  "` → `"multiple spaces here"`.
   - `test_clean_description_passes_nan_through` — `pd.NA` and `float("nan")` inputs return NaN.
   - `test_clean_description_is_vectorized` — given a `pd.Series` of mixed strings + NaN, the helper accepts a Series and returns a Series (not scalar-in-scalar-out).
   - `test_map_gurgaon_emits_all_canonical_columns` — given a small synthetic Gurgaon frame (3 rows, all 67 raw columns populated with literal values), `map_gurgaon` returns a DataFrame whose `columns.tolist()` equals `list(CANONICAL_COLUMNS)` (in the canonical order), every column from `CANONICAL_COLUMNS` is present (no missing columns), and the row count is preserved.
   - `test_map_gurgaon_decodes_furnish_via_step04` — a row with raw `FURNISH=4` produces a `furnish` (and/or canonical furnishing-type) column with the human-readable label, proving the mapper delegates to Step 04's `decode_furnish` and does not reimplement it. Verified by asserting the output column equals the value `decode_furnish(4, facet_frames)` produces.
   - `test_map_gurgaon_decodes_amenities_as_list` — a row with raw `AMENITIES="20,21,32"` produces an `amenities_list` column whose value is a `list[str]` of length 3 with the matching labels.
   - `test_map_gurgaon_parses_price_via_step03` — a row with raw `PRICE="3.5 Cr"` produces a `price_inr` column with value `35_000_000.0`, proving the mapper delegates to Step 03's `parse_price`.
   - `test_map_gurgaon_parses_area_via_step03` — a row with raw `AREA="1450 sq.ft."` produces an `area_sqft` column with value `1450.0`.
   - `test_map_gurgaon_parses_map_details_via_step03` — a row with raw `MAP_DETAILS="{'LATITUDE': '28.4065', 'LONGITUDE': '76.9628'}"` produces `latitude=28.4065`, `longitude=76.9628`.
   - `test_map_gurgaon_cleans_description` — a row with raw `DESCRIPTION="<p>Spacious 3BHK with CLUBHOUSE</p>"` produces a `description_clean` column with the lowercased + HTML-stripped form.
   - `test_map_gurgaon_idempotent` — calling `map_gurgaon` twice on the same `(raw_path, facet_frames)` returns two DataFrames that compare equal under `pd.testing.assert_frame_equal` (same dtypes, same values, same column ordering).
   - `test_map_gurgaon_drops_all_unsafe_columns` — given a synthetic Gurgaon frame containing the documented unsafe columns (PHOTO_URL, MEDIUM_PHOTO_URL, DEALER_PHOTO_URL, PROP_DETAILS_URL, CONTACT_NAME, CONTACT_COMPANY_NAME, DEALER_PHONE, DEALER_EMAIL, SPID), the mapper's output has NONE of them in `df.columns`.
   - `test_map_gurgaon_does_not_write_to_data_raw_or_data_processed` — `os.access("data/raw", os.W_OK)` and `os.access("data/processed", os.W_OK)` are checked before and after the mapper call (via the same `assert_raw_readonly()` pattern from Step 02); no write occurs.
   - `test_map_hyderabad_uses_value_label_for_ownership` — a Hyderabad row with raw `VALUE_LABEL="Freehold"` produces an `ownership_type` column with `"Freehold"` (string pass-through, no facet decode needed — this is the documented Hyderabad quirk).
   - `test_map_hyderabad_extracts_locality_from_nested_dict` — a Hyderabad row with raw `location="{'LOCALITY_NAME': 'Banjara Hills', 'CITY_NAME': 'Hyderabad'}"` produces a `locality` column with `"Banjara Hills"` (nested-dict extraction per the documented quirk).
   - `test_map_kolkata_register_date_is_nan` — a Kolkata row has no `REGISTER_DATE` in the raw frame; the mapper's output has `register_date=NaN` for that row (not missing column, not crash).
   - `test_map_mumbai_emits_all_canonical_columns` — same shape as the Gurgaon test, against a synthetic Mumbai frame.
   - `test_map_mumbai_does_not_drop_local_rows` — a Mumbai-only row (with `CITY="Mumbai"`) is preserved through the mapper; no city-filtering is applied at this step (Step 06 owns concatenation).
   - `test_map_city_dispatches_correctly` — `map_city("Gurgaon", ...)` returns the same DataFrame as `map_gurgaon(...)`; same for the other 3 cities.
   - `test_map_city_emits_log_warning_for_unknown_facet_id` — using `caplog.set_level(logging.WARNING)`, a row with an unmapped `FURNISH=999` produces a log line containing the field name (`"furnish"`), proving the mapper delegates the logging to Step 04's `_log_unparseable`.
   - `test_map_does_not_import_app_or_api` — `ml.cleaning.canonical_mapping` module-level imports do not include anything from `app.*` or `api.*` (asserted via `sys.modules` introspection or a static `ast` scan).
   - `test_canonical_mapping_does_not_touch_filesystem_outside_data_raw` — the mapper's source string contains no `open(`, `to_parquet`, `to_csv`, `to_json`, or `Path("data/processed"` literals — guarantees no accidental write paths.
   - `test_luxury_category_left_as_nan` — given any input row, `map_gurgaon`'s output has `luxury_category=NaN` for every row (no self-report-fill at cleaning time per Rules §10.2).
   - `test_transact_type_preserved_as_string` — given a raw row with `TRANSACT_TYPE="Sale"`, the output `transact_type` is the string `"Sale"` (not a one-hot-encoded 0/1).
   - `test_bedroom_canonical_name_uses_camelcase` — the output column for bedrooms is exactly `"bedRoom"` (not `"bedroom"` or `"bed_room"` or `"BedRoom"`), matching the reference project's contract.
2. `python -m pytest -m "not realdata"` from repo root still passes — confirms no real-data dependency was accidentally introduced.
3. `ruff check ml/cleaning/canonical_mapping.py tests/test_canonical_mapping.py` reports zero issues.
4. `python -c "from ml.cleaning.canonical_mapping import CANONICAL_COLUMNS, CITY_COLUMN_ALIASES, map_city, CITY_FRAME_LOADERS, normalize_columns, clean_description; print(len(CANONICAL_COLUMNS), len(CITY_COLUMN_ALIASES), len(CITY_FRAME_LOADERS))"` from repo root prints `28 4 4` (or whatever the documented `CANONICAL_COLUMNS` count is — the exact number is locked in by test #1's assertion and must match the count reported here) — confirms the public API imports cleanly.
5. `python -c "import pandas as pd; from ml.cleaning.canonical_mapping import map_city, load_facet_frames; ff = load_facet_frames(Path('data/raw/facets')); df = map_city('Gurgaon', Path('data/raw/gurgaon_10k.csv'), ff); print(df.shape, df.columns.tolist()[:6])"` from repo root, when run against the real ~44k-row Gurgaon CSV, completes without error, returns a DataFrame with shape `(~44890, N)` where N equals `len(CANONICAL_COLUMNS)`, and prints the first 6 canonical column names. (This is the **only** DoD item that touches real data — and it is gated on the previous tests passing, so it doesn't slow CI.)
6. `git status` after committing the spec, mapper module, tests, and the modified `scripts/run_pipeline.py` + `ml/cleaning/__init__.py` shows only those files changed (plus this spec). No accidental additions to `data/processed/`, `data/raw/`, `models/`, or `notebooks/`.
7. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes.
8. `07-TRACKER.md` "Week 1 — Data Understanding & Cleaning" Day 4 row's status is updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.
