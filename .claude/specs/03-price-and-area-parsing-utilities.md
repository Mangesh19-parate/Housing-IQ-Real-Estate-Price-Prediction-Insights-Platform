# Spec: Price And Area Parsing Utilities

## Overview
Implement the two field-level string parsers — `parse_price()` and `parse_area()` — that turn the free-text price and area columns from the 4 raw city CSVs into clean numeric values in INR and square-feet respectively, plus the slim normalization layer (`parse_map_details()` is owned by a later step because it produces a different output shape — geo coordinates — but a placeholder seam is reserved here). This is Step 03 of the foundation module, Week 1, Day 2 of the Implementation Plan. It unblocks every downstream pipeline stage that touches price/area (cleaning → imputation → outlier detection → feature engineering → training), and is the first spec whose correctness is verified by a real pytest unit-test suite rather than just code review. Module: **foundation**.

## Depends on
- Step 02 — `02-raw-data-ingestion-and-schema-inventory` (provides the inventory under `data/processed/` that names exactly which raw columns hold free-text prices and areas per city — e.g. `PRICE`, `MIN_PRICE`, `MAX_PRICE`, `PRICE_SQFT`, `AREA`, `MIN_AREA_SQFT`, `MAX_AREA_SQFT`, `SUPER_SQFT`, `BUILTUP_SQFT`, `CARPET_SQFT`).
- Step 01 — `01-repo-scaffolding-and-environment-setup` (provides the `pytest.ini`, `ruff.toml`, `tests/conftest.py`, `scripts/run_pipeline.py` stub, and `ml/cleaning/` package).

## Routes / Endpoints
No new routes/endpoints. This spec is offline-only — pure Python utilities + pytest coverage. FastAPI/Flask are not touched. The parsers are imported by the cleaning pipeline that later specs build.

## Data / Schema changes
**No writes to `data/raw/`** — Rule §1.1 and §1.2 (raw immutable; cleaning writes only to `/processed`) are binding. The parsers are pure functions, no I/O.

**No writes to `data/processed/` either** — this spec produces no derived artifacts (no parquet, no JSON, no CSV). The first spec that writes to `/processed` was Step 02 (inventory); Step 04 (canonical cleaning) is the next one to write a derived dataset.

**No application DB changes** (`data/app.db` and its four tables from Step 01 stay as-is).

**New file under `tests/`:**
- `tests/fixtures/parse_fixtures.py` — a small set of literal sample strings (one per observed format) used by the pytest suite. Pure literals, no real-data dependency — keeps the unit tests fast and CI-friendly.

No new model artifacts (Step 04+ territory).

## Templates / UI
None. No Flask templates, no CSS, no JS, no static assets.

## Files to change / Files to create

**Create:**
- `ml/cleaning/parsing.py` — the parser module. Public API:
  - `parse_price(value, *, fallback_min=None, fallback_max=None) -> float | None` — accepts the observed free-text formats (see Rules below) plus plain numeric strings; returns INR as a float, or `None` if unparseable. Never raises on bad input — `None` is the failure signal that downstream code (the cleaning pipeline) inspects via a `was_parsed` mask.
  - `parse_area(value, *, unit_hint: str = "auto") -> float | None` — accepts the observed area formats (sqft, sq.m., ranges, plain numeric); normalizes everything to square-feet. Returns sqft as a float, or `None` on failure. A `1 sq.m. = 10.7639 sq.ft.` conversion constant is exported as a module-level `SQFT_PER_SQM` for unit testing.
  - `_PRICE_UNIT_MAP: dict[str, float]` — exported dict mapping unit suffixes to multipliers (`"cr" → 1e7`, `"l" → 1e5`, `"lac" → 1e5`). Documented and unit-tested independently of `parse_price()` so a future Lakh/Lac edge case can be fixed in one place.
  - `_AREA_UNIT_MAP: dict[str, float]` — same shape for area (`"sqft" → 1`, `"sq.m." → 10.7639`, `"sqft" → 1`). The dict has a single canonical key per unit; lowercase normalization happens at the regex layer.
  - `_log_unparseable(field: str, value, city: str | None = None) -> None` — internal helper that emits a structured `logging.warning(...)` line for every unparseable row, with the field name, the original value (truncated to 80 chars), and the optional city tag. The full failure list is the responsibility of Step 04's cleaning script (which writes `_parse_failures.csv` per the data-cleaning skill); this spec only emits the log so the failure rate is visible during dev, it does not persist.
  - Module docstring enumerates the exact source columns this parser is intended to handle (`PRICE`, `MIN_PRICE`, `MAX_PRICE`, `AREA`, `MIN_AREA_SQFT`, `MAX_AREA_SQFT`, `SUPER_SQFT`, `BUILTUP_SQFT`, `CARPET_SQFT`, `PRICE_SQFT`) so Step 04 has a one-stop reference.
- `tests/test_parsing.py` — pytest unit tests, all using the `parse_fixtures` fixture (no real-data dependency). Required tests listed in "Definition of done".
- `tests/fixtures/parse_fixtures.py` — sample strings (see "Files to create" — fixtures, not tests).

**Modify:**
- `scripts/run_pipeline.py` — already imports `ingest_raw` from Step 02; this spec adds a no-op wiring placeholder so the orchestrator still runs end-to-end: `from scripts import parse_check  # noqa: F401`. This is intentional — the actual parse run is invoked by Step 04's cleaning stage, not here. No code change beyond the import + a one-line comment that explains why.

**No changes** to:
- `requirements.txt` — `pandas` and `numpy` already pinned; the parsers use only Python stdlib (`re`, `logging`, `typing`).
- `.gitignore`, `pytest.ini`, `ruff.toml`, `app/`, `api/`, `data/`, `models/`, `notebooks/`, `scripts/ingest_raw.py`, `scripts/__init__.py`, `tests/conftest.py`.

## New dependencies
No new dependencies. `re` (regex), `logging`, `typing` are Python stdlib. `pandas` and `numpy` (already in `requirements.txt`) are not imported by this spec — the parsers are dtype-agnostic and operate on scalars/strings, which keeps them trivially unit-testable and reusable at any future pipeline stage.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL is written by this spec.
- **No dealer/contact/media-URL fields ever reach the UI or an export.** N/A — parsers operate on numeric/text columns only.
- **CSS variables only.** N/A — no templates or styles.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol.** N/A — no model artifacts produced.
- **Raw data is immutable.** The parsers never touch `data/raw/`. Step 04 (cleaning) opens raw files read-only and pipes the relevant columns through these parsers — the parsers themselves do no I/O and have no opinion on where data comes from.
- **Pure functions, deterministic, no side effects beyond logging.** `parse_price(x)` called twice on the same `x` returns the same value; no module-level state, no hidden caching, no randomness.
- **Never raise on bad input.** The failure signal is `None`. Any caller that needs the parse-failure reason can check `_log_unparseable` output; the cleaning pipeline in Step 04 builds the persistent `_parse_failures.csv` from the same rows. A `try/except` that swallows `ValueError` is acceptable at the regex boundary; re-raising is not — every bad row is data we want to *count and log*, not crash on.
- **Unit-suffix normalization happens at the regex layer, not in `_PRICE_UNIT_MAP`.** The dict has a single canonical key per unit; the parser lowercases the suffix before lookup, so `'L'`, `'l'`, `'Lac'`, `'lac'`, `'Lakh'`, `'lakh'` all map consistently. This avoids the "inconsistent casing" footgun called out by the data-cleaning skill.
- **Range resolution rule for area is documented and explicit.** Strings like `"1200-1400 sq.ft."` use the **midpoint** rule (per the data-cleaning skill). The parser picks midpoint, the docstring states this in one line, and the test suite covers it. This is a documented choice, not a per-row guess.
- **Fallback semantics for `parse_price()`.** When `value` is unparseable AND both `fallback_min` and `fallback_max` are provided AND they are valid floats, the function averages them and returns that (already-numeric fallback). When `value` is unparseable AND no fallback is provided (or the fallback itself is `None`/non-numeric), the function returns `None`. This keeps the cross-check against `MIN_PRICE`/`MAX_PRICE` (TRD §4.1) inside one function rather than scattered across the cleaning pipeline.
- **No silent truncation of currency precision.** Prices up to ~₹100 crore (₹1e9) and areas up to ~100,000 sqft are within the float64 precision envelope; the parsers do not round. Rounding happens at display time (FastAPI response formatting), not at parse time.
- **Logging uses the stdlib `logging` module, not `print()`.** Every unparseable row is `logging.warning("parse.unparseable field=%s value=%r city=%s", ...)`. The cleaning pipeline in Step 04 wires these warnings through to the `_parse_failures.csv` file (using a `logging.Handler` it owns); this spec does not own that handler.
- **All randomness is seeded.** N/A — no randomness used. (Step 04 may add stratified sampling and must use `random_state=42` per CLAUDE.md.)
- **Determinism.** `parse_price("3.5 Cr") == 35_000_000.0` and `parse_area("1500 sqft") == 1500.0` — exact, no floating-point drift between runs. The midpoint-of-range rule uses exact integer averaging where both bounds are integer-valued (`(1200 + 1400) / 2 == 1300.0`); the `SQFT_PER_SQM` constant is defined once and used identically everywhere.
- **Public-API stability.** The exported names listed under "Files to create" (`parse_price`, `parse_area`, `_PRICE_UNIT_MAP`, `_AREA_UNIT_MAP`, `SQFT_PER_SQM`) are the public contract Step 04 imports from. Anything else in the module is private (underscore-prefixed). Renaming a public name later is a breaking change to be flagged in the next spec.
- **No integration with the rest of the pipeline in this spec.** Step 04 owns the wiring (loading raw CSVs, iterating rows, applying parsers, writing `clean_listings.parquet`). This spec only ships the pure-function parsers + their unit tests. The temptation to "just wire it up to one city CSV to see it work" is a Step 04 concern; doing it here would couple two specs and make it harder to review either in isolation.

## Definition of done
A specific, testable checklist verifiable by running the test suite.

1. `python -m pytest tests/test_parsing.py -v` from repo root runs and passes. Tests required (exact names):
   - `test_parse_price_crore_uppercase` — `parse_price("3.5 Cr") == 35_000_000.0`.
   - `test_parse_price_crore_lowercase` — `parse_price("3.5 cr") == 35_000_000.0`.
   - `test_parse_price_lakh_short` — `parse_price("69.25 L") == 6_925_000.0`.
   - `test_parse_price_lakh_full` — `parse_price("69.25 Lac") == 6_925_000.0` and `parse_price("69.25 Lakh") == 6_925_000.0`.
   - `test_parse_price_plain_numeric` — `parse_price("15000000") == 15_000_000.0` and `parse_price("15000000.0") == 15_000_000.0`.
   - `test_parse_price_with_whitespace` — leading/trailing/inner whitespace stripped: `parse_price("  3.5  Cr  ") == 35_000_000.0`.
   - `test_parse_price_invalid_returns_none` — empty string, `"call for price"`, `"--"` all return `None`; no exception raised.
   - `test_parse_price_unparseable_is_logged` — `caplog` records a `WARNING`-level log line containing the field name and the truncated value when an unparseable row is passed (using `caplog.set_level(logging.WARNING)`).
   - `test_parse_price_fallback_to_min_max` — when `value="N/A"` and `fallback_min=14_000_000`, `fallback_max=16_000_000`, the function returns `15_000_000.0` (midpoint). When `value="N/A"` and either fallback is `None`, returns `None`.
   - `test_parse_price_fallback_min_max_ignore_when_value_parseable` — when `value="3.5 Cr"` and fallbacks are provided, the parsed value (₹35M) wins — fallbacks are only consulted on parse failure.
   - `test_parse_area_sqft` — `parse_area("1500 sq.ft.") == 1500.0` and `parse_area("1500 sqft") == 1500.0`.
   - `test_parse_area_sqm_to_sqft` — `parse_area("100 sq.m.") ≈ 1076.39` (within `1e-6` of `100 * SQFT_PER_SQM`).
   - `test_parse_area_range_midpoint` — `parse_area("1200-1400 sq.ft.") == 1300.0`.
   - `test_parse_area_plain_numeric` — `parse_area("1500") == 1500.0`.
   - `test_parse_area_with_whitespace` — whitespace stripped: `parse_area("  1500  sq.ft.  ") == 1500.0`.
   - `test_parse_area_invalid_returns_none` — empty string, `"--"` return `None`.
   - `test_parse_area_unit_hint_auto` — `unit_hint="auto"` (default) auto-detects; `unit_hint="sqft"` forces sqft interpretation regardless of any embedded suffix; `unit_hint="sqm"` forces sqm interpretation (test with `parse_area("100", unit_hint="sqm") ≈ 1076.39`).
   - `test_price_unit_map_keys_are_canonical_lowercase` — `_PRICE_UNIT_MAP` keys are lowercase; `"cr"`, `"l"`, `"lac"`, `"lakh"` all present; `"Cr"`, `"L"`, `"Lac"` are not (catches accidental upper-case duplication).
   - `test_area_unit_map_includes_sqft_and_sqm` — `_AREA_UNIT_MAP` has `"sqft"` and `"sqm"` keys with correct multipliers.
   - `test_sqft_per_sqm_constant_value` — `SQFT_PER_SQM == 10.7639` exactly.
   - `test_parse_price_idempotent` — `parse_price(parse_price.__doc__ or "3.5 Cr") == parse_price("3.5 Cr")` (calling twice gives the same answer; no module-level cache leaks).
   - `test_parse_area_idempotent` — same idempotency check for `parse_area`.
   - `test_parser_does_not_import_pandas_or_numpy` — `ml.cleaning.parsing` module-level imports do not include `pandas` or `numpy` (asserted via `sys.modules` introspection or a static `ast` scan of the file). Keeps the parser dependency-light per the rule "no new pip packages."
   - `test_parser_does_not_touch_data_raw_or_data_processed` — the parser module's source string contains no `open(`, `Path("data/raw"`, or `Path("data/processed"` literals — guarantees no accidental I/O.
2. `python -m pytest -m "not realdata"` from repo root still passes — confirms no real-data dependency was accidentally introduced.
3. `ruff check ml/cleaning/parsing.py tests/test_parsing.py tests/fixtures/parse_fixtures.py` reports zero issues.
4. `python -c "from ml.cleaning.parsing import parse_price, parse_area, SQFT_PER_SQM; print(parse_price('3.5 Cr'), parse_area('1500 sq.ft.'), SQFT_PER_SQM)"` from repo root prints `35000000.0 1500.0 10.7639` — confirms the public API imports cleanly and the canonical example values are correct.
5. `git status` after committing the spec, parser module, tests, and the modified `scripts/run_pipeline.py` shows only those files changed (plus this spec). No accidental additions to `data/processed/`, `data/raw/`, `models/`, or `notebooks/`.
6. `CLAUDE.md`'s "Implemented vs stub routes" table is **unchanged** by this spec — this spec adds no routes. (Updating that table is the job of the first spec that actually wires a Flask page or FastAPI endpoint.)
7. `07-TRACKER.md` "Week 1 — Data Understanding & Cleaning" Day 2 row's status is updated from `Not Started` to `Done` with the actual date and a note linking to this spec — via `/update-tracker`, not by hand-editing the tracker during this PR.