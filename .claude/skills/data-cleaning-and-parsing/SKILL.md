# Skill: Data Cleaning & Parsing

**Trigger:** Any task touching raw city CSVs, price/area string parsing, or the clean_listings pipeline.

## Use this skill when
- Writing or editing `parse_price()`, `parse_area()`, or `parse_map_details()`
- Investigating null counts, dtypes, or malformed rows in a raw city file
- Adding a new city's raw CSV to the pipeline

## Key conventions (binding for this project)
- Raw CSVs under `data/raw/` are immutable — never write back to them, ever
- All cleaning writes new files to `data/processed/` only
- `parse_price()` must handle `'3.5 Cr'`, `'69.25 L'`, and plain numeric strings — unit-test every format seen in the data
- `parse_area()` must handle `'sq.ft.'`, `'sq.m.'`, and unitless numeric strings; normalize everything to sqft
- Log (don't silently drop) any row that fails parsing — write to a `_parse_failures.csv` for manual review

## Workflow
1. Load the raw file(s) and print `.info()`, dtypes, and null counts before writing any transform
2. Write/extend the relevant `parse_*()` function with unit tests covering every string format observed
3. Run the parser over the full column and inspect the failure log, not just a sample
4. Only after parsing is clean, hand off to the facet-decoding skill for ID→label joins

## Gotchas / things that have bitten us before
- Lakh/Crore suffixes are inconsistently cased ('L', 'l', 'Lac') across cities — normalize case first
- Some rows have area given as a range ('1200-1400 sq.ft.') — decide and document a single resolution rule (e.g. midpoint) rather than guessing per-row
- Never assume a city's raw schema matches another city's — always re-inspect columns per file

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
