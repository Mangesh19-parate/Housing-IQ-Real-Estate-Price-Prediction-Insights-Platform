# Skill: Feature Engineering

**Trigger:** Building or modifying the model-ready feature set for price prediction or classification.

## Use this skill when
- Adding a new engineered feature (e.g. price-per-sqft, locality-level aggregates)
- Aligning a feature to the Finalized Input Schema v3 (16 fields)
- Deciding encoding strategy for a categorical field

## Key conventions (binding for this project)
- The canonical field list is `10-FINALIZED-INPUT-SCHEMA.md` — never invent a field name that isn't mapped there
- Ordinal categoricals (e.g. `balcony`, `agePossession`) are encoded as ordered integers, not one-hot, unless a spec says otherwise
- Nominal categoricals (`property_type`, `sector`) are one-hot or target-encoded — pick one per model and keep it consistent between training and serving
- Any leakage-prone feature (anything derived from `price` itself) is forbidden outside of price-per-sqft style ratios that are recomputed at inference from user input, not looked up

## Workflow
1. Confirm the field against the Finalized Input Schema before adding it
2. Add the transform to the shared feature-engineering module (not duplicated in FastAPI and Flask separately)
3. Add a unit test asserting the same input row produces identical output whether it goes through the training path or the serving path

## Gotchas / things that have bitten us before
- `sector` must be paired with `city` (per schema addition #13) — a bare sector name is ambiguous across cities
- Classification module reuses these fields minus price-derived ones — keep the shared feature module city/module-agnostic so both can import it

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
