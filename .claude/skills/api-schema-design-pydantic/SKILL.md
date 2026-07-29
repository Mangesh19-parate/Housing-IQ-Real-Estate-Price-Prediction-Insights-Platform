# Skill: API Schema Design (Pydantic)

**Trigger:** Defining or changing any FastAPI request/response schema.

## Use this skill when
- Adding a new field to an existing endpoint
- Designing a new endpoint's request/response contract

## Key conventions (binding for this project)
- Field names match the Finalized Input Schema v3 canonical names exactly — no ad hoc renaming per endpoint
- Use explicit types and constraints (e.g. `conint(ge=1, le=10)` for bedRoom) rather than bare `int`
- Optional fields are truly optional with sensible defaults documented in the schema's docstring, not silently required
- Every schema has a short docstring describing the field's business meaning, not just its type

## Workflow
1. Cross-check the field list against `10-FINALIZED-INPUT-SCHEMA.md` before writing the model
2. Add validation constraints matching the documented value ranges
3. Let FastAPI's auto-generated OpenAPI docs be the source of truth for consumers — keep schemas accurate rather than writing separate API docs by hand

## Gotchas / things that have bitten us before
- A schema mismatch between what the Flask form sends and what the Pydantic model expects fails silently as a 422 with a generic message — test the full form→API round trip, not just the API in isolation

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
