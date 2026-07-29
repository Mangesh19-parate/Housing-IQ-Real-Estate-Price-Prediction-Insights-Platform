# Skill: Testing (pytest, Flask + FastAPI)

**Trigger:** Writing or running tests for either the Flask app or the FastAPI service.

## Use this skill when
- Writing tests for a newly implemented spec
- Debugging a failing test in CI or locally

## Key conventions (binding for this project)
- Flask: use `app.test_client()`, never spin up a live server or use Selenium unless a spec explicitly requires JS execution
- FastAPI: use `TestClient` (or `httpx.AsyncClient`) against the app instance, not live network calls
- Tests are anchored to the feature **spec**, not the implementation — every assertion should be traceable to a spec line
- Test file naming: `tests/test_<feature>.py`; function naming: `test_<behavior>_<condition>_<expected>`

## Workflow
1. Locate the spec for the feature under test before writing assertions
2. Cover: happy path, auth boundary, input validation, boundary values, state changes, HTTP contract, authorization, a security smoke test, template/response rendering
3. Run via `/test-feature` and address failures before marking the spec's tracker row done

## Gotchas / things that have bitten us before
- Don't assert on internal helper names or exact SQL strings — test observable behavior only, or tests break on harmless refactors

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
