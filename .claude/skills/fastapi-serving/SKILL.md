# Skill: FastAPI Model Serving

**Trigger:** Any work in `api/` — routers, schemas, or services for the model-serving layer.

## Use this skill when
- Adding a new inference endpoint
- Changing a Pydantic request/response schema
- Wiring a new trained model into the serving layer

## Key conventions (binding for this project)
- FastAPI is for model inference only — never put page rendering or session/auth logic here, that belongs in Flask (`app/`)
- Every endpoint has a Pydantic request model and response model — no bare dicts
- Load model artifacts once at startup (module-level or via a dependency-injected singleton), never per-request
- Async endpoints where the work is I/O bound; model inference itself can run sync inside a threadpool if the library isn't async-friendly

## Workflow
1. Define the Pydantic schema first, matching the Finalized Input Schema field names exactly
2. Write the router function; delegate actual inference to `api/services/`, keep routers thin
3. Add the endpoint to the auto-generated OpenAPI docs (default FastAPI behavior — don't suppress it)
4. Write a request/response round-trip test

## Gotchas / things that have bitten us before
- Field name drift between the Flask form, the FastAPI schema, and the model's training columns is the #1 source of silent bugs — always cross-check against `10-FINALIZED-INPUT-SCHEMA.md`

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
