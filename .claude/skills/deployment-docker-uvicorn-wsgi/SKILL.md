# Skill: Deployment (Docker / Uvicorn / WSGI)

**Trigger:** Preparing or changing deployment configuration for FastAPI (Uvicorn) or Flask (WSGI).

## Use this skill when
- Writing/editing a Dockerfile or process manager config
- Changing environment variables, ports, or worker counts

## Key conventions (binding for this project)
- FastAPI runs under Uvicorn (with Gunicorn worker manager in prod if needed); Flask runs under a production WSGI server, never `flask run` in prod
- Model artifacts are loaded from a mounted/versioned path, never baked into the image with a hardcoded version that can't be rolled back
- Config (DB URL, model version, API base URL) comes from environment variables, never hardcoded per environment

## Workflow
1. Update the Dockerfile / compose config for any new dependency
2. Run `/deploy-check` before merging a deployment-affecting change
3. Verify health-check endpoints exist for both services

## Gotchas / things that have bitten us before
- Forgetting to bump the model-version env var after training a new model means the API silently keeps serving the old one

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
