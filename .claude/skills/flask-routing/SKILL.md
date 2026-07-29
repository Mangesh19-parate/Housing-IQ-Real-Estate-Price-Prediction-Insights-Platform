# Skill: Flask Web App Routing

**Trigger:** Any work in `app/` — page routes, session/auth, or Jinja templates.

## Use this skill when
- Adding a new page route
- Wiring a page to call the FastAPI serving layer
- Working on auth (registration/login/logout/session)

## Key conventions (binding for this project)
- Flask renders pages and owns session/auth; it calls FastAPI over HTTP for any prediction/analytics/recommend/insights data — it does not import model code directly
- Every route function does one thing: fetch data (via API call or `db.py`), render template, done
- DB logic lives only in `app/database/db.py`, never inline in a route
- Every internal link uses `url_for()` — never a hardcoded path string

## Workflow
1. Add the route in `app.py` (or the relevant blueprint if the project has grown past single-file)
2. Call the FastAPI endpoint via a small HTTP client wrapper (with a documented timeout and error fallback)
3. Render a template that extends `base.html`
4. Add a loading state and a failure state per spec `53-loading-and-failure-states-all-modules`

## Gotchas / things that have bitten us before
- If the FastAPI service is down, the Flask page must degrade gracefully (spec 53) — never let an unhandled request exception surface a raw stack trace to the user

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
