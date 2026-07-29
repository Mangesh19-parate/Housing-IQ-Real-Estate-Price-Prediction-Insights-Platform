# Skill: Landing Page & SEO Basics

**Trigger:** Editing the landing page or any publicly indexable page.

## Use this skill when
- Changing landing page copy, meta tags, or structure
- Adding a new publicly accessible page that should be indexable

## Key conventions (binding for this project)
- Every public page has a descriptive `<title>` and meta description — no default/placeholder titles
- Landing page leads with the city selector and module nav per `03-APP-FLOW.md` — don't bury primary navigation below marketing copy

## Workflow
1. Check existing meta tag patterns in `templates/base.html` before adding new ones
2. Verify the page renders sensibly with JS disabled (progressive enhancement, not a JS-only shell) where feasible

## Gotchas / things that have bitten us before
- Terms/Privacy pages are legal content — don't casually reword them without flagging the change; they're linked from real user-facing footers

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
