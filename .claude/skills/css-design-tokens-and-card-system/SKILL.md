# Skill: CSS Design Tokens & Card System

**Trigger:** Any new stylesheet or component that needs colors, spacing, or the shared card look.

## Use this skill when
- Styling a new page or component
- Reviewing whether a hardcoded color/spacing value should instead be a token

## Key conventions (binding for this project)
- Colors, spacing, and typography scale are CSS variables — never hardcode hex values or magic pixel numbers in a component stylesheet
- The card component (rounded corners, consistent padding, subtle shadow) is defined once and reused everywhere; do not fork a near-duplicate card style per page
- Large numerals for price displays (2.5–3rem), regular body text ~0.95–1rem, per the UI/UX doc

## Workflow
1. Check `static/css/style.css` (or the tokens file) for an existing variable before adding a new color/spacing value
2. If a genuinely new token is needed, add it to the shared tokens file, not inline in a page-specific stylesheet

## Gotchas / things that have bitten us before
- Page-specific CSS files (e.g. `analytics.css`) should only contain layout specific to that page — shared look-and-feel belongs in the global stylesheet

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
