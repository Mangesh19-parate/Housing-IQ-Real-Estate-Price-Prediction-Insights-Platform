# Skill: Accessibility Review

**Trigger:** Before shipping any new page or interactive component.

## Use this skill when
- Completing a new page (spec `55-accessibility-pass-all-pages` requires this before sign-off)
- Reviewing a form, chart, or map for keyboard/screen-reader usability

## Key conventions (binding for this project)
- Every form input has an associated `<label>`; every chart has a text alternative (a short data summary, not just a canvas)
- Color is never the only signal — the green/red price-up/down convention must be paired with a +/- sign or icon for colorblind users
- All interactive elements are reachable and operable via keyboard (tab order, visible focus states)

## Workflow
1. Run a quick keyboard-only pass on the new page
2. Check color contrast against the design tokens (navy/slate on off-white, accent on white)
3. Add alt text / aria-labels for icons and map markers

## Gotchas / things that have bitten us before
- Chart.js/Plotly canvases are invisible to screen readers by default — always ship a companion text summary or data table toggle

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
