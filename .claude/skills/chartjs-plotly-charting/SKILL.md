# Skill: Chart.js / Plotly.js Charting

**Trigger:** Building or editing any chart in the Analytics or Insights pages.

## Use this skill when
- Adding a new chart type to the Analytics dashboard
- Rendering a SHAP bar chart or price/locality trend chart

## Key conventions (binding for this project)
- No build step — Chart.js/Plotly.js loaded via CDN script tags, vanilla JS to wire data in
- Color usage matches the app palette: navy/slate for chrome, amber or blue as the single accent, green/red only for price-up/price-down semantics
- Every chart has a loading skeleton and a failure state (no blank/broken canvas)

## Workflow
1. Fetch chart data from the relevant FastAPI/analytics-cache endpoint
2. Render with the smallest chart type that conveys the data shape (bar for comparisons, line for trends, avoid 3D/pie unless the spec explicitly calls for it)
3. Verify responsiveness at mobile width

## Gotchas / things that have bitten us before
- Plotly's default toolbar/branding should be trimmed down (`displayModeBar: false` or a curated subset) to match the app's minimal-chrome design principle

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
