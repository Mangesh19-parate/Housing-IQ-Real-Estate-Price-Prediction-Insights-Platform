# Skill: Security & Data-Privacy Review

**Trigger:** Reviewing any code path that touches raw listing data, dealer/contact fields, or user PII.

## Use this skill when
- Before merging any feature that queries or displays listing data
- Reviewing a new join that could reintroduce a previously-dropped field

## Key conventions (binding for this project)
- Dealer/agent contact fields, phone-like fields, and raw photo/media URLs must never reach the UI or an exported artifact — dropped at cleaning, must not reappear via a later join
- All SQL is parameterized — grep for f-string/`.format()`/`%`-style SQL construction as part of every review
- Raw source CSVs are immutable; verify no code path writes to `data/raw/`
- Every derived table states its computation date and source dataset version

## Workflow
1. Diff the new/changed query or join against the dropped-fields list before approving
2. Run a quick grep for raw SQL string interpolation in the changed files
3. Confirm any new derived table has the required metadata header

## Gotchas / things that have bitten us before
- A later join that re-adds a dropped field (e.g. joining back to the raw file "just to get one column") is the most common way contact fields leak back in — treat this as a review priority every time

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
