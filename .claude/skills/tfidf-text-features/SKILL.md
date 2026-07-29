# Skill: TF-IDF Text Feature Pipeline

**Trigger:** Any work on the TF-IDF vectorizer used for the recommender's text-derived features.

## Use this skill when
- Building the TF-IDF vectorizer over listing descriptions/amenity text
- Debugging why two very different listings are scoring as similar

## Key conventions (binding for this project)
- The fitted vectorizer is a versioned artifact, matched 1:1 with the similarity index build it feeds — never mix vectorizer versions across index builds
- Never surface raw dealer/marketing text verbatim in the UI (Rules doc — no unreviewed scraped text) — TF-IDF is for similarity scoring, not display

## Workflow
1. Fit/refit the vectorizer on the current corpus of cleaned listing text
2. Persist alongside the similarity index build (`/generate-recommender-index` handles both together)

## Gotchas / things that have bitten us before
- Boilerplate marketing phrases repeated across many listings inflate false similarity — consider a custom stopword list beyond the default

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
