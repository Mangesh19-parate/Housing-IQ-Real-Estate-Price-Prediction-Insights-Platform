# Skill: Recommender Similarity Search

**Trigger:** Building or tuning the property recommender's similarity index.

## Use this skill when
- Rebuilding the similarity index after a data refresh
- Adding a new similarity feature (e.g. weighting locality proximity higher)

## Key conventions (binding for this project)
- Similarity index artifacts are versioned under `models/` alongside the price model, same joblib/pickle convention
- Recommendation results must always be paired with a "matched attributes" explanation (spec 39) — never a bare list of similar properties

## Workflow
1. Build/refresh the TF-IDF + numeric feature blend used for similarity
2. Rebuild the nearest-neighbor index via `/generate-recommender-index`
3. Spot-check a handful of known-similar pairs manually before deploying

## Gotchas / things that have bitten us before
- Similarity purely on raw price will just recommend "same price bracket" regardless of type/locality — make sure categorical and locality features are weighted, not just price/area

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
