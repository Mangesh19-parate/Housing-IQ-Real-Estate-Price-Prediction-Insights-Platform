# Skill: Insights Narrative Templater

**Trigger:** Generating the templated "insight sentences" that accompany locality/city stats.

## Use this skill when
- Adding a new insight sentence template
- Debugging a nonsensical or grammatically broken generated insight

## Key conventions (binding for this project)
- Insights are template-based (fill-in-the-blank from precomputed stats), not free-form LLM generation, unless a future spec explicitly changes this
- Every generated sentence must be traceable to a specific stat in `locality_stats` / `amenity_uplift` / `age_price_trend` / `bhk_price_trend` — no invented claims
- Numbers in generated sentences are always human-formatted (₹ Cr/L, not raw integers)

## Workflow
1. Pick the template matching the available stat combination
2. Fill placeholders from the precomputed table, applying the ₹ formatting helper
3. Fall back to a neutral "insufficient data" sentence if a required stat is missing — never render a template with a blank/`None` filled in

## Gotchas / things that have bitten us before
- Small-sample localities can produce misleading stats (e.g. "+40% price uplift" from 2 data points) — apply a minimum sample-size gate before generating a claim

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
