# Skill: Facet Decoding (ID → Label Joins)

**Trigger:** Any task that joins the 15 facet lookup CSVs (AGE, AMENITIES, FURNISH, etc.) onto listing rows.

## Use this skill when
- Decoding a single-value coded column (e.g. `FURNISH_ID` → `furnishing_type`)
- Decoding a multi-value ID list column (e.g. `AMENITIES`, `FEATURES`)
- Adding support for a new facet file

## Key conventions (binding for this project)
- One decode function per facet, named `decode_<facet>()`, living alongside the cleaning code
- Multi-value fields are decoded to a Python list, then exploded or kept as a delimited string depending on downstream need — document which per column
- Never hardcode a facet's ID→label mapping inline — always load from the facet CSV so updates to the lookup propagate everywhere
- Unknown/unmapped IDs decode to `'unknown'`, not `NaN` and not a silent drop

## Workflow
1. Load the facet CSV once per pipeline run and cache it in memory (don't re-read per row)
2. Join on the documented key column; verify join coverage (% matched) before proceeding
3. For multi-value columns, split on the documented delimiter, map each ID, rejoin or explode as needed
4. Report unmapped-ID rate; investigate if it's above ~1%

## Gotchas / things that have bitten us before
- FLOOR_NUM and TOTAL_FLOOR use overlapping ID ranges in some cities — don't assume the same facet file applies to every city without checking
- AMENITIES and FEATURES look similar but are separate facets with separate ID spaces — don't cross-map them

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
