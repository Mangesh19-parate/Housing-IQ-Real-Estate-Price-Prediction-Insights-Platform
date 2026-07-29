# Skill: Git Workflow & Spec-Driven Development

**Trigger:** Starting any new feature, or when the working directory has uncommitted changes and a new task begins.

## Use this skill when
- Beginning work on any numbered spec
- Deciding whether to branch, and how to name the branch

## Key conventions (binding for this project)
- Every feature starts from a clean working directory — `git status` must be clean before creating a new spec/branch
- Branch naming: `feature/<feature-slug>`; if taken, append `-01`, `-02`, etc.
- One spec = one branch = one focused PR; don't bundle unrelated specs into a single branch
- Specs are written and saved to `.claude/specs/<NN>-<slug>.md` *before* implementation begins — see `/create-spec`

## Workflow
1. Check `git status`; stop and ask the user to commit/stash if dirty
2. `git checkout main && git pull origin main`
3. Create the feature branch, then the spec, then the plan, then implement
4. Update `07-TRACKER.md` (via `/update-tracker`) when the step is actually done, with real dates

## Gotchas / things that have bitten us before
- Don't implement a stub route/feature ahead of its turn just because it looks easy — CLAUDE.md's implemented-vs-stub table is binding

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
