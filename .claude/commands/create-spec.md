You are a senior developer spinning up a new feature for HousingIQ.
Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

## Step 1 - Check working directory is clean
Run `git status` and check for uncommitted, unstaged, or untracked files.
If any exist, stop immediately and tell the user to commit or stash
changes before proceeding. DO NOT CONTINUE until the working directory is clean.

## Step 2 - Parse the arguments
From $ARGUMENTS extract:
1. `step_number` - zero-padded to 2 digits: 2 -> 02, 11 -> 11
2. `feature_title` - human readable title in Title Case
3. `feature_slug` - git and file safe slug (lowercase, kebab-case, a-z0-9-, max 40 chars)
4. `module` - one of: foundation, price-prediction, classification, analytics,
   recommender, insights, map, ui-ux, privacy, testing, deployment
5. `branch_name` - format: `feature/<feature_slug>`

If you cannot infer these from $ARGUMENTS, ask the user to clarify before proceeding.

## Step 3 - Check branch name is not taken
Run `git branch`. If `branch_name` is already taken, append a number:
`feature/<slug>-01`, `feature/<slug>-02`, etc.

## Step 4 - Switch to main and pull latest
```
git checkout main
git pull origin main
```

## Step 5 - Create and switch to the feature branch
```
git checkout -b <branch_name>
```

## Step 6 - Research the codebase
Read before writing the spec:
- `CLAUDE.md` - roadmap, conventions, schema
- The relevant docs: `01-PRD.md`, `02-TRD.md`, `05-BACKEND-SCHEMA.md`,
  `08-RULES.md`, `10-FINALIZED-INPUT-SCHEMA.md`, `11-UML-DIAGRAM.md`
- All files in `.claude/specs/` - avoid duplicating existing specs
- The relevant skill(s) in `.claude/skills/`

Check `07-TRACKER.md` to confirm the requested step is not already marked done.
If it is, warn the user and stop.

## Step 7 - Write the spec
Generate a spec document with this exact structure:

---
# Spec: <feature_title>

## Overview
One paragraph: what this feature does, which module it belongs to, and why
it exists at this stage of the HousingIQ roadmap.

## Depends on
Which previous specs/steps this feature requires to be complete.

## Routes / Endpoints
- FastAPI: `METHOD /path` - description
- Flask: `METHOD /path` - description - access level (public/logged-in)
If none: state "No new routes/endpoints".

## Data / Schema changes
Any new tables, columns, cache files, or model artifacts needed.
Verify against `05-BACKEND-SCHEMA.md` and `10-FINALIZED-INPUT-SCHEMA.md` first.
If none: state "No data/schema changes".

## Templates / UI
- **Create:** new templates/components with their path
- **Modify:** existing templates and what changes

## Files to change / Files to create
Every file touched.

## New dependencies
Any new pip/npm packages. If none: state "No new dependencies".

## Rules for implementation
Always include:
- No SQLAlchemy/ORM unless already in use; parameterized queries only
- No dealer/contact/media-URL fields ever reach the UI or an export
- CSS variables only, never hardcoded hex values
- All templates extend `base.html`
- Model changes must reference the fixed evaluation protocol

## Definition of done
A specific, testable checklist verifiable by running the app or the test suite.
---

## Step 8 - Save the spec
Save to: `.claude/specs/<step_number>-<feature_slug>.md`

## Step 9 - Report to the user
```
Branch:    <branch_name>
Spec file: .claude/specs/<step_number>-<feature_slug>.md
Title:     <feature_title>
Module:    <module>
```
Then tell the user: "Review the spec, then enter Plan Mode with Shift+Tab twice
to begin implementation." Do not print the full spec in chat unless explicitly asked.
