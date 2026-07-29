You are turning an already-written spec into a step-by-step implementation plan.

User input: $ARGUMENTS (the spec number or slug)

## Step 1 - Load the spec
Read `.claude/specs/<NN>-<slug>.md` in full. If it doesn't exist, stop and tell
the user to run `/create-spec` first.

## Step 2 - Load context
Read `CLAUDE.md`, the relevant skill(s) in `.claude/skills/`, and any prior
plan in `.claude/plans/` for dependent specs.

## Step 3 - Write the plan
Break the spec's "Files to change / Files to create" into an ordered list of
concrete edits. For each item, state:
- The exact file path
- What changes (in 1-3 sentences, not a full diff)
- Which spec section it satisfies
- Any test that must accompany it

Order the plan so that data/schema changes come before the code that depends
on them, and backend (API/DB) changes come before frontend wiring.

## Step 4 - Flag risks
List anything ambiguous in the spec that should be confirmed with the user
before implementation starts, rather than guessed.

## Step 5 - Save
Save to `.claude/plans/<NN>-<slug>.md`.

## Step 6 - Report
Print the ordered step list to the user and ask for confirmation before
starting implementation.
