You are running a full review pass on a feature branch before merge.

User input: $ARGUMENTS (branch name or spec number)

## Step 1 - Gather the diff
```
git diff main...<branch_name>
```

## Step 2 - Run the reviewer agents
- Launch `housingiq-quality-reviewer` against the diff (style, conventions, CLAUDE.md compliance)
- Launch `housingiq-security-reviewer` against the diff (privacy rules, parameterized queries, immutability of raw data)
- If the diff touches `ml/` or `models/`, also launch `housingiq-ml-evaluator`

## Step 3 - Consolidate findings
Merge all three reports into one list, deduplicated, ordered by severity
(blocking > should-fix > nit).

## Step 4 - Report
Present the consolidated list to the user. Do not approve or merge anything
yourself - this command only produces a report.
