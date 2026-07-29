You are running lint/style checks before a feature is considered ready for review.

## Step 1 - Python
Run the project's configured linter/formatter (e.g. `ruff`/`flake8` + `black`
if configured) against changed files only.

## Step 2 - Templates/CSS
Spot-check changed templates for hardcoded hex values or inline `<style>`
tags (should be flagged per the `css-design-tokens-and-card-system` skill).

## Step 3 - Report
List violations grouped by file; do not auto-fix without the user's
confirmation.
