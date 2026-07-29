# Skill: Classification Model Training

**Trigger:** Training the Classification module's model (e.g. price-tier or luxury-category classification).

## Use this skill when
- Defining or changing the classification target
- Training/tuning a classifier that reuses the price-prediction feature set

## Key conventions (binding for this project)
- Reuses the Finalized Input Schema fields minus price-derived ones — do not leak `price` or `price_per_sqft` into classifier features
- Same 70/15/15 / `random_state=42` split discipline as regression
- Report precision/recall/F1 per class plus a confusion matrix, not just accuracy — class imbalance is expected (luxury categories are rare)

## Workflow
1. Confirm target definition against spec `21-classification-target-definition`
2. Check class balance; apply class weighting or stratified sampling before training if skewed
3. Train, evaluate via `housingiq-ml-evaluator`, version the artifact under `models/`

## Gotchas / things that have bitten us before
- Don't reuse the regression model's exact feature pipeline object without checking for price-derived columns first

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
