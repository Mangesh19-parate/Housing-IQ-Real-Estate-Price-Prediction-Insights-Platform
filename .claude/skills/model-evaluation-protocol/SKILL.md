# Skill: Model Evaluation Protocol

**Trigger:** Any time a model (regression or classification) needs to be scored before being promoted to `models/` or referenced by the API.

## Use this skill when
- A new model artifact is produced
- Comparing two candidate models
- Investigating a production regression report

## Key conventions (binding for this project)
- Regression: R², MAE, RMSE, MAPE, computed on the original price scale, on the held-out 15% test split
- Classification: accuracy, precision/recall/F1 per class, confusion matrix
- Every reported metric set must state the data version and split seed it was computed against
- No model is considered "done" until the `housingiq-ml-evaluator` agent has signed off — see agent memory for the running metric log

## Workflow
1. Load the exact 15% test split (seeded, reproducible) — never eyeball a random sample instead
2. Compute the full metric set, not a subset
3. Write the report next to the model artifact (e.g. `models/price_xgb_v3.metrics.json`)
4. Update `metric-protocol-notes.md` in the evaluator agent's memory with the new version's numbers

## Gotchas / things that have bitten us before
- A high R² with a high MAPE on low-price properties usually means the model is systematically worse for cheaper listings — check error distribution by price bucket, not just aggregate metrics

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
