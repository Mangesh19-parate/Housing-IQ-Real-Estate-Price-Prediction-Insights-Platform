# Skill: SHAP Explainability

**Trigger:** Generating per-prediction "why" explanations for the price prediction or classification UI.

## Use this skill when
- Building the SHAP bar chart shown alongside a prediction
- Debugging why a model gives a surprising prediction for a specific input

## Key conventions (binding for this project)
- Price-up features render green (`#16A34A`), price-down render red (`#DC2626`) — consistent across the whole app, per the UI/UX doc
- Never show a bare number without a SHAP explanation on the Predict page — this is a hard product requirement, not optional polish
- Compute SHAP values once per prediction request; do not recompute the full training-set explainer on every API call — load a precomputed explainer object at startup

## Workflow
1. Load the versioned model + a matching precomputed SHAP explainer (TreeExplainer for XGBoost/LightGBM)
2. For a single prediction, compute local SHAP values, sort by magnitude, take the top N (5–7) for the UI
3. Map feature names to human-readable labels before sending to the frontend (never show raw column names like `bedRoom_enc`)

## Gotchas / things that have bitten us before
- TreeExplainer output shape differs between binary classifiers and regressors — check shape before assuming index 0/1 conventions

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
