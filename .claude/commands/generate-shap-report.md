You are generating a SHAP explainability report for a model.

User input: $ARGUMENTS (model artifact path; optional: single input row for
a per-prediction explanation)

## Step 1 - Load
Load the model and a precomputed TreeExplainer (build one if it doesn't exist
yet, and persist it alongside the model artifact).

## Step 2 - Compute
- If a single input row was given: compute local SHAP values for that
  prediction, sorted by magnitude, top 5-7 features.
- Otherwise: compute a global SHAP summary over the test split.

## Step 3 - Format
Map internal feature names to the human-readable labels used in the UI
(see `shap-explainability` skill) before presenting results.

## Step 4 - Save/report
Save a summary plot/JSON under `models/` (global case) or print the
top-feature table (single-prediction case).
