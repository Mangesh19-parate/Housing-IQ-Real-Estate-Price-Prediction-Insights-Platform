# Skill: Regression Model Training (Price Prediction)

**Trigger:** Training, retraining, or tuning the price prediction regression model.

## Use this skill when
- Training a baseline linear/tree model
- Training or tuning XGBoost/LightGBM for price prediction
- Comparing candidate models before promoting one to `models/`

## Key conventions (binding for this project)
- Every model uses the same protocol: 70/15/15 train/val/test split, `random_state=42`
- Report metrics on the **original price scale**, not just log scale, even if the model trains on log(price)
- Serialize with joblib/pickle under `models/`, versioned filename (e.g. `price_xgb_v3.pkl`), never overwrite a prior version in place
- Every model artifact ships with its preprocessing pipeline bundled (or a matching versioned preprocessor) so serving never drifts from training

## Workflow
1. Load `clean_listings.parquet`, apply the shared feature-engineering module
2. Split 70/15/15 with `random_state=42`; never re-split ad hoc for a "quick check"
3. Train candidate(s), evaluate with the `housingiq-ml-evaluator` agent before considering it done
4. Save artifact + metrics report side by side under `models/`

## Gotchas / things that have bitten us before
- Training on log(price) but forgetting to inverse-transform before computing MAE/RMSE gives meaningless numbers — always check units
- Outlier rows are flagged, not deleted, in the processed store — make sure the training subset explicitly filters `is_outlier` rather than assuming it's already excluded

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
