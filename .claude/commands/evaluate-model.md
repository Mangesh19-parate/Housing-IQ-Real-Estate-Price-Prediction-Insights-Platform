You are running the fixed evaluation protocol against a model artifact.

User input: $ARGUMENTS (model artifact path or version tag)

## Step 1 - Load
Load the model artifact and its matching preprocessing pipeline.

## Step 2 - Load the held-out test split
Use the exact seeded 70/15/15 split (`random_state=42`) - regenerate it
deterministically from `clean_listings.parquet`, do not use an ad hoc sample.

## Step 3 - Score
- Regression: R^2, MAE, RMSE, MAPE, computed on the original price scale
- Classification: accuracy, per-class precision/recall/F1, confusion matrix

## Step 4 - Write the report
Save `models/<artifact_name>.metrics.json` alongside the artifact, including
the data version and split seed used.

## Step 5 - Update agent memory
Append the new version's numbers to
`.claude/agent-memory/housingiq-ml-evaluator/metric-protocol-notes.md`.

## Step 6 - Report
Print the metric set and flag any metric that regressed versus the previous
version.
