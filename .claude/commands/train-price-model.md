You are running the price-prediction training pipeline.

User input: $ARGUMENTS (model type: baseline | xgboost | lightgbm; optional
hyperparameter overrides)

## Step 1 - Load data
Load `data/processed/clean_listings.parquet`, apply the shared
feature-engineering module (see `feature-engineering` skill).

## Step 2 - Split
70/15/15 train/val/test, `random_state=42`. Never deviate from this split
protocol, even for a "quick experiment".

## Step 3 - Train
Train the requested model type with any provided hyperparameter overrides,
otherwise use the last-known-good defaults recorded in the
`housingiq-ml-evaluator` agent's memory.

## Step 4 - Evaluate
Invoke the `housingiq-ml-evaluator` agent (or `/evaluate-model`) before
considering the run complete.

## Step 5 - Save
Save the artifact under `models/` with a new version suffix; never overwrite
an existing version file.

## Step 6 - Report
Print the version filename, the metric set, and a diff against the previous
best version's metrics.
