# housingiq-ml-evaluator — Memory

## Fixed protocol (do not deviate without a spec change)
- 70/15/15 split, `random_state=42`.
- Regression: R², MAE, RMSE, MAPE on original ₹ scale.
- Classification: accuracy + per-class precision/recall/F1 + confusion
  matrix.
- Always check error distribution by price bucket, not just aggregate
  metrics — a model can look good in aggregate while failing on the cheap
  or luxury tails.

## Known pitfalls seen before
- Training on log(price) and forgetting to inverse-transform before scoring
  gives meaningless MAE/RMSE numbers. Always confirm units first.
- Outlier-flagged rows accidentally included in a training run inflate
  apparent performance — confirm `is_outlier` filtering before trusting a
  suspiciously high R².
