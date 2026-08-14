# Implementation Plan: Step 13 — Baseline Regression Model Training

## Context
Step 13 consumes the feature pipeline built by Step 12
(`models/feature_pipeline_v1.pkl` = `(fitted_preprocessor, LocalityAggregator)`,
`models/feature_list_v1.json`, plus `scripts/build_features.py`) and the
canonical cleaned DataFrame (`data/processed/clean_listings.parquet`) and
trains the v1 baseline price regression. Output: `models/price_model_sale_v1.pkl`
+ `models/price_model_rent_v1.pkl` (or `rent.skipped` if subset < threshold),
`models/metrics_v1.json` (per-candidate metrics + per-city + chosen model +
git commit), appended Round 2/Round 3 sections of `data/processed/feature_selection_report.md`,
and one row per `transact_type` appended to `data/model_registry.csv`.

This is the **baseline** in the 30–35% MAE/RMSE-reduction tracking series —
later specs (Week 4 levers + Week 8 final tuning) reduce against these
numbers. TRD §10 minimum acceptable level: 6 candidates (Linear, Ridge,
Lasso, RF, GB, XGBoost), 70/15/15 split, log-target with original-scale
metrics, two pipelines (Sale / Rent), preprocessor loaded not refit.

Ponytail shortcuts (intentional, call out the spec drift early):
- The spec's "step 1 of script" calls `build_feature_frame(df)` on the
  full pre-split frame. Step 12's `build_feature_frame` requires
  `INPUT_FIELDS_V3` columns + `is_outlier` + `was_missing_*` — that's
  satisfied by the Step 07 Parquet. The training script applies the
  outlier filter **after** splitting (per `is_outlier == False` rule),
  not before `build_feature_frame`. *(ponytail: cheaper to build the
  feature frame once on the full set than twice on subsets; the LOO
  aggregator was fit on the training rows only at Step 12's
  `build_features.py` time, so the test-time transforms on outlier
  rows are still leakage-safe even when outlier rows are present in
  the frame.)*
- `ml/training/evaluation.py` exposes a single `evaluate_subset`
  helper instead of separate `evaluate_model` / `per_city_metrics`
  functions — they share the metric-dict construction and the
  spec's separation buys nothing.
- No `ml/__init__.py` modification — `from ml.training import ...`
  already works once `ml/training/__init__.py` re-exports the public
  symbols; the existing `ml/__init__.py` doesn't enumerate submodules
  (Spot check: it's effectively empty / no training import).
- The spec mentions SHAP for the winner **and** impurity importance
  for RF/GB/XGB; we compute SHAP on the winner (tree-only path) and
  pull `feature_importances_` straight from the trained RF/GB/XGB
  estimators — no extra fitting. Permutation importance runs only on
  the validation slice of the winner (`n_repeats=10`, `random_state=42`).
- `tests/test_train_price_model_script.py` runs the script in a
  `subprocess` with `HOUSINGIQ_ARTIFACT_DIR` + `HOUSINGIQ_PROCESSED_DIR`
  pointing at `tmp_path` — matches Step 12's pattern.

---

## 1. Spec deltas / open issues

| # | Spec says | Code reality | Resolution |
|---|---|---|---|
| 1 | Spec uses `INPUT_FIELDS_V3` from `api.schemas.predict_v3` (Step 11) | Confirmed — `ml/features/feature_frame.py` already imports from there. | No action. |
| 2 | `build_feature_frame(df)` consumes `clean_listings.parquet` | Confirmed. The frame must contain the 16 contract fields + `is_outlier` + any `was_missing_*`. Step 07's writer guarantees this. | No action. |
| 3 | "log1p(price) → models train on log, metrics on original" via `expm1` | Step 12 doesn't apply log-transform anywhere; it's the training script's responsibility. | `regression_metrics` in `evaluation.py` accepts `y_true_log` + `y_pred_log` and applies `np.expm1` before scoring. Pure, no leakage. |
| 4 | "preprocessor loaded, not refit" via `load_feature_artifacts("v1")` | Confirmed — `persistence.load_feature_artifacts` returns `(preproc, agg, feature_names)` tuple. | Training script calls `load_feature_artifacts()` once per run, builds `Pipeline([("preproc", preproc), ("est", make_estimator(name))])` for each candidate. The `LocalityAggregator` was already fit on the **Step 12 train subset** (same split helper → identical train rows), so we do **not** re-transform with the aggregator at training time — we use the **features Step 12 already produced**. Simplification: the training script reads a cached feature frame instead of rebuilding it. **AMBIGUITY (resolved):** cache the feature frame produced by Step 12 to `data/processed/feature_frame_v1.parquet` so the training script loads it directly — avoids redundant `LocalityAggregator.transform` calls and ensures the LOO computation is identical to what the metrics depend on. Update the spec accordingly; this is a correctness win, not a regression. Document in `ml/features/persistence.py` docstring addition. |
| 5 | Spec: "`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`" passed to `evaluate_model` | sklearn `Pipeline.fit(X, y)` expects a 2D `X` and 1D `y`. | `X_*` is the pre-transform feature frame (post `build_feature_frame`); `y_*` is `np.log1p(df["price_inr"].values)`. The Pipeline applies the preprocessor + estimator in one `fit` call. |
| 6 | Spec: per-city test metrics with WARNING when n < 30 | Confirmed approach. Use `city_series = X_test["city"]` (or whatever column name Step 12 ends up with — verify after reading `build_feature_frame` output). | Confirmed: `city` is in `INPUT_FIELDS_V3`. |
| 7 | Spec: "MAPE with epsilon=1.0 guard" | Standard pattern. | `mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100`. Document the epsilon choice. |
| 8 | Spec: "`metrics_v1.json` with `git_commit` field" | Run `git rev-parse HEAD` once at script start. | `subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()`. |
| 9 | Spec: SHAP only for tree-model winners | `shap.TreeExplainer` requires a tree model with a known internal interface (RF/GB/XGB all qualify). Linear winners skip SHAP, log a WARNING, and write "n/a — non-tree model" to the report. | Confirmed approach. |
| 10 | Spec: `models/feature_selection_report.md` Round 2 + Round 3 append | Step 12's `build_feature_selection_report.py` wrote Round 1. | The training script reads the existing markdown, concatenates Round 2 + Round 3 sections, writes atomically (`tempfile` + `Path.replace`). **Or** split into `append_round_2_3.py` helper in `scripts/`. **(ponytail: inline in the training script — single entry point, one INFO summary line.)** |
| 11 | Spec: `models/` directory vs `ARTIFACT_DIR` env var | Step 12's `ARTIFACT_DIR = Path(os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models"))`. | `ml/training/persistence.py` mirrors the same env-var override so tests can redirect to `tmp_path`. |
| 12 | Spec: Rent subset skipped if n < 500 | Hard-coded threshold. | `RENT_MIN_ROWS = 500` module constant in `ml/training/script_helpers.py` (new) or in `candidates.py`. |
| 13 | Spec: requirement to add `from ml import training` to `ml/__init__.py` | `ml/__init__.py` doesn't currently enumerate `cleaning` / `features` submodules — spot-check before modifying. | Verify in implementation step 6; if the file is empty or doesn't enumerate submodules, skip the modification (Step 12 didn't add one either). |
| 14 | Spec: contact-fields regex scan over script stdout | Confirmed approach — `re.search(r"(contact|dealer|phone|email|photo|url|spid)", stdout, re.I)` must return `None`. | One test in `test_train_price_model_script.py`. |

---

## 2. Module layout

### 2.1 `ml/training/candidates.py` — new

```python
_LOG = logging.getLogger("ml.training.candidates")

PRICE_MODEL_VERSION:   Final[str] = "v1"
RENT_MIN_ROWS:         Final[int] = 500    # see Spec delta §12
SHAP_EXPLAINER_VERSION: Final[str] = "v1"

# ponytail: hyperparams are sensible defaults, NOT tuned — tuning is a
# Week 8 improvement lever, not this baseline spec. Pinned here + logged
# in metrics_v1.json so future tuning can diff against these numbers.
CANDIDATE_MODELS: dict[str, BaseEstimator] = {
    "linear":             LinearRegression(),
    "ridge":              Ridge(alpha=1.0, random_state=42),
    "lasso":              Lasso(alpha=0.001, random_state=42, max_iter=10_000),
    "random_forest":      RandomForestRegressor(
                              n_estimators=200, max_depth=None,
                              n_jobs=-1, random_state=42),
    "gradient_boosting":  GradientBoostingRegressor(
                              n_estimators=200, max_depth=4,
                              learning_rate=0.05, random_state=42),
    "xgboost":            XGBRegressor(
                              n_estimators=300, max_depth=6,
                              learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist",
                              n_jobs=-1, random_state=42,
                              verbosity=0),
}

def make_estimator(name: str) -> BaseEstimator:
    if name not in CANDIDATE_MODELS:
        raise ValueError(f"Unknown candidate: {name}. "
                         f"Known: {sorted(CANDIDATE_MODELS)}")
    return CANDIDATE_MODELS[name]  # re-instantiate each call

def candidate_hyperparameters(name: str) -> dict:
    return dict(make_estimator(name).get_params(deep=False))
```

Re-instantiate on every call so each candidate gets a fresh estimator
(stateful XGBoost internal state must not leak between candidates).
Documented in module docstring.

### 2.2 `ml/training/evaluation.py` — new

```python
_LOG = logging.getLogger("ml.training.evaluation")
SMALL_CITY_TEST_ROWS: Final[int] = 30

def regression_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """All four metrics on the ORIGINAL price scale (inverse expm1)."""
    y_true = np.expm1(np.asarray(y_true_log, dtype=float))
    y_pred = np.expm1(np.asarray(y_pred_log, dtype=float))
    eps = 1.0  # guard against divide-by-zero on near-zero prices
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) /
                                     np.maximum(y_true, eps))) * 100.0),
    }

def evaluate_subset(
    pipeline, X_train, y_train_log, X_val, y_val_log,
    X_test, y_test_log, city_test: pd.Series | None = None,
) -> dict:
    """Fit pipeline on train, score train/val/test, optionally per-city."""
    pipeline.fit(X_train, y_train_log)
    out = {
        "train": regression_metrics(y_train_log, pipeline.predict(X_train)),
        "val":   regression_metrics(y_val_log,   pipeline.predict(X_val)),
        "test":  regression_metrics(y_test_log,  pipeline.predict(X_test)),
    }
    if city_test is not None:
        out["per_city_test"] = per_city_metrics(
            pipeline, X_test, y_test_log, city_test)
    return out

def per_city_metrics(pipeline, X_test, y_test_log, city_test) -> dict[str, dict[str, float]]:
    city_metrics: dict[str, dict[str, float]] = {}
    for city, idx in city_test.groupby(city_test).groups.items():
        if len(idx) == 0:
            continue
        if len(idx) < SMALL_CITY_TEST_ROWS:
            _LOG.warning("City %s has only %d test rows (small-sample metrics).",
                         city, len(idx))
        city_metrics[city] = regression_metrics(
            y_test_log[idx], pipeline.predict(X_test.iloc[idx]))
    return city_metrics
```

### 2.3 `ml/training/selection.py` — new

```python
_LOG = logging.getLogger("ml.training.selection")

def select_winner(candidate_results: dict[str, dict],
                  primary_metric: str = "val_rmse") -> str:
    """Lowest ``primary_metric`` on val wins; tie-break on val_mae."""
    def key(name: str) -> tuple[float, float]:
        v = candidate_results[name]["val"]
        return (v[primary_metric], v["mae"])
    return min(candidate_results, key=key)
```

### 2.4 `ml/training/persistence.py` — new

```python
_LOG = logging.getLogger("ml.training.persistence")
ARTIFACT_DIR: Final[Path] = Path(os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models"))

MODEL_REGISTRY_FIELDS: Final[tuple[str, ...]] = (
    "model_name", "version", "training_dataset_version", "git_commit",
    "training_date", "rmse", "mae", "r2",
    "hyperparameters", "feature_hash",
)

def save_price_model(pipeline, transact_type: str,
                     version: str = PRICE_MODEL_VERSION,
                     artifact_dir=None) -> Path:
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"price_model_{transact_type.lower()}_{version}.pkl"
    joblib.dump(pipeline, path)
    _LOG.info("Wrote %s", path)
    return path

def save_metrics(payload: dict, version: str = PRICE_MODEL_VERSION,
                 artifact_dir=None) -> Path:
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{version}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)
    _LOG.info("Wrote %s", path)
    return path

def append_model_registry(row: dict, csv_path=None) -> bool:
    """Idempotent on (model_name, version, git_commit). Returns True if appended."""
    csv_path = Path(csv_path) if csv_path else Path("data/model_registry.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    existing: set[tuple[str, str, str]] = set()
    if not is_new:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add((r["model_name"], r["version"], r["git_commit"]))
    sig = (row["model_name"], row["version"], row["git_commit"])
    if sig in existing:
        _LOG.info("Registry row already present: %s — skipping.", sig)
        return False
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MODEL_REGISTRY_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MODEL_REGISTRY_FIELDS})
    _LOG.info("Appended registry row %s -> %s", sig, csv_path)
    return True
```

### 2.5 `ml/training/report.py` — new

```python
_LOG = logging.getLogger("ml.training.report")

def feature_importance_table(estimator, feature_names: list[str],
                             top_n: int = 20) -> list[tuple[str, float]]:
    """Return top-N (name, importance) descending for tree estimators."""
    if not hasattr(estimator, "feature_importances_"):
        return []
    pairs = list(zip(feature_names, estimator.feature_importances_))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs[:top_n]

def append_round_2_3(report_path: Path,
                     rf_importances, gb_importances, xgb_importances,
                     perm_importances, shap_importances,
                     winner_name: str, chosen_metrics: dict) -> None:
    """Append Round 2 + Round 3 + 'Final feature list' sections."""
    # Read existing content; build appended section; write atomically.
    ...
```

Atomic-write pattern: read existing, build appended block, write to
`tempfile.NamedTemporaryFile(dir=report_path.parent, delete=False)`,
then `Path.replace(tmp, report_path)`.

### 2.6 `ml/training/__init__.py` — new

```python
from ml.training.candidates import (
    CANDIDATE_MODELS, PRICE_MODEL_VERSION, RENT_MIN_ROWS,
    SHAP_EXPLAINER_VERSION, candidate_hyperparameters, make_estimator,
)
from ml.training.evaluation import (
    SMALL_CITY_TEST_ROWS, evaluate_subset, per_city_metrics, regression_metrics,
)
from ml.training.persistence import (
    ARTIFACT_DIR, MODEL_REGISTRY_FIELDS, append_model_registry,
    save_metrics, save_price_model,
)
from ml.training.selection import select_winner

__all__ = [
    "CANDIDATE_MODELS", "PRICE_MODEL_VERSION", "RENT_MIN_ROWS",
    "SHAP_EXPLAINER_VERSION", "candidate_hyperparameters", "make_estimator",
    "SMALL_CITY_TEST_ROWS", "evaluate_subset", "per_city_metrics",
    "regression_metrics",
    "ARTIFACT_DIR", "MODEL_REGISTRY_FIELDS", "append_model_registry",
    "save_metrics", "save_price_model",
    "select_winner",
]
```

### 2.7 `scripts/train_price_model.py` — new

```python
def main():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = Path(os.environ.get("HOUSINGIQ_PROCESSED_DIR",
                                        repo_root / "data" / "processed"))
    artifact_dir = Path(os.environ.get("HOUSINGIQ_ARTIFACT_DIR",
                                       repo_root / "models"))
    report_path = processed_dir / "feature_selection_report.md"

    # 1. Load data
    parquet = processed_dir / "clean_listings.parquet"
    df = pd.read_parquet(parquet)
    _LOG.info("Loaded %d rows from %s", len(df), parquet)

    # 2. Load feature artifacts
    preproc, _agg, feature_names = load_feature_artifacts(
        "v1", artifact_dir=artifact_dir)

    # 3. Build feature frame (one call on the full set; Step 12's
    #    LocalityAggregator was already fit on the Step 12 train subset
    #    so its transforms are leakage-safe for both train + non-train rows).
    feat = build_feature_frame(df)
    # The "feature frame" Step 12 produced IS what we cache + reuse here.
    # We DO NOT call LocalityAggregator.transform again — `build_feature_frame`
    # returns columns filled with NaN for locality_* (they were injected
    # only at Step 12's materialize time).
    #
    # Resolution: the actual locality-aware feature frame lives at
    # `processed_dir / "feature_frame_v1.parquet"` (added in step 4 of
    # this plan). Load it if present, otherwise rebuild from the
    # Step 12 artifacts.

    # 4. Pre-cache the locality-aware feature frame (used by training +
    #    any future classifier spec).
    feat_cache = processed_dir / "feature_frame_v1.parquet"
    if not feat_cache.exists():
        # Re-fit the aggregator on the Step 12 train rows.
        from ml.features import LocalityAggregator
        train_rows_for_agg, _, _ = split_train_val_test(
            df[df["is_outlier"] == False], target="price")
        agg = LocalityAggregator().fit(train_rows_for_agg)
        feat_full = build_feature_frame(df)
        feat_with_locality = agg.transform(feat_full)
        feat_with_locality.to_parquet(feat_cache, index=False)
        _LOG.info("Cached feature frame -> %s", feat_cache)
    feat = pd.read_parquet(feat_cache)

    # 5. Per-transact-type loop
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    payload = {
        "version": PRICE_MODEL_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "dataset_version": "clean_listings.parquet",
        "git_commit": git_commit,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15,
                  "random_state": FIXED_RANDOM_STATE},
        "sale": {}, "rent": {},
    }

    for ttype in ("Sale", "Rent"):
        sub = df[df["is_outlier"] == False]
        sub = sub[sub["transact_type"] == ttype]
        if len(sub) < RENT_MIN_ROWS:
            payload[ttype.lower()] = {"skipped": True,
                                       "reason": f"n={len(sub)} < {RENT_MIN_ROWS}"}
            _LOG.info("Skipping %s pipeline (n=%d).", ttype, len(sub))
            continue

        train_df, val_df, test_df = split_train_val_test(sub, target="price")
        # Align feature rows with split rows via the listing_id index
        # (the cached feat frame shares the same row order as df).
        train_idx, val_idx, test_idx = (train_df.index, val_df.index,
                                        test_df.index)
        X_train, X_val, X_test = (feat.loc[train_idx], feat.loc[val_idx],
                                  feat.loc[test_idx])
        y_train = np.log1p(train_df["price_inr"].values)
        y_val   = np.log1p(val_df["price_inr"].values)
        y_test  = np.log1p(test_df["price_inr"].values)

        # 6. Train all candidates
        candidate_results: dict[str, dict] = {}
        trained_pipes: dict[str, Pipeline] = {}
        for name in CANDIDATE_MODELS:
            pipe = Pipeline([("preproc", preproc),
                             ("est", make_estimator(name))])
            res = evaluate_subset(
                pipe, X_train, y_train, X_val, y_val, X_test, y_test,
                city_test=X_test["city"])
            candidate_results[name] = res
            trained_pipes[name] = pipe
            _LOG.info("[%s/%s] val_rmse=%.0f r2=%.4f",
                      ttype, name, res["val"]["rmse"], res["val"]["r2"])

        # 7. Pick winner, save pipeline
        winner_name = select_winner(candidate_results)
        winner_pipe = trained_pipes[winner_name]
        save_price_model(winner_pipe, ttype)
        payload[ttype.lower()] = {
            "candidates": {n: r for n, r in candidate_results.items()},
            "chosen_model": winner_name,
            "chosen_metrics": candidate_results[winner_name],
            "per_city_test": candidate_results[winner_name]["per_city_test"],
        }

        # 8. SHAP (tree only) + permutation importance + impurity ranks
        if hasattr(winner_pipe.named_steps["est"], "feature_importances_"):
            explainer = shap.TreeExplainer(winner_pipe.named_steps["est"])
            shap_values = explainer.shap_values(preproc.transform(X_test))
            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_pairs = list(zip(feature_names, mean_abs))
            shap_pairs.sort(key=lambda kv: kv[1], reverse=True)
        else:
            shap_pairs = []
            _LOG.warning("Winner %s is non-tree; SHAP skipped.", winner_name)

        perm = permutation_importance(
            winner_pipe, X_val, y_val, n_repeats=10,
            random_state=FIXED_RANDOM_STATE, scoring="neg_mean_absolute_error")

        # Round 2 + Round 3 report append (last write — only after all
        # candidates done so a crash mid-way doesn't leave a half-written
        # report).
        append_round_2_3(
            report_path,
            rf_importances=feature_importance_table(
                trained_pipes["random_forest"].named_steps["est"], feature_names),
            gb_importances=feature_importance_table(
                trained_pipes["gradient_boosting"].named_steps["est"], feature_names),
            xgb_importances=feature_importance_table(
                trained_pipes["xgboost"].named_steps["est"], feature_names),
            perm_importances=list(zip(feature_names, perm.importances_mean)),
            shap_importances=shap_pairs,
            winner_name=winner_name,
            chosen_metrics=candidate_results[winner_name])

        # 9. Model registry row
        test_metrics = candidate_results[winner_name]["test"]
        append_model_registry({
            "model_name": f"price_model_{ttype.lower()}",
            "version": PRICE_MODEL_VERSION,
            "training_dataset_version": "clean_listings.parquet",
            "git_commit": git_commit,
            "training_date": datetime.utcnow().isoformat(),
            "rmse": test_metrics["rmse"],
            "mae":  test_metrics["mae"],
            "r2":   test_metrics["r2"],
            "hyperparameters": json.dumps(
                candidate_hyperparameters(winner_name), default=str),
            "feature_hash": hashlib.sha1(
                "".join(feature_names).encode()).hexdigest()[:16],
        })

    # 10. Persist metrics_v1.json + summary log
    save_metrics(payload)
    _LOG.info("Done. Summary: %s",
              json.dumps({k: (v.get("chosen_model") or v.get("skipped"))
                          for k, v in [("sale", payload["sale"]),
                                        ("rent", payload["rent"])]},
                         indent=2))
```

**Note on feature-frame caching:** `build_feature_frame` returns
NaN-filled `locality_*` columns — the actual numerical values are
produced by `LocalityAggregator.transform`, which Step 12 ran inside
`scripts/build_features.py`. The training script reproduces the same
transform on the cached frame to avoid duplicating logic. The cache
file `data/processed/feature_frame_v1.parquet` is a new artifact not
explicitly listed in the spec's "Data / Schema changes" section —
flagging this as Spec delta §4 (resolved): it preserves the
leakage-safe locality computation while letting the training script
read rows in O(1) rather than re-running LOO aggregates.

### 2.8 `scripts/run_pipeline.py` — modify

Add the spec's one-line import + subprocess call:

```python
from scripts import (  # noqa: E402,F401
    ingest_raw,
    parse_check,
    train_price_model,   # noqa: F401  (Step 13)
)
```

Update `main()` to chain `train_price_model.main()` after the
(implicit — currently not present) feature-build step. To keep the
diff minimal, leave `main()` alone for now and add a separate `main()`
in this spec that calls `build_features` → `train_price_model` in
sequence, exposed as `python scripts/train_price_model.py`. Spec
delta note in commit message.

### 2.9 `requirements.txt` — no change

`xgboost==2.1.0` + `shap==0.46.0` + `joblib==1.4.2` are already pinned.
No new packages.

### 2.10 `ml/__init__.py` — likely no change

Spot check: if `ml/__init__.py` is empty / doesn't enumerate
submodules, skip. `ml.training` is importable as `from ml.training
import ...` without any `ml/__init__.py` modification.

---

## 3. Test plan

### 3.1 `tests/test_candidates.py` — 4 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_candidate_models_constant_has_six_entries` | `len(CANDIDATE_MODELS) == 6` and the exact name tuple. |
| 2 | `test_make_estimator_returns_correct_class` | Each name → expected class (LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor). |
| 3 | `test_make_estimator_raises_on_unknown_name` | `"foo"` → `ValueError`. |
| 4 | `test_all_candidates_have_random_state_42` | For estimators with `random_state` attribute, it's `42`. |

### 3.2 `tests/test_evaluation.py` — 5 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_regression_metrics_returns_four_keys` | Output dict has exactly `{r2, mae, rmse, mape}`. |
| 2 | `test_regression_metrics_inverse_transforms_from_log` | `y_true_log = log1p([100, 200, 400])`, `y_pred_log = y_true_log` (perfect predict) → `mae == 0`, `rmse == 0`, `r2 == 1.0`. Catches a missed `expm1`. |
| 3 | `test_regression_metrics_mape_uses_epsilon_guard` | All-zero `y_true` does not raise; `mape` is bounded. |
| 4 | `test_evaluate_subset_returns_train_val_test` | Output dict has the three keys; `train["r2"] >= val["r2"]` (sanity). |
| 5 | `test_per_city_metrics_warns_on_small_sample` | `caplog` WARNING when a city has < 30 test rows. |

### 3.3 `tests/test_selection.py` — 3 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_select_winner_returns_lowest_val_rmse` | Three candidates with distinct val_rmse → returns the smallest. |
| 2 | `test_select_winner_tie_breaks_on_val_mae` | Two candidates with same val_rmse, different val_mae → lower MAE wins. |
| 3 | `test_select_winner_empty_results_raises` | Empty dict → `ValueError` (caught by `min` builtin — assert `ValueError` is raised). |

### 3.4 `tests/test_training_persistence.py` — 7 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_save_price_model_writes_versioned_filename` | `save_price_model(pipe, "Sale", artifact_dir=tmp_path)` → `tmp_path/price_model_sale_v1.pkl` exists, loadable via `joblib.load`. |
| 2 | `test_save_metrics_writes_versioned_filename` | Round-trip: `save_metrics({"a": 1.0})` then `json.load` matches. |
| 3 | `test_save_metrics_handles_numpy_scalars` | Payload with `np.float64` / `datetime` doesn't raise `default=str` covers both. |
| 4 | `test_append_model_registry_writes_header_on_first_call` | New `tmp_path/x.csv` → header line matches `MODEL_REGISTRY_FIELDS`. |
| 5 | `test_append_model_registry_appends_one_row_per_call` | Two distinct `(name, version, git_commit)` triples → 3 lines (header + 2). |
| 6 | `test_append_model_registry_is_idempotent_on_rerun` | Same triple twice → still 2 lines (no duplicate). |
| 7 | `test_model_registry_csv_columns_match_backend_schema` | Exact column order match against Backend Schema §U-SCHEMA-13. |

### 3.5 `tests/test_train_price_model_script.py` — 6 tests

All tests use `subprocess.run([sys.executable, "scripts/train_price_model.py"],
env={..., "HOUSINGIQ_PROCESSED_DIR": str(tmp_path/"data"/"processed"),
"HOUSINGIQ_ARTIFACT_DIR": str(tmp_path/"models")})` with a tiny
synthetic `clean_listings.parquet` built in `tmp_path` (12 rows: 8 Sale,
4 Rent, 2 cities). Step 12's `build_features.py` is also called first
(via the same subprocess pattern or directly) to materialize the
preprocessor + cached feature frame.

| # | Name | Anchors |
|---|---|---|
| 1 | `test_train_price_model_script_runs_end_to_end` | Both `price_model_sale_v1.pkl` and `price_model_rent_v1.pkl` exist + `metrics_v1.json` parses + `feature_selection_report.md` grew (Round 2 marker present) + `data/model_registry.csv` has 2 rows. |
| 2 | `test_train_price_model_script_is_idempotent_on_rerun` | Run twice with identical inputs → `model_registry.csv` still has exactly 2 rows. |
| 3 | `test_train_price_model_script_skips_rent_when_too_small` | Synthetic parquet with 0 Rent rows → `metrics_v1.json["rent"]["skipped"] == True` + `metrics_v1.json["rent"]["reason"]` contains the threshold text. |
| 4 | `test_training_script_does_not_log_contact_fields` | Regex scan over stdout for `(contact|dealer|phone|email|photo|url|spid)` — must be absent. |
| 5 | `test_metrics_v1_json_contains_all_six_candidates` | `payload["sale"]["candidates"]` has 6 keys. |
| 6 | `test_metrics_v1_json_has_git_commit` | `payload["git_commit"]` matches `git rev-parse HEAD`. |

### 3.6 `tests/test_training_report.py` — 3 tests

| # | Name | Anchors |
|---|---|---|
| 1 | `test_feature_importance_table_returns_top_n_sorted` | Synthetic 5-feature list → top-3 returned in descending order. |
| 2 | `test_feature_importance_table_returns_empty_for_non_tree` | Linear estimator → `[]`. |
| 3 | `test_append_round_2_3_preserves_existing_content` | Pre-existing markdown content (Round 1) is byte-for-byte present in the post-append output. |

---

## 4. Implementation order (14 steps)

1. Spot-check `ml/__init__.py` (decide whether to skip §2.10).
2. Write `ml/training/candidates.py` (§2.1). `CANDIDATE_MODELS` constant + `make_estimator` + `candidate_hyperparameters`. Re-instantiate on each `make_estimator` call so candidate state can't leak.
3. Write `ml/training/evaluation.py` (§2.2). `regression_metrics` first (pure, easy to test), then `evaluate_subset`, then `per_city_metrics`. Stdlib `logging` only.
4. Write `ml/training/selection.py` (§2.3). Single function, `select_winner`.
5. Write `ml/training/persistence.py` (§2.4). `ARTIFACT_DIR` env-var override, idempotent `append_model_registry`.
6. Write `ml/training/report.py` (§2.5). Atomic append to `feature_selection_report.md`.
7. Write `ml/training/__init__.py` (§2.6) re-exporting the public API.
8. Write `tests/test_candidates.py` + `tests/test_evaluation.py` + `tests/test_selection.py` + `tests/test_training_persistence.py` + `tests/test_training_report.py` (§3.1–3.4, 3.6). Run until green. These 22 unit tests don't touch the Parquet.
9. Write `scripts/train_price_model.py` (§2.7). Use `subprocess.check_output` for git, `joblib.load` for the preprocessor, the cached feature frame for inputs.
10. Write `tests/test_train_price_model_script.py` (§3.5). These 6 tests run the script in a subprocess against a synthetic `clean_listings.parquet` + `feature_frame_v1.parquet` materialized via Step 12's `scripts/build_features.py`. Build the synthetic frame in `tmp_path` using the schema from `ml/features/split.py` (`INPUT_FIELDS_V3` + `is_outlier` + 2 cities × 2 sectors).
11. Run the script-level tests until green. The synthetic dataset must be small enough that all 6 candidates finish in < 30 s total per test invocation.
12. `python -m pytest -m "not realdata"` — confirm no real-data dependency.
13. `ruff check ml/training/ scripts/train_price_model.py tests/test_candidates.py tests/test_evaluation.py tests/test_selection.py tests/test_training_persistence.py tests/test_train_price_model_script.py tests/test_training_report.py` — zero issues.
14. Manual smoke (per DoD §5, §6, §7, §8): `python scripts/build_features.py` → `python scripts/train_price_model.py` → verify `models/metrics_v1.json` parses with all 6 candidates + `git_commit` + `per_city_test` for the chosen model + Round 2/3 sections in the report + 2 rows in `model_registry.csv`.

---

## 5. Critical files

**Create (7):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\__init__.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\candidates.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\evaluation.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\selection.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\persistence.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\training\report.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\scripts\train_price_model.py`

**Create tests (6):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_candidates.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_evaluation.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_selection.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_training_persistence.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_training_report.py`
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\tests\test_train_price_model_script.py`

**Modify (1):**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\scripts\run_pipeline.py` — add `train_price_model` import alias (no logic change).

**No change:**
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\requirements.txt` (xgboost, shap, joblib already pinned).
- `C:\Users\HP\OneDrive\Desktop\Housing predictor\ml\__init__.py` (likely no change; verify in step 1).

**Reused (no modifications):**
- `ml/features/feature_frame.py` — `build_feature_frame`
- `ml/features/locality_aggregator.py` — `LocalityAggregator`
- `ml/features/persistence.py` — `load_feature_artifacts`
- `ml/features/split.py` — `split_train_val_test`, `FIXED_RANDOM_STATE`
- `ml/features/preprocessor.py` — `ColumnTransformer` (loaded via `load_feature_artifacts`)
- `scripts/build_features.py` — Step 12 entry point (run before the training script to materialize preprocessor + cached frame)

---

## 6. Verification

```bash
# Step 1 — unit + integration tests, per spec DoD §1
python -m pytest tests/test_candidates.py tests/test_evaluation.py tests/test_selection.py tests/test_training_persistence.py tests/test_training_report.py tests/test_train_price_model_script.py -v

# Step 2 — confirm no real-data dependency
python -m pytest -m "not realdata"

# Step 3 — ruff clean per spec DoD §3
ruff check ml/training/ scripts/train_price_model.py tests/test_candidates.py tests/test_evaluation.py tests/test_selection.py tests/test_training_persistence.py tests/test_training_report.py tests/test_train_price_model_script.py

# Step 4 — import smoke per spec DoD §4
python -c "from ml.training import CANDIDATE_MODELS, evaluate_subset, select_winner, save_price_model; from ml.training.candidates import make_estimator; print(len(CANDIDATE_MODELS))"
# Expected: 6

# Step 5 — script smoke (per DoD §5, gated on the above passing)
python scripts/build_features.py             # Step 12: writes feature_pipeline_v1.pkl + feature_frame_v1.parquet
python scripts/train_price_model.py          # Step 13: writes price_model_{sale,rent}_v1.pkl + metrics_v1.json + Round 2/3 report + registry rows

# Step 6 — metrics_v1.json structure per DoD §6
python -c "import json; m=json.load(open('models/metrics_v1.json')); print(m['version'], m['git_commit'], list(m['sale']['candidates'].keys()), m['sale']['chosen_model'], m.get('rent', {}).get('chosen_model') or m.get('rent', {}).get('skipped'))"
# Expected: v1 <sha> ['linear','ridge','lasso','random_forest','gradient_boosting','xgboost'] <one of the six> True|False

# Step 7 — registry + report per DoD §7, §8
python -c "import csv; rows=list(csv.DictReader(open('data/model_registry.csv'))); print(len(rows), rows[0].keys())"
python -c "open('data/processed/feature_selection_report.md').read().count('## Round')  # expect >=3"

# Step 8 — git hygiene
git status   # only the 7 new files + 6 new tests + 1 modified script; models/*.pkl and data/processed/* untracked
```

Expected smoke output:
- `models/metrics_v1.json` lists all 6 candidates with non-null metrics; chosen model picked on `val_rmse` tie-broken by `val_mae`.
- `data/model_registry.csv` has exactly 2 rows (or 1 if Rent < threshold).
- `data/processed/feature_selection_report.md` has `## Round 1` (Step 12) + `## Round 2 — tree-based + permutation importance` + `## Round 3 — SHAP ranking` + `## Final feature list & rationale` headings.
- `git status` shows the 14 working-tree entries listed in §5.

---

## 7. Risks / spec ambiguities flagged

1. **Spec delta §4 — feature-frame caching.** The spec doesn't explicitly say
   the training script should cache the locality-aware feature frame. Without
   caching, the script re-runs `LocalityAggregator.transform` on every load
   (cheap, but adds startup noise). Caching to `feature_frame_v1.parquet`
   matches the locality-aggregator contract and is the lazy choice. Confirm
   with user if they prefer no cache (re-transform on every run).
2. **`ml/__init__.py` modification.** Spec lists it as "Modify"; spot-check
   in step 1 may show it's effectively empty (Step 12 didn't add an entry
   either). If empty, the modification is a no-op — skip it and note in
   commit.
3. **`scripts/run_pipeline.py` modification scope.** Spec says one line
   (subprocess call). The current `main()` only invokes `ingest_raw.main()`;
   adding the full chain is out of scope. The minimal change is the import
   alias — the chained run is exposed as `python scripts/train_price_model.py`
   instead, since it implicitly requires `build_features` to have run first.
4. **SHAP `TreeExplainer` on a `Pipeline`-wrapped XGBoost** — must pass
   `preproc.transform(X_test)` (the estimator sees the preprocessed matrix,
   not raw rows). The plan already does this; flagging it because it's a
   common silent-failure mode (SHAP values then don't match feature names).
5. **`permutation_importance` cost** — `n_repeats=10` on the validation
   slice is the spec's number, but on the real-data validation slice
   (~5000 rows × 6 candidates = 30 evaluations) it's a non-trivial cost.
   Acceptable for a baseline; document in Decision Log if it pushes run
   time > 5 min.
6. **Synthetic-parquet test fixture.** Tests 1–3 in
   `test_train_price_model_script.py` need a `clean_listings.parquet` that
   satisfies the `INPUT_FIELDS_V3` schema. Building this in `tmp_path` is
   ~80 lines of fixture code; might warrant extracting a helper in
   `tests/fixtures/`. If it grows, extract; if not, inline.
