# Spec: Price Model Evaluation Protocol

## Overview
Make the fixed 70/15/15 evaluation protocol (`random_state=42`) from
`02-TRD.md` §10 and `08-RULES.md` §2.1 the **single, authoritative gate**
that every trained price model must clear before being copied into
`models/` for serving or referenced by the FastAPI `/predict` route.
Specs 13 and 14 currently each ship their own training script with
inline metric computation; this spec consolidates that logic into one
`ml/evaluation/` package — a pure, reusable protocol — so the same code
path enforces the protocol for v1, v2, and any future vN. It also
introduces one CLI gate (`scripts/evaluate_price_model.py`) the
`housingiq-ml-evaluator` agent (and humans) run to certify a model,
plus a thin regression test that any drift in protocol behavior (e.g.,
a leaked `random_state`, a wrong split ratio, a missing metric) fails
CI immediately. Module: **price-prediction**.

## Depends on
- **Step 13** — `13-baseline-regression-model-training` — supplies the
  `ml/training/evaluation.py` + `persistence.py` + `candidates.py`
  surface this spec consumes (kept unchanged, no behavior edits). The
  v1 baseline pipeline (`models/price_model_sale_v1.pkl` /
  `metrics_v1.json`) is the first artifact this spec certifies.
- **Step 14** — `14-xgboost-lightgbm-price-model-training` — supplies
  `ml/training/levers/` (stacking, Optuna, geospatial, target
  encoding) + `ml/training/report.py`. The v2 candidate models are
  the second set of artifacts this spec certifies.
- **Step 12** — `12-feature-engineering-price-model` — supplies
  `ml/features/split.py` (70/15/15 with `random_state=42`) and
  `ml/features/feature_frame.py` + `ml/features/persistence.py` for
  the fitted preprocessor + locality aggregator. The protocol
  package reuses `split_train_val_test` unchanged so v1 + v2 + vN
  all split identically on the same input.
- **Step 07** — `07-clean-listings-parquet-pipeline` — supplies
  `data/processed/clean_listings.parquet`, the canonical input.
- **Step 06** — `06-data-deduplication-and-outlier-flagging` —
  supplies the `is_outlier` filter the protocol enforces.
- **Step 11** — `11-price-prediction-input-schema-v3` — locks the
  16-field input contract; the protocol's `evaluate()` validates
  that every required field is present in the input frame before
  scoring.
- **`02-TRD.md` §10** — the protocol's source of truth (split
  ratios, `random_state=42`, the four headline metrics on the
  original ₹ scale, train-only feature aggregation).
- **`08-RULES.md` §2.1** — fixed evaluation protocol binding rules.
- **`08-RULES.md` §5.4** — `random_state=42` everywhere.
- **`08-RULES.md` §2.4** — production model = exact `Pipeline`
  object used at evaluation, no re-implementation.
- **`08-RULES.md` §2.5** — versioned artifacts, never overwritten.
- **`08-RULES.md` §9.2** — honest logging of shortfalls, no
  aspirational claims.
- **`05-BACKEND-SCHEMA.md` §6** — model artifact filenames +
  §U-SCHEMA-13 — `model_registry.csv` schema.
- **Skill: `model-evaluation-protocol`** — the project's
  documented procedure for certifying models; this spec
  consolidates that procedure into code.

## Routes / Endpoints
No new routes/endpoints. This spec is offline tooling only:
- A new Python package `ml/evaluation/` (pure functions + one
  `evaluate()` entry point).
- A new CLI `scripts/evaluate_price_model.py` (manual + agent-
  invokable gate).
- New pytest tests + one CLI integration test.
The FastAPI route layer that consumes the certified model is a
follow-on spec.

## Data / Schema changes
- **Read** `data/processed/clean_listings.parquet` (Step 07).
- **Read** `models/feature_pipeline_v1.pkl` (Step 12 — fitted
  preprocessor + locality aggregator).
- **Read** `models/feature_list_v1.json` (Step 12).
- **Read** `models/price_model_{sale,rent}_v{n}.pkl` and
  `models/metrics_v{n}.json` for any version the user/agent
  passes to the gate.
- **Write** `models/evaluation_report_{version}.json` — one
  per certification run, with the same schema as `metrics_v{n}.json`
  + a `protocol` block pinning the evaluated split, `random_state`,
  metric definitions, dataset version, git commit, evaluator
  version, and a `passed` boolean. Filename is pinned by Rules
  §2.5 (versioned, never overwritten — `v1` report is
  preserved; a re-run writes `evaluation_report_v1_rerun_2026-08-15.json`
  with a timestamp suffix if the existing file is to be preserved).
- **Append** `data/processed/feature_selection_report.md` — one
  new section "Protocol Certification" per certified version
  with the pass/fail result + the four headline metrics + the
  threshold check + per-city R². Append-only.
- **Append** `data/model_registry.csv` — unchanged from Spec 13:
  the registry row is added at training time, not at evaluation
  time. The evaluation gate is **read-only** on
  `model_registry.csv` (it does not create new registry rows; it
  reads them to confirm the certified version exists). If the
  registry row is missing, the gate fails with a clear message
  ("Run the training script first; the evaluation gate only
  certifies models already trained and registered").
- **No writes** to `data/raw/`, to
  `data/processed/clean_listings.parquet`, to
  `data/processed/analytics_cache/`, or to `data/app.db`.
- **No new model artifacts** — the gate certifies existing
  artifacts; it does not train or persist a new model.

## Templates / UI
None. The evaluation gate is offline tooling — no Flask
templates, no static assets, no HTML. Its output is the
`evaluation_report_{version}.json` file + the appended
`feature_selection_report.md` section + a stdout summary line,
all of which the `/evaluate-model` skill consumes for the
human/agent reviewer.

## Files to change / Files to create

**Create:**
- `ml/evaluation/` — new sub-package, one file per concern, so
  the protocol is testable in isolation from any training
  pipeline. Re-exports from `ml/evaluation/__init__.py`.
  - `__init__.py` — empty; re-exports the four public symbols:
    `PROTOCOL_VERSION`, `evaluate`, `EvaluationResult`,
    `protocol_thresholds`.
  - `protocol.py` — the pinned protocol constants. Public API:
    - `PROTOCOL_VERSION: str = "1.0.0"` — semver-pinned. The
      `evaluate()` entry point emits this into every
      `evaluation_report_{version}.json`'s `protocol` block so a
      reviewer can confirm what protocol was applied.
    - `SPLIT_RATIOS: dict[str, float] = {"train": 0.70,
      "val": 0.15, "test": 0.15}` — pinned literal. Any drift
      in the ratio fails a test.
    - `RANDOM_STATE: int = 42` — pinned literal. Any drift
      fails a test.
    - `METRIC_NAMES: tuple[str, ...] = ("r2", "mae", "rmse",
      "mape")` — the four headline metrics the protocol
      requires (per `02-TRD.md` §10 + `08-RULES.md` §2.1).
    - `protocol_thresholds: dict[str, float]` — the pass/fail
      thresholds from `02-TRD.md` §10 / PRD §3:
      - `"r2_min": 0.80` (PRD: ≥ 0.80, stretch 0.85)
      - `"r2_stretch": 0.85`
      - `"mae_pct_within_15_at_least": 0.70` (PRD: MAE
        within ±15% of actual price for 70% of test listings).
      - `"p95_latency_ms_max": 300.0` (PRD: `/predict` p95
        latency < 300ms). Latency is measured by the gate
        only when run against a served FastAPI instance; for
        offline scoring the gate emits `latency_p95_ms: null`
        and a note that latency is measured at serve time,
        not training time.
      - `"rent_min_rows": 500` — same constant as Specs 13/14;
      re-imported from `ml/training/persistence.py` so the
      threshold lives in exactly one place.
    - `PROTOCOL_DOC_PATH: str = "docs/02-TRD.md"` — the source-
      of-truth doc the protocol claims to mirror. Used in
      error messages and the report's `protocol.source_doc`
      field.
  - `splits.py` — the split helper that enforces the protocol's
    split contract. Public API:
    - `protocol_split(df: pd.DataFrame, target: str = "price",
      random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame,
      pd.DataFrame, pd.DataFrame]` — thin wrapper over
      `ml.features.split.split_train_val_test` that **asserts**
      the returned split ratios match `SPLIT_RATIOS` (within
      ±1 row, to tolerate floor rounding on small synthetic
      fixtures) and that `random_state == RANDOM_STATE`. The
      assertion is a hard `pytest.fail` in tests and a logged
      `ERROR` + raise in production. This is the gate's
      enforcement hook — Specs 13/14's training scripts can
      continue to use the raw `ml.features.split` helper (no
      behavior change), but the gate always uses
      `protocol_split` so the protocol is enforced at
      certification time even if a training script's
      `random_state` drifts.
  - `scoring.py` — the metric-scoring layer. Public API:
    - `score_predictions(y_true: np.ndarray, y_pred_log: np.ndarray,
      invert_log: bool = True) -> dict[str, float]` — pure
      function; computes `{r2, mae, rmse, mape}` on the
      original ₹ scale via `expm1` (per `08-RULES.md` §2.1)
      when `invert_log=True`, else on the log scale. Returns
      the dict in `METRIC_NAMES` key order.
    - `within_tolerance_pct(y_true: np.ndarray, y_pred: np.ndarray,
      tolerance: float = 0.15) -> float` — returns the
      fraction of test rows where `|y_pred - y_true| / y_true
      <= tolerance`. Used to score the PRD "MAE within ±15% for
      70% of test listings" threshold.
    - `per_city_metrics(...)` — re-imported from
      `ml.training.evaluation` so the gate and the training
      scripts agree on per-city math (no second copy).
  - `gate.py` — the `evaluate()` entry point + result model.
    Public API:
    - `@dataclass(frozen=True) class ProtocolThresholds`
      mirroring `protocol_thresholds`.
    - `@dataclass(frozen=True) class EvaluationResult` —
      fields:
      - `version: str` — model version certified (e.g. `"v1"`,
        `"v2"`).
      - `transact_type: str` — `"sale"` or `"rent"`.
      - `protocol_version: str` — from `PROTOCOL_VERSION`.
      - `dataset_version: str` — from the parquet's filename
        + sha1 of the first 1 MB (cheap content fingerprint;
        per Rules §3 every derived table states its source).
      - `git_commit: str` — `git rev-parse HEAD`.
      - `split_sizes: dict[str, int]` — `{train, val, test}`
        row counts.
      - `metrics: dict[str, float]` — the four headline
        metrics.
      - `per_city_test: dict[str, dict[str, float]]` — per
        city on the test slice.
      - `within_tol_15_pct: float` — fraction of test rows
        within ±15%.
      - `latency_p95_ms: float | None` — null in offline mode;
        populated only when run against a live FastAPI
        instance.
      - `thresholds_passed: dict[str, bool]` — per threshold,
        `True` iff the metric cleared the gate.
      - `overall_passed: bool` — `True` iff every threshold
        passed. The model is **certified** iff
        `overall_passed == True`.
      - `evaluated_at: str` — ISO timestamp.
      - `evaluator_version: str` — this package's version
        (from `ml/evaluation/__init__.py`'s `__version__`).
    - `evaluate(model_path: Path | str, version: str,
      transact_type: str, processed_dir: Path | str = ...,
      models_dir: Path | str = ...,
      parquet_path: Path | str = ...,
      fastapi_url: str | None = None) -> EvaluationResult` —
      the single entry point. Steps:
      1. Load `clean_listings.parquet` from
         `processed_dir / "clean_listings.parquet"`.
      2. Filter to `is_outlier == False`.
      3. Split per `transact_type` (sale/rent). If Rent
         subset has n < `RENT_MIN_ROWS`, return an
         `EvaluationResult` with
         `overall_passed=False`, `thresholds_passed={"rent_min_rows":
         False}`, and a note in `metrics` (`{"skipped": true,
         "reason": "n=X < 500"}`). Same shape as the
         training scripts' rent-skip behavior (consistency
         with Specs 13/14).
      4. For each subset with n ≥ 500: call
         `protocol_split(df, target="price")` to enforce
         the split contract.
      5. Load the fitted preprocessor + locality aggregator
         from `models/feature_pipeline_v1.pkl` via
         `ml.features.persistence.load_feature_artifacts("v1")`.
         Assert the loaded object equals the in-memory
         reference (Rules §2.4: production model is the
         exact object used at evaluation). If it isn't,
         raise — the gate's `overall_passed` is `False` with
         a clear `preprocessor_drift` reason.
      6. Load `joblib.load(model_path)` — the model artifact
         to certify. Type-check it has `predict()` and (if
         tree-based) `feature_importances_`.
      7. Score `X_train` / `X_val` / `X_test` against
         `y_train` / `y_val` / `y_test`, calling
         `score_predictions` with `invert_log=True`.
      8. Compute `per_city_test` via the re-imported helper.
      9. Compute `within_tol_15_pct` on the test slice.
      10. Optionally measure latency: if `fastapi_url` is
          provided, POST `/predict` with a 50-row random
          sample (per city) and record p50/p95. Else
          `latency_p95_ms = None`.
      11. Check each threshold in `protocol_thresholds`
          against the measured value; populate
          `thresholds_passed` and `overall_passed`.
      12. Return `EvaluationResult`. The function is pure
          with respect to its inputs — it does not write
          files. Persistence is the caller's job (the CLI
          below).
    - `format_summary(result: EvaluationResult) -> str` —
      one-line stdout summary used by the CLI (and
      `/evaluate-model` skill): `"[PASS|FAIL] v{sale,rent}_v{N}
      R²={r2:.4f} (≥0.80) MAE=₹{mae:.0f} within±15%={pct:.1%}
      (≥70%) latency_p95={lat}ms (<300ms)"`.
  - `report.py` — the report writer. Public API:
    - `write_evaluation_report(result: EvaluationResult,
      out_dir: Path | str) -> Path` — JSON-dumps the result
      to `out_dir / f"evaluation_report_{result.version}_{result.transact_type}.json"`.
      Idempotent re-run with the same inputs writes the same
      content. Re-run with different `evaluated_at` writes a
      timestamp-suffixed file
      (`..._v1_sale_rerun_2026-08-15T14-22-03.json`) per Rules
      §2.5.
    - `append_protocol_section(result: EvaluationResult,
      report_path: Path | str) -> None` — appends a
      "Protocol Certification" section to
      `data/processed/feature_selection_report.md` with the
      pass/fail result + per-city R² + threshold check. The
      file is opened in append mode; never overwrites prior
      content.
  - `__init__.py` — `__version__ = "1.0.0"` + re-exports the
    four public symbols from the sibling modules.

- `scripts/evaluate_price_model.py` — the CLI gate. Invoked as
  `python scripts/evaluate_price_model.py --version v1
  --transact-type sale [--transact-type rent] [--fastapi-url
  http://localhost:8000] [--processed-dir ...] [--models-dir
  ...]`. Steps:
  1. Parse args (`argparse`, one positional group per
     `transact_type`).
  2. For each `transact_type`: call `evaluate(...)` to get an
     `EvaluationResult`.
  3. Write `evaluation_report_{version}_{transact_type}.json`
     via `write_evaluation_report`.
  4. Append the "Protocol Certification" section to
     `data/processed/feature_selection_report.md` via
     `append_protocol_section`.
  5. Print the `format_summary` line per transact_type.
  6. Exit 0 if every result's `overall_passed == True`;
     exit 1 otherwise. CI / `/evaluate-model` callers check
     the exit code.

- `tests/test_protocol.py` — pytest tests for the pinned
  constants + the `protocol_split` enforcement. Required
  tests (exact names):
  - `test_protocol_version_is_pinned` — `PROTOCOL_VERSION
    == "1.0.0"`.
  - `test_split_ratios_match_trd_section_10` — exact literal
    match.
  - `test_random_state_is_42` — exact literal match.
  - `test_metric_names_match_rules_section_2_1` — exact
    tuple match.
  - `test_protocol_split_returns_three_frames` — sanity.
  - `test_protocol_split_enforces_ratios_within_one_row` —
    synthetic 1000-row fixture; assert actual ratios within
    ±1 row of pinned.
  - `test_protocol_split_asserts_random_state` — passing
    `random_state=7` raises (logged ERROR + raise).

- `tests/test_scoring.py` — pytest tests for
  `score_predictions` + `within_tolerance_pct`. Required
  tests (exact names):
  - `test_score_predictions_returns_four_keys_in_pinned_order`
    — dict key order == `METRIC_NAMES`.
  - `test_score_predictions_inverts_log_target` — synthetic
    log-scale targets; assert the returned MAE equals the
    original-scale MAE within ±0.5% (catches a missed
    `expm1`).
  - `test_within_tolerance_pct_returns_fraction` — synthetic
    y_true/y_pred; assert the returned fraction is between
    0 and 1 inclusive.
  - `test_within_tolerance_pct_is_zero_when_all_outside_band`
    — perfect inverse predictions → fraction = 0.
  - `test_within_tolerance_pct_is_one_when_all_inside_band`
    — perfect predictions → fraction = 1.

- `tests/test_gate.py` — pytest tests for `evaluate()`. Uses
  a tiny synthetic `clean_listings.parquet` (built in
  `tmp_path` by the test) + a tiny fitted preprocessor +
  tiny trained XGBRegressor. Required tests (exact names):
  - `test_evaluate_returns_evaluation_result_dataclass` —
    type check.
  - `test_evaluate_overall_passed_false_when_r2_below_threshold`
    — synthesize a model whose test R² < 0.80 by training
    on noise-only targets; assert `overall_passed=False`
    + `thresholds_passed["r2_min"] == False`.
  - `test_evaluate_overall_passed_true_when_all_thresholds_met`
    — synthesize a near-perfect model (XGB on a tight
    synthetic linear target); assert
    `overall_passed=True`.
  - `test_evaluate_skips_rent_when_too_small` — 0 Rent rows
    → `overall_passed=False` with
    `thresholds_passed["rent_min_rows"] == False` +
    `metrics["skipped"] == True`.
  - `test_evaluate_rejects_preprocessor_drift` — fit one
    preprocessor on the fixture, load a *different* one
    from disk; assert `evaluate()` raises with a
    `preprocessor_drift` message.
  - `test_evaluate_records_protocol_version_in_result` —
    `result.protocol_version == PROTOCOL_VERSION`.
  - `test_evaluate_records_git_commit_in_result` —
    `result.git_commit == git rev-parse HEAD`.

- `tests/test_report_writer.py` — pytest tests for
  `write_evaluation_report` + `append_protocol_section`.
  Required tests (exact names):
  - `test_write_evaluation_report_writes_versioned_filename`
    — saves to `tmp_path`, asserts
    `evaluation_report_v1_sale.json` exists.
  - `test_write_evaluation_report_rerun_uses_timestamp_suffix`
    — write twice with a mocked timestamp; assert
    second file is `..._v1_sale_rerun_<timestamp>.json`.
  - `test_append_protocol_section_appends_not_overwrites`
    — pre-populate the report with a sentinel line, call
    `append_protocol_section`, assert sentinel is still
    present (file was opened in append mode).

- `tests/test_evaluate_price_model_cli.py` — pytest test for
  the CLI gate. Required tests (exact names):
  - `test_cli_runs_end_to_end_on_synthetic_artifacts` —
    build a tiny synthetic `clean_listings.parquet` +
    fitted v1 preprocessor + trained XGBRegressor +
    `metrics_v1.json` in `tmp_path`, invoke the CLI via
    `subprocess.run` with `--processed-dir` +
    `--models-dir` pointing at the fixture, assert exit
    code 0 or 1 (depending on the synthetic model's
    quality) + the JSON report file lands in
    `models/`.
  - `test_cli_exits_nonzero_when_r2_below_threshold` —
    synthetic noise-only model → CLI exits 1 + stdout
    summary line begins with `[FAIL]`.
  - `test_cli_exits_zero_when_all_thresholds_met` —
    synthetic near-perfect model → CLI exits 0 + stdout
    summary begins with `[PASS]`.

**Modify:**
- `scripts/run_pipeline.py` — append one CLI gate line after
  the v2-training line:
  ```python
  subprocess.run(
      [sys.executable, "scripts/evaluate_price_model.py",
       "--version", "v2", "--transact-type", "sale",
       "--transact-type", "rent"],
      check=False,  # gate exits 1 on FAIL; do not fail the pipeline
  )
  ```
  The pipeline prints the gate's stdout summary regardless
  and proceeds to the next stage. The gate's exit code is
  surfaced in the pipeline's final summary, not raised as
  a hard error — a FAIL is a signal for a human/agent
  reviewer, not a reason to abort other stages.
- `ml/__init__.py` — add `from ml import evaluation  #
  noqa: F401` so the new submodule is importable
  consistently with `ml.training` / `ml.features` /
  `ml.cleaning`.
- `requirements.txt` — no new packages; this spec only
  uses numpy/pandas/scikit-learn/joblib, all already
  pinned. Verify with `pip freeze | grep -E
  "(numpy|pandas|scikit-learn|joblib)"` and flag
  explicitly per CLAUDE.md "no new packages without
  checking first."

**No changes** to:
- `app/`, `api/`, `data/raw/`,
  `data/processed/clean_listings.parquet`,
  `data/processed/feature_selection_report.md`'s
  pre-existing Round 1/2/3 / v2 sections (the new
  "Protocol Certification" section is **appended**,
  never overwrites prior content).
- `notebooks/`, `migrations/`,
  `tests/conftest.py` (existing fixtures remain; new
  fixtures live in the new test files).
- `ml/training/` — Specs 13/14's training scripts are
  unchanged. The gate reads their outputs; it does not
  edit them.
- `CLAUDE.md`'s "Implemented vs stub routes" table —
  this spec adds **no routes**. The `POST /predict`
  FastAPI route stays a Stub until a follow-on spec
  wires whichever model version is current.

## New dependencies
None. The spec uses numpy / pandas / scikit-learn / joblib
(already pinned) + stdlib (`argparse`, `dataclasses`,
`logging`, `json`, `pathlib`, `subprocess`). The
`/evaluate-model` skill (existing) drives the gate from
agent contexts. **No** new pip/npm packages, **no**
Flask/FastAPI route additions, **no** new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The gate reads +
  appends JSON files only.
- **No dealer/contact/media-URL fields ever reach the
  UI or an export.** The gate is offline; it does not
  log any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. Pinned
  by a test that greps the CLI's captured stdout for
  the regex — must be absent.
- **CSS variables only.** N/A — no templates.
- **All templates extend `base.html`.** N/A — no
  templates.
- **Model evaluation must reference the fixed
  evaluation protocol (Rules §2.1).** Every metric the
  gate emits comes from `protocol.py`'s pinned
  constants; no inline literals. The four headline
  metrics, the split ratios, the `random_state`, and
  the threshold values are read from one place. The
  gate's `EvaluationResult.protocol_version` field
  records which protocol version was applied, so a
  reviewer can audit drift over time.
- **Single source of truth for the protocol.** The
  constants in `ml/evaluation/protocol.py` are the only
  place these literals live. Specs 13/14's training
  scripts keep their own constants for training-time
  use (no behavior change there); the gate
  independently enforces the protocol at
  certification time so a drift in a training
  script's constants does not silently ship a model
  that fails the protocol.
- **`random_state=42` everywhere (Rules §5.4).** The
  pinned `RANDOM_STATE = 42` is the gate's only
  `random_state`. Any call to `protocol_split(...
  random_state=...)` with a non-42 value raises.
- **Outliers excluded from training (Rules §1.4).**
  Step 2 of `evaluate()` filters to
  `is_outlier == False`. Same rule as Specs 13/14 —
  the gate's behavior is byte-equivalent to the
  training scripts' filter.
- **`transact_type` is a routing key (Rules §10.3).**
  The gate runs independently per `transact_type`;
  it never mixes Sale and Rent in one model scoring
  pass. The CLI accepts multiple `--transact-type`
  flags; each runs its own `evaluate()` call.
- **Production model is the exact `Pipeline` used at
  evaluation (Rules §2.4).** Step 5 of `evaluate()`
  asserts the loaded preprocessor equals the
  in-memory reference. Drift → raise, never silent.
- **Versioned artifacts, never overwritten (Rules
  §2.5).** `write_evaluation_report` writes
  `evaluation_report_{version}_{transact_type}.json`;
  a re-run with the same content writes the same
  file; a re-run with new `evaluated_at` writes a
  timestamp-suffixed sibling. v1 reports are
  preserved when v2 is certified.
- **`model_registry.csv` is read-only at gate time.**
  The gate checks the registry row exists; it does
  **not** append new rows. Training time is the only
  place the registry grows (Specs 13/14 contract).
- **Honest logging of shortfalls (Rules §9.2).**
  `evaluate()` returns `overall_passed=False` if any
  threshold is missed; the CLI exits 1 and prints
  `[FAIL]` + the actual measured value next to the
  target. No "we probably would have hit it with more
  compute" rationalization in the report or the
  Decision Log.
- **Per-city test metrics are required (Rules §8.5,
  adapted).** Step 8 of `evaluate()` always computes
  `per_city_test` for the certified model. Cities
  with < 30 test rows get a logged WARNING but the
  row is still included (zero metrics would be more
  misleading than small-sample metrics). Same
  threshold + behavior as Specs 13/14's training
  scripts.
- **No FastAPI imports in `ml/evaluation/`.** The
  latency-measurement path is optional and uses
  `urllib.request` (stdlib) against the provided
  `fastapi_url` — no FastAPI / httpx client
  imports. Keeps the gate runnable in offline CI
  without a live inference service.
- **Logging uses stdlib `logging` only.** One module-
  level logger per file (`logger =
  logging.getLogger(__name__)`). INFO for stage
  boundaries; WARNING for expected-but-noteworthy
  conditions (Rent skipped, per-city n < 30,
  latency not measured); ERROR for hard failures
  (preprocessor drift, random_state drift, missing
  registry row).
- **CLI never aborts the pipeline on FAIL.** The
  pipeline calls the CLI with `check=False`. The
  gate's stdout summary is printed regardless; the
  pipeline proceeds. FAIL is a signal for review,
  not a reason to abort unrelated stages. The
  pipeline's final summary line surfaces the gate's
  pass/fail state.
- **No notebook-only steps (Rules §5.3).** Everything
  in this spec is reproducible via
  `python scripts/evaluate_price_model.py --version
  v2 --transact-type sale --transact-type rent`. No
  Jupyter cell scores a model or computes a metric
  the script can't reproduce.

## Definition of done

1. `python -m pytest tests/test_protocol.py
   tests/test_scoring.py tests/test_gate.py
   tests/test_report_writer.py
   tests/test_evaluate_price_model_cli.py -v` from repo
   root runs and passes. Tests required (exact names):
   - **Protocol constants** (`test_protocol.py`):
     - `test_protocol_version_is_pinned`
     - `test_split_ratios_match_trd_section_10`
     - `test_random_state_is_42`
     - `test_metric_names_match_rules_section_2_1`
     - `test_protocol_split_returns_three_frames`
     - `test_protocol_split_enforces_ratios_within_one_row`
     - `test_protocol_split_asserts_random_state`
   - **Scoring** (`test_scoring.py`):
     - `test_score_predictions_returns_four_keys_in_pinned_order`
     - `test_score_predictions_inverts_log_target`
     - `test_within_tolerance_pct_returns_fraction`
     - `test_within_tolerance_pct_is_zero_when_all_outside_band`
     - `test_within_tolerance_pct_is_one_when_all_inside_band`
   - **Gate** (`test_gate.py`):
     - `test_evaluate_returns_evaluation_result_dataclass`
     - `test_evaluate_overall_passed_false_when_r2_below_threshold`
     - `test_evaluate_overall_passed_true_when_all_thresholds_met`
     - `test_evaluate_skips_rent_when_too_small`
     - `test_evaluate_rejects_preprocessor_drift`
     - `test_evaluate_records_protocol_version_in_result`
     - `test_evaluate_records_git_commit_in_result`
   - **Report writer** (`test_report_writer.py`):
     - `test_write_evaluation_report_writes_versioned_filename`
     - `test_write_evaluation_report_rerun_uses_timestamp_suffix`
     - `test_append_protocol_section_appends_not_overwrites`
   - **CLI** (`test_evaluate_price_model_cli.py`):
     - `test_cli_runs_end_to_end_on_synthetic_artifacts`
     - `test_cli_exits_nonzero_when_r2_below_threshold`
     - `test_cli_exits_zero_when_all_thresholds_met`
     - `test_cli_does_not_log_contact_fields` — grep
       the captured stdout for any column name matching
       `(contact|dealer|phone|email|photo|url|spid)`
       — must be absent.
2. `python -m pytest -m "not realdata"` from repo root
   still passes (no real-data dependency introduced
   by this spec).
3. `ruff check ml/evaluation/
   scripts/evaluate_price_model.py tests/test_protocol.py
   tests/test_scoring.py tests/test_gate.py
   tests/test_report_writer.py
   tests/test_evaluate_price_model_cli.py` reports zero
   issues.
4. `python -c "from ml.evaluation import
   PROTOCOL_VERSION, evaluate, EvaluationResult,
   protocol_thresholds; print(PROTOCOL_VERSION)"` from
   repo root prints `1.0.0` without error — public API
   imports cleanly.
5. `python scripts/evaluate_price_model.py --version v2
   --transact-type sale --transact-type rent` from repo
   root exits 0 if v2 clears all thresholds, exits 1
   otherwise, and prints the `[PASS|FAIL]` summary line
   per `transact_type` — manual smoke test of the CLI
   gate.
6. After running step 5,
   `models/evaluation_report_v2_sale.json` (and
   `_v2_rent.json` if Rent ran) parse as valid JSON with
   `protocol_version == "1.0.0"`, `git_commit` matching
   `git rev-parse HEAD`, `dataset_version` from the
   parquet, and the `thresholds_passed` dict populated
   for every key in `protocol_thresholds`.
7. `data/processed/feature_selection_report.md` after
   running step 5 contains Specs 12/13/14's prior
   sections (preserved) **plus** an appended "Protocol
   Certification" section with the v2 pass/fail result
   + per-city R².
8. `data/model_registry.csv` after running step 5 is
   unchanged — the gate does not append new registry
   rows.
9. `git status` after committing shows only the new
   files listed above, the modified
   `scripts/run_pipeline.py`, and the modified
   `ml/__init__.py`. No accidental additions to
   `app/`, `api/`, `data/processed/clean_listings.parquet`,
   `data/raw/`, `notebooks/`, or `ml/training/`.
10. `CLAUDE.md`'s "Implemented vs stub routes" table
    is unchanged — this spec adds **no routes**. The
    `POST /predict` FastAPI route stays a Stub until
    a follow-on spec wires whichever model version
    is current.
11. `07-TRACKER.md` is updated via `/update-tracker`
    to mark Day 25 (Final model selection + SHAP
    explainer validation) and Day 28 (`metrics_v1.json`
    checkpoint) as **Certified** (not "Done" — the
    training was done in Specs 13/14; this spec adds
    the certification gate on top), with the
    actual measured R² / MAE / within-15% values from
    the `evaluation_report_v{1,2}_{sale,rent}.json`
    files. Days 22–27 remain Done from Specs 13/14.
