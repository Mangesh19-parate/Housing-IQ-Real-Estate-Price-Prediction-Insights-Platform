# Spec: SHAP Explainability — Price Model

## Overview
Land the SHAP explainability layer for the price regression model that
PRD §6.1 (FR3) and the `shap-explainability` skill already call for, and
that `05-BACKEND-SCHEMA.md` §6 already promised the artifact for
(`models/shap_explainer_v{n}.pkl`) but no prior spec has delivered.
Today Spec 13/14's training scripts write a Round 3 SHAP ranking into
`data/processed/feature_selection_report.md` as a side effect of model
selection — that's training-time *feature-selection* signal, not the
**per-prediction explanation** the Predict page UI must show alongside
the price number. This spec produces a reusable per-prediction
explanation function (`explain_one(model, request) -> list[ShapContribution]`)
backed by a precomputed `shap.TreeExplainer` artifact persisted under
`models/shap_explainer_v{n}.pkl`, plus a single-row CLI
(`scripts/explain_prediction.py`) the FastAPI route layer and the
`housingiq-ml-evaluator` agent can invoke directly. Per the skill, the
artifact is built **once** at training time and **loaded** at every
prediction request — no per-request `TreeExplainer(model)` construction.
Module: **price-prediction**.

## Depends on
- **Step 13** — `13-baseline-regression-model-training` — produces
  `models/price_model_{sale,rent}_v1.pkl` + the v1 SHAP Round 3 section.
  This spec consumes both: it wraps the same `TreeExplainer` into a
  persisted artifact so future explainability requests don't rebuild it.
- **Step 14** — `14-xgboost-lightgbm-price-model-training` — produces
  `price_model_{sale,rent}_v2.pkl` + Optuna-tuned XGBoost/LightGBM
  candidates. The v2 pipeline is the **primary beneficiary** of this
  spec (per `shap-explainability` skill: "load precomputed TreeExplainer
  at startup"). v2 SHAP rankings are appended to the report (consistent
  with Step 14's spec); the per-prediction helper works against v1 and v2
  identically.
- **Step 15** — `15-price-model-evaluation-protocol` — the evaluation
  gate that certifies v1/v2. This spec does **not** modify the gate, but
  the SHAP artifact's existence is verified by a follow-up check inside
  the gate's `evaluate()` entry point (out of scope here — the gate
  keeps reading the model `.pkl` and computing a fresh `TreeExplainer`
  for its internal SHAP ranking; this spec adds the *persisted* artifact
  for serving-time use).
- **Step 12** — `12-feature-engineering-price-model` — supplies the
  fitted preprocessor + `feature_list_v1.json` so the explainer is
  built against the same transformed feature order the model consumes.
- **Step 11** — `11-price-prediction-input-schema-v3` — locks the 16
  input fields; the per-prediction helper takes a `PredictionRequest`
  matching this contract.
- **`02-TRD.md` §10.3 + §16 (TRD Update v1)** — the productionization
  checklist that already names SHAP as a required production artifact.
- **`05-BACKEND-SCHEMA.md` §6** — `models/shap_explainer_v{n}.pkl` is
  the artifact this spec lands; the schema's promise is fulfilled here.
- **`05-BACKEND-SCHEMA.md` §7** — `PredictionResponse.shap_contributions`
  shape (`list[{feature, impact}]`); this spec's output type is exactly
  that.
- **`08-RULES.md` §2.6** — SHAP explanations shown to users must come
  from the same model instance making the prediction; this spec builds
  the explainer **from the persisted model artifact**, never a proxy.
- **`08-RULES.md` §2.5** — versioned artifacts, never overwritten; the
  explainer filename pins `{version}_{transact_type}`.
- **`08-RULES.md` §5.4** — `random_state=42` everywhere (background
  dataset sampling in tests is the only randomness surface; pinned).
- **`shap-explainability` skill** — color tokens, top-N (5–7) rule,
  precomputed-explainer-at-startup convention, human-readable label
  mapping.
- **`generate-shap-report` command** — existing agent skill for ad-hoc
  global SHAP summaries; this spec is the *programmatic* counterpart
  the FastAPI route + per-prediction UI consume.

## Routes / Endpoints
No new routes/endpoints. This spec ships:
- A new Python module `ml/explainability/` (pure functions, no FastAPI
  surface).
- A new model artifact `models/shap_explainer_{sale,rent}_v{n}.pkl`
  (built once, loaded forever).
- A new CLI `scripts/explain_prediction.py` (manual + agent-invokable).
- A new pytest suite.
The FastAPI route that calls this in `/predict` is a follow-on spec;
this spec is its dependency.

## Data / Schema changes
- **Read** `data/processed/clean_listings.parquet` (Steps 07, 13, 14).
- **Read** `models/feature_pipeline_v1.pkl` (Step 12) — the fitted
  preprocessor; explainer input must use the same transformed shape.
- **Read** `models/feature_list_v1.json` (Step 12).
- **Read** `models/price_model_{sale,rent}_v{n}.pkl` (Step 13/14) —
  the model instance the explainer is built from. Version arg is
  required; defaults to `"v2"` (current) per Step 15's gate output.
- **Write** `models/shap_explainer_sale_v{n}.pkl` —
  `joblib.dump(shap.TreeExplainer(model))` for the Sale pipeline's
  final estimator (the last step of the `Pipeline`). The
  `Pipeline`'s preprocessor step is **not** included in the
  explainer artifact — the explainer is a `TreeExplainer` over the
  tree estimator alone, matching the standard SHAP-on-pipeline pattern
  (call `pipeline[:-1].transform(X)` before passing to
  `explainer.shap_values(...)`).
- **Write** `models/shap_explainer_rent_v{n}.pkl` — same, for Rent.
  Skipped with INFO if Rent was skipped at training time (Step 13/14
  `metrics_v{n}.json.rent.skipped == true`).
- **Write** `models/feature_label_map_v{n}.json` — pinned JSON
  `dict[str, str]` mapping internal feature names (post-preprocessor,
  e.g. `num__built_up_area`, `cat__city_Gurgaon`,
  `ord__furnishing_type`) to human-readable labels used in the UI
  (`"Built-up Area (sqft)"`, `"City: Gurgaon"`,
  `"Furnishing Type"`). Same map used by the per-prediction helper
  and by future FastAPI/Flask code (one source of truth for labels).
- **Append** `data/processed/feature_selection_report.md` — one new
  section "SHAP Explainability Artifact v{n}" with (a) the artifact
  filename, (b) the explainer type, (c) the top-10 `mean |SHAP value|`
  features from a held-out test slice (200 rows, `random_state=42`),
  (d) a one-line note that the artifact is consumed by the Predict
  page UI via the FastAPI route layer (not yet wired). Specs 13/14's
  Round 2/3 sections are preserved above this new section.
- **Append** `data/model_registry.csv` — one row per
  `(transact_type, version)` with `model_name="shap_explainer"`,
  `version=v{n}`, `training_dataset_version` + `git_commit` +
  `training_date` from the underlying model's registry row, and
  `hyperparameters` = `{"explainer_type": "TreeExplainer", "top_n":
  7, "label_map_hash": "<sha1 of feature_label_map.json contents>"}`.
  Reuses `append_model_registry` from Step 13; idempotent on
  `(model_name, version, git_commit)`.
- **No writes** to `data/raw/`,
  `data/processed/clean_listings.parquet`,
  `data/processed/analytics_cache/`, or `data/app.db`.
- **No writes** to the trained model `.pkl` files — this spec only
  reads them and writes the sibling explainer artifact.
- **No new DB columns** — `prediction_log.latency_ms` (Backend Schema §5)
  remains unchanged; logging the per-prediction SHAP compute time is
  out of scope (FastAPI observability is a follow-on spec).

## Templates / UI
None. This spec ships the artifact + helper + CLI. The Jinja2 template
that *renders* the SHAP bar chart on the Predict result page lives in
the follow-on "Predict result page" spec (Day 38 in `06-IMPLEMENTATION-PLAN.md`).
The colors (`#16A34A` price-up / `#DC2626` price-down per the
`shap-explainability` skill) and the top-N=7 rule are **documented here
in the helper's output type** so the template can consume them directly
when it's built, but no template is added by this spec.

## Files to change / Files to create

**Create:**
- `ml/explainability/` — new sub-package, one file per concern, so the
  explainability layer is testable in isolation from any FastAPI or
  Flask code. Re-exports from `ml/explainability/__init__.py`.
  - `__init__.py` — empty; re-exports the four public symbols:
    `build_explainer`, `explain_one`, `global_summary`,
    `FEATURE_LABEL_MAP_V2`.
  - `explainer.py` — the `TreeExplainer` factory + artifact
    persistence. Public API:
    - `EXPLAINER_VERSION: str = "1.0.0"` — pinned semver, emitted
      into the artifact's metadata block.
    - `SHAP_TOP_N: int = 7` — pinned UI top-N (skill: 5–7). 7 is the
      high end of the skill's range; matches the reference project's
      Predict page chart.
    - `build_explainer(model: Pipeline | BaseEstimator) ->
      shap.TreeExplainer` — pure factory. Asserts the input has a
      tree estimator in the last `Pipeline` step (or is itself a
      tree estimator). For non-tree models (e.g. Ridge baseline),
      raises `ValueError` with a message naming the model's
      `type(model).__name__`; the per-spec-13/14 winner is a tree
      model so this is a defensive guard, not a code path. Returns
      the unfitted `TreeExplainer` ready to load from artifact.
    - `save_explainer(explainer: shap.TreeExplainer,
      transact_type: str, version: str, out_dir: Path | str) ->
      Path` — `joblib.dump(explainer, out_dir /
      f"shap_explainer_{transact_type}_{version}.pkl")`. Filename
      pinned by Rules §2.5 (versioned, never overwritten).
      Returns the path.
    - `load_explainer(transact_type: str, version: str,
      models_dir: Path | str) -> shap.TreeExplainer` — inverse of
      `save_explainer`. Raises `FileNotFoundError` with the expected
      path on miss. Does **not** refit.
  - `contributions.py` — the per-prediction explanation helper.
    Public API:
    - `ShapContribution` — `@dataclass(frozen=True)` matching
      Backend Schema §7's shape:
      `{feature: str, impact: float, direction: Literal["up",
      "down"], label: str}`. `label` is the human-readable form
      (from `feature_label_map_v{n}.json`); `feature` is the raw
      internal name; `impact` is the SHAP value in log-price space
      (the model trains on `log1p(price)`, so SHAP values are in
      log units — the FastAPI route's display layer is responsible
      for re-scaling, e.g. `pct_impact ≈ exp(impact) - 1`, which is
      out of scope here). `direction` is the sign of `impact`
      (`"up"` if `impact > 0`, `"down"` otherwise), so the UI can
      apply the color tokens from the skill without re-computing
      the sign per row.
    - `explain_one(model: Pipeline, explainer: shap.TreeExplainer,
      request_features: np.ndarray, feature_names: list[str],
      label_map: dict[str, str],
      top_n: int = SHAP_TOP_N) -> list[ShapContribution]` — compute
      `explainer.shap_values(request_features)` for a single row
      (1 × n_features), build a list of `(name, impact)` pairs, map
      names through `label_map`, sort by `abs(impact)` descending,
      slice top-N. Returns an empty list with a logged WARNING if
      `request_features` has zero rows or a row whose sum is NaN
      (defensive — the FastAPI Pydantic layer validates inputs, but
      this is the trust-boundary guard). Pure function; no I/O.
    - `direction_breakdown(contributions: list[ShapContribution]) ->
      dict[str, int]` — returns `{"up": <count>, "down": <count>}`
      for the helper's test (and for any future UI that wants to
      summarize "n features pushing price up / down" in one
      headline). Pinned by a test.
  - `summary.py` — the global SHAP summary helper. Public API:
    - `global_summary(model: Pipeline, explainer: shap.TreeExplainer,
      X_background: np.ndarray, feature_names: list[str],
      n_samples: int = 200, random_state: int = 42) ->
      dict[str, float]` — returns `mean |SHAP value|` per feature
      across `n_samples` rows sampled from `X_background` with the
      pinned seed. Used by the CLI's global-mode path and by the
      appended `feature_selection_report.md` section. Pure function;
      no I/O.
    - `write_summary_section(summary: dict[str, float], version:
      str, out_path: Path | str, top_k: int = 10) -> None` —
      appends the "SHAP Explainability Artifact v{n}" section to
      `data/processed/feature_selection_report.md`. Renders a
      markdown table with the top-`k` features. Pin the section
      header string in the test.
  - `labels.py` — the feature-label mapping. Public API:
    - `FEATURE_LABEL_MAP_V2: dict[str, str]` — pinned literal
      mapping every internal feature name (post-preprocessor) to a
      human-readable label. Sourced from
      `models/feature_label_map_v2.json` at module import time (the
      module reads the JSON once; the constant is a `dict[str,
      str]` for ergonomic use). Keys are the post-preprocessor
      names: numeric block (`num__bedRoom`, `num__bathroom`,
      `num__built_up_area`, `num__servant_room`, `num__store_room`,
      `num__n_amenities`, `num__distance_to_cbd_km`,
      `num__distance_to_nearest_metro_km`,
      `num__sector_smoothed_price`, `num__locality_smoothed_price`,
      `num__area_per_bedroom`, `num__bath_bed_ratio`,
      `num__floor_ratio`, `num__age_bucket_ord`,
      `num__price_per_sqft`), ordinal block (`ord__luxury_category`,
      `ord__floor_category`, `ord__furnishing_type`), one-hot block
      (one key per `cat__<column>_<value>` pair across `city`,
      `sector`, `property_type`, `balcony`, `agePossession`,
      `facing`, `transact_type` — populated dynamically from the
      fitted preprocessor's `OneHotEncoder.categories_`, so the
      module exposes a `build_label_map(preprocessor: ColumnTransformer)
      -> dict[str, str]` helper that returns the full map; the
      module constant is a fallback for tests). Unknown internal
      names fall through to the raw internal name with a logged
      WARNING (defensive — never raise, since the per-prediction
      path can hit a name the label-map builder didn't anticipate).
  - `__init__.py` — `__version__ = "1.0.0"` + re-exports the four
    public symbols.

- `scripts/build_shap_explainer.py` — the artifact-build CLI.
  Invoked as `python scripts/build_shap_explainer.py --version v2
  [--transact-type sale] [--transact-type rent] [--models-dir ...]`.
  Steps:
  1. Parse args (`argparse`).
  2. For each `--transact-type`:
     a. Load `models/price_model_{transact_type}_v{version}.pkl` via
        `joblib.load`. Assert it is a `Pipeline` whose last step is
        a tree estimator (XGBoost / LightGBM / RandomForest / GBR).
     b. Load `models/feature_pipeline_v1.pkl` and
        `models/feature_list_v1.json` from `models/`. Build the
        feature label map via `labels.build_label_map(preproc)` and
        write `models/feature_label_map_v{version}.json`.
     c. Call `explainability.build_explainer(model)` to get the
        `TreeExplainer`.
     d. Call `explainability.save_explainer(explainer,
        transact_type, version, models_dir)`.
     e. Sample 200 rows from the test split
        (`data/processed/clean_listings.parquet`, filtered to
        `is_outlier == False` + the matching `transact_type`,
        then `protocol_split` from Spec 15's gate). Call
        `explainability.global_summary(model, explainer, X_test,
        feature_names)` to get `mean |SHAP|` per feature. Call
        `explainability.write_summary_section(summary, version,
        report_path)` to append to
        `data/processed/feature_selection_report.md`.
     f. Append one `model_registry.csv` row per
        `(shap_explainer, transact_type, version)`.
  3. Print a summary line per `transact_type`:
     `[OK] shap_explainer_{sale,rent}_v{N}.pkl — top feature: {name}
     (mean |SHAP|={value:.4f})`.
  4. Exit 0 if both artifacts wrote; exit 1 if any failed.

- `scripts/explain_prediction.py` — the single-row CLI. Invoked as
  `python scripts/explain_prediction.py --version v2 --transact-type
  sale --input-json '{"city": "Gurgaon", "sector": "sector 84", ...
  }'`. Steps:
  1. Parse args.
  2. Load the model `.pkl` + the explainer `.pkl` + the label map.
  3. Convert the input JSON to a 1-row numpy array via the same
     fitted preprocessor (`pipeline[:-1].transform(...)`). Reuses
     the preprocessor from `models/feature_pipeline_v1.pkl`.
  4. Call `explainability.explain_one(model, explainer, X_row,
     feature_names, label_map)` to get a `list[ShapContribution]`.
  5. Print the top-`SHAP_TOP_N` as a markdown table (feature, label,
     impact, direction).
  6. Exit 0. Manual smoke test of the helper for a known input.

- `tests/test_explainer.py` — pytest tests for `build_explainer`,
  `save_explainer`, `load_explainer`. Required tests (exact names):
  - `test_explainer_version_is_pinned` — `EXPLAINER_VERSION ==
    "1.0.0"`.
  - `test_shap_top_n_is_seven` — `SHAP_TOP_N == 7`.
  - `test_build_explainer_returns_tree_explainer` — type check on
    a tiny synthetic tree-estimator pipeline.
  - `test_build_explainer_rejects_non_tree_model` — `Ridge(alpha=1)`
    → `ValueError` with the model class name in the message.
  - `test_save_and_load_explainer_round_trip` — save + load + assert
    `isinstance` + assert `explainer.model.equal(...)` (the
    underlying model reference is preserved across `joblib`).
  - `test_save_explainer_writes_versioned_filename` — assert
    `shap_explainer_sale_v2.pkl` lands in `tmp_path`.
  - `test_load_explainer_raises_with_expected_path_on_miss` —
    `FileNotFoundError` message contains the resolved path.

- `tests/test_contributions.py` — pytest tests for
  `ShapContribution` + `explain_one` + `direction_breakdown`.
  Required tests (exact names):
  - `test_shap_contribution_dataclass_has_expected_fields` — assert
    the four field names.
  - `test_explain_one_returns_top_n_contributions` — synthetic
    single-row input; assert the returned list length == `top_n`
    (or fewer if fewer features passed the magnitude threshold).
  - `test_explain_one_sorts_by_abs_impact_descending` — pin the
    sort order.
  - `test_explain_one_maps_feature_names_through_label_map` —
    `contributions[0].label` is the human-readable form, not the
    raw internal name.
  - `test_explain_one_returns_empty_list_for_empty_input` —
    defensive: 0 rows → `[]` + WARNING logged.
  - `test_explain_one_direction_up_for_positive_impact` and
    `test_explain_one_direction_down_for_negative_impact` — sign
    mapping.
  - `test_direction_breakdown_counts_up_and_down` — synthetic
    contributions; assert the returned counts.

- `tests/test_summary.py` — pytest tests for `global_summary` +
  `write_summary_section`. Required tests (exact names):
  - `test_global_summary_returns_dict_of_mean_abs_shap` — type +
    key count check.
  - `test_global_summary_uses_pinned_n_samples` — assert
    `n_samples == 200` by default (and configurable via arg).
  - `test_global_summary_deterministic_with_random_state` —
    two calls with the same seed return the same `mean |SHAP|`
    dict (within a small float tolerance).
  - `test_write_summary_section_appends_not_overwrites` —
    pre-populate the report with a sentinel line, call the
    function, assert the sentinel is still present.
  - `test_write_summary_section_includes_section_header` —
    output contains the "SHAP Explainability Artifact v{N}"
    header.

- `tests/test_labels.py` — pytest tests for the label map.
  Required tests (exact names):
  - `test_feature_label_map_v2_is_a_dict` — type check.
  - `test_label_map_covers_numeric_block` — every
    `num__<expected_name>` key is present.
  - `test_label_map_covers_ordinal_block` — every
    `ord__<expected_name>` key is present.
  - `test_label_map_unknown_name_falls_through_to_raw` —
    `get("num__made_up", raw)` returns `"num__made_up"` (not
    raise).
  - `test_build_label_map_emits_one_hot_keys_for_fitted_categories`
    — build the map from a fitted `ColumnTransformer` with a tiny
    `OneHotEncoder(categories=[["A", "B"]])`; assert
    `cat__<col>_A` and `cat__<col>_B` are present in the output.

- `tests/test_build_shap_explainer_cli.py` — pytest test for the
  artifact-build CLI. Required tests (exact names):
  - `test_cli_builds_explainer_artifact_end_to_end` — write a tiny
    synthetic `clean_listings.parquet` + a tiny fitted preprocessor
    + a tiny XGB `Pipeline` to `tmp_path`, invoke the CLI via
    `subprocess.run` with `--models-dir` pointing at the fixture,
    assert the artifact `.pkl` + the label-map JSON + an appended
    `feature_selection_report.md` section + a `model_registry.csv`
    row all land.
  - `test_cli_skips_rent_when_training_skipped_rent` — when the
    underlying `price_model_rent_v{n}.pkl` is missing, the CLI
    skips Rent with an INFO log and exits 0 (single-transact
    success).
  - `test_cli_exits_nonzero_when_no_transact_type_provided` —
    defensive: missing `--transact-type` → exit 2 (argparse).

- `tests/test_explain_prediction_cli.py` — pytest test for the
  single-row CLI. Required tests (exact names):
  - `test_cli_prints_top_n_table_for_valid_input` — feed a
    synthetic input JSON, assert stdout contains a markdown table
    with `SHAP_TOP_N` rows + the "Feature | Label | Impact |
    Direction" header.
  - `test_cli_exits_nonzero_when_explainer_artifact_missing` —
    `FileNotFoundError` surfaced, exit 1.

**Modify:**
- `scripts/run_pipeline.py` — append two lines after the Step 15
  evaluation-gate line:
  ```python
  subprocess.run(
      [sys.executable, "scripts/build_shap_explainer.py",
       "--version", "v2", "--transact-type", "sale",
       "--transact-type", "rent"],
      check=True,
  )
  ```
  Same pattern as the existing gate registration; the build step
  is part of the reproducible end-to-end pipeline (TRD §13).
- `ml/__init__.py` — add `from ml import explainability  #
  noqa: F401` so the new submodule is importable consistently with
  `ml.training` / `ml.features` / `ml.cleaning` /
  `ml.evaluation`.
- `requirements.txt` — verify `shap` is already pinned (Step 13's
  spec already added it; `pip freeze | grep -i shap`). If absent,
  add with a pinned minor version. Flag the addition explicitly
  per CLAUDE.md "no new packages without checking first." No
  other additions — `joblib` and `numpy` are already pinned.

**No changes** to:
- `app/`, `api/`, `data/raw/`,
  `data/processed/clean_listings.parquet`, `data/app.db`.
- `data/processed/feature_selection_report.md`'s Round 1/2/3 +
  Protocol Certification sections (this spec **appends** the new
  "SHAP Explainability Artifact v{n}" section; never overwrites
  prior content).
- `notebooks/`, `migrations/`, `tests/conftest.py` (existing
  fixtures remain; new fixtures live in the new test files).
- `ml/training/`, `ml/evaluation/`, `ml/features/`, `ml/cleaning/`
  — no behavior edits; this spec is a sibling package.
- `CLAUDE.md`'s "Implemented vs stub routes" table — this spec
  adds **no routes**. The `POST /predict` FastAPI route stays a
  Stub; wiring the new explainer into it is a follow-on spec.

## New dependencies
None. The spec uses `shap` (already pinned per Spec 13) + `joblib` /
`numpy` / `pandas` / `scikit-learn` (already pinned) + stdlib
(`argparse`, `dataclasses`, `logging`, `pathlib`). No new pip/npm
packages, no Flask/FastAPI route additions, no new DB drivers.

## Rules for implementation

- **No SQLAlchemy/ORM.** N/A — no SQL. The `model_registry.csv`
  append reuses `ml.training.persistence.append_model_registry`.
- **No dealer/contact/media-URL fields ever reach the UI or an
  export.** The explainer is offline; the per-prediction CLI does
  not log any column whose name matches the regex
  `(contact|dealer|phone|email|photo|url|spid)`. Pinned by a test
  that greps the CLI's captured stdout for the regex — must be
  absent.
- **CSS variables only.** N/A — no templates. The
  `#16A34A` / `#DC2626` color tokens from the
  `shap-explainability` skill are **documented in the spec's
  per-prediction output type** so the future Predict result
  template consumes them via CSS variables, not inline hex.
- **All templates extend `base.html`.** N/A — no templates.
- **Model changes must reference the fixed evaluation protocol
  (Rules §2.1).** The artifact-build CLI uses the same test
  split as Spec 15's gate (`protocol_split` from
  `ml.evaluation.splits`, with `random_state=42`), so the global
  SHAP summary uses a sample drawn from the same test rows the
  gate scored. No second split definition.
- **Outliers excluded (Rules §1.4).** The artifact-build CLI
  filters to `is_outlier == False` before sampling the test
  rows, matching the training pipeline's filter.
- **`transact_type` is a routing key (Rules §10.3).** Sale and
  Rent each get their own `shap_explainer_*.pkl` artifact; the
  per-prediction helper is called with one pipeline at a time.
  Mixing Sale + Rent in a single explainer is a hard rule
  violation.
- **Single source of truth for labels.** `feature_label_map_v{n}.json`
  is the only place the internal-name → human-readable mapping
  lives; the per-prediction helper, the global summary writer,
  and the future FastAPI route all read the same file. A drift
  between layers would silently show raw column names in the UI
  (Rules §2.6 spirit — explanations must be interpretable).
- **Precomputed explainer, not built per request (skill).** The
  FastAPI route layer loads `shap_explainer_*.pkl` at startup;
  it does **not** call `shap.TreeExplainer(model)` inside the
  request handler. The CLI scripts build the artifact once;
  the helper only consumes it.
- **SHAP comes from the same model as the prediction (Rules
  §2.6).** The explainer is built from the exact persisted
  `Pipeline` instance in `models/price_model_*.pkl`. No proxy
  model, no simplified surrogate, no separate "fast" estimator.
  `save_explainer` + `load_explainer` round-trip preserves the
  underlying model reference (pinned by
  `test_save_and_load_explainer_round_trip`).
- **Versioned artifacts, never overwritten (Rules §2.5).**
  `shap_explainer_sale_v{n}.pkl` is a sibling of
  `price_model_sale_v{n}.pkl`. A v3 cert (future spec) writes
  a new artifact; v1/v2 are preserved.
- **`random_state=42` everywhere (Rules §5.4).** Background
  sampling for the global summary uses
  `np.random.default_rng(42)`; the test pinning
  `test_global_summary_deterministic_with_random_state` enforces
  this.
- **No FastAPI imports in `ml/explainability/`.** This spec is
  offline + a CLI. Wiring the helper into the FastAPI route is a
  follow-on spec.
- **Logging uses stdlib `logging` only.** One module-level
  logger per file. INFO for artifact build + per-prediction
  start/finish; WARNING for expected-but-noteworthy conditions
  (Rent skipped, label-map fallthrough, empty input to
  `explain_one`); ERROR for hard failures (model load miss,
  explainer artifact miss, non-tree model passed to
  `build_explainer`).
- **No notebook-only steps (Rules §5.3).** Everything is
  reproducible via
  `python scripts/build_shap_explainer.py --version v2
  --transact-type sale --transact-type rent` + a follow-up
  `python scripts/explain_prediction.py --version v2
  --transact-type sale --input-json '{...}'`. No Jupyter cell
  computes a SHAP value the script can't reproduce.

## Definition of done

1. `python -m pytest tests/test_explainer.py
   tests/test_contributions.py tests/test_summary.py
   tests/test_labels.py tests/test_build_shap_explainer_cli.py
   tests/test_explain_prediction_cli.py -v` from repo root runs
   and passes. Tests required (exact names):
   - **Explainer** (`test_explainer.py`):
     - `test_explainer_version_is_pinned`
     - `test_shap_top_n_is_seven`
     - `test_build_explainer_returns_tree_explainer`
     - `test_build_explainer_rejects_non_tree_model`
     - `test_save_and_load_explainer_round_trip`
     - `test_save_explainer_writes_versioned_filename`
     - `test_load_explainer_raises_with_expected_path_on_miss`
   - **Contributions** (`test_contributions.py`):
     - `test_shap_contribution_dataclass_has_expected_fields`
     - `test_explain_one_returns_top_n_contributions`
     - `test_explain_one_sorts_by_abs_impact_descending`
     - `test_explain_one_maps_feature_names_through_label_map`
     - `test_explain_one_returns_empty_list_for_empty_input`
     - `test_explain_one_direction_up_for_positive_impact`
     - `test_explain_one_direction_down_for_negative_impact`
     - `test_direction_breakdown_counts_up_and_down`
   - **Summary** (`test_summary.py`):
     - `test_global_summary_returns_dict_of_mean_abs_shap`
     - `test_global_summary_uses_pinned_n_samples`
     - `test_global_summary_deterministic_with_random_state`
     - `test_write_summary_section_appends_not_overwrites`
     - `test_write_summary_section_includes_section_header`
   - **Labels** (`test_labels.py`):
     - `test_feature_label_map_v2_is_a_dict`
     - `test_label_map_covers_numeric_block`
     - `test_label_map_covers_ordinal_block`
     - `test_label_map_unknown_name_falls_through_to_raw`
     - `test_build_label_map_emits_one_hot_keys_for_fitted_categories`
   - **Build CLI** (`test_build_shap_explainer_cli.py`):
     - `test_cli_builds_explainer_artifact_end_to_end`
     - `test_cli_skips_rent_when_training_skipped_rent`
     - `test_cli_exits_nonzero_when_no_transact_type_provided`
     - `test_build_shap_explainer_cli_does_not_log_contact_fields`
       — grep the captured stdout for any column name matching
       `(contact|dealer|phone|email|photo|url|spid)` — must be
       absent.
   - **Explain CLI** (`test_explain_prediction_cli.py`):
     - `test_cli_prints_top_n_table_for_valid_input`
     - `test_cli_exits_nonzero_when_explainer_artifact_missing`
2. `python -m pytest -m "not realdata"` from repo root still
   passes (no real-data dependency introduced by this spec).
3. `ruff check ml/explainability/ scripts/build_shap_explainer.py
   scripts/explain_prediction.py tests/test_explainer.py
   tests/test_contributions.py tests/test_summary.py
   tests/test_labels.py tests/test_build_shap_explainer_cli.py
   tests/test_explain_prediction_cli.py` reports zero issues.
4. `python -c "from ml.explainability import build_explainer,
   explain_one, global_summary, FEATURE_LABEL_MAP_V2;
   print(EXPLAINER_VERSION)"` from repo root prints `1.0.0`
   without error — public API imports cleanly.
5. `python scripts/build_shap_explainer.py --version v2
   --transact-type sale --transact-type rent` from repo root
   exits 0 (if v2 trained and certified) and prints the summary
   line per `transact_type` — manual smoke test of the
   artifact-build CLI.
6. After running step 5, the following artifacts exist:
   - `models/shap_explainer_sale_v2.pkl`
   - `models/shap_explainer_rent_v2.pkl` (or skipped with INFO
     if Rent was skipped at training time)
   - `models/feature_label_map_v2.json` — parses as JSON, has
     keys for every `num__`/`ord__`/`cat__` column the v2
     preprocessor emits.
7. `data/processed/feature_selection_report.md` after running
   step 5 contains Specs 12–15's prior sections (preserved) **plus**
   an appended "SHAP Explainability Artifact v2" section with a
   top-10 `mean |SHAP value|` table.
8. `data/model_registry.csv` after running step 5 has 2 new rows
   appended (one per `transact_type` for `model_name=
   "shap_explainer"`, `version="v2"`); re-running the CLI does
   not duplicate them.
9. `python scripts/explain_prediction.py --version v2
   --transact-type sale --input-json '{"bedRoom": 3, "bathroom":
   3, "built_up_area": 1450, ...}'` (a synthetic minimal valid
   payload — exact JSON shape covered by the test) prints a
   markdown table with `SHAP_TOP_N` rows, exits 0 — manual smoke
   test of the per-prediction CLI.
10. `git status` after committing shows only the new files listed
    above, the modified `scripts/run_pipeline.py`, and the
    modified `ml/__init__.py` (+ `requirements.txt` if `shap`
    needed pinning). No accidental additions to `app/`, `api/`,
    `data/processed/clean_listings.parquet`, `data/raw/`, or
    `notebooks/`.
11. `CLAUDE.md`'s "Implemented vs stub routes" table is
    unchanged — this spec adds **no routes**. The `POST
    /predict` FastAPI route stays a Stub until a follow-on
    spec wires the precomputed explainer into the request
    handler.
12. `07-TRACKER.md` is updated via `/update-tracker` to mark
    Day 25 ("Final model selection + SHAP explainer validation")
    as **Done** (not just "Certified" per Spec 15) with the
    actual artifact filenames + the top feature from the global
    SHAP summary. The Decision Log gets one new entry: "Built
    precomputed `shap_explainer_v{n}.pkl` artifacts; the
    per-prediction helper reads from the artifact at serving
    time, never rebuilding `TreeExplainer` per request" — per
    the `shap-explainability` skill's "load precomputed
    explainer at startup" convention.
