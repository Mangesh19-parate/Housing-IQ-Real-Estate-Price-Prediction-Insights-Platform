"""Public API for the price-model training submodule (Spec 13).

Re-exports the 6 candidate factory + evaluation + selection +
persistence + report symbols so the script + tests can write
``from ml.training import CANDIDATE_MODELS, evaluate_subset, ...``
without touching the submodule layout.
"""

from ml.training.candidates import (
    CANDIDATE_MODELS,
    PRICE_MODEL_VERSION,
    RENT_MIN_ROWS,
    SHAP_EXPLAINER_VERSION,
    V2_CANDIDATE_MODELS,
    candidate_hyperparameters,
    make_estimator,
    make_v2_estimator,
)
from ml.training.evaluation import (
    IMPROVEMENT_TARGET_PCT,
    SMALL_CITY_TEST_ROWS,
    evaluate_subset,
    improvement_target_met,
    per_city_metrics,
    regression_metrics,
    vs_v1_metrics,
)
from ml.training.levers import (  # noqa: E402
    CITY_CENTERS,
    GEO_NUMERIC_FEATURES,
    METRO_STATIONS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    SECTOR_OUTPUT_COLUMN,
    SECTOR_SMOOTHING_PRIOR_WEIGHT,
    STACKING_CV,
    STACKING_META_ALPHA,
    SectorTargetEncoder,
    add_distance_features,
    haversine_km,
    make_stacking_regressor,
    optuna_search_lgbm,
    optuna_search_xgb,
)
from ml.training.persistence import (
    ARTIFACT_DIR,
    MODEL_REGISTRY_FIELDS,
    PRICE_MODEL_VERSION_V1,
    PRICE_MODEL_VERSION_V2,
    REGISTRY_CSV_PATH,
    append_model_registry,
    load_metrics,
    save_metrics,
    save_price_model,
)
from ml.training.report import (
    DEFAULT_REPORT_PATH,
    append_round_2_3,
    feature_importance_table,
    write_v2_lever_section,
)
from ml.training.selection import select_winner

__all__ = [
    # candidates
    "CANDIDATE_MODELS",
    "PRICE_MODEL_VERSION",
    "RENT_MIN_ROWS",
    "SHAP_EXPLAINER_VERSION",
    "V2_CANDIDATE_MODELS",
    "candidate_hyperparameters",
    "make_estimator",
    "make_v2_estimator",
    # evaluation
    "IMPROVEMENT_TARGET_PCT",
    "SMALL_CITY_TEST_ROWS",
    "evaluate_subset",
    "improvement_target_met",
    "per_city_metrics",
    "regression_metrics",
    "vs_v1_metrics",
    # persistence
    "ARTIFACT_DIR",
    "MODEL_REGISTRY_FIELDS",
    "PRICE_MODEL_VERSION_V1",
    "PRICE_MODEL_VERSION_V2",
    "REGISTRY_CSV_PATH",
    "append_model_registry",
    "load_metrics",
    "save_metrics",
    "save_price_model",
    # report
    "DEFAULT_REPORT_PATH",
    "append_round_2_3",
    "feature_importance_table",
    "write_v2_lever_section",
    # selection
    "select_winner",
    # levers (Spec 14)
    "CITY_CENTERS",
    "GEO_NUMERIC_FEATURES",
    "METRO_STATIONS",
    "OPTUNA_N_TRIALS",
    "OPTUNA_TIMEOUT_SEC",
    "SECTOR_OUTPUT_COLUMN",
    "SECTOR_SMOOTHING_PRIOR_WEIGHT",
    "STACKING_CV",
    "STACKING_META_ALPHA",
    "SectorTargetEncoder",
    "add_distance_features",
    "haversine_km",
    "make_stacking_regressor",
    "optuna_search_lgbm",
    "optuna_search_xgb",
]
