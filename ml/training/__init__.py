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
    candidate_hyperparameters,
    make_estimator,
)
from ml.training.evaluation import (
    SMALL_CITY_TEST_ROWS,
    evaluate_subset,
    per_city_metrics,
    regression_metrics,
)
from ml.training.persistence import (
    ARTIFACT_DIR,
    MODEL_REGISTRY_FIELDS,
    REGISTRY_CSV_PATH,
    append_model_registry,
    save_metrics,
    save_price_model,
)
from ml.training.report import (
    DEFAULT_REPORT_PATH,
    append_round_2_3,
    feature_importance_table,
)
from ml.training.selection import select_winner

__all__ = [
    # candidates
    "CANDIDATE_MODELS",
    "PRICE_MODEL_VERSION",
    "RENT_MIN_ROWS",
    "SHAP_EXPLAINER_VERSION",
    "candidate_hyperparameters",
    "make_estimator",
    # evaluation
    "SMALL_CITY_TEST_ROWS",
    "evaluate_subset",
    "per_city_metrics",
    "regression_metrics",
    # persistence
    "ARTIFACT_DIR",
    "MODEL_REGISTRY_FIELDS",
    "REGISTRY_CSV_PATH",
    "append_model_registry",
    "save_metrics",
    "save_price_model",
    # report
    "DEFAULT_REPORT_PATH",
    "append_round_2_3",
    "feature_importance_table",
    # selection
    "select_winner",
]
