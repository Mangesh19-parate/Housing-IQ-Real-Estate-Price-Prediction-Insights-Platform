"""Improvement levers for the v2 price regression (Spec 14).

Re-exports the four lever constructors so callers can write
``from ml.training.levers import optuna_search_xgb, ...`` without
touching the submodule layout. Each lever is independently testable
and composable — the v2 training script wires them together.

Levers implemented here:
    - Lever 1: ``make_stacking_regressor`` (stacking ensemble).
    - Lever 2: ``optuna_search_xgb``, ``optuna_search_lgbm``
      (Bayesian hyperparameter search).
    - Lever 3: ``add_distance_features`` (geo: distance-to-CBD /
      distance-to-nearest-metro).
    - Lever 4: ``SectorTargetEncoder`` (smoothed target encoding for
      ``(city, sector)``).

Each lever is a pure module — no I/O, no global state mutation beyond
the Optuna verbosity silencing (one-time at import).
"""

from ml.training.levers.geospatial import (
    CITY_CENTERS,
    GEO_NUMERIC_FEATURES,
    METRO_STATIONS,
    add_distance_features,
    haversine_km,
)
from ml.training.levers.optuna_search import (
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    optuna_search_lgbm,
    optuna_search_xgb,
)
from ml.training.levers.stacking import (
    STACKING_CV,
    STACKING_META_ALPHA,
    make_stacking_regressor,
)
from ml.training.levers.target_encoding import (
    SECTOR_OUTPUT_COLUMN,
    SECTOR_SMOOTHING_PRIOR_WEIGHT,
    SectorTargetEncoder,
)

__all__ = [
    # Lever 1
    "make_stacking_regressor",
    "STACKING_CV",
    "STACKING_META_ALPHA",
    # Lever 2
    "optuna_search_xgb",
    "optuna_search_lgbm",
    "OPTUNA_N_TRIALS",
    "OPTUNA_TIMEOUT_SEC",
    # Lever 3
    "add_distance_features",
    "haversine_km",
    "METRO_STATIONS",
    "CITY_CENTERS",
    "GEO_NUMERIC_FEATURES",
    # Lever 4
    "SectorTargetEncoder",
    "SECTOR_OUTPUT_COLUMN",
    "SECTOR_SMOOTHING_PRIOR_WEIGHT",
]
