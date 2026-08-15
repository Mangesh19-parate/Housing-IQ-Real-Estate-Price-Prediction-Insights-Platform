"""Lever 2 — Bayesian hyperparameter search (Spec 14).

Two thin Optuna wrappers:
    - ``optuna_search_xgb`` — search space for ``XGBRegressor``.
    - ``optuna_search_lgbm`` — search space for ``LGBMRegressor``.

Both return ``{"best_params": {...}, "best_value": float}`` and use
``TPESampler(seed=42)`` for determinism (Rules §5.4).

Pinned constants per literature (S8):
    - ``OPTUNA_N_TRIALS = 40`` — S8 recommends 30–50 for boosted trees.
    - ``OPTUNA_TIMEOUT_SEC = 600`` — 10-minute upper bound prevents
      runaway CI runs; if a trial hits the timeout Optuna returns the
      best-so-far and continues (no error).

Optuna's study-level logging is silenced at import
(``optuna.logging.set_verbosity(WARNING)``) so trial-by-trial output
doesn't drown the stdlib logs. The ``best_value`` is logged at INFO
once per search by the caller.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Final

import numpy as np
import optuna

# Silence Optuna's chatty default logging — one-time at import. The
# caller logs the best value at INFO so the script's summary line
# still surfaces the result.
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Optuna also emits Python warnings on some platforms; suppress those
# too so the training script's stdout stays clean.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)

#: Pinned per spec — S8 recommends 30–50 trials for boosted trees.
OPTUNA_N_TRIALS: Final[int] = 40

#: Pinned per spec — 10 minutes prevents runaway CI runs. If a trial
#: hits the timeout Optuna returns the best-so-far and continues.
OPTUNA_TIMEOUT_SEC: Final[int | None] = 600


def _make_objective(
    estimator_cls,
    fixed_params: dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    search_space,
    random_state: int,
):
    """Build an Optuna objective that fits + scores one trial.

    The objective uses ``neg_root_mean_squared_error`` (sklearn 1.4+)
    so higher is better; the caller negates back when reporting.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params}
        for name, value in search_space(trial).items():
            params[name] = value
        try:
            from sklearn.metrics import root_mean_squared_error

            def score_fn(y, p):
                return -float(root_mean_squared_error(y, p))
        except ImportError:  # sklearn < 1.4 fallback
            from sklearn.metrics import mean_squared_error

            def score_fn(y, p):
                return -float(np.sqrt(mean_squared_error(y, p)))

        est = estimator_cls(random_state=random_state, **params)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_val)
        # Higher-is-better for Optuna (negate RMSE).
        return score_fn(y_val, y_pred)

    return objective


def optuna_search_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout_sec: int | None = OPTUNA_TIMEOUT_SEC,
    random_state: int = 42,
) -> dict[str, Any]:
    """Search XGBoost hyperparameters; return best params + best value.

    Best value is the **negative** RMSE on the validation slice
    (Optuna maximizes). Callers can negate back to get RMSE.

    Search space (per spec, loose bounds from literature S8):
        - ``max_depth ∈ [3, 10]``
        - ``learning_rate ∈ loguniform[0.01, 0.3]``
        - ``n_estimators ∈ [100, 1000]``
        - ``subsample ∈ [0.6, 1.0]``
        - ``colsample_bytree ∈ [0.6, 1.0]``
        - ``min_child_weight ∈ [1, 10]``
        - ``reg_alpha ∈ loguniform[1e-8, 1.0]``
        - ``reg_lambda ∈ loguniform[1e-8, 1.0]``

    Fixed params: ``objective="reg:squarederror"``, ``tree_method="hist"``,
    ``n_jobs=-1``, ``verbosity=0``.
    """
    from xgboost import XGBRegressor

    fixed = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }

    def _space(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 1.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 1.0, log=True
            ),
        }

    objective = _make_objective(
        XGBRegressor,
        fixed,
        X_train,
        y_train,
        X_val,
        y_val,
        _space,
        random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=False,
    )

    best_params = {**fixed, **dict(study.best_params)}
    best_value = float(study.best_value)  # negative RMSE
    logger.info(
        "optuna_search_xgb: best_value=%.4f (neg RMSE), n_trials=%d",
        best_value,
        len(study.trials),
    )
    return {"best_params": best_params, "best_value": best_value}


def optuna_search_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout_sec: int | None = OPTUNA_TIMEOUT_SEC,
    random_state: int = 42,
) -> dict[str, Any]:
    """Search LightGBM hyperparameters; return best params + best value.

    Search space (LGBM-flavored):
        - ``num_leaves ∈ [15, 255]``
        - ``max_depth ∈ [3, 10]``
        - ``learning_rate ∈ loguniform[0.01, 0.3]``
        - ``n_estimators ∈ [100, 1000]``
        - ``subsample ∈ [0.6, 1.0]``
        - ``colsample_bytree ∈ [0.6, 1.0]``
        - ``min_child_samples ∈ [5, 50]``
        - ``reg_alpha ∈ loguniform[1e-8, 1.0]``
        - ``reg_lambda ∈ loguniform[1e-8, 1.0]``

    Fixed params: ``objective="regression"``, ``metric="rmse"``,
    ``n_jobs=-1``, ``verbose=-1``.
    """
    from lightgbm import LGBMRegressor

    fixed = {
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": -1,
        "verbose": -1,
    }

    def _space(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 50
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 1.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 1.0, log=True
            ),
        }

    objective = _make_objective(
        LGBMRegressor,
        fixed,
        X_train,
        y_train,
        X_val,
        y_val,
        _space,
        random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        show_progress_bar=False,
    )

    best_params = {**fixed, **dict(study.best_params)}
    best_value = float(study.best_value)
    logger.info(
        "optuna_search_lgbm: best_value=%.4f (neg RMSE), n_trials=%d",
        best_value,
        len(study.trials),
    )
    return {"best_params": best_params, "best_value": best_value}


__all__ = [
    "OPTUNA_N_TRIALS",
    "OPTUNA_TIMEOUT_SEC",
    "optuna_search_xgb",
    "optuna_search_lgbm",
]
