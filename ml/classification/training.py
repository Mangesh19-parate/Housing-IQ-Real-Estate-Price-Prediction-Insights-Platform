"""Classification model training (Spec 23).

Authority:
    - .claude/specs/23-classification-model-training.md
    - docs/02-TRD.md §U-TRD-8 (classification targets + selection metrics)
    - docs/10-FINALIZED-INPUT-SCHEMA.md §3 (label definitions)

Two classifiers:
    1. good_deal_verdict (primary) — 3-class: Good Deal, Fair Deal, Overpriced
       Selection metric: **recall on "Good Deal" class (index 0)**
    2. price_tier (secondary) — 4-class: Budget, Mid-Range, Premium, Luxury
       Selection metric: **macro-F1**

Both reuse the leakage-safe feature frame from Spec 22 and the
ColumnTransformer from ml.features.preprocessor.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.features.preprocessor import fit_preprocessor, transform_with_preprocessor
from ml.features.locality_aggregator import LocalityAggregator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned label constants (must match Spec 21: tiers.py + verdicts.py)
# ---------------------------------------------------------------------------

#: 3-class good_deal_verdict labels in integer encoding order (0, 1, 2)
GOOD_DEAL_LABELS: tuple[str, ...] = ("Good Deal", "Fair Deal", "Overpriced")

#: 4-class price_tier labels in integer encoding order (0, 1, 2, 3)
PRICE_TIER_LABELS: tuple[str, ...] = ("Budget", "Mid-Range", "Premium", "Luxury")

# ---------------------------------------------------------------------------
# Model candidates (pinned hyperparameters for reproducibility)
# ---------------------------------------------------------------------------

#: Baseline model — Logistic Regression
_LOGREG_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

#: Candidate 1 — Random Forest
_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

#: Candidate 2 — XGBoost
_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

#: All candidates for good_deal_verdict (primary)
_GOOD_DEAL_CANDIDATES: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(**_LOGREG_PARAMS),
    "RandomForestClassifier": RandomForestClassifier(**_RF_PARAMS),
    "XGBClassifier": XGBClassifier(**_XGB_PARAMS),
}

#: All candidates for price_tier (secondary)
_PRICE_TIER_CANDIDATES: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(**_LOGREG_PARAMS),
    "RandomForestClassifier": RandomForestClassifier(**_RF_PARAMS),
    "XGBClassifier": XGBClassifier(**_XGB_PARAMS),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    preprocessor: ColumnTransformer,
    classifier: Any,
) -> Pipeline:
    """Wrap a fitted preprocessor + classifier into a single Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
    primary_class_index: int = 0,
) -> dict[str, Any]:
    """Compute full metric suite for a classifier.

    Args:
        y_true: Ground truth labels (integer encoded).
        y_pred: Predicted labels (integer encoded).
        labels: Human-readable label names in integer order.
        primary_class_index: Index of the "primary" class for recall tracking
            (0 for Good Deal in good_deal_verdict).

    Returns:
        Dict with keys:
        - accuracy, macro_f1
        - per_class_precision, per_class_recall, per_class_f1 (dicts keyed by label)
        - confusion_matrix (list[list[int]])
        - primary_class_recall (float) — recall on the primary class
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    primary_recall = float(per_class_recall[primary_class_index])

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_precision": {
            labels[i]: float(per_class_precision[i]) for i in range(len(labels))
        },
        "per_class_recall": {
            labels[i]: float(per_class_recall[i]) for i in range(len(labels))
        },
        "per_class_f1": {
            labels[i]: float(per_class_f1[i]) for i in range(len(labels))
        },
        "confusion_matrix": cm.tolist(),
        "primary_class_recall": primary_recall,
    }


def _select_best_model(
    candidates: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    locality_aggregator: LocalityAggregator,
    selection_metric: str,
    primary_class_index: int,
    labels: tuple[str, ...],
) -> tuple[str, Pipeline, dict[str, Any]]:
    """Train all candidates, evaluate on validation, pick best by selection_metric.

    Args:
        candidates: Dict of model_name -> unfitted estimator.
        X_train, y_train: Training split (raw feature frame + labels).
        X_val, y_val: Validation split (raw feature frame + labels).
        locality_aggregator: Fitted LocalityAggregator for preprocessing.
        selection_metric: "primary_class_recall" or "macro_f1".
        primary_class_index: Index of primary class (0 for Good Deal).
        labels: Label names for metric computation.

    Returns:
        (best_model_name, best_pipeline, best_metrics_on_val)
    """
    # Fit preprocessor on training data once
    preprocessor = fit_preprocessor(X_train, locality_aggregator)

    best_score = -1.0
    best_name = None
    best_pipeline = None
    best_metrics = None

    for name, estimator in candidates.items():
        logger.info("Training candidate: %s", name)

        # Fit classifier on preprocessed training data
        pipe = _make_pipeline(preprocessor, estimator)
        pipe.fit(X_train, y_train)

        # Evaluate on validation
        y_val_pred = pipe.predict(X_val)
        metrics = _compute_metrics(
            y_val.values,
            y_val_pred,
            labels=labels,
            primary_class_index=primary_class_index,
        )

        score = metrics[selection_metric]
        logger.info(
            "  %s: %s=%.4f, accuracy=%.4f, macro_f1=%.4f",
            name,
            selection_metric,
            score,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe
            best_metrics = metrics

    logger.info(
        "Selected best model: %s (%s=%.4f)",
        best_name,
        selection_metric,
        best_score,
    )
    return best_name, best_pipeline, best_metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_good_deal_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    locality_aggregator: LocalityAggregator,
    class_weight: dict[int, float] | str = "balanced",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the primary 3-class good_deal_verdict classifier.

    Selection criterion: **maximize recall on "Good Deal" class (index 0)**.

    Args:
        X_train: Training feature frame (raw, output of build_classifier_feature_frame).
        y_train: Training labels (integer 0,1,2 for Good Deal, Fair Deal, Overpriced).
        X_val: Validation feature frame (same schema as X_train).
        y_val: Validation labels.
        locality_aggregator: Fitted LocalityAggregator for target encoding.
        class_weight: Passed to all candidates (default "balanced").
        random_state: Random seed (default 42).

    Returns:
        (best_pipeline, best_val_metrics) where best_pipeline is a fitted
        Pipeline(preprocessor + classifier) and best_val_metrics contains
        the full validation metrics dict.
    """
    # Override class_weight in candidate params if provided
    candidates = {}
    for name, estimator in _GOOD_DEAL_CANDIDATES.items():
        # Clone with updated class_weight
        params = estimator.get_params()
        params["class_weight"] = class_weight
        params["random_state"] = random_state
        if name == "LogisticRegression":
            candidates[name] = LogisticRegression(**params)
        elif name == "RandomForestClassifier":
            candidates[name] = RandomForestClassifier(**params)
        else:
            candidates[name] = XGBClassifier(**params)

    best_name, best_pipe, best_metrics = _select_best_model(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=locality_aggregator,
        selection_metric="primary_class_recall",  # Good Deal recall
        primary_class_index=0,  # Good Deal = index 0
        labels=GOOD_DEAL_LABELS,
    )

    # Attach model name for downstream logging
    best_pipe.best_model_name_ = best_name  # type: ignore[attr-defined]
    return best_pipe, best_metrics


def train_price_tier_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    locality_aggregator: LocalityAggregator,
    class_weight: dict[int, float] | str = "balanced",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the secondary 4-class price_tier classifier.

    Selection criterion: **maximize macro-F1** (balanced across 4 classes).

    Args:
        X_train: Training feature frame (raw, output of build_classifier_feature_frame).
        y_train: Training labels (integer 0,1,2,3 for Budget, Mid-Range, Premium, Luxury).
        X_val: Validation feature frame (same schema as X_train).
        y_val: Validation labels.
        locality_aggregator: Fitted LocalityAggregator for target encoding.
        class_weight: Passed to all candidates (default "balanced").
        random_state: Random seed (default 42).

    Returns:
        (best_pipeline, best_val_metrics) where best_pipeline is a fitted
        Pipeline(preprocessor + classifier) and best_val_metrics contains
        the full validation metrics dict.
    """
    candidates = {}
    for name, estimator in _PRICE_TIER_CANDIDATES.items():
        params = estimator.get_params()
        params["class_weight"] = class_weight
        params["random_state"] = random_state
        if name == "LogisticRegression":
            candidates[name] = LogisticRegression(**params)
        elif name == "RandomForestClassifier":
            candidates[name] = RandomForestClassifier(**params)
        else:
            candidates[name] = XGBClassifier(**params)

    best_name, best_pipe, best_metrics = _select_best_model(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        locality_aggregator=locality_aggregator,
        selection_metric="macro_f1",
        primary_class_index=0,  # not used for macro_f1 selection, but required
        labels=PRICE_TIER_LABELS,
    )

    best_pipe.best_model_name_ = best_name  # type: ignore[attr-defined]
    return best_pipe, best_metrics


def evaluate_classifier(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, ...],
    primary_class_index: int = 0,
) -> dict[str, Any]:
    """Evaluate a fitted classifier pipeline on a dataset.

    Args:
        pipeline: Fitted Pipeline (preprocessor + classifier).
        X: Feature frame (raw, same schema as training).
        y: True labels (integer encoded).
        labels: Label names in integer order.
        primary_class_index: Index of primary class for recall tracking.

    Returns:
        Full metrics dict (same structure as _compute_metrics).
    """
    y_pred = pipeline.predict(X)
    return _compute_metrics(
        y.values,
        y_pred,
        labels=labels,
        primary_class_index=primary_class_index,
    )


__all__ = [
    "GOOD_DEAL_LABELS",
    "PRICE_TIER_LABELS",
    "train_good_deal_classifier",
    "train_price_tier_classifier",
    "evaluate_classifier",
]