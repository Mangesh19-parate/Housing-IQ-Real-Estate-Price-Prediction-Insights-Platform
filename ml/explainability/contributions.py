"""Per-prediction SHAP contributions for the Predict page UI.

Output shape matches ``05-BACKEND-SCHEMA.md`` §7
``PredictionResponse.shap_contributions`` extended with the
``direction`` field the ``shap-explainability`` skill calls for
(``"up"`` / ``"down"`` so the UI can apply the color tokens
without re-computing the sign).

Ponytail: one ``sorted(...)`` + one slice + one comprehension.
No numpy juggling; the SHAP library already returned a numpy array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import shap

from ml.explainability.labels import resolve_label

logger = logging.getLogger(__name__)


Direction = Literal["up", "down"]


@dataclass(frozen=True)
class ShapContribution:
    """One feature's contribution to a single prediction.

    Fields:
        feature: raw internal feature name (e.g. ``"num__built_up_area"``).
        label:   human-readable label from the label map.
        impact:  SHAP value in log-price space (model trains on
                 ``log1p(price)``). Sign carries direction; the
                 FastAPI display layer re-scales to ``pct_impact``
                 (``exp(impact) - 1``) for user-facing copy.
        direction: ``"up"`` if ``impact > 0``, else ``"down"``.
    """

    feature: str
    label: str
    impact: float
    direction: Direction


def explain_one(
    explainer: shap.TreeExplainer,
    request_features: np.ndarray,
    feature_names: list[str],
    label_map: dict[str, str],
    top_n: int = 7,
) -> list[ShapContribution]:
    """Compute the top-``top_n`` SHAP contributions for a single prediction.

    ``request_features`` is expected to be a 1-row numpy array
    (shape ``(1, n_features)``) — the preprocessor-transformed
    feature vector for the request. The function is shape-tolerant:
    a multi-row input is explained row-by-row and the *first* row's
    contributions are returned (pinned by DoD test
    ``test_explain_one_returns_top_n_contributions``).

    Empty input → ``[]`` + WARNING (defensive). Zero-variance row
    where all SHAP values are NaN → ``[]`` + WARNING.
    """
    if request_features is None or len(request_features) == 0:
        logger.warning("explain_one called with empty input; returning []")
        return []
    if request_features.shape[0] > 1:
        logger.warning(
            "explain_one received %d rows; explaining only the first row",
            request_features.shape[0],
        )
    if len(feature_names) != request_features.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"input columns ({request_features.shape[1]})"
        )

    shap_values = explainer.shap_values(request_features)
    # SHAP occasionally returns a list (e.g. multi-output classifiers);
    # for regressors it's always a single ndarray. Defensive unwrap.
    if isinstance(shap_values, list):
        if not shap_values:
            logger.warning("explainer returned empty shap_values list; returning []")
            return []
        shap_values = shap_values[0]

    # Single-row case: take row 0.
    row = np.asarray(shap_values[0], dtype=float)
    if not np.isfinite(row).any():
        logger.warning("explain_one row is all NaN/Inf; returning []")
        return []

    # Sort by absolute impact descending, slice top_n, map labels.
    order = np.argsort(np.abs(row))[::-1][:top_n]
    contributions: list[ShapContribution] = []
    for idx in order:
        name = feature_names[int(idx)]
        impact = float(row[int(idx)])
        contributions.append(
            ShapContribution(
                feature=name,
                label=resolve_label(name, label_map),
                impact=impact,
                direction="up" if impact > 0 else "down",
            )
        )
    return contributions


def direction_breakdown(contributions: list[ShapContribution]) -> dict[str, int]:
    """Count contributions by direction. Used by the per-prediction UI
    summary line and pinned by ``test_direction_breakdown_counts_up_and_down``.
    """
    up = sum(1 for c in contributions if c.direction == "up")
    down = sum(1 for c in contributions if c.direction == "down")
    return {"up": up, "down": down}


__all__ = ["ShapContribution", "explain_one", "direction_breakdown", "Direction"]


# ponytail: no `format_for_template` helper — the future FastAPI route
# owns JSON serialization (per the Backend Schema §7 wire format).
# This module's output type is the only thing the route needs.
