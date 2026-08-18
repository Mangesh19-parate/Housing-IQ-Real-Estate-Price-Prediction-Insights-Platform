"""Tests for ``ml.explainability.contributions`` — per-prediction SHAP helper."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.explainability.contributions import (
    ShapContribution,
    direction_breakdown,
    explain_one,
)


_LABEL_MAP = {
    "feat_a": "Feature A",
    "feat_b": "Feature B",
    "feat_c": "Feature C",
}


def _make_fake_explainer(values: np.ndarray) -> Any:
    """Build a MagicMock that mimics ``shap.TreeExplainer.shap_values``."""
    mock = MagicMock()
    mock.shap_values = MagicMock(return_value=values)
    return mock


# --- ShapContribution dataclass ---


def test_shap_contribution_dataclass_has_expected_fields():
    names = {f.name for f in fields(ShapContribution)}
    assert names == {"feature", "label", "impact", "direction"}


# --- explain_one ---


def test_explain_one_returns_top_n_contributions():
    """A 3-feature row returns at most ``top_n`` (default 7) contributions."""
    values = np.array([[0.30, -0.20, 0.05]])
    explainer = _make_fake_explainer(values)
    contribs = explain_one(
        explainer,
        request_features=np.array([[1.0, 2.0, 3.0]]),
        feature_names=["feat_a", "feat_b", "feat_c"],
        label_map=_LABEL_MAP,
    )
    assert len(contribs) == 3  # only 3 features exist
    assert all(isinstance(c, ShapContribution) for c in contribs)


def test_explain_one_sorts_by_abs_impact_descending():
    """Largest-magnitude SHAP value comes first."""
    values = np.array([[0.05, 0.30, -0.20]])
    explainer = _make_fake_explainer(values)
    contribs = explain_one(
        explainer,
        request_features=np.array([[1.0, 2.0, 3.0]]),
        feature_names=["feat_a", "feat_b", "feat_c"],
        label_map=_LABEL_MAP,
        top_n=2,
    )
    assert len(contribs) == 2
    assert contribs[0].feature == "feat_b"  # 0.30 — largest abs
    assert contribs[1].feature == "feat_c"  # |-0.20| = 0.20 — second
    assert contribs[0].impact == pytest.approx(0.30)
    assert contribs[1].impact == pytest.approx(-0.20)


def test_explain_one_maps_feature_names_through_label_map():
    values = np.array([[0.10, 0.05]])
    explainer = _make_fake_explainer(values)
    contribs = explain_one(
        explainer,
        request_features=np.array([[1.0, 2.0]]),
        feature_names=["feat_a", "feat_b"],
        label_map=_LABEL_MAP,
    )
    assert contribs[0].label == "Feature A"  # not "feat_a"


def test_explain_one_returns_empty_list_for_empty_input():
    explainer = _make_fake_explainer(np.zeros((0, 3)))
    contribs = explain_one(
        explainer,
        request_features=np.zeros((0, 3)),
        feature_names=["feat_a", "feat_b", "feat_c"],
        label_map=_LABEL_MAP,
    )
    assert contribs == []


def test_explain_one_direction_up_for_positive_impact():
    values = np.array([[0.5]])
    explainer = _make_fake_explainer(values)
    contribs = explain_one(
        explainer,
        request_features=np.array([[1.0]]),
        feature_names=["feat_a"],
        label_map=_LABEL_MAP,
    )
    assert contribs[0].direction == "up"


def test_explain_one_direction_down_for_negative_impact():
    values = np.array([[-0.5]])
    explainer = _make_fake_explainer(values)
    contribs = explain_one(
        explainer,
        request_features=np.array([[1.0]]),
        feature_names=["feat_a"],
        label_map=_LABEL_MAP,
    )
    assert contribs[0].direction == "down"


# --- direction_breakdown ---


def test_direction_breakdown_counts_up_and_down():
    contribs = [
        ShapContribution("a", "A", 0.10, "up"),
        ShapContribution("b", "B", 0.05, "up"),
        ShapContribution("c", "C", -0.20, "down"),
    ]
    counts = direction_breakdown(contribs)
    assert counts == {"up": 2, "down": 1}
