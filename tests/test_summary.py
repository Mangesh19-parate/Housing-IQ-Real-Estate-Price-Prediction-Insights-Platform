"""Tests for ``ml.explainability.summary`` — global SHAP summary + report writer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.explainability.summary import (
    GLOBAL_N_SAMPLES,
    TOP_K_SUMMARY,
    global_summary,
    write_summary_section,
)


_FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]


def _make_fake_explainer(values: np.ndarray) -> Any:
    mock = MagicMock()
    mock.shap_values = MagicMock(return_value=values)
    return mock


def test_global_summary_returns_dict_of_mean_abs_shap():
    """mean |SHAP| per feature over the sampled rows."""
    values = np.array([
        [0.10, 0.05, 0.01],
        [0.20, 0.05, 0.02],
    ])
    explainer = _make_fake_explainer(values)
    summary = global_summary(
        explainer,
        X_background=np.zeros((10, 3)),
        feature_names=_FEATURE_NAMES,
        n_samples=2,
        random_state=42,
    )
    assert set(summary.keys()) == set(_FEATURE_NAMES)
    assert summary["feat_a"] == pytest.approx(0.15)
    assert summary["feat_b"] == pytest.approx(0.05)
    assert summary["feat_c"] == pytest.approx(0.015)


def test_global_summary_uses_pinned_n_samples():
    """Default n_samples is 200 (spec-pinned)."""
    assert GLOBAL_N_SAMPLES == 200


def test_global_summary_deterministic_with_random_state():
    """Two calls with the same seed return the same summary."""
    rng_vals = np.random.default_rng(0)
    rng_bg = np.random.default_rng(1)
    values = np.abs(rng_vals.normal(size=(20, 3)))
    explainer = _make_fake_explainer(values)
    bg = rng_bg.normal(size=(50, 3))
    s1 = global_summary(explainer, bg, _FEATURE_NAMES, n_samples=10, random_state=42)
    s2 = global_summary(explainer, bg, _FEATURE_NAMES, n_samples=10, random_state=42)
    for name in _FEATURE_NAMES:
        assert s1[name] == pytest.approx(s2[name])


def test_write_summary_section_appends_not_overwrites(tmp_path):
    """The report is opened in append mode — prior content is preserved."""
    report = tmp_path / "feature_selection_report.md"
    sentinel = "ROUND 1 SENTINEL LINE — preserved across append"
    report.write_text(sentinel + "\n", encoding="utf-8")

    summary = {"feat_a": 0.5, "feat_b": 0.2, "feat_c": 0.1}
    write_summary_section(summary, version="v2", out_path=report, top_k=3)

    content = report.read_text(encoding="utf-8")
    assert sentinel in content
    assert "SHAP Explainability Artifact v2" in content


def test_write_summary_section_includes_section_header(tmp_path):
    report = tmp_path / "feature_selection_report.md"
    summary = {"feat_a": 0.5, "feat_b": 0.2, "feat_c": 0.1}
    write_summary_section(summary, version="v2", out_path=report, top_k=3)
    content = report.read_text(encoding="utf-8")
    assert "## SHAP Explainability Artifact v2" in content


def test_top_k_summary_default_is_ten():
    assert TOP_K_SUMMARY == 10
