"""Tests for ``ml.explainability.explainer`` — TreeExplainer factory + persistence."""

from __future__ import annotations

import numpy as np
import pytest
import shap
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.explainability.explainer import (
    EXPLAINER_VERSION,
    SHAP_TOP_N,
    build_explainer,
    load_explainer,
    save_explainer,
)


def _tiny_xgb_pipeline() -> Pipeline:
    """Build a 2-step Pipeline(preprocessor, tree) for tests.

    Uses a 2-feature numeric matrix + an XGBRegressor with
    ``n_estimators=2`` so the explainer fits in milliseconds.
    """
    import xgboost  # noqa: WPS433  (lazy — only the XGB tests need this)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + 0.1 * X[:, 1]
    pipe = Pipeline(
        steps=[
            ("preproc", StandardScaler()),
            ("est", xgboost.XGBRegressor(n_estimators=2, max_depth=2, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# --- pinned constants ---


def test_explainer_version_is_pinned():
    assert EXPLAINER_VERSION == "1.0.0"


def test_shap_top_n_is_seven():
    assert SHAP_TOP_N == 7


# --- build_explainer ---


def test_build_explainer_returns_tree_explainer(tmp_path):
    model = _tiny_xgb_pipeline()
    explainer = build_explainer(model)
    assert isinstance(explainer, shap.TreeExplainer)


def test_build_explainer_rejects_non_tree_model():
    """Defensive guard: a Ridge pipeline is not a tree estimator."""
    pipe = Pipeline(steps=[("preproc", StandardScaler()), ("est", Ridge(alpha=1.0))])
    pipe.fit(np.random.default_rng(0).normal(size=(20, 2)), np.random.default_rng(1).normal(size=20))
    with pytest.raises(ValueError) as excinfo:
        build_explainer(pipe)
    # The Ridge class name is in the error message — pinned by spec.
    assert "Ridge" in str(excinfo.value)


# --- save / load round-trip ---


def test_save_explainer_writes_versioned_filename(tmp_path):
    model = _tiny_xgb_pipeline()
    explainer = build_explainer(model)
    path = save_explainer(explainer, transact_type="sale", version="v2", out_dir=tmp_path)
    assert path.exists()
    assert path.name == "shap_explainer_sale_v2.pkl"


def test_save_and_load_explainer_round_trip(tmp_path):
    """Round-trip preserves the underlying model reference (Rules §2.6)."""
    model = _tiny_xgb_pipeline()
    explainer = build_explainer(model)
    save_explainer(explainer, transact_type="sale", version="v2", out_dir=tmp_path)

    loaded = load_explainer("sale", "v2", tmp_path)
    assert isinstance(loaded, shap.TreeExplainer)
    # Underlying estimator preserved — same predictions on a test row.
    rng = np.random.default_rng(123)
    X_test = rng.normal(size=(1, 2))
    base_pred = explainer.model.predict(X_test)
    loaded_pred = loaded.model.predict(X_test)
    np.testing.assert_allclose(base_pred, loaded_pred, rtol=1e-5)


def test_load_explainer_raises_with_expected_path_on_miss(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_explainer("sale", "v9", tmp_path)
    # The resolved path appears in the message (pinned by spec).
    assert "shap_explainer_sale_v9.pkl" in str(excinfo.value)
