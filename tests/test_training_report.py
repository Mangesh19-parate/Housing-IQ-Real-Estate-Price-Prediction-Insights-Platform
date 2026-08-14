"""Tests for ml.training.report (Spec 13 Phase A)."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from ml.training.report import (
    DEFAULT_REPORT_PATH,
    append_round_2_3,
    feature_importance_table,
)

# ---------------------------------------------------------------------------
# feature_importance_table
# ---------------------------------------------------------------------------


def test_feature_importance_table_returns_top_n_sorted():
    est = DecisionTreeRegressor(random_state=42)
    # Fit on a 3-feature toy so feature_importances_ is non-trivial.
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 3 + X[:, 2] * 1.5 + rng.normal(scale=0.1, size=50)
    est.fit(X, y)

    rows = feature_importance_table(est, ["a", "b", "c"], top_n=2)
    assert len(rows) == 2
    # Sorted descending.
    assert rows[0][1] >= rows[1][1]
    # Names + values.
    assert all(isinstance(r[0], str) for r in rows)
    assert all(isinstance(r[1], float) for r in rows)


def test_feature_importance_table_returns_empty_for_non_tree():
    est = LinearRegression()
    rows = feature_importance_table(est, ["a", "b"], top_n=5)
    assert rows == []


# ---------------------------------------------------------------------------
# append_round_2_3
# ---------------------------------------------------------------------------


def test_append_round_2_3_preserves_existing_content(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "# Feature selection report\n\n"
        "## Round 1 — Correlation filter\n\n"
        "Existing Step 12 content that must survive.\n",
        encoding="utf-8",
    )

    append_round_2_3(
        report,
        rf_importances=[("a", 0.5), ("b", 0.3)],
        gb_importances=[("a", 0.6), ("b", 0.2)],
        xgb_importances=[("a", 0.7), ("b", 0.1)],
        perm_importances=[("a", 0.4), ("b", 0.35)],
        shap_importances=[("a", 0.9), ("b", 0.05)],
        winner_name="xgboost",
        chosen_metrics={
            "train": {"r2": 0.95, "mae": 100.0, "rmse": 200.0, "mape": 5.0},
            "val":   {"r2": 0.85, "mae": 150.0, "rmse": 250.0, "mape": 8.0},
            "test":  {"r2": 0.80, "mae": 200.0, "rmse": 300.0, "mape": 10.0},
        },
    )

    text = report.read_text(encoding="utf-8")
    assert "## Round 1 — Correlation filter" in text
    assert "Existing Step 12 content that must survive." in text
    assert "## Round 2 — Tree-based" in text
    assert "## Round 3 — SHAP ranking" in text
    assert "## Final feature list & rationale" in text
    assert "xgboost" in text


def test_append_round_2_3_handles_non_tree_winner(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Header\n", encoding="utf-8")

    append_round_2_3(
        report,
        rf_importances=[("a", 0.5)],
        gb_importances=[("a", 0.5)],
        xgb_importances=[("a", 0.5)],
        perm_importances=[("a", 0.5)],
        shap_importances=[],  # non-tree winner
        winner_name="ridge",
    )
    text = report.read_text(encoding="utf-8")
    assert "non-tree model" in text


def test_default_report_path_constant_is_pinned():
    assert DEFAULT_REPORT_PATH.as_posix().endswith(
        "data/processed/feature_selection_report.md"
    )
