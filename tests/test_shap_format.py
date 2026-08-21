"""Tests for the SHAP template formatter (Spec 19).

Pins the Flask-side SHAP renderer's behaviour: human-readable
label lookup (static + on-disk overlay), up/down/neutral
direction labelling, magnitude-normalised ``pct``, top-N cap,
input-order preservation, and the no-``ml.*``-imports guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.shap_format import (
    TOP_N_DEFAULT,
    format_shap_for_template,
    summarize_direction,
)

# ---------- helpers ----------------------------------------------------------

def _row(feature: str, impact: float) -> dict:
    return {"feature": feature, "impact": impact}


# ---------- format_shap_for_template -----------------------------------------

def test_format_shap_for_template_assigns_label_from_map() -> None:
    rows = format_shap_for_template([_row("num__built_up_area", 0.18)])
    assert rows[0]["label"] == "Built-up Area (sqft)"


def test_format_shap_for_template_assigns_label_from_disk_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "feature_label_map_v2.json").write_text(
        # JSON requires double-quoted keys; safe to write directly.
        '{"num__built_up_area": "Custom Overlay Label"}',
        encoding="utf-8",
    )
    # Point the helper's module-level models dir at tmp_path; clear
    # the lru_cache so the new path is picked up.
    import app.services.shap_format as shap_format

    monkeypatch.setattr(shap_format, "_LABEL_MAP_DIR", tmp_path)
    shap_format._get_label_map.cache_clear()
    try:
        rows = format_shap_for_template([_row("num__built_up_area", 0.18)])
    finally:
        shap_format._get_label_map.cache_clear()
    assert rows[0]["label"] == "Custom Overlay Label"


def test_format_shap_for_template_falls_back_to_raw_feature_name() -> None:
    rows = format_shap_for_template([_row("num__exotic_thing", 0.05)])
    # No label in the map → raw feature code is shown verbatim.
    assert rows[0]["label"] == "num__exotic_thing"


def test_format_shap_for_template_marks_direction_up_for_positive() -> None:
    rows = format_shap_for_template([_row("num__built_up_area", 0.10)])
    assert rows[0]["direction"] == "up"


def test_format_shap_for_template_marks_direction_down_for_negative() -> None:
    rows = format_shap_for_template([_row("num__built_up_area", -0.10)])
    assert rows[0]["direction"] == "down"


def test_format_shap_for_template_marks_direction_neutral_for_zero() -> None:
    rows = format_shap_for_template([_row("num__built_up_area", 0.0)])
    assert rows[0]["direction"] == "neutral"


def test_format_shap_for_template_caps_at_top_n() -> None:
    rows = [_row(f"num__feat_{i}", float(i)) for i in range(12)]
    out = format_shap_for_template(rows, top_n=5)
    assert len(out) == 5


def test_format_shap_for_template_pct_normalises_to_max_abs() -> None:
    rows = [
        _row("num__a", 0.20),
        _row("num__b", -0.10),
        _row("num__c", 0.05),
    ]
    out = format_shap_for_template(rows)
    abs_pcts = [abs(r["pct"]) for r in out]
    # The maximum-magnitude row always normalises to ±1.0.
    assert max(abs_pcts) == pytest.approx(1.0)
    # The −0.10 row scales linearly: 0.10 / 0.20 = 0.5.
    assert out[1]["pct"] == pytest.approx(-0.5)
    # Direction preserved.
    assert out[0]["direction"] == "up"
    assert out[1]["direction"] == "down"
    assert out[2]["direction"] == "up"


def test_format_shap_for_template_preserves_input_order() -> None:
    rows = [
        _row("num__a", 0.5),
        _row("num__b", 0.3),
        _row("num__c", 0.1),
    ]
    out = format_shap_for_template(rows)
    assert [r["feature"] for r in out] == ["num__a", "num__b", "num__c"]


def test_format_shap_for_template_returns_empty_for_empty_input() -> None:
    assert format_shap_for_template([]) == []


def test_format_shap_for_template_handles_all_zero_impacts() -> None:
    rows = [_row("num__a", 0.0), _row("num__b", 0.0)]
    out = format_shap_for_template(rows)
    assert all(r["direction"] == "neutral" for r in out)
    # All-zero magnitude → ``pct`` falls back to 0.0 (max_abs guard).
    assert all(r["pct"] == 0.0 for r in out)


def test_format_shap_for_template_does_not_import_ml_or_models() -> None:
    # The Flask-side helper must not load model code (Rules §5.1).
    # We allow ``ml.explainability.labels`` (the display-only label
    # map) but block every other ``ml.*`` submodule and any
    # ``models.*`` import. We check the module's *own* top-level
    # imports via AST (so other test modules already loaded into
    # ``sys.modules`` don't pollute the result), not the global
    # import state.
    import ast

    helper_path = Path(__file__).parent.parent / "app" / "services" / "shap_format.py"
    source = helper_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    # Anything under ``ml.*`` except ``ml.explainability.labels`` is
    # forbidden. ``models.*`` (top-level) is forbidden.
    forbidden = [
        m
        for m in imported
        if (m.startswith("ml.") and m != "ml.explainability.labels")
        or m.split(".")[0] == "models"
    ]
    assert forbidden == [], (
        f"shap_format directly imports forbidden modules: {forbidden}"
    )
    # Sanity: the allowlisted label-map import is present.
    assert "ml.explainability.labels" in imported


def test_top_n_default_matches_explainer_constant() -> None:
    # If Spec 16 ever bumps SHAP_TOP_N, the formatter's default
    # should match. Pin the current value so any drift is a
    # deliberate edit to both files.
    assert TOP_N_DEFAULT == 7


# ---------- summarize_direction ----------------------------------------------

def test_summarize_direction_counts_up_and_down() -> None:
    rows = [
        {"direction": "up"},
        {"direction": "up"},
        {"direction": "up"},
        {"direction": "down"},
        {"direction": "down"},
    ]
    assert summarize_direction(rows) == {"up": 3, "down": 2}


def test_summarize_direction_handles_empty_list() -> None:
    assert summarize_direction([]) == {"up": 0, "down": 0}


def test_summarize_direction_ignores_zero_impact() -> None:
    rows = [
        {"direction": "up"},
        {"direction": "neutral"},
        {"direction": "down"},
    ]
    # Neutral rows are intentionally excluded — the summary line
    # always reflects what actually pushed the price.
    assert summarize_direction(rows) == {"up": 1, "down": 1}
