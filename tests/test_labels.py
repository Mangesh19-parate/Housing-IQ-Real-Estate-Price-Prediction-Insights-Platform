"""Tests for ``ml.explainability.labels`` — feature label mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.explainability.labels import (
    FEATURE_LABEL_MAP_V2,
    build_label_map,
    label_map_hash,
    load_label_map_from_disk,
    resolve_label,
    save_label_map,
)


def test_feature_label_map_v2_is_a_dict():
    assert isinstance(FEATURE_LABEL_MAP_V2(), dict)
    assert len(FEATURE_LABEL_MAP_V2()) > 0


def test_label_map_covers_numeric_block():
    label_map = FEATURE_LABEL_MAP_V2()
    for name in ("num__bedRoom", "num__bathroom", "num__built_up_area"):
        assert name in label_map, f"{name} missing from static map"


def test_label_map_covers_ordinal_block():
    label_map = FEATURE_LABEL_MAP_V2()
    for name in ("ord__luxury_category", "ord__floor_category", "ord__furnishing_type"):
        assert name in label_map


def test_label_map_unknown_name_falls_through_to_raw():
    """Defensive: unknown names return the raw internal name, never raise."""
    label_map = FEATURE_LABEL_MAP_V2()
    assert resolve_label("num__made_up_feature", label_map) == "num__made_up_feature"


def test_build_label_map_emits_one_hot_keys_for_fitted_categories():
    """A fitted ColumnTransformer with a tiny OneHotEncoder emits one
    entry per fitted category, with the block name prefix."""
    df = pd.DataFrame({"x": [1.0, 2.0, 1.5], "city": ["A", "B", "A"]})
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["x"]),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["city"]),
        ],
        remainder="drop",
    )
    preprocessor.fit(df)

    label_map = build_label_map(preprocessor)
    # The fitted OneHotEncoder emits keys like ``cat__city_A`` and ``cat__city_B``.
    has_city_a = any(k == "cat__city_A" for k in label_map.keys())
    has_city_b = any(k == "cat__city_B" for k in label_map.keys())
    assert has_city_a, f"Missing cat__city_A in {list(label_map.keys())}"
    assert has_city_b, f"Missing cat__city_B in {list(label_map.keys())}"


def test_load_label_map_from_disk_returns_static_when_missing(tmp_path):
    """No file on disk → static fallback only (no raise)."""
    label_map = load_label_map_from_disk(tmp_path)
    assert "num__bedRoom" in label_map


def test_load_label_map_from_disk_overlays_existing_file(tmp_path):
    """An on-disk file overrides + extends the static map."""
    overlay_path = tmp_path / "feature_label_map_v2.json"
    overlay_path.write_text(json.dumps({"custom_feat": "Custom Label"}), encoding="utf-8")
    label_map = load_label_map_from_disk(tmp_path)
    assert label_map["custom_feat"] == "Custom Label"
    # Static keys still present.
    assert "num__bedRoom" in label_map


def test_save_label_map_is_round_trip(tmp_path):
    label_map = {"a": "A", "b": "B"}
    out = save_label_map(label_map, tmp_path / "feature_label_map_v2.json")
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == label_map


def test_label_map_hash_is_stable():
    """Same dict content → same sha1."""
    label_map = {"b": "B", "a": "A"}
    h1 = label_map_hash(label_map)
    h2 = label_map_hash(label_map)
    assert h1 == h2
    assert len(h1) == 40  # sha1 hex