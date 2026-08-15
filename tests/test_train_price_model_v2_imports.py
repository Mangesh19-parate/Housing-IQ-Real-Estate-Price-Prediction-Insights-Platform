"""Smoke tests for the v2 training script (Spec 14).

Validates import + CLI surface without running the actual sweep (the
sweep requires a populated ``clean_listings.parquet`` + Step 12
artifacts, which the unit-test harness does not have).
"""

from __future__ import annotations

import pytest


def test_train_price_model_v2_imports():
    from scripts import train_price_model_v2

    assert hasattr(train_price_model_v2, "main")
    assert hasattr(train_price_model_v2, "_build_v2_feature_frame")
    assert hasattr(train_price_model_v2, "_train_one_transact_type")


def test_train_price_model_v2_module_constants_present():
    """Sanity: script imports its own helpers without circular deps."""
    import importlib

    m = importlib.import_module("scripts.train_price_model_v2")
    # ``_make_v2_pipeline`` is the closure factory.
    assert callable(getattr(m, "_make_v2_pipeline"))
    assert callable(getattr(m, "_SerializableV2Pipeline"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
