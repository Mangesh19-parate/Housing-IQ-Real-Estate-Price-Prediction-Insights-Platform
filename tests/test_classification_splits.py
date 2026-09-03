"""Tests for classification split enforcement (Spec 24)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.classification.splits import protocol_split


@pytest.fixture
def synthetic_good_deal_df():
    """1000-row DataFrame with good_deal_verdict (3 classes)."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "listing_id": [f"L{i:06d}" for i in range(n)],
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "good_deal_verdict": np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2]),
        "price_tier": np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1]),
        "is_outlier": False,
    })


@pytest.fixture
def synthetic_price_tier_df():
    """1000-row DataFrame with price_tier (4 classes)."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "listing_id": [f"L{i:06d}" for i in range(n)],
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "good_deal_verdict": np.random.choice([0, 1, 2], n, p=[0.2, 0.6, 0.2]),
        "price_tier": np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.35, 0.3, 0.1]),
        "is_outlier": False,
    })


def test_protocol_split_good_deal_sizes(synthetic_good_deal_df):
    """protocol_split returns correct sizes for good_deal_verdict (700/150/150 ±1)."""
    train, val, test = protocol_split(synthetic_good_deal_df, target="good_deal_verdict")
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_protocol_split_price_tier_sizes(synthetic_price_tier_df):
    """protocol_split returns correct sizes for price_tier (700/150/150 ±1)."""
    train, val, test = protocol_split(synthetic_price_tier_df, target="price_tier")
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_protocol_split_stratification_good_deal(synthetic_good_deal_df):
    """Stratification preserves class proportions for good_deal_verdict."""
    train, val, test = protocol_split(synthetic_good_deal_df, target="good_deal_verdict")
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        props = split_df["good_deal_verdict"].value_counts(normalize=True).sort_index()
        orig_props = synthetic_good_deal_df["good_deal_verdict"].value_counts(normalize=True).sort_index()
        # Each class proportion should be within 2% of original
        for cls in [0, 1, 2]:
            assert abs(props.get(cls, 0) - orig_props.get(cls, 0)) < 0.02, (
                f"{split_name}: class {cls} proportion drift"
            )


def test_protocol_split_stratification_price_tier(synthetic_price_tier_df):
    """Stratification preserves class proportions for price_tier."""
    train, val, test = protocol_split(synthetic_price_tier_df, target="price_tier")
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        props = split_df["price_tier"].value_counts(normalize=True).sort_index()
        orig_props = synthetic_price_tier_df["price_tier"].value_counts(normalize=True).sort_index()
        for cls in [0, 1, 2, 3]:
            assert abs(props.get(cls, 0) - orig_props.get(cls, 0)) < 0.02, (
                f"{split_name}: class {cls} proportion drift"
            )


def test_protocol_split_deterministic(synthetic_good_deal_df):
    """random_state=42 yields identical splits across calls."""
    train1, val1, test1 = protocol_split(synthetic_good_deal_df, target="good_deal_verdict")
    train2, val2, test2 = protocol_split(synthetic_good_deal_df, target="good_deal_verdict")
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_protocol_split_indices_reset(synthetic_good_deal_df):
    """Returned DataFrames have reset indices."""
    train, val, test = protocol_split(synthetic_good_deal_df, target="good_deal_verdict")
    assert list(train.index) == list(range(len(train)))
    assert list(val.index) == list(range(len(val)))
    assert list(test.index) == list(range(len(test)))


def test_protocol_split_raises_on_missing_target(synthetic_good_deal_df):
    """Raises ValueError if target column not found."""
    with pytest.raises(ValueError, match="target column 'missing' not found"):
        protocol_split(synthetic_good_deal_df, target="missing")


def test_protocol_split_raises_on_insufficient_test_class():
    """Raises ValueError if any class would have < 2 members total (sklearn requirement)."""
    # Create a DataFrame where one class has only 1 sample total
    # sklearn's StratifiedShuffleSplit requires at least 2 samples per class
    df = pd.DataFrame({
        "listing_id": [f"L{i:06d}" for i in range(100)],
        "feature_1": np.random.randn(100),
        "target": [0] * 97 + [1, 2, 3],  # class 1,2,3 only have 1 each
    })
    with pytest.raises(ValueError, match="The least populated classes in y have only 1 member"):
        protocol_split(df, target="target")