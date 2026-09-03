"""Stratified 70/15/15 split enforcement — Spec 24.

Pure function mirroring the price-model protocol's split logic, adapted
for classification targets.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.evaluation.classification.protocol import (
    RANDOM_STATE,
    SPLIT_RATIOS,
)

#: Minimum number of samples per class required in the test split.
#: If any class would have fewer, ``protocol_split`` raises ValueError.
_MIN_TEST_PER_CLASS: Final[int] = 2


def protocol_split(
    df: pd.DataFrame,
    target: str,                    # "good_deal_verdict" or "price_tier"
    *,
    ratios: dict[str, float] = SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split on ``target`` with pinned ``random_state``.

    Uses two ``train_test_split`` calls:
      1. 70% train, 30% temp (stratified)
      2. 50/50 split of temp -> 15% val, 15% test (stratified)

    Returns ``(train_df, val_df, test_df)`` with reset indices.

    Raises:
        ValueError: If ``target`` column not in ``df``, or if any class
            would have fewer than ``_MIN_TEST_PER_CLASS`` members in the
            test split.
    """
    if target not in df.columns:
        raise ValueError(f"protocol_split: target column '{target}' not found in DataFrame")

    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]

    # Sanity check: ratios must sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"protocol_split: ratios must sum to 1.0, got {total}")

    # Step 1: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df[target],
    )

    # Step 2: val vs test from the 30% temp (50/50 = 15%/15% of original)
    # val_ratio / (val_ratio + test_ratio) = 0.15 / 0.30 = 0.5
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction,
        random_state=random_state,
        stratify=temp_df[target],
    )

    # Validate minimum test representation per class
    test_counts = test_df[target].value_counts()
    min_count = test_counts.min()
    if min_count < _MIN_TEST_PER_CLASS:
        raise ValueError(
            f"protocol_split: class '{test_counts.idxmin()}' has only {min_count} "
            f"samples in test split (minimum {_MIN_TEST_PER_CLASS}). "
            f"Consider more data or fewer classes."
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


__all__ = ["protocol_split"]