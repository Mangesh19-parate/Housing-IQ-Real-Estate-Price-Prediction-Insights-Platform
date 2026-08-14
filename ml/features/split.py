"""Train/val/test split helper (Spec 12 Phase 4).

Single source of truth for the 70/15/15 split. The fixed evaluation
protocol (Rules §5.4, TRD §10) requires ``random_state=42`` and city
stratification; this helper encodes both so the feature-build script
(Phase 5) and the future training script use the same boundaries.

The classifier spec (Week 8) reuses the same helper so model and
classifier share one source of truth on which rows are train/val/test.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

#: Pinned constant per Rules §5.4 + TRD §10. Train and serve must
#: agree on the split.
FIXED_RANDOM_STATE: int = 42


def split_train_val_test(
    df,
    target: str = "price",
):
    """Return ``(train_df, val_df, test_df)`` — 70/15/15, stratified on ``city``.

    Two sklearn calls: first yields 70% train + 30% temp; second splits
    the 30% in half to 15% val + 15% test. Both calls use
    ``random_state=FIXED_RANDOM_STATE`` and stratify on ``city``.

    The ``target`` argument is preserved for API compatibility with the
    future training spec; this helper does not actually drop the target
    column (the caller does that).
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["city"],
        random_state=FIXED_RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["city"],
        random_state=FIXED_RANDOM_STATE,
    )
    return train_df, val_df, test_df


__all__ = ["FIXED_RANDOM_STATE", "split_train_val_test"]
