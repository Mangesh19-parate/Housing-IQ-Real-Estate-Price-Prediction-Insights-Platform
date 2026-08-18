"""Split enforcement — the gate's hook on the 70/15/15 protocol.

Thin wrapper around ``ml.features.split.split_train_val_test`` that
**asserts** the returned split ratios match ``SPLIT_RATIOS`` (within
±1 row, to tolerate floor rounding on small synthetic fixtures) and
that ``random_state == RANDOM_STATE``. Drift → logged ERROR + raise.

Specs 13/14's training scripts use the raw ``ml.features.split``
helper directly (no behavior change there). The gate always uses
``protocol_split`` so the protocol is enforced at certification time
even if a training script's ``random_state`` drifts.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.evaluation.protocol import RANDOM_STATE, SPLIT_RATIOS
from ml.features.split import split_train_val_test

logger = logging.getLogger(__name__)


def protocol_split(
    df: pd.DataFrame,
    target: str = "price",
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, val_df, test_df)`` and enforce the protocol.

    The split ratios are pinned via ``SPLIT_RATIOS`` (70/15/15) and the
    seed via ``RANDOM_STATE`` (42). A non-default ``random_state`` is
    rejected at the call site — the gate is the protocol enforcer,
    not a flexible split API.

    The split tolerance is ±1 row per partition to tolerate floor
    rounding on small synthetic fixtures used in tests. A drift larger
    than ±1 row indicates a real protocol violation and raises.
    """
    if random_state != RANDOM_STATE:
        logger.error(
            "protocol_split called with random_state=%d; expected %d. "
            "Drift is a protocol violation — refusing to split.",
            random_state,
            RANDOM_STATE,
        )
        raise ValueError(
            f"protocol_split requires random_state={RANDOM_STATE}, "
            f"got {random_state}"
        )

    train_df, val_df, test_df = split_train_val_test(df, target=target)
    n_total = len(df)
    if n_total == 0:
        return train_df, val_df, test_df

    expected = {
        "train": int(round(SPLIT_RATIOS["train"] * n_total)),
        "val": int(round(SPLIT_RATIOS["val"] * n_total)),
        "test": int(round(SPLIT_RATIOS["test"] * n_total)),
    }
    actual = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    for partition, exp in expected.items():
        if abs(actual[partition] - exp) > 1:
            logger.error(
                "protocol_split: %s partition has %d rows, expected "
                "~%d (±1 tolerance). Drift indicates a split helper "
                "change without a PROTOCOL_VERSION bump.",
                partition,
                actual[partition],
                exp,
            )
            raise AssertionError(
                f"protocol_split: {partition} partition has "
                f"{actual[partition]} rows, expected ~{exp} "
                f"(±1 tolerance). PROTOCOL_VERSION bump required."
            )
    return train_df, val_df, test_df


__all__ = ["protocol_split"]
