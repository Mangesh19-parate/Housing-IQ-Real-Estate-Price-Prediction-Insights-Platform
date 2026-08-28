"""``price_tier`` label builder (Spec 21).

Quantile-bins ``price_per_sqft`` per ``(city, transact_type)`` to
produce a 4-class affordability label:

    Budget  /  Mid-Range  /  Premium  /  Luxury

Boundaries are computed on the **training subset only**
(``is_outlier == False`` AND ``split == "train"``) per Rules §1.4 +
§8.2 — the same fixed 70/15/15 split the price regression uses, so a
reviewer auditing "what training rows defined these labels?" gets the
same answer as "what training rows defined the price model?".

Authority:
    - docs/01-PRD.md §U4.1 (price_tier is supporting/secondary)
    - docs/02-TRD.md §U-TRD-1 (per-city quantile binning)
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-8 (price_tier field)
    - docs/08-RULES.md §8.1–§8.3 (leakage ban, train-set boundaries,
      per-city definition)
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from ml.features.split import split_train_val_test

logger = logging.getLogger(__name__)

#: Pinned tier labels in canonical order. Spec 22's classifier training
#: will ordinal-encode from this order; the order is the encoding.
#: (Spec 21 DoD §TIER_LABELS.)
TIER_LABELS: Final[tuple[str, ...]] = (
    "Budget",
    "Mid-Range",
    "Premium",
    "Luxury",
)

#: Pinned quartile cut-points. Quartile boundaries → 4 tiers; mirrors
#: the ``protocol.py`` precedent (Spec 15) of pinning protocol
#: constants in one place.
QUANTILE_CUTPOINTS: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)


def compute_per_city_quantile_boundaries(
    train_df: pd.DataFrame,
    price_per_sqft_col: str = "price_per_sqft",
    group_keys: tuple[str, ...] = ("city", "transact_type"),
    min_rows: int = 100,
) -> dict[str, dict[str, list[float]]]:
    """Return per-``(city, transact_type)`` 25/50/75 ``price_per_sqft`` cut-points.

    Skips any group with fewer than ``min_rows`` rows and emits a
    logged WARNING (so the caller can either substitute defaults or
    drop the group from evaluation).

    Pure function — does not mutate the input frame.

    Returns:
        ``{city: {transact_type: [q1, q2, q3]}}``. Missing groups
        are simply absent from the dict; callers must handle that
        case (the label builder maps missing groups to
        ``price_tier = null``).
    """
    if price_per_sqft_col not in train_df.columns:
        raise KeyError(
            f"compute_per_city_quantile_boundaries: "
            f"missing column {price_per_sqft_col!r}"
        )
    for key in group_keys:
        if key not in train_df.columns:
            raise KeyError(
                f"compute_per_city_quantile_boundaries: "
                f"missing group key {key!r}"
            )

    boundaries: dict[str, dict[str, list[float]]] = {}
    # Drop rows with null price_per_sqft before grouping — a null
    # would propagate into the quantile computation and is not
    # meaningful for a tier assignment.
    df = train_df.dropna(subset=[price_per_sqft_col])

    for keys, group in df.groupby(list(group_keys), sort=True):
        city = str(keys[group_keys.index("city")])
        transact = str(keys[group_keys.index("transact_type")])
        n = len(group)
        if n < min_rows:
            logger.warning(
                "Skipping quantile calibration for (city=%s, "
                "transact_type=%s): n=%d < min_rows=%d",
                city,
                transact,
                n,
                min_rows,
            )
            continue
        qs = group[price_per_sqft_col].quantile(list(QUANTILE_CUTPOINTS))
        boundaries.setdefault(city, {})[transact] = [
            float(qs.iloc[0]),
            float(qs.iloc[1]),
            float(qs.iloc[2]),
        ]
        logger.info(
            "Computed %s/%s quantile boundaries: q1=%.2f q2=%.2f q3=%.2f "
            "(n=%d)",
            city,
            transact,
            qs.iloc[0],
            qs.iloc[1],
            qs.iloc[2],
            n,
        )

    return boundaries


def _assign_tier(
    price_per_sqft: float,
    boundaries: list[float],
) -> str | None:
    """Assign a tier label given one row's ``price_per_sqft`` and the
    three quantile cut-points.

    Boundaries are ``[q1, q2, q3]``. Assignment:
        - ``< q1``            → ``"Budget"``
        - ``q1 <= x < q2``    → ``"Mid-Range"``
        - ``q2 <= x < q3``    → ``"Premium"``
        - ``>= q3``           → ``"Luxury"``
    """
    if price_per_sqft < boundaries[0]:
        return "Budget"
    if price_per_sqft < boundaries[1]:
        return "Mid-Range"
    if price_per_sqft < boundaries[2]:
        return "Premium"
    return "Luxury"


def build_price_tier_labels(
    clean_listings: pd.DataFrame,
    boundaries: dict[str, dict[str, list[float]]] | None = None,
    processed_dir: Path | str | None = None,  # noqa: F821 — placeholder for spec parity; unused here
    split_col: str = "split",
    min_rows: int = 100,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    """Build the ``price_tier_labels`` frame.

    Steps:
        1. Tag rows with ``split_train_val_test(..., random_state=42)``
           — the same split the price regression uses, so the
           boundaries this spec computes are byte-equivalent with
           whatever Spec 22 will see.
        2. Filter to ``is_outlier == False`` for boundary computation
           (Rules §1.4 + §8.2: train-set boundaries on the
           non-outlier training subset).
        3. Compute per-city quantile boundaries on the training
           subset. If ``boundaries`` is passed (e.g., from a previous
           run's JSON), reuse them — never recompute on val/test
           (Rules §8.2).
        4. Assign ``price_tier`` for every row in the **full** frame
           (incl. outliers + val/test) using the persisted boundaries.
           Outlier rows get a tier label too (the analytics store
           stays complete) but their per-row ``tier_quantile_q{1,2,3}``
           columns are null.

    Pure function — does not mutate the input frame or write to disk.

    Args:
        clean_listings: the canonical cleaned listings frame (or a
            pre-loaded DataFrame when called from tests). When
            ``None``, the function reads
            ``processed_dir / "clean_listings.parquet"``.
        boundaries: optional pre-computed boundaries; if ``None``,
            the function computes them from the training subset.
        processed_dir: kept for spec parity; unused at the moment
            (the function never auto-reads).
        split_col: name of the split-tag column added to the output
            (default ``"split"``).
        min_rows: minimum non-outlier train rows per
            ``(city, transact_type)`` group for per-city calibration.

    Returns:
        ``(labeled_df, boundaries_dict)``. The labeled frame has the
        11 columns defined in the spec's "Data / Schema changes"
        section. The boundaries dict is the dict passed in (or the
        newly computed one) for downstream persistence.
    """
    # ponytail: nothing fancy, just a working builder. ~60 lines.
    df = clean_listings.copy()

    if "is_outlier" not in df.columns:
        raise KeyError(
            "build_price_tier_labels: missing 'is_outlier' column"
        )
    if "city" not in df.columns:
        raise KeyError(
            "build_price_tier_labels: missing 'city' column"
        )
    if "transact_type" not in df.columns:
        raise KeyError(
            "build_price_tier_labels: missing 'transact_type' column"
        )
    if "price_per_sqft" not in df.columns:
        raise KeyError(
            "build_price_tier_labels: missing 'price_per_sqft' column"
        )

    # Step 1: tag rows with the fixed 70/15/15 split (Rules §2.1, §5.4).
    # split_train_val_test returns three frames; concatenate back with
    # a 'split' column so the boundary computation filters on the
    # training slice.
    train_df, val_df, test_df = split_train_val_test(df, target="price")
    df = df.copy()
    df[split_col] = None
    df.loc[train_df.index, split_col] = "train"
    df.loc[val_df.index, split_col] = "val"
    df.loc[test_df.index, split_col] = "test"

    # Step 2: filter to non-outlier training rows for boundary compute.
    boundary_input = df[
        (df["is_outlier"] == False)  # noqa: E712 — explicit, matches spec
        & (df[split_col] == "train")
    ]

    # Step 3: compute or reuse boundaries.
    if boundaries is None:
        boundaries = compute_per_city_quantile_boundaries(
            boundary_input,
            price_per_sqft_col="price_per_sqft",
            min_rows=min_rows,
        )
    else:
        logger.info(
            "build_price_tier_labels: reusing %d cities' boundaries "
            "(skipping quantile recomputation per Rules §8.2).",
            len(boundaries),
        )

    # Step 4: assign tier + per-row quantile columns for every row.
    def _row_tier(row: pd.Series) -> str | None:
        pps = row["price_per_sqft"]
        if pd.isna(pps):
            return None
        city_bounds = boundaries.get(row["city"])
        if city_bounds is None:
            return None
        per_transact = city_bounds.get(row["transact_type"])
        if per_transact is None:
            return None
        return _assign_tier(float(pps), per_transact)

    df["price_tier"] = df.apply(_row_tier, axis=1)

    # Per-row quantile columns. Non-outlier rows that had a boundary
    # lookup get the actual values; everyone else gets null.
    def _row_q(row: pd.Series, idx: int) -> float | None:
        if row["is_outlier"]:
            return None
        city_bounds = boundaries.get(row["city"])
        if city_bounds is None:
            return None
        per_transact = city_bounds.get(row["transact_type"])
        if per_transact is None:
            return None
        return float(per_transact[idx])

    df["tier_quantile_q1"] = df.apply(lambda r: _row_q(r, 0), axis=1)
    df["tier_quantile_q2"] = df.apply(lambda r: _row_q(r, 1), axis=1)
    df["tier_quantile_q3"] = df.apply(lambda r: _row_q(r, 2), axis=1)

    # Select + order the 11 output columns.
    out = pd.DataFrame(
        {
            "listing_id": df["listing_id"],
            "city": df["city"],
            "locality": df["locality"],
            "transact_type": df["transact_type"],
            "price_per_sqft": df["price_per_sqft"],
            "price_tier": df["price_tier"],
            "tier_quantile_q1": df["tier_quantile_q1"],
            "tier_quantile_q2": df["tier_quantile_q2"],
            "tier_quantile_q3": df["tier_quantile_q3"],
            "is_outlier": df["is_outlier"],
        }
    )
    return out, boundaries


__all__ = [
    "QUANTILE_CUTPOINTS",
    "TIER_LABELS",
    "build_price_tier_labels",
    "compute_per_city_quantile_boundaries",
]
