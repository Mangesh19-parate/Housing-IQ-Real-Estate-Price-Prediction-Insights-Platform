"""Classifier feature-frame builder (Spec 22).

Reuses the price-model feature pipeline (``ml.features.feature_frame`` +
``ml.features.locality_aggregator.LocalityAggregator`` + the
fixed-70/15/15 ``split_train_val_test``) and drops every price-derived
column before the classifier sees the result. The output is the
deterministic ``data/processed/classifier/feature_frame.parquet`` that
Day 52's classifier-training spec consumes directly.

Leakage ban (``docs/08-RULES.md`` §8.1 + §12.4 + the
``model-training-classification`` skill):

- ``price`` — the regression target.
- ``price_per_sqft`` — price-derived ratio.
- ``locality_avg_price_sqft`` — locality aggregate of
  ``price_per_sqft``.
- ``locality_smoothed_price`` — locality aggregate of
  ``price_inr``.

``locality_listing_count`` is **kept** because a count is not
price-derived — it describes locality popularity, not price level.
``luxury_category`` is **kept** because Rules §10.2's server-derived
rule applies to the API surface (the client never sends it), not to
offline consumption of the resolved value. Both exceptions are
documented in the Decision Log entry for Spec 22.

Function ordering (critical — confirmed against ``ml/features/*``):

1. ``split_train_val_test`` returns 3 frames, not a frame with a
   ``split`` column. Derive ``is_train_mask`` + ``split`` via
   ``clean_listings.index.isin(train_df.index)``.
2. ``LocalityAggregator.fit`` reads ``is_outlier`` to filter at fit
   time. Run ``fit_transform`` **before** ``build_feature_frame`` so
   the locality columns are present.
3. ``build_feature_frame`` drops ``is_outlier`` and ``was_missing_*``
   from its output. Re-attach ``is_outlier`` after the call so the
   Day-52 training script can apply the ``is_outlier == False``
   filter.
4. Drop ``DROPPED_FOR_LEAKAGE`` columns. Assert zero survive.

Pure function: no I/O, no DB writes. The CLI
(``scripts/build_classifier_features.py``) handles persistence + the
report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from api.schemas.predict_v3 import INPUT_FIELDS_V3
from ml.features.feature_frame import build_feature_frame
from ml.features.locality_aggregator import LocalityAggregator
from ml.features.split import split_train_val_test

logger = logging.getLogger(__name__)

#: Price-derived columns that must NEVER appear in the classifier
#: feature frame (Rules §8.1 + §12.4). Single source of truth — the
#: CLI summary reads from it; the report's leakage-audit line reads
#: from it; the production-zero-overlap test asserts on it.
DROPPED_FOR_LEAKAGE: frozenset[str] = frozenset(
    {"price", "price_per_sqft", "locality_avg_price_sqft", "locality_smoothed_price"}
)

#: The one locality aggregate the classifier feature frame retains.
#: A count describes locality popularity, not price level — it does
#: not leak the target. Documented in the Decision Log.
RETAINED_LOCALITY_COLUMN: str = "locality_listing_count"

#: Columns ordered at the head of the output frame for downstream
#: readability: ``[listing_id, split, is_outlier, *INPUT_FIELDS_V3]``.
_HEADER_COLS: tuple[str, ...] = ("listing_id", "split", "is_outlier")

__all__ = [
    "DROPPED_FOR_LEAKAGE",
    "RETAINED_LOCALITY_COLUMN",
    "build_classifier_feature_frame",
]


def build_classifier_feature_frame(
    clean_listings: pd.DataFrame,
    locality_aggregator: LocalityAggregator | None = None,
    processed_dir: Path | str | None = None,
    split_fn=None,
) -> pd.DataFrame:
    """Build the leakage-safe classifier feature frame.

    Parameters
    ----------
    clean_listings:
        The canonical ``clean_listings`` DataFrame. If ``None``,
        ``processed_dir`` must point at a directory containing
        ``clean_listings.parquet``.
    locality_aggregator:
        An optional pre-fitted ``LocalityAggregator`` (e.g. the
        price model's fitted aggregator). When passed, ``transform``
        is called without refit — leakage-safe "apply, don't
        recompute". When ``None``, a fresh aggregator is fitted on
        ``is_outlier == False AND split == "train"`` rows.
    processed_dir:
        Directory containing ``clean_listings.parquet``. Used only
        when ``clean_listings`` is ``None``.
    split_fn:
        Test injection point — defaults to
        ``split_train_val_test(target="price")``.

    Returns
    -------
    pd.DataFrame
        Deterministic feature frame: ``[listing_id, split,
        is_outlier, *INPUT_FIELDS_V3, retained engineered + locality_*]``
        columns, with all four price-derived columns dropped.
    """
    if split_fn is None:
        split_fn = split_train_val_test

    if clean_listings is None:
        if processed_dir is None:
            raise ValueError(
                "build_classifier_feature_frame: provide either "
                "`clean_listings` or `processed_dir`."
            )
        clean_listings = pd.read_parquet(Path(processed_dir) / "clean_listings.parquet")

    # ---- 1. Derive the split column from the 70/15/15 helper ----
    train_df, val_df, test_df = split_fn(clean_listings, target="price")
    is_train_mask = clean_listings.index.isin(train_df.index)
    work = clean_listings.copy()

    # Build as plain strings first, then convert to categorical at the end.
    # This avoids the "Cannot setitem on a Categorical with a new category" error.
    split_vals = ["train" if m else "valtest" for m in is_train_mask]
    split_vals = [
        "val" if i in val_df.index else "test" if i in test_df.index else v
        for i, v in zip(work.index, split_vals)
    ]
    work["split"] = pd.Categorical(split_vals, categories=["train", "val", "test"])

    # ---- 2. LocalityAggregator: fit (if needed) + transform ----
    if locality_aggregator is None:
        locality_aggregator = LocalityAggregator()
        locality_aggregator.fit(work[is_train_mask & (work["is_outlier"] == False)])  # noqa: E712
    work = locality_aggregator.transform(work)

    # Capture columns added by LocalityAggregator that build_feature_frame
    # would drop (it only keeps INPUT_FIELDS_V3 + ENGINEERED_COLUMNS).
    # We need to re-attach these after the call.
    locality_extra_cols = [
        c
        for c in work.columns
        if c not in clean_listings.columns
        and c not in ["split"]  # split handled separately
        and c not in ("is_outlier", "listing_id")  # re-attached below
    ]
    split_col = work["split"].copy()

    # ---- 3. build_feature_frame — drops is_outlier + was_missing_* + extra cols ----
    featured = build_feature_frame(work)

    # ---- 4. Re-attach is_outlier + listing_id + split + aggregator extras ----
    if "is_outlier" not in featured.columns and "is_outlier" in work.columns:
        featured["is_outlier"] = work["is_outlier"].values
    if "listing_id" not in featured.columns and "listing_id" in work.columns:
        featured["listing_id"] = work["listing_id"].values
    if "split" not in featured.columns:
        featured["split"] = split_col.values
    for col in locality_extra_cols:
        if col not in featured.columns and col in work.columns:
            featured[col] = work[col].values

    # ---- 5. Drop the price-derived columns (idempotent) ----
    drop_cols = [c for c in DROPPED_FOR_LEAKAGE if c in featured.columns]
    if drop_cols:
        featured = featured.drop(columns=drop_cols)

    # ---- 6. Hard assertion: zero leakage columns survive ----
    leak = [c for c in DROPPED_FOR_LEAKAGE if c in featured.columns]
    if leak:
        raise ValueError(
            f"build_classifier_feature_frame: leakage columns survived "
            f"the drop step: {leak}. This is a bug — check that "
            f"DROPPED_FOR_LEAKAGE matches every column produced by "
            f"build_feature_frame + LocalityAggregator."
        )

    # ---- 7. Deterministic column order ----
    head = list(_HEADER_COLS)
    head_present = [c for c in head if c in featured.columns]
    contract = [c for c in INPUT_FIELDS_V3 if c in featured.columns]
    tail = [
        c
        for c in featured.columns
        if c not in head_present and c not in contract
    ]
    return featured[head_present + contract + tail]
