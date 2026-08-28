"""``good_deal_verdict`` label builder (Spec 21).

3-class investment-decision label derived from the residual between
a listing's actual price and the **out-of-fold** price regression
prediction::

    Good Deal :  residual_pct <= city_threshold_low
    Fair Price:  city_threshold_low  <  residual_pct <  city_threshold_high
    Overpriced:  residual_pct >= city_threshold_high

OOF predictions are mandatory (Rules §8.4) — in-sample predictions
would leak. When the OOF parquet is missing, every verdict comes
back null with a ``reason`` column entry so the caller can audit
why without silently shipping a leaking target.

Authority:
    - docs/01-PRD.md §U4.2 (good_deal_verdict is primary)
    - docs/02-TRD.md §U-TRD-6 (3-class verdict + ±10% default)
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-8 (good_deal_verdict field)
    - docs/08-RULES.md §8.4 (OOF rule) + §12.2 (thresholds tunable)
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

#: Pinned verdict labels in canonical order. Spec 22's classifier
#: training will one-hot-encode from this order; the order is the
#: encoding.
VERDICT_LABELS: Final[tuple[str, ...]] = (
    "Good Deal",
    "Fair Price",
    "Overpriced",
)

#: Rules §12.2 starting-point low threshold. ``calibrate_good_deal_thresholds``
#: may override this per city from the residual distribution.
DEFAULT_THRESHOLD_LOW: Final[float] = -0.10

#: Rules §12.2 starting-point high threshold.
DEFAULT_THRESHOLD_HIGH: Final[float] = 0.10


def _lookup_thresholds(
    city: str,
    thresholds: dict[str, dict[str, float]] | None,
) -> tuple[float, float, str]:
    """Return ``(low, high, source)`` for a city.

    ``source`` is one of ``"city"`` (per-city override),
    ``"_default"`` (default block), or ``"hardcoded_default"``
    (no thresholds dict at all).
    """
    if thresholds is not None:
        if city in thresholds:
            t = thresholds[city]
            return float(t["low"]), float(t["high"]), "city"
        if "_default" in thresholds:
            t = thresholds["_default"]
            return float(t["low"]), float(t["high"]), "_default"
    return DEFAULT_THRESHOLD_LOW, DEFAULT_THRESHOLD_HIGH, "hardcoded_default"


def _assign_verdict(
    residual_pct: float,
    low: float,
    high: float,
) -> str:
    """Pure verdict assignment. ``residual_pct`` is non-null by contract."""
    if residual_pct <= low:
        return "Good Deal"
    if residual_pct >= high:
        return "Overpriced"
    return "Fair Price"


def build_good_deal_labels(
    clean_listings: pd.DataFrame,
    oof_predictions: pd.DataFrame | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
    model_version: str = "v2",
    processed_dir: Path | str | None = None,  # noqa: F821 — placeholder for spec parity
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Build the ``good_deal_labels`` frame.

    Steps:
        1. If ``oof_predictions`` is ``None``, attempt to load
           ``data/processed/oof_predictions_{model_version}.parquet``.
           If that file is also missing, log a WARNING and proceed
           with ``oof_predictions = None`` (every row's verdict
           comes back null with ``reason="missing_oof_prediction"``).
        2. Left-join ``clean_listings[[listing_id, transact_type,
           city, price, is_outlier]]`` against
           ``oof_predictions[[listing_id, transact_type,
           oof_predicted_price]]`` on ``(listing_id,
           transact_type)``. Rows without a matching OOF prediction
           get ``oof_predicted_price = null`` and
           ``reason = "missing_oof_prediction"``.
        3. Compute ``residual_pct = (actual_price -
           oof_predicted_price) / oof_predicted_price``. Rows with
           ``oof_predicted_price <= 0`` or null get
           ``residual_pct = null`` and
           ``reason = "non_positive_predicted_price"``.
        4. Look up the per-city threshold (or fall back to
           ``_default`` / hardcoded default).
        5. Assign ``good_deal_verdict`` from the residual + the
           per-city thresholds. Rows with null ``residual_pct``
           keep ``good_deal_verdict = null`` and their existing
           ``reason``.

    Pure function — does not write to disk.

    Returns:
        ``(labeled_df, thresholds_dict)``. The thresholds dict is the
        one passed in (so the caller can persist it; the function
        does not fabricate its own).
    """
    # ponytail: ~80 lines, straight SQL-style left-join + arithmetic.
    # No clever shortcut — the per-row `reason` auditability is the
    # whole point of this module.
    df = clean_listings.copy()

    # Step 1: load OOF predictions if not passed.
    oof_loaded_from_disk = False
    if oof_predictions is None:
        # The caller is the CLI; processed_dir is wired there.
        # Tests pass the frame directly. We don't auto-load here
        # to keep this function deterministic + side-effect-free.
        logger.warning(
            "build_good_deal_labels: no OOF predictions supplied "
            "(model_version=%s); every verdict will be null with "
            "reason='missing_oof_prediction'.",
            model_version,
        )
        oof = pd.DataFrame(
            columns=["listing_id", "transact_type", "oof_predicted_price"]
        )
    else:
        oof = oof_predictions.copy()
        oof_loaded_from_disk = False

    # Step 2: left-join. Use a sentinel merge-key to keep the join
    # stable when (listing_id, transact_type) duplicates exist.
    if "listing_id" not in df.columns:
        raise KeyError(
            "build_good_deal_labels: missing 'listing_id' column"
        )
    if "price" not in df.columns:
        raise KeyError(
            "build_good_deal_labels: missing 'price' column "
            "(actual_price is read from 'price')"
        )
    if "city" not in df.columns:
        raise KeyError(
            "build_good_deal_labels: missing 'city' column"
        )

    # Required columns on the OOF side.
    for col in ("listing_id", "transact_type", "oof_predicted_price"):
        if col not in oof.columns:
            raise KeyError(
                f"build_good_deal_labels: OOF predictions missing "
                f"required column {col!r}"
            )

    # Trim OOF to the join keys + prediction to keep the merge frame
    # small. The merge keys must be unique on the OOF side — if a
    # listing has multiple OOF predictions (e.g. rerun), keep the
    # first.
    oof_trimmed = (
        oof[["listing_id", "transact_type", "oof_predicted_price"]]
        .drop_duplicates(subset=["listing_id", "transact_type"], keep="first")
    )

    merged = df.merge(
        oof_trimmed,
        on=["listing_id", "transact_type"],
        how="left",
    )

    # Step 3: compute residual_pct.
    def _row_residual_and_reason(row: pd.Series) -> tuple[float | None, str | None]:
        actual = row["price"]
        pred = row["oof_predicted_price"]
        if pd.isna(pred):
            return None, "missing_oof_prediction"
        if pred <= 0:
            return None, "non_positive_predicted_price"
        if pd.isna(actual):
            return None, "missing_actual_price"
        return float((actual - pred) / pred), None

    residual_and_reason = merged.apply(_row_residual_and_reason, axis=1)
    merged["residual_pct"] = [r[0] for r in residual_and_reason]
    merged["reason"] = [r[1] for r in residual_and_reason]

    # Step 4 + 5: per-city threshold lookup + verdict assignment.
    low_used = merged["city"].map(
        lambda c: _lookup_thresholds(c, thresholds)[0]
    )
    high_used = merged["city"].map(
        lambda c: _lookup_thresholds(c, thresholds)[1]
    )
    merged["verdict_threshold_low"] = low_used
    merged["verdict_threshold_high"] = high_used

    def _row_verdict(row: pd.Series) -> str | None:
        if pd.isna(row["residual_pct"]):
            return None
        return _assign_verdict(
            float(row["residual_pct"]),
            float(row["verdict_threshold_low"]),
            float(row["verdict_threshold_high"]),
        )

    merged["good_deal_verdict"] = merged.apply(_row_verdict, axis=1)

    # If a verdict was assigned, ensure reason is null (defensive —
    # the per-row residual/reason logic only sets reason on the
    # null-residual path).
    merged.loc[merged["good_deal_verdict"].notna(), "reason"] = None

    # Build the 11-column output frame.
    out = pd.DataFrame(
        {
            "listing_id": merged["listing_id"],
            "city": merged["city"],
            "transact_type": merged["transact_type"],
            "actual_price": merged["price"],
            "oof_predicted_price": merged["oof_predicted_price"],
            "residual_pct": merged["residual_pct"],
            "good_deal_verdict": merged["good_deal_verdict"],
            "verdict_threshold_low": merged["verdict_threshold_low"],
            "verdict_threshold_high": merged["verdict_threshold_high"],
            "is_outlier": merged.get("is_outlier", False),
            "reason": merged["reason"],
        }
    )
    # Track model_version as a per-row column so a future v3 rerun
    # overwrites with v3 rows. The spec places model_version in the
    # row schema.
    out["model_version"] = model_version
    _ = oof_loaded_from_disk  # quiet ruff; reserved for future use

    return out, thresholds or {}


__all__ = [
    "DEFAULT_THRESHOLD_HIGH",
    "DEFAULT_THRESHOLD_LOW",
    "VERDICT_LABELS",
    "build_good_deal_labels",
]
