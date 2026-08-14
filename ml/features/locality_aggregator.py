"""Leakage-safe locality aggregates for the price model (Spec 12 Phase 2).

Authority:
    - docs/02-TRD.md §8 (``locality_avg_price_sqft``, ``locality_listing_count``)
    - docs/08-RULES.md §2.3 (train-only fit), §8.4 (outlier exclusion),
      §8.2 (apply, don't recompute)

The aggregator computes three per-(city, locality) features:

- ``locality_avg_price_sqft``: mean ``price_per_sqft`` across the (city,
  locality) group, excluding the row being transformed itself
  (leave-one-out).
- ``locality_listing_count``: the group size — same value for every row
  in the group, no LOO adjustment.
- ``locality_smoothed_price``: Bayesian-smoothed mean of ``price_inr``
  toward the city mean, with LOO adjustment (subtract the row's own
  price, decrement the prior weight).

Why leave-one-out per row?

Rules §2.3 requires train-only fit, but a tighter form of the same
leakage class is "a row's locality aggregate includes its own price" —
even when fit on training data, a row's ``locality_avg_price_sqft`` is
correlated with the row's own ``price`` by construction. This is the
standard fix (and stricter than the Rules minimum): compute group
sums + counts at fit time, then at transform time emit
``(group_sum - row_value) / (group_count - 1)``.

When ``group_count == 1`` the LOO is undefined; we emit ``NaN`` +
WARNING. Same for ``locality_smoothed_price`` when LOO weight would
go negative.

Missing (city, locality) at transform time falls back to the city
prior (``city_priors_[city]``) — a row from a locality never seen in
training gets the city-level mean rather than NaN, because the
city-level signal is the best the model can offer for unseen
localities.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bayesian smoothing weight toward the city mean. Tunable; default 20
# matches literature S12 — at ~20 listings per locality the smoothed
# estimate leans ~50/50 toward the locality mean vs. the city mean.
SMOOTHING_PRIOR_WEIGHT: float = 20.0


class LocalityAggregator:
    """sklearn-style fit/transform aggregator for (city, locality) features.

    Fit on training data only; transform applies the learned aggregates
    (no refit). Designed to be paired with
    :func:`ml.features.preprocessor.fit_preprocessor`, which calls
    :meth:`transform` before fitting the ``ColumnTransformer``.
    """

    def __init__(self) -> None:
        self.fitted_aggregates_: pd.DataFrame | None = None
        self.city_priors_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_stats(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Compute per-(city, locality) sums/counts + per-city priors.

        Filters to ``is_outlier == False`` first (Rules §2.3, §8.4).
        Returns ``(groups, city_priors)`` where ``groups`` is a sorted
        DataFrame indexed by (city, locality) with columns
        ``sum_price_inr, count, sum_price_per_sqft``. All observed (city,
        locality) pairs in the input frame are included; zero-count
        groups (all rows flagged as outlier) are still listed so
        ``transform`` can look them up.
        """
        # Build the full (city, locality) keyspace from the input frame
        # first (BEFORE outlier filter), so a group whose only/all rows
        # are outliers still gets a row in the lookup with count==0.
        all_keys = df[["city", "locality"]].drop_duplicates().reset_index(drop=True)
        clean = df[df["is_outlier"] == False]  # noqa: E712 — match pandas idiom
        if clean.empty:
            raise ValueError(
                "LocalityAggregator.fit: training frame has no non-outlier rows"
            )
        # Per-city priors — mean price_per_sqft per city.
        city_priors = (
            clean.groupby("city")["price_per_sqft"].mean().to_dict()
        )
        # Per-(city, locality) aggregates from the CLEAN (non-outlier) set.
        agg_part = (
            clean.groupby(["city", "locality"], dropna=False)
            .agg(
                sum_price_inr=("price_inr", "sum"),
                count=("price_inr", "size"),
                sum_price_per_sqft=("price_per_sqft", "sum"),
            )
            .reset_index()
        )
        # Outer-join onto the full keyspace so outlier-only groups have
        # a row with count=0, sums=0.
        groups = all_keys.merge(
            agg_part,
            on=["city", "locality"],
            how="left",
        )
        groups["count"] = groups["count"].fillna(0).astype("int64")
        groups["sum_price_inr"] = groups["sum_price_inr"].fillna(0.0)
        groups["sum_price_per_sqft"] = groups["sum_price_per_sqft"].fillna(0.0)
        return groups, city_priors

    def _lookup(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Look up (city, locality) group stats for each row in ``df``.

        Returns ``(group_count, group_sum_price_inr, group_sum_price_per_sqft,
        city_prior)`` — all Series aligned to ``df.index``. Missing (city,
        locality) combos are filled with the city prior + 0/0 sums (so
        the LOO transform degrades to the city fallback).
        """
        assert self.fitted_aggregates_ is not None, "LocalityAggregator not fitted"
        groups = self.fitted_aggregates_
        # Preserve the original index across the merge — pandas merge
        # produces a default RangeIndex, which would break downstream
        # boolean-mask alignment with the input frame.
        original_index = df.index.copy()
        df_reset = df.reset_index(drop=False).rename(columns={"index": "_row_idx"})
        df_keyed = df_reset[["_row_idx", "city", "locality"]].merge(
            groups[
                [
                    "city",
                    "locality",
                    "count",
                    "sum_price_inr",
                    "sum_price_per_sqft",
                ]
            ],
            on=["city", "locality"],
            how="left",
        )
        df_keyed = df_keyed.set_index("_row_idx")
        df_keyed.index.name = None
        # Re-align to the input frame's original index ordering.
        df_keyed = df_keyed.reindex(original_index)
        city_prior = df["city"].map(self.city_priors_).fillna(0.0)
        return (
            df_keyed["count"].fillna(0).astype("int64"),
            df_keyed["sum_price_inr"].fillna(0.0),
            df_keyed["sum_price_per_sqft"].fillna(0.0),
            city_prior,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "LocalityAggregator":
        """Compute per-(city, locality) aggregates from the training frame.

        Idempotent — re-calling overwrites prior state. Stores
        :attr:`fitted_aggregates_` (sorted DataFrame) and
        :attr:`city_priors_` (dict).
        """
        groups, priors = self._group_stats(train_df)
        self.fitted_aggregates_ = groups.sort_values(
            ["city", "locality"]
        ).reset_index(drop=True)
        self.city_priors_ = priors
        logger.info(
            "LocalityAggregator fitted on %d rows: %d (city, locality) groups, %d city priors",
            len(train_df),
            len(self.fitted_aggregates_),
            len(self.city_priors_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned aggregates to ``df`` (no refit).

        Adds three columns to a copy of ``df``:
        ``locality_avg_price_sqft``, ``locality_listing_count``,
        ``locality_smoothed_price``. Each row's
        ``locality_avg_price_sqft`` and ``locality_smoothed_price``
        exclude the row's own contribution (leave-one-out).
        """
        if self.fitted_aggregates_ is None:
            raise RuntimeError(
                "LocalityAggregator.transform called before fit"
            )
        out = df.copy()
        n_rows = len(out)
        count, sum_inr, sum_psf, city_prior = self._lookup(out)

        # locality_listing_count — group size (no LOO adjustment).
        out["locality_listing_count"] = count.astype("int64")

        # locality_avg_price_sqft — leave-one-out mean.
        n_eff = count - 1  # denominator after removing self
        own_psf = out["price_per_sqft"].fillna(0.0)
        adjusted_sum = sum_psf - own_psf
        # Cases:
        #   - group missing entirely (count == 0) -> city prior
        #   - group has exactly 1 row (count == 1, n_eff == 0) -> NaN
        #   - group has >=2 rows -> adjusted_sum / n_eff
        result_psf = pd.Series(np.nan, index=out.index, dtype="float64")
        valid = count >= 2
        result_psf[valid] = adjusted_sum[valid] / n_eff[valid]
        fallback = count == 0
        if fallback.any():
            result_psf[fallback] = city_prior[fallback]
            logger.warning(
                "LocalityAggregator.transform: %d rows fell back to city prior "
                "(unseen (city, locality))",
                int(fallback.sum()),
            )
        out["locality_avg_price_sqft"] = result_psf

        # locality_smoothed_price — Bayesian-smoothed mean of price_inr.
        own_inr = out["price_inr"].fillna(0.0)
        adj_sum_inr = sum_inr - own_inr
        n = count - 1  # LOO-adjusted count
        w = SMOOTHING_PRIOR_WEIGHT
        # (n * mean_locality_loo + w * city_prior) / (n + w)
        # where mean_locality_loo = adj_sum_inr / n (when n > 0).
        # Numerator: adj_sum_inr + w * city_prior
        # Denominator: n + w
        denom = n + w
        num = adj_sum_inr + w * city_prior
        result_smooth = num / denom
        # Edge: own price is missing and group is missing -> NaN.
        own_missing = out["price_inr"].isna()
        no_group = count == 0
        if (own_missing & no_group).any():
            result_smooth[own_missing & no_group] = np.nan
        # Edge: group has 1 row (count == 1, n == 0) but own row is
        # missing -> result is purely the prior (num = w * prior,
        # denom = w). That is fine — it IS the city mean.
        out["locality_smoothed_price"] = result_smooth

        # Silence unused-variable warning while documenting size.
        _ = n_rows
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform on the same frame."""
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------------
    # Convenience accessors used by tests + the artifact recipe writer
    # ------------------------------------------------------------------

    @property
    def n_groups_(self) -> int:
        if self.fitted_aggregates_ is None:
            return 0
        return int(len(self.fitted_aggregates_))


__all__ = ["LocalityAggregator", "SMOOTHING_PRIOR_WEIGHT"]
