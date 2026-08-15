"""Lever 4 — sector-level smoothed target encoding (Spec 14).

Extends Step 12's ``LocalityAggregator`` with a coarser-grained
``(city, sector)`` Bayesian-smoothed mean of ``price_per_sqft`` toward
the city mean. Same LOO semantics: each row's encoded value excludes
its own contribution from the group mean (Rules §2.3 + stricter LOO
rule from Step 12's locality aggregator docstring).

Why a separate encoder for ``sector`` vs Step 12's
``LocalityAggregator``?
    - Step 12 already produces ``locality_smoothed_price`` keyed on
      ``(city, locality)``. That is the model-ready aggregate.
    - Lever 4 adds a **sector-level** aggregate (``sector_smoothed_price``,
      keyed on ``(city, sector)``) that complements the locality-level
      one. The two carry different signals: locality is finer-grained
      (often a single building), sector is broader (a neighborhood
      with multiple buildings). The literature (S12) supports both
      levels in a single model.

Leakage rules (Rules §2.3, §8.4):
    - ``fit`` filters to ``is_outlier == False`` first.
    - ``transform`` is a pure join; never refits.
    - Missing ``(city, sector)`` at transform time falls back to the
      city-level mean.

Public API:
    - ``SECTOR_SMOOTHING_PRIOR_WEIGHT: float = 20.0`` — pinned
      constant; same value as Step 12's ``SMOOTHING_PRIOR_WEIGHT``.
    - ``class SectorTargetEncoder`` — sklearn-style fit/transform.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Bayesian smoothing weight toward the city mean. Matches Step 12's
#: ``SMOOTHING_PRIOR_WEIGHT`` (literature S12 — at ~20 listings per
#: sector the smoothed estimate leans ~50/50 toward sector vs. city).
SECTOR_SMOOTHING_PRIOR_WEIGHT: Final[float] = 20.0

#: Output column name produced by ``transform``.
SECTOR_OUTPUT_COLUMN: Final[str] = "sector_smoothed_price"


class SectorTargetEncoder:
    """sklearn-style fit/transform encoder for ``(city, sector)``.

    Computes a per-(city, sector) Bayesian-smoothed mean of
    ``price_per_sqft`` toward the city mean, with leave-one-out
    semantics per row (own contribution excluded from the group
    mean).
    """

    def __init__(self) -> None:
        self.fitted_aggregates_: pd.DataFrame | None = None
        self.city_priors_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_stats(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Compute per-(city, sector) sums/counts + per-city priors.

        Filters to ``is_outlier == False`` first (Rules §2.3, §8.4).
        Returns ``(groups, city_priors)``.
        """
        all_keys = df[["city", "sector"]].drop_duplicates().reset_index(drop=True)
        clean = df[df["is_outlier"] == False]  # noqa: E712 — match pandas idiom
        if clean.empty:
            raise ValueError(
                "SectorTargetEncoder.fit: training frame has no non-outlier rows"
            )
        city_priors = (
            clean.groupby("city")["price_per_sqft"].mean().to_dict()
        )
        agg_part = (
            clean.groupby(["city", "sector"], dropna=False)
            .agg(
                count=("price_per_sqft", "size"),
                sum_psf=("price_per_sqft", "sum"),
            )
            .reset_index()
        )
        groups = all_keys.merge(agg_part, on=["city", "sector"], how="left")
        groups["count"] = groups["count"].fillna(0).astype("int64")
        groups["sum_psf"] = groups["sum_psf"].fillna(0.0)
        return groups, city_priors

    def _lookup(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Look up (city, sector) group stats for each row in ``df``."""
        assert self.fitted_aggregates_ is not None, (
            "SectorTargetEncoder not fitted"
        )
        original_index = df.index.copy()
        df_reset = df.reset_index(drop=False).rename(columns={"index": "_row_idx"})
        df_keyed = df_reset[["_row_idx", "city", "sector"]].merge(
            self.fitted_aggregates_[
                ["city", "sector", "count", "sum_psf"]
            ],
            on=["city", "sector"],
            how="left",
        )
        df_keyed = df_keyed.set_index("_row_idx")
        df_keyed.index.name = None
        df_keyed = df_keyed.reindex(original_index)
        city_prior = df["city"].map(self.city_priors_).fillna(0.0)
        return (
            df_keyed["count"].fillna(0).astype("int64"),
            df_keyed["sum_psf"].fillna(0.0),
            city_prior,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "SectorTargetEncoder":
        """Compute per-(city, sector) aggregates from the training frame.

        Idempotent — re-calling overwrites prior state. Stores
        :attr:`fitted_aggregates_` and :attr:`city_priors_`.
        """
        groups, priors = self._group_stats(train_df)
        self.fitted_aggregates_ = groups.sort_values(
            ["city", "sector"]
        ).reset_index(drop=True)
        self.city_priors_ = priors
        logger.info(
            "SectorTargetEncoder fitted on %d rows: %d (city, sector) groups",
            len(train_df),
            len(self.fitted_aggregates_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``sector_smoothed_price`` (LOO-adjusted) to ``df``.

        Returns a **copy** of ``df`` with the new column appended.

        Edge cases (handled in order):
            - ``count == 0`` (unseen ``(city, sector)``): emit the
              city prior directly. No LOO adjustment (the row's own
              price is the only price; we have no group to
              subtract from).
            - ``count >= 2``: LOO-adjusted smoothed mean, with the
              row's own contribution removed from the group sum.
            - ``count == 1``: LOO has no defined group mean (own
              contribution is the entire group); emit the city
              prior directly (same rationale as ``count == 0``).
        """
        if self.fitted_aggregates_ is None:
            raise RuntimeError(
                "SectorTargetEncoder.transform called before fit"
            )
        out = df.copy()
        count, sum_psf, city_prior = self._lookup(out)
        w = SECTOR_SMOOTHING_PRIOR_WEIGHT
        own_psf = out["price_per_sqft"].fillna(0.0)
        adj_sum = sum_psf - own_psf
        n_eff = count - 1  # LOO-adjusted count

        # Default result: city prior for unseen / singleton groups,
        # LOO-smoothed mean for >=2-row groups.
        result = pd.Series(city_prior.to_numpy(), index=out.index, dtype="float64")
        valid = count >= 2
        if valid.any():
            denom = n_eff[valid] + w
            num = adj_sum[valid] + w * city_prior[valid]
            result.loc[valid] = num / denom

        # Own price missing AND no group -> NaN.
        own_missing = out["price_per_sqft"].isna()
        no_group = count == 0
        if (own_missing & no_group).any():
            result.loc[own_missing & no_group] = np.nan
        out[SECTOR_OUTPUT_COLUMN] = result

        # Warning if many rows fell back to the city prior (likely
        # the model sees a sector not in training).
        fallback = count < 2
        if fallback.any():
            logger.warning(
                "SectorTargetEncoder.transform: %d rows fell back to "
                "city prior (unseen or singleton (city, sector))",
                int(fallback.sum()),
            )
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform on the same frame."""
        return self.fit(train_df).transform(train_df)

    @property
    def n_groups_(self) -> int:
        if self.fitted_aggregates_ is None:
            return 0
        return int(len(self.fitted_aggregates_))


__all__ = [
    "SECTOR_OUTPUT_COLUMN",
    "SECTOR_SMOOTHING_PRIOR_WEIGHT",
    "SectorTargetEncoder",
]
