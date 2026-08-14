"""Candidate-selection helper (Spec 13).

``select_winner`` picks the candidate with the lowest
``primary_metric`` (default ``val_rmse``) on the validation slice;
ties break on ``val_mae``. Pure function — no estimator fitting, no
I/O.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_winner(
    candidate_results: dict[str, dict],
    primary_metric: str = "rmse",
) -> str:
    """Return the candidate name with the lowest ``val[primary_metric]``.

    ``candidate_results[name]["val"]`` must contain the keys
    ``primary_metric`` (default ``"rmse"``) and ``"mae"``. The
    ``primary_metric`` value is the bare metric name (e.g. ``"rmse"``,
    ``"mape"``); the ``"val"`` split prefix is implied.

    Tie-break: lowest ``val["mae"]``. Empty input raises
    ``ValueError``.
    """
    if not candidate_results:
        raise ValueError("select_winner: candidate_results is empty.")

    def _sort_key(name: str) -> tuple[float, float]:
        val = candidate_results[name]["val"]
        if primary_metric not in val or "mae" not in val:
            raise KeyError(
                f"select_winner: candidate {name!r} missing "
                f"{primary_metric!r} or 'mae' in its 'val' dict."
            )
        return (val[primary_metric], val["mae"])

    winner = min(candidate_results, key=_sort_key)
    logger.info(
        "select_winner: chose %s (val_%s=%.4f, val_mae=%.4f)",
        winner,
        primary_metric,
        candidate_results[winner]["val"][primary_metric],
        candidate_results[winner]["val"]["mae"],
    )
    return winner


__all__ = ["select_winner"]
