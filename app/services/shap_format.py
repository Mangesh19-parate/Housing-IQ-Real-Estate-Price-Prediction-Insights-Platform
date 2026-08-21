"""Flask-side formatter for FastAPI ``shap_contributions``.

Spec 19 wires the SHAP bar chart on ``predict_result.html``. The
FastAPI ``POST /predict`` response (Spec 17) carries only
``{feature, impact}`` per ``docs/05-BACKEND-SCHEMA.md`` §7 — this
helper adds the human-readable label, an up/down/neutral direction,
and a magnitude-normalised ``pct`` the template iterates over.

Per Rules §6.4 the bars always carry a ``+``/``−``/``±`` text
glyph alongside the colour, so the colour is never the sole
carrier of meaning. The direction logic mirrors
``ml.explainability.contributions.explain_one`` verbatim — three
lines, identical to Spec 16's per-prediction helper.

Imports: stdlib only + ``ml.explainability.labels`` (the
display-only label-map module Spec 16 already established).
**No** ``ml.training.*``, no model ``.pkl`` loaders, no
``api.services.*`` imports — Rules §5.1.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from ml.explainability.labels import (
    FEATURE_LABEL_MAP_V2,
    load_label_map_from_disk,
)

#: Default number of top SHAP contributions shown on the result page.
#: Matches ``ml.explainability.explainer.SHAP_TOP_N`` (Spec 16). The
#: Flask helper caps the list defensively in case FastAPI ever sends
#: more than expected.
TOP_N_DEFAULT: Final[int] = 7

#: Where the on-disk label-map overlay lives. Mirrors the directory
#: the training scripts use (``scripts/train_price_model_v2.py``).
#: Override in tests via ``monkeypatch.setattr`` on the module-level
#: ``_LABEL_MAP_DIR``.
_LABEL_MAP_DIR: Final[Path] = Path("models")


@lru_cache(maxsize=1)
def _get_label_map() -> dict[str, str]:
    """Return the merged (static + on-disk overlay) feature→label map.

    Cached so the JSON overlay is read at most once per process.
    Returns the static map alone when the on-disk file is missing
    (the helper must not crash on a fresh checkout before the
    training CLI has landed the artifact).
    """
    base = FEATURE_LABEL_MAP_V2()
    overlay = load_label_map_from_disk(_LABEL_MAP_DIR)
    base.update(overlay)
    return base


def format_shap_for_template(
    contributions: list[dict],
    *,
    top_n: int = TOP_N_DEFAULT,
) -> list[dict]:
    """Turn the API's ``{feature, impact}`` pairs into template rows.

    Each output row is::

        {"feature": <raw>,
         "label": <human-friendly>,
         "impact": <float>,
         "direction": "up" | "down" | "neutral",
         "pct": <float in -1.0..1.0>}

    The list is capped at ``top_n`` (input is assumed pre-sorted
    by absolute impact desc by Spec 17's service). Order is
    preserved.
    """
    if not contributions:
        return []

    label_map = _get_label_map()
    sliced = contributions[: max(0, int(top_n))]

    # Magnitude normalisation — the longest bar is always ±1.0 so
    # the chart scale is stable across currencies and models.
    max_abs = max((abs(float(row.get("impact", 0.0))) for row in sliced), default=0.0)
    if max_abs == 0.0:
        # All-zero contributions — every bar collapses to a point.
        # Keep them visible as "neutral" so the section never
        # appears empty when the model genuinely produced zeros.
        max_abs = 1.0

    out: list[dict] = []
    for row in sliced:
        feature = str(row.get("feature", ""))
        impact = float(row.get("impact", 0.0))
        direction = "up" if impact > 0 else ("down" if impact < 0 else "neutral")
        out.append(
            {
                "feature": feature,
                "label": label_map.get(feature, feature),
                "impact": impact,
                "direction": direction,
                "pct": impact / max_abs,
            }
        )
    return out


def summarize_direction(rows: list[dict]) -> dict[str, int]:
    """Count SHAP rows by direction.

    Returns ``{"up": <int>, "down": <int>}``. ``neutral`` rows
    (zero-impact contributions) are intentionally excluded so the
    summary line always reflects what actually pushed the price.
    Empty input → ``{"up": 0, "down": 0}``.
    """
    up = sum(1 for r in rows if r.get("direction") == "up")
    down = sum(1 for r in rows if r.get("direction") == "down")
    return {"up": up, "down": down}


__all__ = [
    "TOP_N_DEFAULT",
    "format_shap_for_template",
    "summarize_direction",
]
