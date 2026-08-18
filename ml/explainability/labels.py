"""Human-readable label map for post-preprocessor feature names.

Single source of truth used by both the per-prediction helper
(`contributions.explain_one`) and the global summary writer
(`summary.write_summary_section`). Built dynamically from a fitted
`ColumnTransformer` so the map is always in sync with what the v2
preprocessor actually emits — never hardcoded twice.

Ponytail: stdlib + numpy only; no external label-mapping libraries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Static fallback map — covers the numeric + ordinal blocks the v2
# preprocessor emits (Step 12's NUMERIC_FEATURES + Spec 14's
# GEO_NUMERIC_FEATURES + ORDINAL_FEATURES). One-hot block entries
# are built dynamically from the fitted OneHotEncoder's
# ``categories_``.
_STATIC_LABEL_MAP: dict[str, str] = {
    # numeric block (v2)
    "num__bedRoom": "Bedrooms",
    "num__bathroom": "Bathrooms",
    "num__built_up_area": "Built-up Area (sqft)",
    "num__servant_room": "Servant Room",
    "num__store_room": "Store Room",
    "num__n_amenities": "Number of Amenities",
    "num__distance_to_cbd_km": "Distance to City Center (km)",
    "num__distance_to_nearest_metro_km": "Distance to Nearest Metro (km)",
    "num__sector_smoothed_price": "Sector Average Price (smoothed)",
    "num__locality_smoothed_price": "Locality Average Price (smoothed)",
    "num__area_per_bedroom": "Area per Bedroom (sqft)",
    "num__bath_bed_ratio": "Bath-to-Bed Ratio",
    "num__floor_ratio": "Floor Ratio",
    "num__age_bucket_ord": "Property Age (ordinal)",
    "num__price_per_sqft": "Price per Sqft",
    # ordinal block
    "ord__luxury_category": "Luxury Category",
    "ord__floor_category": "Floor Category",
    "ord__furnishing_type": "Furnishing Type",
}


def FEATURE_LABEL_MAP_V2() -> dict[str, str]:  # noqa: N802  (snake-case would shadow the constant convention)
    """Return the static fallback label map.

    Kept as a function (not a top-level constant) so tests can call it
    without triggering the on-import file read; the on-disk JSON
    overlay is loaded lazily by :func:`load_label_map_from_disk`.
    """
    return dict(_STATIC_LABEL_MAP)


def load_label_map_from_disk(models_dir: Path | str) -> dict[str, str]:
    """Load ``models/feature_label_map_v{n}.json`` if present, else the static map.

    Never raises on a missing file — falls back to the static map so
    tests + the per-prediction CLI can run before the build CLI has
    landed the artifact.
    """
    base = dict(_STATIC_LABEL_MAP)
    models_dir = Path(models_dir)
    # Spec writes the file with a versioned name; the per-prediction CLI
    # defaults to v2. If newer versions exist, callers pass the version
    # explicitly. We probe v2 first; if absent, the caller passes a
    # different path.
    candidate = models_dir / "feature_label_map_v2.json"
    if candidate.exists():
        try:
            overlay = json.loads(candidate.read_text(encoding="utf-8"))
            base.update(overlay)
        except (OSError, json.JSONDecodeError) as exc:  # ponytail: defensive — one log line, never raise
            logger.warning("feature_label_map_v2.json unreadable (%s); using static fallback", exc)
    return base


def build_label_map(preprocessor: Any) -> dict[str, str]:
    """Walk a fitted ``ColumnTransformer`` and return the full label map.

    The transformer exposes ``transformers_`` (list of
    ``(name, transformer, columns)`` triples after fit). For each
    block we read:

    - numeric / passthrough blocks: column names from the third
      tuple element (list[str]).
    - ordinal blocks: same — column names from the third tuple element.
    - one-hot blocks: column names from
      ``OneHotEncoder.get_feature_names_out(columns)`` after fit.

    Unknown block types fall through with a logged WARNING so the
    per-prediction path never raises on an unseen block name (Rules
    §2.6 spirit — explanations must be interpretable).
    """
    label_map = dict(_STATIC_LABEL_MAP)
    # ColumnTransformer.transformers_ is only populated post-fit; the
    # spec guarantees a fitted preprocessor.
    transformers = getattr(preprocessor, "transformers_", None)
    if not transformers:
        logger.warning("preprocessor has no transformers_; returning static label map only")
        return label_map

    for block_name, transformer, columns in transformers:
        if columns is None:
            # 'remainder' passthrough block — skip; columns would be
            # an integer slice that doesn't translate to a name.
            continue
        # sklearn >=1.2 returns ndarray for get_feature_names_out; older
        # versions return a list[str]. Normalise to a list.
        try:
            from sklearn.preprocessing import OneHotEncoder  # noqa: WPS433  (lazy import — only needed when this block fires)
        except ImportError:  # pragma: no cover  (scikit-learn is pinned; this is defensive)
            OneHotEncoder = None  # type: ignore[assignment]

        if OneHotEncoder is not None and isinstance(transformer, OneHotEncoder):
            try:
                feat_names = list(transformer.get_feature_names_out(columns))
            except (AttributeError, TypeError):
                # Older sklearn fallback.
                feat_names = [f"{c}_{v}" for c in columns for v in transformer.categories_[columns.index(c)]]
            for raw in feat_names:
                key = f"{block_name}__{raw}"
                col, _, value = raw.partition("_")
                label_map[key] = f"{col.replace('_', ' ').title()}: {value}"
        else:
            for col in columns:
                key = f"{block_name}__{col}"
                if key not in label_map:
                    label_map[key] = col.replace("_", " ").title()

    return label_map


def resolve_label(name: str, label_map: dict[str, str]) -> str:
    """Resolve an internal feature name to its human-readable label.

    Defensive fall-through: unknown names return the raw internal name
    with a logged WARNING (never raise). One call site in
    ``contributions.explain_one`` — kept as a tiny helper so the
    fall-through rule lives in one place.
    """
    if name in label_map:
        return label_map[name]
    logger.warning("feature label missing for %r; falling through to raw name", name)
    return name


def save_label_map(label_map: dict[str, str], out_path: Path | str) -> Path:
    """Persist the label map to ``feature_label_map_v{n}.json``.

    Deterministic key order (sorted) so the file's sha1 is reproducible
    — the model_registry row records ``label_map_hash`` for audit.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(label_map, indent=2, sort_keys=True, ensure_ascii=False)
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def label_map_hash(label_map: dict[str, str]) -> str:
    """Stable sha1 of the label map for the registry row."""
    import hashlib

    canonical = json.dumps(label_map, indent=2, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "FEATURE_LABEL_MAP_V2",
    "load_label_map_from_disk",
    "build_label_map",
    "resolve_label",
    "save_label_map",
    "label_map_hash",
]


# ponytail: this module is intentionally pure data + a few small helpers.
# The dynamic build path matters because Step 14's v2 preprocessor emits
# one-hot keys whose names depend on the fitted encoder's categories_;
# hardcoding "Gurgaon"/"Mumbai" twice (once here, once in the v2 fit
# step) is exactly the drift the spec is built to prevent.
