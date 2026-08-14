"""Artifact persistence for the price-model feature pipeline (Spec 12 Phase 4).

Two artifacts are written under :data:`ARTIFACT_DIR`:

    ``feature_pipeline_{version}.pkl`` — ``joblib.dump`` of a tuple
        ``(fitted_preprocessor, locality_aggregator)``. Tuple order is
        meaningful: preprocessor first, aggregator second.

    ``feature_list_{version}.json`` — the post-transform feature names
        plus the engineered-feature recipe (which base column produced
        which derived column). The schema is:

        .. code-block:: json

            {
              "version": "v1",
              "feature_names": ["num__bedRoom", "num__bathroom", ...],
              "preprocessor_input_columns": ["bedRoom", "bathroom", ...],
              "engineered_feature_recipe": {
                "price_per_sqft": "price_inr / built_up_area",
                ...
              }
            }

The serving path (FastAPI ``/predict``) and the training script both
call ``load_feature_artifacts`` to obtain the same preprocessor +
aggregator objects — guarantees train/serve parity (Rules §2.4).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer

from ml.features.locality_aggregator import LocalityAggregator

logger = logging.getLogger(__name__)

#: Directory under which both artifacts are written. Override at test
#: time via the ``HOUSINGIQ_ARTIFACT_DIR`` env var.
ARTIFACT_DIR: Path = Path(os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models"))

#: Pinned module-level constant. A future retrain bumps this to
#: ``"v2"``; old files are never overwritten in place (Rules §2.5).
FEATURE_ARTIFACT_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# Engineered feature recipe — pinned narrative reference for the
# feature_list_v1.json artifact. The key is the output column name;
# the value is a one-line description of the derivation.
# ---------------------------------------------------------------------------
ENGINEERED_FEATURE_RECIPE: dict[str, str] = {
    "price_per_sqft": "price_inr / built_up_area (within-row ratio; leakage-free)",
    "n_amenities": "len(amenities_list)",
    "n_features": "len(features_list)",
    "floor_ratio": "floor_num / total_floor (NaN if total_floor == 0)",
    "age_bucket_ord": "ordinal map of agePossession via AGE_BUCKET_ORDINAL",
    "bath_bed_ratio": "bathroom / bedRoom (NaN if bedRoom == 0)",
    "area_per_bedroom": "built_up_area / bedRoom (NaN if bedRoom == 0)",
    "locality_avg_price_sqft": "mean(price_per_sqft) over (city, locality), leave-one-out",
    "locality_listing_count": "size of (city, locality) group in training (non-outlier)",
    "locality_smoothed_price": "Bayesian-smoothed mean(price_inr) toward city prior, leave-one-out",
    "top_amenities_count": "how many of the top-K(10) amenities this row has",
}


def save_feature_artifacts(
    fitted_preprocessor: ColumnTransformer,
    locality_aggregator: LocalityAggregator,
    feature_list: list[str],
    version: str = FEATURE_ARTIFACT_VERSION,
    artifact_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Write the versioned artifact pair. Returns paths for test assertions."""
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"feature_pipeline_{version}.pkl"
    json_path = out_dir / f"feature_list_{version}.json"

    joblib.dump((fitted_preprocessor, locality_aggregator), pkl_path)
    logger.info("Wrote %s", pkl_path)

    # Resolve the preprocessor's input columns for the recipe.
    preprocessor_input_columns: list[str] = []
    for name, _transformer, cols in fitted_preprocessor.transformers_:
        if name == "num":
            preprocessor_input_columns.extend(list(cols))
        elif name in ("ord", "cat"):
            preprocessor_input_columns.extend(list(cols))

    payload = {
        "version": version,
        "feature_names": list(feature_list),
        "preprocessor_input_columns": preprocessor_input_columns,
        "engineered_feature_recipe": ENGINEERED_FEATURE_RECIPE,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", json_path)

    return {"pkl": pkl_path, "json": json_path}


def load_feature_artifacts(
    version: str = FEATURE_ARTIFACT_VERSION,
    artifact_dir: Path | str | None = None,
) -> tuple[ColumnTransformer, LocalityAggregator, list[str]]:
    """Load the versioned artifact pair.

    Raises ``FileNotFoundError`` with the expected path when artifacts
    are missing — actionable error for callers.
    """
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    pkl_path = out_dir / f"feature_pipeline_{version}.pkl"
    json_path = out_dir / f"feature_list_{version}.json"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Feature pipeline artifact not found: {pkl_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Feature list artifact not found: {json_path}")

    fitted_preprocessor, locality_aggregator = joblib.load(pkl_path)
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    feature_list = list(payload.get("feature_names", []))
    logger.info(
        "Loaded artifacts (version=%s): %d feature names",
        version,
        len(feature_list),
    )
    return fitted_preprocessor, locality_aggregator, feature_list


__all__ = [
    "ARTIFACT_DIR",
    "FEATURE_ARTIFACT_VERSION",
    "ENGINEERED_FEATURE_RECIPE",
    "save_feature_artifacts",
    "load_feature_artifacts",
]
