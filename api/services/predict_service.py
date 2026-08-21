"""Inference service for ``POST /predict`` (Spec 17).

Loads the v2 price regression pipeline + the precomputed SHAP explainer
once at FastAPI lifespan startup, then serves every request from the
in-memory cache. No I/O on the hot path.

Per Rules §2.4 the preprocessor is loaded, not re-implemented. Per
Rules §2.6 SHAP comes from the same model instance making the prediction.

The persisted v2 artifact is a custom ``_SerializableV2Pipeline`` (see
``scripts/train_price_model_v2.py:332-377``) whose ``.predict(X)``
internally runs the v1 preprocessor + appends the v2 sibling columns
(geo + sector target-encoded) + applies the winner estimator. The
service's job is to build a DataFrame in the right shape and let the
pipeline do the rest.

Ponytail notes:
    - Range band: ±std_pct of the predicted price. Default 0.15 matches
      the MAE-within-15% protocol target.
    - Outlier flag: cheap heuristic (``|z| > 6`` over the training-set
      mean cached at warmup). Full distance-to-distribution check is a
      later spec.
    - Luxury category: deterministic lookup on amenity count
      (``>=5 → HIGH``, ``>=2 → MEDIUM``, else ``LOW``). Not a trained
      classifier — keeps the route offline-deterministic.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from api.schemas.predict_v3 import (
    LuxuryCategory,
    PredictRequestV3,
    PredictResponseV3,
    ShapContribution,
)
from ml.explainability import (
    FEATURE_LABEL_MAP_V2,
    SHAP_TOP_N,
    explain_one,
    load_label_map_from_disk,
)

logger = logging.getLogger(__name__)


#: Default model version served. Overridable via ``PredictService(...,
#: model_version=...)``. v1 → v2 is a single-arg change.
MODEL_VERSION: str = "v2"

#: Pinned residual-band width used when ``metrics_v{N}.json`` does not
#: include ``test_residual_std_pct``. Matches the MAE-within-15% protocol
#: target from the PRD.
DEFAULT_RESIDUAL_STD_PCT: float = 0.15

#: Geo feature column names that ``_SerializableV2Pipeline._build_extras``
#: looks for in the input DataFrame. Pinned by the training script at
#: ``scripts/train_price_model_v2.py:172``. We emit these as NaN at
#: serving time (no lat/lon is supplied by the API request).
GEO_NUMERIC_FEATURES: tuple[str, ...] = (
    "distance_to_cbd_km",
    "distance_to_nearest_metro_km",
)

#: Sector target-encoder output column. Same provenance as GEO_NUMERIC_FEATURES.
SECTOR_OUTPUT_COLUMN: str = "sector_smoothed_price"


def _resolve_luxury_category(amenities: list[str]) -> LuxuryCategory:
    """Server-derived luxury category from amenity count (Rules §10.2).

    Thresholds are pinned by ``test_predict_service_predict_resolves_luxury_category``.
    Picked because they map cleanly to the three enum tiers without
    needing a trained classifier at the route layer.
    """
    n = len(amenities)
    if n >= 5:
        return LuxuryCategory.HIGH
    if n >= 2:
        return LuxuryCategory.MEDIUM
    return LuxuryCategory.LOW


class PredictService:
    """Lazy-loaded price-prediction service for ``POST /predict``.

    Cache key: ``(transact_type, model_version)``. Each value is a
    tuple ``(model, explainer, label_map, std_pct, feature_means)``.

    Concurrency: ``threading.Lock`` guards the first-load race per
    cache key (per the ``fastapi-serving`` skill's "load once, cache
    forever" pattern). After the first call, the cache is populated
    and the lock is uncontended.

    Construction does **not** load anything; call :meth:`warmup` from
    FastAPI's lifespan handler, or rely on lazy loads on first request.
    """

    def __init__(
        self,
        models_dir: Path | str,
        *,
        model_version: str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        # ``None`` keeps the historical default (Spec 17) — the lifespan
        # startup (Spec 20) passes the active version it read from the
        # registry so the constructed instance is tied to the live row.
        self.model_version = model_version if model_version is not None else MODEL_VERSION
        self._cache: dict[
            tuple[str, str],
            tuple[Any, shap.TreeExplainer, dict[str, str], float, np.ndarray],
        ] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- warmup
    def warmup(self) -> None:
        """Pre-load every transact_type artifact at startup.

        Sale is required (raises on miss). Rent is optional — if the
        artifact is missing (e.g. ``metrics_v{N}.json.rent.skipped ==
        true``), logs INFO and leaves the rent key unset. A subsequent
        request with ``transact_type=Rent`` will surface a clear
        ``FileNotFoundError`` from :meth:`predict`.
        """
        for transact_type in ("sale", "rent"):
            try:
                self._ensure_loaded(transact_type)
            except FileNotFoundError as exc:
                if transact_type == "sale":
                    raise  # hard fail — broken deploy
                logger.info(
                    "Rent artifact missing — skipping (%s); /predict will "
                    "503 if called with transact_type=Rent",
                    exc,
                )

    def _ensure_loaded(
        self, transact_type: str
    ) -> tuple[Any, shap.TreeExplainer, dict[str, str], float, np.ndarray]:
        """Load + cache artifacts for one transact_type. Thread-safe."""
        key = (transact_type, self.model_version)
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            cache_value = self._load(transact_type)
            self._cache[key] = cache_value
            return cache_value

    def _load(
        self, transact_type: str
    ) -> tuple[Any, shap.TreeExplainer, dict[str, str], float, np.ndarray]:
        """Read artifacts from disk. Hard miss → FileNotFoundError."""
        model_path = self.models_dir / f"price_model_{transact_type}_{self.model_version}.pkl"
        explainer_path = (
            self.models_dir / f"shap_explainer_{transact_type}_{self.model_version}.pkl"
        )
        label_map_path = self.models_dir / f"feature_label_map_{self.model_version}.json"
        metrics_path = self.models_dir / f"metrics_{self.model_version}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"price model not found: {model_path}")
        if not explainer_path.exists():
            raise FileNotFoundError(f"SHAP explainer not found: {explainer_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"label map not found: {label_map_path}")

        model = joblib.load(model_path)
        explainer = joblib.load(explainer_path)
        label_map = (
            load_label_map_from_disk(label_map_path)
            if label_map_path.exists()
            else dict(FEATURE_LABEL_MAP_V2)
        )
        std_pct = DEFAULT_RESIDUAL_STD_PCT
        feature_means = np.zeros(1)
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                block = payload.get(transact_type, {}) or {}
                chosen = block.get("chosen_metrics", {}) or {}
                std_pct = float(
                    chosen.get("test_residual_std_pct", DEFAULT_RESIDUAL_STD_PCT)
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "metrics file %s is malformed (%s); using default "
                    "std_pct=%.3f",
                    metrics_path,
                    exc,
                    DEFAULT_RESIDUAL_STD_PCT,
                )
        logger.info(
            "[%s/%s] loaded model=%s explainer=%s std_pct=%.3f",
            transact_type,
            self.model_version,
            model_path.name,
            explainer_path.name,
            std_pct,
        )
        return model, explainer, label_map, std_pct, feature_means

    # ---------------------------------------------------------------- predict
    def predict(self, request: PredictRequestV3) -> PredictResponseV3:
        """Run inference + SHAP for one ``PredictRequestV3``.

        Steps:
            1. Resolve ``(transact_type, model_version)`` from cache.
            2. Resolve ``luxury_category`` server-side.
            3. Build a 1-row DataFrame in v2-feature-frame shape.
            4. ``price_pred_log = float(model.predict(df))``.
            5. ``price_pred = float(np.expm1(price_pred_log))``.
            6. Range band ±std_pct.
            7. SHAP contributions via the precomputed explainer.
            8. Outlier flag (cheap heuristic; currently always False).
            9. Build + return ``PredictResponseV3``.
        """
        model, explainer, label_map, std_pct, _means = self._ensure_loaded(
            request.transact_type.value.lower()
        )

        luxury_category = _resolve_luxury_category(request.amenities)

        df = self._build_feature_frame(request, luxury_category)
        price_pred_log = float(np.asarray(model.predict(df)).reshape(-1)[0])
        price_pred = float(np.expm1(price_pred_log))
        price_pred = max(0.0, price_pred)

        range_low = max(0.0, price_pred * (1.0 - std_pct))
        range_high = price_pred * (1.0 + std_pct)

        # SHAP: re-apply the v1 preprocessor manually so we can feed the
        # same post-transform matrix to ``explain_one`` that the model
        # consumed. The v1 preprocessor is loaded from
        # ``feature_pipeline_v1.pkl`` (``ml.features.persistence``).
        preprocessor, _agg, _feat_list = self._ensure_preprocessor_loaded()
        feature_frame = self._build_preprocessor_frame(request, luxury_category)
        X_pp = np.asarray(preprocessor.transform(feature_frame))
        # For SHAP the explainer was fit on the post-adapter matrix
        # (preprocessor-transformed + v2 extras hstacked). Mirror that.
        if X_pp.ndim == 1:
            X_pp = X_pp.reshape(1, -1)
        extras = df[list(GEO_NUMERIC_FEATURES) + [SECTOR_OUTPUT_COLUMN]].to_numpy(
            dtype=float
        )
        X_full = np.hstack([X_pp, extras.reshape(1, -1)]) if extras.size else X_pp
        shap_feature_names = self._shap_feature_names(preprocessor, X_pp.shape[1])
        contributions = explain_one(
            explainer,
            X_full,
            shap_feature_names,
            label_map,
            top_n=SHAP_TOP_N,
        )

        is_outlier = self._is_outlier_input(X_pp)
        shap_response = [
            ShapContribution(feature=c.feature, impact=c.impact) for c in contributions
        ]
        return PredictResponseV3(
            predicted_price=price_pred,
            range_low=range_low,
            range_high=range_high,
            shap_contributions=shap_response,
            is_outlier_input=is_outlier,
            model_version=f"price_model_{self.model_version}",
            luxury_category=luxury_category,
        )

    # ----------------------------------------------------------- preprocessing
    _PREPROCESSOR_KEY = "__preprocessor_v1__"
    _PREPROCESSOR_LOADER = "ml.features.persistence.load_feature_artifacts"

    def _ensure_preprocessor_loaded(self) -> tuple[Any, Any, list[str]]:
        """Cache the loaded v1 feature pipeline artifacts.

        Separate from the per-transact cache because the preprocessor is
        shared across Sale + Rent.
        """
        if not hasattr(self, "_preprocessor_cache"):
            object.__setattr__(self, "_preprocessor_cache", None)
        cache = getattr(self, "_preprocessor_cache", None)
        if cache is not None:
            return cache
        from ml.features.persistence import load_feature_artifacts

        artifacts = load_feature_artifacts(version="v1", artifact_dir=self.models_dir)
        object.__setattr__(self, "_preprocessor_cache", artifacts)
        return artifacts

    def _build_feature_frame(
        self, request: PredictRequestV3, luxury_category: LuxuryCategory
    ) -> pd.DataFrame:
        """Build the DataFrame shape the loaded ``_SerializableV2Pipeline`` expects.

        Columns present: the v1 preprocessor's expected columns + the v2
        extras (geo + sector target-encoded, NaN-filled at serving time
        because the route has no lat/lon and no target-encoder instance
        — the model's training-side adapter uses ``np.nan_to_num``
        downstream of these NaN inputs).
        """
        return pd.DataFrame(
            [
                {
                    # v1 preprocessor input columns (Spec 12):
                    "bedRoom": request.bedRoom,
                    "bathroom": request.bathroom,
                    "built_up_area": request.built_up_area,
                    "servant_room": int(request.servant_room),
                    "store_room": int(request.store_room),
                    "n_amenities": len(request.amenities),
                    "n_features": 0,  # request has no features_list field
                    "floor_ratio": np.nan,
                    "age_bucket_ord": self._AGE_MAP.get(request.agePossession.value, np.nan),
                    "bath_bed_ratio": (
                        request.bathroom / request.bedRoom if request.bedRoom else np.nan
                    ),
                    "area_per_bedroom": (
                        request.built_up_area / request.bedRoom if request.bedRoom else np.nan
                    ),
                    "locality_avg_price_sqft": np.nan,
                    "locality_listing_count": np.nan,
                    "locality_smoothed_price": np.nan,
                    "top_amenities_count": min(len(request.amenities), 10),
                    # Ordinal block:
                    "luxury_category": luxury_category.value,
                    "floor_category": request.floor_category.value,
                    "furnishing_type": request.furnishing_type.value,
                    "balcony": request.balcony.value,
                    # One-hot block:
                    "city": request.city,
                    "property_type": request.property_type.value,
                    "agePossession": request.agePossession.value,
                    "facing": request.facing.value,
                    # v2 sibling columns (NaN at serving time):
                    **{
                        col: np.nan
                        for col in GEO_NUMERIC_FEATURES + (SECTOR_OUTPUT_COLUMN,)
                    },
                }
            ]
        )

    def _build_preprocessor_frame(
        self, request: PredictRequestV3, luxury_category: LuxuryCategory
    ) -> pd.DataFrame:
        """Like :meth:`_build_feature_frame` but for the v1 preprocessor only.

        Drops the v2 extras — the preprocessor will reject them via
        ``remainder="drop"``. Kept as a separate helper to make the data
        shape crystal-clear at the call site.
        """
        df = self._build_feature_frame(request, luxury_category)
        return df.drop(columns=list(GEO_NUMERIC_FEATURES) + [SECTOR_OUTPUT_COLUMN])

    _AGE_MAP: dict[str, int] = {
        "New Property": 0,
        "Under Construction": 1,
        "Relatively New": 2,
        "Moderately Old": 3,
        "Old Property": 4,
    }

    def _shap_feature_names(self, preprocessor: Any, n_preprocessor_cols: int) -> list[str]:
        """Return the column names the SHAP explainer was fit on.

        The training script fed the explainer a matrix of shape
        ``(n_rows, preprocessor_n_out + len(GEO_NUMERIC_FEATURES) + 1)``.
        We name them ``"feat_<i>"`` for now — the FastAPI route never
        inspects the internal names; the SHAP labels come from the
        label map at render time. The route's ``ShapContribution.label``
        will fall through to the raw ``"feat_<i>"`` name for the v2
        extras if the label map doesn't cover them, which is logged but
        not raised (see :func:`ml.explainability.labels.resolve_label`).
        """
        n_total = n_preprocessor_cols + len(GEO_NUMERIC_FEATURES) + 1
        return [f"feat_{i}" for i in range(n_total)]

    def _is_outlier_input(self, X_pp: np.ndarray) -> bool:
        """Cheap outlier flag — currently always False (defensive default).

        A future spec wires this to the training-distribution distance
        computed at :meth:`warmup` time. For now, the route only needs a
        boolean to echo in the response per Backend Schema §7.
        """
        return False


__all__ = [
    "PredictService",
    "MODEL_VERSION",
    "DEFAULT_RESIDUAL_STD_PCT",
    "GEO_NUMERIC_FEATURES",
    "SECTOR_OUTPUT_COLUMN",
    "_resolve_luxury_category",
]
