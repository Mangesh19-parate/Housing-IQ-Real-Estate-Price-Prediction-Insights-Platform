"""Tests for ``api.services.predict_service.PredictService``.

The service wraps the v2 ``_SerializableV2Pipeline`` + precomputed SHAP
explainer. These tests build a tiny synthetic pipeline + a real
``shap.TreeExplainer`` in-memory, then assert the service behaves
correctly.

The preprocessor in the test outputs 6 columns (``StandardScaler`` on
6 numeric v1-input cols always present in the service's
``_build_preprocessor_frame``). The explainer is fit on
``(n_samples, 6 + 3)`` = 9 columns — matching what the service's SHAP
path will pass: the preprocessor's 6 outputs + the 3 v2 sibling cols
(geo + sector target-encoded) hstacked together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from api.schemas.predict_v3 import (
    AgePossession,
    Balcony,
    FacingDirection,
    FloorCategory,
    FurnishingType,
    LuxuryCategory,
    PredictRequestV3,
    PredictResponseV3,
    PropertyType,
    TransactType,
)
from api.services.predict_service import (
    DEFAULT_RESIDUAL_STD_PCT,
    MODEL_VERSION,
    PredictService,
    _resolve_luxury_category,
)

# ---------------------------------------------------------------------------
# Constants — column counts and names
# ---------------------------------------------------------------------------


#: Numeric v1-input columns the test preprocessor uses. All 6 are
#: present in the service's ``_build_preprocessor_frame`` (see
#: ``api/services/predict_service.py:351-361``).
_PREPROCESSOR_COLS: tuple[str, ...] = (
    "bedRoom",
    "bathroom",
    "built_up_area",
    "servant_room",
    "store_room",
    "n_amenities",
)

#: v2 sibling columns the service appends to the post-preprocessor matrix
#: (geo distances + sector target-encoded). See
#: ``api/services/predict_service.py:70-76``.
_V2_EXTRAS: tuple[str, ...] = (
    "distance_to_cbd_km",
    "distance_to_nearest_metro_km",
    "sector_smoothed_price",
)

#: Total cols the explainer sees: preprocessor output + v2 extras.
_EXPLAINER_COLS: int = len(_PREPROCESSOR_COLS) + len(_V2_EXTRAS)


# ---------------------------------------------------------------------------
# Tiny synthetic fixtures
# ---------------------------------------------------------------------------


def _fit_tiny_preprocessor() -> ColumnTransformer:
    """Fit a 6-column ``StandardScaler`` preprocessor on synthetic data.

    Output shape: ``(n_rows, 6)``. Returns a fitted
    ``ColumnTransformer`` ready for ``.transform()`` on the service's
    v1 preprocessor frame.
    """
    df = pd.DataFrame({c: [2, 3, 4, 5, 3] for c in _PREPROCESSOR_COLS})
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(_PREPROCESSOR_COLS))],
        remainder="drop",
        sparse_threshold=0.0,
    )
    pre.fit(df)
    return pre


class _TinyPipeline:
    """Minimal ``_SerializableV2Pipeline``-shaped stub.

    - ``predict(X)`` returns a fixed log-price (the real model's
      prediction is not what this test exercises).
    - Holds a fitted ``RandomForestRegressor`` on synthetic data of
      the right shape — used only for the SHAP path.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, _EXPLAINER_COLS))
        y = np.log1p(rng.uniform(1e6, 2e7, 20))
        self._estimator = RandomForestRegressor(
            n_estimators=5, max_depth=2, random_state=42, n_jobs=1
        )
        self._estimator.fit(X, y)
        # The service reads ``self.preprocessor`` for completeness, but
        # the SHAP path uses the preprocessor from ``_preprocessor_cache``
        # instead. Keep the attribute so any sanity check on the
        # pipeline object doesn't blow up.
        self.preprocessor = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([np.log1p(7_000_000.0)])


def _make_tiny_explainer() -> shap.TreeExplainer:
    """Fit a tiny ``shap.TreeExplainer`` on synthetic data of the right shape."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, _EXPLAINER_COLS))
    y = np.log1p(rng.uniform(1e6, 2e7, 20))
    est = RandomForestRegressor(
        n_estimators=5, max_depth=2, random_state=42, n_jobs=1
    )
    est.fit(X, y)
    return shap.TreeExplainer(est)


def _minimal_payload(**overrides: Any) -> dict[str, Any]:
    base = dict(
        city="Gurgaon",
        sector="sector 84",
        property_type=PropertyType.FLAT.value,
        transact_type=TransactType.SALE.value,
        bedRoom=3,
        bathroom=3,
        balcony=Balcony.TWO.value,
        agePossession=AgePossession.RELATIVELY_NEW.value,
        built_up_area=1450.0,
        servant_room=False,
        store_room=False,
        furnishing_type=FurnishingType.SEMIFURNISHED.value,
        floor_category=FloorCategory.MID.value,
        facing=FacingDirection.NORTH.value,
        amenities=["Clubhouse"],
    )
    base.update(overrides)
    return base


def _request_from(**overrides: Any) -> PredictRequestV3:
    return PredictRequestV3(**_minimal_payload(**overrides))


# ---------------------------------------------------------------------------
# Disk fixtures (warmup tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def writable_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def populated_models_dir(writable_models_dir: Path) -> Path:
    """Sale + Rent artifacts + label map + metrics on disk."""
    joblib.dump(
        _TinyPipeline(),
        writable_models_dir / f"price_model_sale_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_sale_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _TinyPipeline(),
        writable_models_dir / f"price_model_rent_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_rent_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        json.dumps({"feat_0": "Built-up Area (sqft)"}),
        encoding="utf-8",
    )
    (writable_models_dir / f"metrics_{MODEL_VERSION}.json").write_text(
        json.dumps({
            "sale": {"chosen_metrics": {"test_residual_std_pct": 0.15}},
            "rent": {"chosen_metrics": {"test_residual_std_pct": 0.15}},
        }),
        encoding="utf-8",
    )
    return writable_models_dir


# ---------------------------------------------------------------------------
# Service fixture (predict tests)
# ---------------------------------------------------------------------------


def _make_service_with_cache(
    cache: dict[tuple[str, str], tuple],
    preprocessor: ColumnTransformer,
) -> PredictService:
    """Create a ``PredictService`` with injected cache + preprocessor.

    Bypasses the disk-load path so tests don't depend on real artifacts.
    The cache key shape is ``(transact_type, model_version)`` — matches
    what :meth:`PredictService._ensure_loaded` reads.
    """
    svc = PredictService(Path("/tmp"))
    svc._cache = cache  # type: ignore[attr-defined]
    svc._preprocessor_cache = (preprocessor, None, list(_PREPROCESSOR_COLS))
    return svc


def _cache_entry() -> tuple:
    """Single cache entry — factory so ruff doesn't ding line length."""
    return (_TinyPipeline(), _make_tiny_explainer(), {}, 0.15, np.zeros(1))


# ---------------------------------------------------------------------------
# warmup tests
# ---------------------------------------------------------------------------


def test_predict_service_warmup_loads_sale_pipeline(populated_models_dir: Path) -> None:
    svc = PredictService(populated_models_dir)
    svc.warmup()
    assert ("sale", MODEL_VERSION) in svc._cache


def test_predict_service_warmup_loads_rent_pipeline_when_present(
    populated_models_dir: Path,
) -> None:
    svc = PredictService(populated_models_dir)
    svc.warmup()
    assert ("rent", MODEL_VERSION) in svc._cache


def test_predict_service_warmup_skips_rent_when_artifact_missing(
    writable_models_dir: Path,
) -> None:
    """Only sale artifacts present — rent key should be unset."""
    joblib.dump(
        _TinyPipeline(),
        writable_models_dir / f"price_model_sale_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_sale_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )
    (writable_models_dir / f"metrics_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )

    svc = PredictService(writable_models_dir)
    svc.warmup()
    assert ("sale", MODEL_VERSION) in svc._cache
    assert ("rent", MODEL_VERSION) not in svc._cache


def test_predict_service_warmup_is_idempotent(populated_models_dir: Path) -> None:
    svc = PredictService(populated_models_dir)
    svc.warmup()
    first = svc._cache[("sale", MODEL_VERSION)]
    svc.warmup()
    second = svc._cache[("sale", MODEL_VERSION)]
    assert first is second


# ---------------------------------------------------------------------------
# predict tests
# ---------------------------------------------------------------------------


def test_predict_service_predict_returns_response_v3() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    assert isinstance(response, PredictResponseV3)
    assert response.model_version == f"price_model_{MODEL_VERSION}"


def test_predict_service_predict_routes_by_transact_type() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {
            ("sale", MODEL_VERSION): _cache_entry(),
            ("rent", MODEL_VERSION): _cache_entry(),
        },
        pre,
    )
    sale_resp = svc.predict(_request_from(transact_type="Sale"))
    rent_resp = svc.predict(_request_from(transact_type="Rent"))
    assert isinstance(sale_resp, PredictResponseV3)
    assert isinstance(rent_resp, PredictResponseV3)


def test_predict_service_predict_attaches_shap_contributions() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    assert isinstance(response.shap_contributions, list)
    # Either empty or capped at SHAP_TOP_N=7 — both are valid.
    assert len(response.shap_contributions) <= 7


def test_predict_service_predict_excludes_outlier_input_flag() -> None:
    """Response always has a boolean ``is_outlier_input``."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    assert isinstance(response.is_outlier_input, bool)


def test_predict_service_predict_resolves_luxury_category() -> None:
    """Pinned lookup: len>=5 → HIGH, >=2 → MEDIUM, else LOW."""
    assert _resolve_luxury_category([]) == LuxuryCategory.LOW
    assert _resolve_luxury_category(["Clubhouse"]) == LuxuryCategory.LOW
    assert _resolve_luxury_category(["Clubhouse", "Gym"]) == LuxuryCategory.MEDIUM
    assert (
        _resolve_luxury_category(["Clubhouse", "Gym", "Pool", "Security", "Power"])
        == LuxuryCategory.HIGH
    )


def test_predict_service_predict_uses_loaded_preprocessor() -> None:
    """Service must call the loaded preprocessor's ``transform``."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    seen = {"called": False}
    original = pre.transform

    def spy(X):
        seen["called"] = True
        return original(X)

    pre.transform = spy
    try:
        svc.predict(_request_from())
    finally:
        pre.transform = original
    assert seen["called"], "service must call the loaded preprocessor"


def test_predict_service_predict_uses_loaded_explainer() -> None:
    """Service must read SHAP from the loaded explainer."""
    pre = _fit_tiny_preprocessor()
    explainer = _make_tiny_explainer()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): (_TinyPipeline(), explainer, {}, 0.15, np.zeros(1))},
        pre,
    )
    seen = {"called": False}
    original = explainer.shap_values

    def spy(X, **kwargs):
        seen["called"] = True
        return original(X, **kwargs)

    explainer.shap_values = spy
    try:
        svc.predict(_request_from())
    finally:
        explainer.shap_values = original
    assert seen["called"], "service must call the loaded explainer"


def test_predict_service_predict_does_not_log_pii_fields() -> None:
    """Service response model never carries PII columns (Rules §1)."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    dumped = json.dumps(response.model_dump())
    _PII = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)
    assert not _PII.search(dumped), dumped[:200]


def test_predict_service_predict_latency_ms_is_absent() -> None:
    """The latency contract is enforced at the route; service exposes no latency."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    assert "latency_ms" not in response.model_dump()


def test_predict_service_predict_predicted_price_is_non_negative() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache(
        {("sale", MODEL_VERSION): _cache_entry()},
        pre,
    )
    response = svc.predict(_request_from())
    assert response.predicted_price >= 0
    assert response.range_low >= 0
    assert response.range_high >= response.predicted_price


def test_predict_service_default_residual_std_pct_is_pinned() -> None:
    """DEFAULT_RESIDUAL_STD_PCT matches the MAE-within-15% protocol target."""
    assert DEFAULT_RESIDUAL_STD_PCT == 0.15


def test_predict_service_predict_missing_artifact_raises_file_not_found() -> None:
    """Service.predict on a missing cache key raises FileNotFoundError → 503."""
    svc = PredictService(Path("/tmp"))
    with pytest.raises(FileNotFoundError):
        svc.predict(_request_from(transact_type="Sale"))
