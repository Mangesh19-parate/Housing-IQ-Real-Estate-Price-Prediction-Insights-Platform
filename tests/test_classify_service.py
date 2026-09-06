"""Tests for ``api.services.classify_service.ClassifyService``.

The service wraps two classifier pipelines + a shared preprocessor + a
precomputed SHAP TreeExplainer. These tests build tiny synthetic artifacts
in-memory, then assert the service behaves correctly.

The test preprocessor outputs a fixed set of columns (matching the
classifier preprocessor's expected feature names from training). The
explainer is fit on the same column count.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import shap
from pathlib import Path
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from api.schemas.classify_v1 import (
    ClassifyResponseV1,
    SHAP_TOP_N,
    TierLabel,
    VerdictLabel,
)
from api.schemas.predict_v3 import (
    AgePossession,
    Balcony,
    FacingDirection,
    FloorCategory,
    FurnishingType,
    LuxuryCategory,
    PredictRequestV3,
    PropertyType,
    ShapContribution,
    TransactType,
)
from api.services.classify_service import (
    ClassifyService,
    MODEL_VERSION,
    _resolve_luxury_category,
)
from ml.classification import GOOD_DEAL_LABELS, PRICE_TIER_LABELS


# ---------------------------------------------------------------------------
# Constants — column names matching the classifier preprocessor output
# ---------------------------------------------------------------------------

# These are the columns the classifier preprocessor was fitted on during
# training. The test preprocessor must output the same column names so the
# downstream classifier + SHAP paths work.
_PREPROCESSOR_COLS: tuple[str, ...] = (
    "bedRoom",
    "bathroom",
    "built_up_area",
    "servant_room",
    "store_room",
    "n_amenities",
    "luxury_category",
    "floor_category",
    "furnishing_type",
    "balcony",
    "city",
    "property_type",
    "agePossession",
    "facing",
    "age_bucket_ord",
    "bath_bed_ratio",
    "area_per_bedroom",
    "locality_avg_price_sqft",
    "locality_listing_count",
    "locality_smoothed_price",
    "top_amenities_count",
    "floor_ratio",
)

_EXPLAINER_COLS: int = len(_PREPROCESSOR_COLS)


# ---------------------------------------------------------------------------
# Tiny synthetic fixtures
# ---------------------------------------------------------------------------


def _fit_tiny_preprocessor() -> ColumnTransformer:
    """Fit a ColumnTransformer on synthetic data matching _PREPROCESSOR_COLS.

    Uses StandardScaler on numeric cols, OrdinalEncoder on ordinal cols,
    OneHotEncoder on categorical cols. Returns a fitted transformer ready
    for .transform() on the service's feature frame.
    """
    # Create synthetic training data with all expected columns
    rng = np.random.default_rng(42)
    n_rows = 50

    data = {}
    for col in _PREPROCESSOR_COLS:
        if col in ("city", "property_type", "agePossession", "facing"):
            # Categorical
            data[col] = rng.choice(["A", "B", "C"], n_rows)
        elif col in ("luxury_category", "floor_category", "furnishing_type", "balcony"):
            # Ordinal (strings)
            data[col] = rng.choice(["Low", "Medium", "High"], n_rows)
        elif col in ("servant_room", "store_room"):
            # Binary
            data[col] = rng.integers(0, 2, n_rows)
        else:
            # Numeric
            data[col] = rng.uniform(1, 100, n_rows)

    df = pd.DataFrame(data)

    # Build the same column transformer structure the training script uses
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    numeric_cols = [
        "bedRoom",
        "bathroom",
        "built_up_area",
        "n_amenities",
        "age_bucket_ord",
        "bath_bed_ratio",
        "area_per_bedroom",
        "locality_avg_price_sqft",
        "locality_listing_count",
        "locality_smoothed_price",
        "top_amenities_count",
        "floor_ratio",
    ]
    ordinal_cols = [
        "luxury_category",
        "floor_category",
        "furnishing_type",
        "balcony",
    ]
    categorical_cols = [
        "city",
        "property_type",
        "agePossession",
        "facing",
    ]
    binary_cols = [
        "servant_room",
        "store_room",
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    pre.fit(df)
    return pre


class _TinyPipeline:
    """Minimal sklearn.Pipeline-shaped stub for a classifier.

    - predict(X) returns a fixed class index
    - predict_proba(X) returns a fixed probability distribution
    - Holds a fitted RandomForestClassifier on synthetic data of the right
      shape — used only for the SHAP path.
    """

    def __init__(self, n_classes: int, pred_class: int = 0) -> None:
        self.n_classes = n_classes
        self.pred_class = pred_class
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, _EXPLAINER_COLS))
        y = rng.integers(0, n_classes, 30)
        self._estimator = RandomForestClassifier(
            n_estimators=5, max_depth=2, random_state=42, n_jobs=1
        )
        self._estimator.fit(X, y)

    def predict(self, X):
        return np.array([self.pred_class])

    def predict_proba(self, X):
        # Return a valid probability distribution
        proba = np.zeros((1, self.n_classes))
        proba[0, self.pred_class] = 1.0
        return proba


def _make_tiny_explainer() -> shap.TreeExplainer:
    """Fit a tiny shap.TreeExplainer on synthetic data of the right shape."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, _EXPLAINER_COLS))
    y = rng.integers(0, 3, 30)
    est = RandomForestClassifier(
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
        bathroom=2,
        balcony=Balcony.TWO.value,
        agePossession=AgePossession.RELATIVELY_NEW.value,
        built_up_area=1450.0,
        servant_room=False,
        store_room=False,
        furnishing_type=FurnishingType.SEMIFURNISHED.value,
        floor_category=FloorCategory.MID.value,
        facing=FacingDirection.NORTH.value,
        amenities=["Clubhouse", "Swimming Pool"],
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
    """Both classifier artifacts + shared preprocessor + SHAP explainer + label map on disk."""
    joblib.dump(
        _TinyPipeline(n_classes=3, pred_class=0),
        writable_models_dir / f"good_deal_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _TinyPipeline(n_classes=4, pred_class=1),
        writable_models_dir / f"price_tier_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _fit_tiny_preprocessor(),
        writable_models_dir / f"classifier_preprocessor_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_good_deal_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        json.dumps({"feat_0": "Built-up Area (sqft)"}), encoding="utf-8"
    )
    return writable_models_dir


# ---------------------------------------------------------------------------
# Service fixture (classify tests)
# ---------------------------------------------------------------------------


def _make_service_with_cache(
    cache: dict[str, tuple],
    preprocessor: ColumnTransformer,
) -> ClassifyService:
    """Create a ClassifyService with injected cache + preprocessor.

    Bypasses the disk-load path so tests don't depend on real artifacts.
    The preprocessor is injected into the cache tuple so the service uses it.
    """
    svc = ClassifyService(Path("/tmp"))
    # Replace the preprocessor in the cache entry with the one we're passing
    new_cache = {}
    for k, v in cache.items():
        # v is (good_deal_pipe, price_tier_pipe, preprocessor, explainer, label_map)
        new_cache[k] = (v[0], v[1], preprocessor, v[3], v[4])
    svc._cache = new_cache  # type: ignore[attr-defined]
    return svc


def _cache_entry() -> tuple:
    """Single cache entry — factory so ruff doesn't ding line length."""
    return (
        _TinyPipeline(n_classes=3, pred_class=0),  # good_deal
        _TinyPipeline(n_classes=4, pred_class=1),  # price_tier
        _fit_tiny_preprocessor(),
        _make_tiny_explainer(),
        {"feat_0": "Built-up Area (sqft)"},
    )


# ---------------------------------------------------------------------------
# warmup tests
# ---------------------------------------------------------------------------


def test_classify_service_warmup_loads_all_artifacts(populated_models_dir: Path) -> None:
    svc = ClassifyService(populated_models_dir)
    svc.warmup()
    assert MODEL_VERSION in svc._cache


def test_classify_service_warmup_is_idempotent(populated_models_dir: Path) -> None:
    svc = ClassifyService(populated_models_dir)
    svc.warmup()
    first = svc._cache[MODEL_VERSION]
    svc.warmup()
    second = svc._cache[MODEL_VERSION]
    assert first is second


def test_classify_service_warmup_raises_on_missing_good_deal(writable_models_dir: Path) -> None:
    """Missing good_deal artifact → FileNotFoundError (hard fail)."""
    # Only provide price_tier + preprocessor
    joblib.dump(
        _TinyPipeline(n_classes=4, pred_class=1),
        writable_models_dir / f"price_tier_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _fit_tiny_preprocessor(),
        writable_models_dir / f"classifier_preprocessor_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_good_deal_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )

    svc = ClassifyService(writable_models_dir)
    with pytest.raises(FileNotFoundError, match="good_deal"):
        svc.warmup()


def test_classify_service_warmup_raises_on_missing_price_tier(writable_models_dir: Path) -> None:
    """Missing price_tier artifact → FileNotFoundError (hard fail)."""
    joblib.dump(
        _TinyPipeline(n_classes=3, pred_class=0),
        writable_models_dir / f"good_deal_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _fit_tiny_preprocessor(),
        writable_models_dir / f"classifier_preprocessor_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_good_deal_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )

    svc = ClassifyService(writable_models_dir)
    with pytest.raises(FileNotFoundError, match="price_tier"):
        svc.warmup()


def test_classify_service_warmup_raises_on_missing_preprocessor(writable_models_dir: Path) -> None:
    """Missing shared preprocessor → FileNotFoundError (hard fail)."""
    joblib.dump(
        _TinyPipeline(n_classes=3, pred_class=0),
        writable_models_dir / f"good_deal_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _TinyPipeline(n_classes=4, pred_class=1),
        writable_models_dir / f"price_tier_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_good_deal_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )

    svc = ClassifyService(writable_models_dir)
    with pytest.raises(FileNotFoundError, match="preprocessor"):
        svc.warmup()


def test_classify_service_warmup_skips_shap_explainer_when_missing(writable_models_dir: Path) -> None:
    """Missing SHAP explainer logs WARNING but doesn't fail warmup."""
    joblib.dump(
        _TinyPipeline(n_classes=3, pred_class=0),
        writable_models_dir / f"good_deal_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _TinyPipeline(n_classes=4, pred_class=1),
        writable_models_dir / f"price_tier_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _fit_tiny_preprocessor(),
        writable_models_dir / f"classifier_preprocessor_{MODEL_VERSION}.pkl",
    )
    (writable_models_dir / f"feature_label_map_{MODEL_VERSION}.json").write_text(
        "{}", encoding="utf-8"
    )

    svc = ClassifyService(writable_models_dir)
    svc.warmup()  # Should not raise
    assert MODEL_VERSION in svc._cache
    _, _, _, explainer, _ = svc._cache[MODEL_VERSION]
    assert explainer is None


def test_classify_service_warmup_skips_label_map_when_missing(writable_models_dir: Path) -> None:
    """Missing label map logs WARNING but doesn't fail warmup."""
    joblib.dump(
        _TinyPipeline(n_classes=3, pred_class=0),
        writable_models_dir / f"good_deal_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _TinyPipeline(n_classes=4, pred_class=1),
        writable_models_dir / f"price_tier_classifier_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _fit_tiny_preprocessor(),
        writable_models_dir / f"classifier_preprocessor_{MODEL_VERSION}.pkl",
    )
    joblib.dump(
        _make_tiny_explainer(),
        writable_models_dir / f"shap_explainer_good_deal_{MODEL_VERSION}.pkl",
    )

    svc = ClassifyService(writable_models_dir)
    svc.warmup()  # Should not raise
    assert MODEL_VERSION in svc._cache
    _, _, _, _, label_map = svc._cache[MODEL_VERSION]
    assert label_map == {}


# ---------------------------------------------------------------------------
# classify tests
# ---------------------------------------------------------------------------


def test_classify_service_classify_returns_response_v1() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert isinstance(response, ClassifyResponseV1)


def test_classify_service_classify_verdict_is_valid_enum() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert response.good_deal_verdict in VerdictLabel


def test_classify_service_classify_tier_is_valid_enum() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert response.price_tier in TierLabel


def test_classify_service_classify_verdict_probabilities_sum_to_one() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert abs(sum(response.verdict_probabilities.values()) - 1.0) < 1e-9


def test_classify_service_classify_tier_probabilities_sum_to_one() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert abs(sum(response.tier_probabilities.values()) - 1.0) < 1e-9


def test_classify_service_classify_verdict_probabilities_has_all_classes() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert set(response.verdict_probabilities.keys()) == set(VerdictLabel)


def test_classify_service_classify_tier_probabilities_has_all_classes() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert set(response.tier_probabilities.keys()) == set(TierLabel)


def test_classify_service_classify_attaches_shap_contributions() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert isinstance(response.shap_contributions, list)
    # Capped at SHAP_TOP_N=7
    assert len(response.shap_contributions) <= SHAP_TOP_N
    if response.shap_contributions:
        assert isinstance(response.shap_contributions[0], ShapContribution)
        assert hasattr(response.shap_contributions[0], "feature")
        assert hasattr(response.shap_contributions[0], "impact")


def test_classify_service_classify_resolves_luxury_category() -> None:
    """Pinned lookup: len>=5 → HIGH, >=2 → MEDIUM, else LOW."""
    assert _resolve_luxury_category([]) == LuxuryCategory.LOW
    assert _resolve_luxury_category(["Clubhouse"]) == LuxuryCategory.LOW
    assert _resolve_luxury_category(["Clubhouse", "Gym"]) == LuxuryCategory.MEDIUM
    assert (
        _resolve_luxury_category(["Clubhouse", "Gym", "Pool", "Security", "Power"])
        == LuxuryCategory.HIGH
    )


def test_classify_service_classify_echoes_luxury_category_in_response() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from(amenities=["Clubhouse", "Gym", "Pool"]))
    assert response.luxury_category == LuxuryCategory.MEDIUM


def test_classify_service_classify_uses_loaded_preprocessor() -> None:
    """Service must call the loaded preprocessor's transform."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    seen = {"called": False}
    original = pre.transform

    def spy(X):
        seen["called"] = True
        return original(X)

    pre.transform = spy
    try:
        svc.classify(_request_from())
    finally:
        pre.transform = original
    assert seen["called"], "service must call the loaded preprocessor"


def test_classify_service_classify_uses_loaded_explainer() -> None:
    """Service must read SHAP from the loaded explainer."""
    pre = _fit_tiny_preprocessor()
    explainer = _make_tiny_explainer()
    cache_val = (
        _TinyPipeline(n_classes=3, pred_class=0),
        _TinyPipeline(n_classes=4, pred_class=1),
        pre,
        explainer,
        {"feat_0": "Built-up Area (sqft)"},
    )
    svc = _make_service_with_cache({MODEL_VERSION: cache_val}, pre)
    seen = {"called": False}
    original = explainer.shap_values

    def spy(X, **kwargs):
        seen["called"] = True
        return original(X, **kwargs)

    explainer.shap_values = spy
    try:
        svc.classify(_request_from())
    finally:
        explainer.shap_values = original
    assert seen["called"], "service must call the loaded explainer"


def test_classify_service_classify_does_not_log_pii_fields() -> None:
    """Response model never carries PII columns (Rules §1)."""
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    dumped = json.dumps(response.model_dump())
    _PII = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)
    assert not _PII.search(dumped), dumped[:200]


def test_classify_service_classify_model_version_in_response() -> None:
    pre = _fit_tiny_preprocessor()
    svc = _make_service_with_cache({MODEL_VERSION: _cache_entry()}, pre)
    response = svc.classify(_request_from())
    assert response.model_version == f"good_deal_classifier_{MODEL_VERSION}"


def test_classify_service_classify_missing_artifact_raises_file_not_found() -> None:
    """Service.classify on a missing cache key raises FileNotFoundError → 503."""
    svc = ClassifyService(Path("/tmp"))
    with pytest.raises(FileNotFoundError):
        svc.classify(_request_from())


# ---------------------------------------------------------------------------
# Import re for PII regex in classify tests
# ---------------------------------------------------------------------------
import re