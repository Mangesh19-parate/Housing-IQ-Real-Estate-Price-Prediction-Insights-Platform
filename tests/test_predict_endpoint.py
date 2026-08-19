"""Tests for the FastAPI ``POST /predict`` route (TestClient).

Uses ``tmp_clean_db`` (from ``tests/conftest.py:14-26``) for the
prediction-log assertion, monkeypatches ``get_predict_service`` per-test
so the route's lifespan doesn't try to load real artifacts from disk.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers import predict as predict_router
from api.schemas.predict_v3 import (
    AgePossession,
    Balcony,
    FacingDirection,
    FloorCategory,
    FurnishingType,
    LuxuryCategory,
    PredictResponseV3,
    PropertyType,
    ShapContribution,
    TransactType,
)
from app.database.db import get_db

_PAYLOAD: dict[str, Any] = {
    "city": "Gurgaon",
    "sector": "sector 84",
    "property_type": PropertyType.FLAT.value,
    "transact_type": TransactType.SALE.value,
    "bedRoom": 3,
    "bathroom": 3,
    "balcony": Balcony.TWO.value,
    "agePossession": AgePossession.RELATIVELY_NEW.value,
    "built_up_area": 1450.0,
    "servant_room": True,
    "store_room": False,
    "furnishing_type": FurnishingType.SEMIFURNISHED.value,
    "floor_category": FloorCategory.MID.value,
    "facing": FacingDirection.NORTH.value,
    "amenities": ["Clubhouse", "Swimming Pool"],
}


def _stub_response(**overrides: Any) -> PredictResponseV3:
    base = dict(
        predicted_price=14_200_000.0,
        range_low=12_800_000.0,
        range_high=15_600_000.0,
        shap_contributions=[
            ShapContribution(feature="num__built_up_area", impact=0.18),
        ],
        is_outlier_input=False,
        model_version="price_model_v2",
        luxury_category=LuxuryCategory.MEDIUM,
    )
    base.update(overrides)
    return PredictResponseV3(**base)


class _StubService:
    """Minimal PredictService-shaped stub for route tests.

    Avoids loading real artifacts; records calls so tests can assert
    that the route delegated correctly.
    """

    def __init__(self, response: PredictResponseV3 | None = None,
                 raise_exc: BaseException | None = None) -> None:
        self._response = response or _stub_response()
        self._raise = raise_exc
        self.calls: list[Any] = []
        self.warmed = False

    def warmup(self) -> None:
        self.warmed = True

    def predict(self, request):  # noqa: ANN001
        self.calls.append(request)
        if self._raise is not None:
            raise self._raise
        return self._response


@pytest.fixture
def stub_service() -> _StubService:
    return _StubService()


@pytest.fixture
def client(monkeypatch, stub_service, tmp_path) -> TestClient:
    """FastAPI TestClient with a stub service + temp DB.

    Skips lifespan by overriding the FastAPI ``lifespan`` to a no-op via
    monkeypatch on ``predict_router.get_predict_service`` so the test
    never triggers real artifact loading.

    The DB path is patched on **both** ``app.config`` (where it's
    declared) and ``app.database.db`` (where it's bound by
    ``from app.config import APP_DB_PATH`` at import time). Patching only
    ``app_config`` leaves the route's ``get_db()`` reading the default
    path.
    """
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    from app.database import db as app_db
    monkeypatch.setattr(app_db, "APP_DB_PATH", str(db_file))
    from app.database.db import init_db
    init_db(db_path=str(db_file))

    monkeypatch.setattr(predict_router, "_service", stub_service)
    monkeypatch.setattr(predict_router, "get_predict_service", lambda: stub_service)

    # Also patch the symbol imported by api.main so the lifespan no-ops.
    import api.main as api_main
    monkeypatch.setattr(api_main, "get_predict_service", lambda: stub_service)

    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------- happy paths


def test_predict_endpoint_returns_200_for_valid_payload(client, stub_service) -> None:
    resp = client.post("/predict", json=_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["predicted_price"] == 14_200_000.0
    assert body["model_version"] == "price_model_v2"
    assert len(stub_service.calls) == 1


def test_predict_endpoint_response_includes_model_version(client) -> None:
    resp = client.post("/predict", json=_PAYLOAD)
    assert resp.json()["model_version"].startswith("price_model_")


def test_predict_endpoint_response_includes_shap_contributions(client) -> None:
    resp = client.post("/predict", json=_PAYLOAD)
    contribs = resp.json()["shap_contributions"]
    assert isinstance(contribs, list)
    assert len(contribs) >= 1
    assert {"feature", "impact"} <= contribs[0].keys()


# -------------------------------------------------------------- 422 validation


def test_predict_endpoint_returns_422_on_missing_field(client) -> None:
    bad = dict(_PAYLOAD)
    del bad["built_up_area"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_endpoint_returns_422_on_extra_field(client) -> None:
    bad = dict(_PAYLOAD, **{"bedrooms": 3})  # typo, missing 'R'
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_endpoint_returns_422_on_bedroom_bathroom_violation(client) -> None:
    bad = dict(_PAYLOAD, bedRoom=5, bathroom=1)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_endpoint_returns_422_on_area_over_20000(client) -> None:
    bad = dict(_PAYLOAD, built_up_area=50_000.0)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------- 503 paths


def test_predict_endpoint_returns_503_when_model_artifact_missing(
    monkeypatch, tmp_path
) -> None:
    """Patch the service to raise FileNotFoundError → 503, not 500."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    from app.database import db as app_db
    monkeypatch.setattr(app_db, "APP_DB_PATH", str(db_file))
    from app.database.db import init_db
    init_db(db_path=str(db_file))

    stub = _StubService(raise_exc=FileNotFoundError("model not found"))
    monkeypatch.setattr(predict_router, "_service", stub)
    monkeypatch.setattr(predict_router, "get_predict_service", lambda: stub)
    import api.main as api_main
    monkeypatch.setattr(api_main, "get_predict_service", lambda: stub)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/predict", json=_PAYLOAD)
    assert resp.status_code == 503
    assert "model" in resp.json()["detail"].lower()


def test_predict_endpoint_returns_503_no_500_on_runtime_error(
    monkeypatch, tmp_path
) -> None:
    """RuntimeError from service → 503 (fastapi-serving skill rule)."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    from app.database import db as app_db
    monkeypatch.setattr(app_db, "APP_DB_PATH", str(db_file))
    from app.database.db import init_db
    init_db(db_path=str(db_file))

    stub = _StubService(raise_exc=RuntimeError("scaler misfit"))
    monkeypatch.setattr(predict_router, "_service", stub)
    monkeypatch.setattr(predict_router, "get_predict_service", lambda: stub)
    import api.main as api_main
    monkeypatch.setattr(api_main, "get_predict_service", lambda: stub)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/predict", json=_PAYLOAD)
    assert resp.status_code == 503


# ---------------------------------------------------------------- DB logging


_PII_REGEX = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)


def test_predict_endpoint_does_not_log_contact_fields(client, tmp_path) -> None:
    client.post("/predict", json=_PAYLOAD)
    # The row was inserted into the temp DB by ``client``'s fixture.
    db_path = tmp_path / "app.db"
    with get_db(db_path=str(db_path)) as conn:
        rows = conn.execute(
            "SELECT input_features_json FROM prediction_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert rows, "prediction_log should have one row after /predict"
    parsed = json.loads(rows[0]["input_features_json"])
    dumped = json.dumps(parsed)
    assert not _PII_REGEX.search(dumped), (
        f"PII regex matched in logged payload: {dumped[:200]}"
    )


def test_predict_endpoint_logs_one_row_per_request(client) -> None:
    import os
    client.post("/predict", json=_PAYLOAD)
    client.post("/predict", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM prediction_log").fetchall()
    assert rows[0]["n"] == 2


def test_predict_endpoint_logs_latency_ms(client) -> None:
    import os
    client.post("/predict", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT latency_ms FROM prediction_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert isinstance(rows[0]["latency_ms"], int)
    assert rows[0]["latency_ms"] >= 0


def test_predict_endpoint_happy_path_does_not_log_luxury_category(client) -> None:
    """Server-resolved luxury_category appears in response, never in the log row."""
    import os
    body = client.post("/predict", json=_PAYLOAD).json()
    assert body["luxury_category"] in {"Low", "Medium", "High"}
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT input_features_json FROM prediction_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    parsed = json.loads(rows[0]["input_features_json"])
    assert "luxury_category" not in parsed
