"""Tests for the FastAPI ``POST /classify`` route (TestClient).

Uses ``tmp_clean_db`` (from ``tests/conftest.py:14-26``) for the
classification-log assertion, monkeypatches ``get_classify_service``
per-test so the route's lifespan doesn't try to load real artifacts
from disk.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers import classify as classify_router
from api.schemas.classify_v1 import (
    ClassifyResponseV1,
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
    "bathroom": 2,
    "balcony": Balcony.TWO.value,
    "agePossession": AgePossession.RELATIVELY_NEW.value,
    "built_up_area": 1450.0,
    "servant_room": False,
    "store_room": False,
    "furnishing_type": FurnishingType.SEMIFURNISHED.value,
    "floor_category": FloorCategory.MID.value,
    "facing": FacingDirection.NORTH.value,
    "amenities": ["Clubhouse", "Swimming Pool"],
}


def _stub_response(**overrides: Any) -> ClassifyResponseV1:
    base = dict(
        good_deal_verdict=VerdictLabel.GOOD_DEAL,
        verdict_probabilities={
            VerdictLabel.GOOD_DEAL: 0.6,
            VerdictLabel.FAIR_DEAL: 0.3,
            VerdictLabel.OVERPRICED: 0.1,
        },
        price_tier=TierLabel.MID_RANGE,
        tier_probabilities={
            TierLabel.BUDGET: 0.1,
            TierLabel.MID_RANGE: 0.6,
            TierLabel.PREMIUM: 0.2,
            TierLabel.LUXURY: 0.1,
        },
        shap_contributions=[
            ShapContribution(feature="num__built_up_area", impact=0.18),
        ],
        model_version="good_deal_classifier_v1",
        luxury_category=LuxuryCategory.MEDIUM,
    )
    base.update(overrides)
    return ClassifyResponseV1(**base)


class _StubService:
    """Minimal ClassifyService-shaped stub for route tests.

    Avoids loading real artifacts; records calls so tests can assert
    that the route delegated correctly.
    """

    def __init__(
        self, response: ClassifyResponseV1 | None = None, raise_exc: BaseException | None = None
    ) -> None:
        self._response = response or _stub_response()
        self._raise = raise_exc
        self.calls: list[Any] = []
        self.warmed = False

    def warmup(self) -> None:
        self.warmed = True

    def classify(self, request):  # noqa: ANN001
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

    Skips lifespan by overriding ``classify_router.get_classify_service``
    so the test never triggers real artifact loading.

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

    monkeypatch.setattr(classify_router, "_service", stub_service)
    monkeypatch.setattr(classify_router, "get_classify_service", lambda: stub_service)

    # Also patch the symbol imported by api.main so the lifespan no-ops.
    import api.main as api_main

    monkeypatch.setattr(api_main, "get_classify_service", lambda: stub_service)

    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------- happy paths


def test_classify_endpoint_returns_200_for_valid_payload(client, stub_service) -> None:
    resp = client.post("/classify", json=_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["good_deal_verdict"] == "Good Deal"
    assert body["model_version"] == "good_deal_classifier_v1"
    assert len(stub_service.calls) == 1


def test_classify_endpoint_response_includes_model_version(client) -> None:
    resp = client.post("/classify", json=_PAYLOAD)
    assert resp.json()["model_version"].startswith("good_deal_classifier_")


def test_classify_endpoint_response_includes_shap_contributions(client) -> None:
    resp = client.post("/classify", json=_PAYLOAD)
    contribs = resp.json()["shap_contributions"]
    assert isinstance(contribs, list)
    assert len(contribs) >= 1
    assert {"feature", "impact"} <= contribs[0].keys()


def test_classify_endpoint_response_verdict_first_field_order(client) -> None:
    """Field order is significant: good_deal_verdict leads (Backend Schema §U-SCHEMA-10)."""
    resp = client.post("/classify", json=_PAYLOAD)
    body = resp.json()
    keys = list(body.keys())
    assert keys[0] == "good_deal_verdict", f"First field must be good_deal_verdict, got {keys[0]}"
    assert keys[1] == "verdict_probabilities", f"Second field must be verdict_probabilities, got {keys[1]}"


def test_classify_endpoint_response_includes_all_verdict_probabilities(client) -> None:
    resp = client.post("/classify", json=_PAYLOAD)
    probs = resp.json()["verdict_probabilities"]
    assert set(probs.keys()) == {"Good Deal", "Fair Deal", "Overpriced"}


def test_classify_endpoint_response_includes_all_tier_probabilities(client) -> None:
    resp = client.post("/classify", json=_PAYLOAD)
    probs = resp.json()["tier_probabilities"]
    assert set(probs.keys()) == {"Budget", "Mid-Range", "Premium", "Luxury"}


# -------------------------------------------------------------- 422 validation


def test_classify_endpoint_returns_422_on_missing_field(client) -> None:
    bad = dict(_PAYLOAD)
    del bad["built_up_area"]
    resp = client.post("/classify", json=bad)
    assert resp.status_code == 422


def test_classify_endpoint_returns_422_on_extra_field(client) -> None:
    bad = dict(_PAYLOAD, **{"bedrooms": 3})  # typo, missing 'R'
    resp = client.post("/classify", json=bad)
    assert resp.status_code == 422


def test_classify_endpoint_returns_422_on_bedroom_bathroom_violation(client) -> None:
    bad = dict(_PAYLOAD, bedRoom=5, bathroom=1)
    resp = client.post("/classify", json=bad)
    assert resp.status_code == 422


def test_classify_endpoint_returns_422_on_area_over_20000(client) -> None:
    bad = dict(_PAYLOAD, built_up_area=50_000.0)
    resp = client.post("/classify", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------- 503 paths


def test_classify_endpoint_returns_503_when_model_artifact_missing(
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
    monkeypatch.setattr(classify_router, "_service", stub)
    monkeypatch.setattr(classify_router, "get_classify_service", lambda: stub)
    import api.main as api_main

    monkeypatch.setattr(api_main, "get_classify_service", lambda: stub)

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/classify", json=_PAYLOAD)
    assert resp.status_code == 503
    assert "model" in resp.json()["detail"].lower()


def test_classify_endpoint_returns_503_no_500_on_runtime_error(
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
    monkeypatch.setattr(classify_router, "_service", stub)
    monkeypatch.setattr(classify_router, "get_classify_service", lambda: stub)
    import api.main as api_main

    monkeypatch.setattr(api_main, "get_classify_service", lambda: stub)

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/classify", json=_PAYLOAD)
    assert resp.status_code == 503


# ---------------------------------------------------------------- DB logging


_PII_REGEX = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)


def test_classify_endpoint_does_not_log_contact_fields(client, tmp_path) -> None:
    client.post("/classify", json=_PAYLOAD)
    # The row was inserted into the temp DB by ``client``'s fixture.
    db_path = tmp_path / "app.db"
    with get_db(db_path=str(db_path)) as conn:
        rows = conn.execute(
            "SELECT input_features_json FROM classification_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert rows, "classification_log should have one row after /classify"
    parsed = json.loads(rows[0]["input_features_json"])
    dumped = json.dumps(parsed)
    assert not _PII_REGEX.search(dumped), (
        f"PII regex matched in logged payload: {dumped[:200]}"
    )


def test_classify_endpoint_logs_one_row_per_request(client) -> None:
    import os

    client.post("/classify", json=_PAYLOAD)
    client.post("/classify", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM classification_log").fetchall()
    assert rows[0]["n"] == 2


def test_classify_endpoint_logs_latency_ms(client) -> None:
    import os

    client.post("/classify", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT latency_ms FROM classification_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert isinstance(rows[0]["latency_ms"], int)
    assert rows[0]["latency_ms"] >= 0


def test_classify_endpoint_happy_path_does_not_log_luxury_category(client) -> None:
    """Server-resolved luxury_category appears in response, never in the log row."""
    import os

    body = client.post("/classify", json=_PAYLOAD).json()
    assert body["luxury_category"] in {"Low", "Medium", "High"}
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT input_features_json FROM classification_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    parsed = json.loads(rows[0]["input_features_json"])
    assert "luxury_category" not in parsed


def test_classify_endpoint_logs_verdict_and_tier(client) -> None:
    import os

    client.post("/classify", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT predicted_verdict, predicted_tier FROM classification_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    assert rows[0]["predicted_verdict"] == "Good Deal"
    assert rows[0]["predicted_tier"] == "Mid-Range"


def test_classify_endpoint_logs_probabilities_json(client) -> None:
    import os

    client.post("/classify", json=_PAYLOAD)
    db_path = os.environ["APP_DB_PATH"]
    with get_db(db_path=db_path) as conn:
        rows = conn.execute(
            "SELECT verdict_probabilities_json, tier_probabilities_json FROM classification_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
    verdict_probs = json.loads(rows[0]["verdict_probabilities_json"])
    tier_probs = json.loads(rows[0]["tier_probabilities_json"])
    assert abs(sum(verdict_probs.values()) - 1.0) < 1e-9
    assert abs(sum(tier_probs.values()) - 1.0) < 1e-9