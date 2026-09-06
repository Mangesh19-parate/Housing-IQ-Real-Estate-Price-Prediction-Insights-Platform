"""Tests for ``api.schemas.classify_log_entry.to_classification_log_row``.

Pure-function serializer — no DB access, no side effects. Verifies the
column mapping matches ``classification_log`` (docs/05-BACKEND-SCHEMA.md
§U-SCHEMA-11) and that the JSON payloads are valid + PII-free.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from api.schemas.classify_log_entry import to_classification_log_row
from api.schemas.classify_v1 import ClassifyResponseV1, TierLabel, VerdictLabel
from api.schemas.predict_v3 import (
    AgePossession,
    Balcony,
    FacingDirection,
    FloorCategory,
    FurnishingType,
    LuxuryCategory,
    PredictRequestV3,
    PropertyType,
    TransactType,
)

_PII_REGEX = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)


def _minimal_request(**overrides) -> PredictRequestV3:
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
    return PredictRequestV3(**base)


def _minimal_response(**overrides) -> ClassifyResponseV1:
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
        shap_contributions=[],
        model_version="good_deal_classifier_v1",
        luxury_category=LuxuryCategory.MEDIUM,
    )
    base.update(overrides)
    return ClassifyResponseV1(**base)


def test_to_classification_log_row_returns_all_columns() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, latency_ms=123)

    expected_cols = {
        "timestamp",
        "city",
        "input_features_json",
        "predicted_verdict",
        "predicted_tier",
        "verdict_probabilities_json",
        "tier_probabilities_json",
        "model_version",
    }
    assert set(row.keys()) == expected_cols


def test_to_classification_log_row_city_matches_request() -> None:
    req = _minimal_request(city="Hyderabad")
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)
    assert row["city"] == "Hyderabad"


def test_to_classification_log_row_verdict_matches_response() -> None:
    req = _minimal_request()
    resp = _minimal_response(good_deal_verdict=VerdictLabel.OVERPRICED)
    row = to_classification_log_row(req, resp, 0)
    assert row["predicted_verdict"] == "Overpriced"


def test_to_classification_log_row_tier_matches_response() -> None:
    req = _minimal_request()
    resp = _minimal_response(price_tier=TierLabel.LUXURY)
    row = to_classification_log_row(req, resp, 0)
    assert row["predicted_tier"] == "Luxury"


def test_to_classification_log_row_verdict_probabilities_json_is_valid() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    parsed = json.loads(row["verdict_probabilities_json"])
    assert set(parsed.keys()) == {"Good Deal", "Fair Deal", "Overpriced"}
    assert abs(sum(parsed.values()) - 1.0) < 1e-9
    assert all(isinstance(v, float) for v in parsed.values())


def test_to_classification_log_row_tier_probabilities_json_is_valid() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    parsed = json.loads(row["tier_probabilities_json"])
    assert set(parsed.keys()) == {"Budget", "Mid-Range", "Premium", "Luxury"}
    assert abs(sum(parsed.values()) - 1.0) < 1e-9
    assert all(isinstance(v, float) for v in parsed.values())


def test_to_classification_log_row_input_features_json_excludes_luxury_category() -> None:
    """Server-derived luxury_category is echoed in response, never logged."""
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    parsed = json.loads(row["input_features_json"])
    assert "luxury_category" not in parsed


def test_to_classification_log_row_input_features_json_is_valid_json() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    # Should not raise
    json.loads(row["input_features_json"])


def test_to_classification_log_row_input_features_json_no_pii() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    dumped = json.dumps(json.loads(row["input_features_json"]))
    assert not _PII_REGEX.search(dumped)


def test_to_classification_log_row_timestamp_is_iso8601_utc() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, 0)

    # Parse without raising
    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
    assert ts.tzinfo == timezone.utc


def test_to_classification_log_row_latency_ms_recorded() -> None:
    req = _minimal_request()
    resp = _minimal_response()
    row = to_classification_log_row(req, resp, latency_ms=456)
    # latency_ms is not a column in classification_log — it's only in prediction_log.
    # This test documents that the function accepts the parameter but doesn't include it.
    assert "latency_ms" not in row


def test_to_classification_log_row_model_version_from_response() -> None:
    req = _minimal_request()
    resp = _minimal_response(model_version="good_deal_classifier_v2")
    row = to_classification_log_row(req, resp, 0)
    assert row["model_version"] == "good_deal_classifier_v2"