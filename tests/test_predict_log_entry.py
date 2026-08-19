"""Tests for ``api.schemas.predict_log_entry.to_prediction_log_row``.

Pure serialiser — no I/O, no DB. Verifies the contract that the route
layer relies on (column-name mapping + JSON shape).
"""

from __future__ import annotations

import json

from api.schemas.predict_log_entry import to_prediction_log_row
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
    ShapContribution,
    TransactType,
)

_PREDICTION_LOG_COLUMNS = frozenset(
    {
        "id",  # autoincrement — filled by SQLite, omitted from dict
        "timestamp",
        "city",
        "locality",
        "input_features_json",
        "predicted_price",
        "predicted_range_low",
        "predicted_range_high",
        "model_version",
        "is_outlier_input",
        "latency_ms",
    }
)


def _minimal_request(**overrides) -> PredictRequestV3:
    base = dict(
        city="Gurgaon",
        sector="sector 84",
        property_type=PropertyType.FLAT,
        transact_type=TransactType.SALE,
        bedRoom=3,
        bathroom=3,
        balcony=Balcony.TWO,
        agePossession=AgePossession.RELATIVELY_NEW,
        built_up_area=1450.0,
        servant_room=False,
        store_room=False,
        furnishing_type=FurnishingType.SEMIFURNISHED,
        floor_category=FloorCategory.MID,
        facing=FacingDirection.NORTH,
        amenities=["Clubhouse"],
    )
    base.update(overrides)
    return PredictRequestV3(**base)


def _minimal_response(**overrides) -> PredictResponseV3:
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


def test_to_prediction_log_row_keys_match_db_columns() -> None:
    """Returned dict's keys are exactly the prediction_log columns (minus id)."""
    row = to_prediction_log_row(_minimal_request(), _minimal_response(), latency_ms=42)
    assert set(row.keys()) == _PREDICTION_LOG_COLUMNS - {"id"}


def test_to_prediction_log_row_serialises_features_as_json() -> None:
    """``input_features_json`` is a JSON string parsable by ``json.loads``."""
    row = to_prediction_log_row(_minimal_request(), _minimal_response(), latency_ms=1)
    assert isinstance(row["input_features_json"], str)
    parsed = json.loads(row["input_features_json"])
    assert parsed["city"] == "Gurgaon"
    assert parsed["sector"] == "sector 84"
    assert parsed["amenities"] == ["Clubhouse"]


def test_to_prediction_log_row_excludes_luxury_category_from_input() -> None:
    """The dumped input JSON does **not** contain ``luxury_category``.

    Confirms Spec 11's ``Field(exclude=True)`` + pre-validator carry
    through the serialiser — client-supplied luxury_category would never
    be logged even if a client tried.
    """
    row = to_prediction_log_row(
        _minimal_request(luxury_category=LuxuryCategory.HIGH),
        _minimal_response(),
        latency_ms=1,
    )
    parsed = json.loads(row["input_features_json"])
    assert "luxury_category" not in parsed


def test_to_prediction_log_row_latency_ms_is_int() -> None:
    """``latency_ms`` field is a Python ``int`` (matches ``INTEGER`` column)."""
    row = to_prediction_log_row(_minimal_request(), _minimal_response(), latency_ms=7)
    assert isinstance(row["latency_ms"], int)
    assert row["latency_ms"] == 7


def test_to_prediction_log_row_maps_sector_to_locality() -> None:
    """``request.sector`` is written under the ``locality`` key.

    The DB column is singular ``locality`` (app/database/db.py:48) —
    this serialiser is the only place that mapping lives.
    """
    row = to_prediction_log_row(_minimal_request(), _minimal_response(), latency_ms=0)
    assert row["locality"] == "sector 84"
    assert "sector" not in row


def test_to_prediction_log_row_is_outlier_serialised_as_int() -> None:
    """``is_outlier_input`` is stored as 0/1 int (SQLite has no bool)."""
    row_true = to_prediction_log_row(
        _minimal_request(), _minimal_response(is_outlier_input=True), latency_ms=0
    )
    row_false = to_prediction_log_row(
        _minimal_request(), _minimal_response(is_outlier_input=False), latency_ms=0
    )
    assert row_true["is_outlier_input"] == 1
    assert row_false["is_outlier_input"] == 0


def test_to_prediction_log_row_price_range_passthrough() -> None:
    """predicted_price / range_low / range_high pass through verbatim."""
    response = _minimal_response(
        predicted_price=1.0,
        range_low=0.5,
        range_high=1.5,
    )
    row = to_prediction_log_row(_minimal_request(), response, latency_ms=0)
    assert row["predicted_price"] == 1.0
    assert row["predicted_range_low"] == 0.5
    assert row["predicted_range_high"] == 1.5
