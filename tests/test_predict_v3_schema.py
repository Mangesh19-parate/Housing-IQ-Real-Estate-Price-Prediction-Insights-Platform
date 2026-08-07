"""Tests for api/schemas/predict_v3.py — Step 11, Price Prediction Input Schema v3.

Pure in-memory Pydantic validation. No DB, no HTTP, no filesystem writes.
The single filesystem read is `docs/10-FINALIZED-INPUT-SCHEMA.md` (a markdown
file in the repo root, NOT under data/raw/), so none of these tests are
marked @pytest.mark.realdata.

Mirrors the function-based, `# A. Constants`-style layout used in
tests/test_canonical_mapping.py. The ast.walk import-scan pattern follows
the same file's `test_map_does_not_import_app_or_api` (line 463).

Note: `FacingDirection` has 8 members per
`data/raw/facets/FACING_DIRECTION.csv`, not 9 as the spec prose states.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pydantic
import pytest

from api.schemas import (
    INPUT_FIELD_TYPES_V3,
    INPUT_FIELDS_V3,
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DOC = REPO_ROOT / "docs" / "10-FINALIZED-INPUT-SCHEMA.md"
SCHEMA_MODULE_PATH = REPO_ROOT / "api" / "schemas" / "predict_v3.py"

EXPECTED_16_TUPLE = (
    "property_type",
    "sector",
    "city",
    "transact_type",
    "bedRoom",
    "bathroom",
    "balcony",
    "agePossession",
    "built_up_area",
    "servant_room",
    "store_room",
    "furnishing_type",
    "luxury_category",
    "floor_category",
    "facing",
    "amenities",
)


def _valid_payload(**overrides: object) -> dict[str, object]:
    """Return a dict with all 15 visible (non-excluded) fields valid."""
    base: dict[str, object] = {
        "city": "Gurgaon",
        "sector": "Sector 84",
        "property_type": "flat",
        "transact_type": "Sale",
        "bedRoom": 3,
        "bathroom": 3,
        "balcony": "2",
        "agePossession": "Relatively New",
        "built_up_area": 1450.0,
        "servant_room": True,
        "store_room": False,
        "furnishing_type": "Semifurnished",
        "floor_category": "Mid Floor",
        "facing": "North",
        "amenities": ["Clubhouse", "Swimming Pool"],
    }
    base.update(overrides)
    return base


# A. Constants ---------------------------------------------------------------


def test_input_fields_v3_has_exactly_sixteen_entries() -> None:
    assert len(INPUT_FIELDS_V3) == 16


def test_input_fields_v3_is_tuple() -> None:
    assert isinstance(INPUT_FIELDS_V3, tuple)


def test_input_fields_v3_order_matches_reference_project() -> None:
    assert INPUT_FIELDS_V3 == EXPECTED_16_TUPLE


# The schema doc uses the reference project's spelling for some fields
# (`servant room` / `store room` with a space); the API uses snake_case.
# This map normalizes for the doc-lookup assertion only.
_DOC_NAME_ALIASES = {
    "servant_room": "servant room",
    "store_room": "store room",
}


def test_input_fields_v3_names_match_input_schema_doc() -> None:
    doc_text = SCHEMA_DOC.read_text(encoding="utf-8")
    for name in INPUT_FIELDS_V3:
        needle = _DOC_NAME_ALIASES.get(name, name)
        assert needle in doc_text, f"{name!r} (or alias {needle!r}) not found in {SCHEMA_DOC}"


def test_input_field_types_v3_covers_all_input_fields() -> None:
    assert set(INPUT_FIELD_TYPES_V3.keys()) == set(INPUT_FIELDS_V3)


def test_enums_have_expected_string_values() -> None:
    assert [m.value for m in TransactType] == ["Sale", "Rent"]
    assert [m.value for m in PropertyType] == ["flat", "house"]
    assert [m.value for m in Balcony] == ["0", "1", "2", "3", "3+"]
    assert [m.value for m in AgePossession] == [
        "New Property",
        "Relatively New",
        "Moderately Old",
        "Old Property",
        "Under Construction",
    ]
    assert [m.value for m in FurnishingType] == [
        "Unfurnished",
        "Semifurnished",
        "Furnished",
    ]
    assert [m.value for m in LuxuryCategory] == ["Low", "Medium", "High"]
    assert [m.value for m in FloorCategory] == [
        "Low Floor",
        "Mid Floor",
        "High Floor",
    ]
    # 8 values per data/raw/facets/FACING_DIRECTION.csv (not 9 as spec prose).
    assert [m.value for m in FacingDirection] == [
        "North",
        "South",
        "East",
        "West",
        "North-East",
        "North-West",
        "South-East",
        "South-West",
    ]


# B. Request validation ------------------------------------------------------


def test_predict_request_v3_minimal_valid_payload() -> None:
    req = PredictRequestV3(**_valid_payload())
    assert req.city == "Gurgaon"
    assert req.bedRoom == 3
    assert req.amenities == ["Clubhouse", "Swimming Pool"]


def test_predict_request_v3_rejects_missing_required_field() -> None:
    payload = _valid_payload()
    del payload["built_up_area"]
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**payload)


def test_predict_request_v3_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(unknown_field="x"))


def test_predict_request_v3_rejects_bedroom_zero() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(bedRoom=0))


def test_predict_request_v3_rejects_bedroom_over_15() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(bedRoom=16))


def test_predict_request_v3_rejects_negative_area() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(built_up_area=-100.0))


def test_predict_request_v3_rejects_area_over_20000() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(built_up_area=50000.0))


def test_predict_request_v3_bedroom_bathroom_sanity_check() -> None:
    with pytest.raises(pydantic.ValidationError) as exc_info:
        PredictRequestV3(**_valid_payload(bedRoom=5, bathroom=1))
    assert "bathroom" in str(exc_info.value)


def test_predict_request_v3_balcony_accepts_three_plus() -> None:
    req = PredictRequestV3(**_valid_payload(balcony="3+"))
    assert req.balcony == Balcony.THREE_PLUS


def test_predict_request_v3_transact_type_enum_values() -> None:
    PredictRequestV3(**_valid_payload(transact_type="Sale"))
    PredictRequestV3(**_valid_payload(transact_type="Rent"))
    with pytest.raises(pydantic.ValidationError):
        PredictRequestV3(**_valid_payload(transact_type="sale"))


def test_predict_request_v3_strips_string_whitespace() -> None:
    req = PredictRequestV3(**_valid_payload(city=" Gurgaon "))
    assert req.city == "Gurgaon"


def test_predict_request_v3_amenities_defaults_to_empty_list() -> None:
    payload = _valid_payload()
    del payload["amenities"]
    req = PredictRequestV3(**payload)
    assert req.amenities == []


def test_predict_request_v3_amenities_accepts_list_of_strings() -> None:
    req = PredictRequestV3(
        **_valid_payload(amenities=["Clubhouse", "Swimming Pool"])
    )
    assert req.amenities == ["Clubhouse", "Swimming Pool"]
    assert isinstance(req.amenities, list)


def test_predict_request_v3_luxury_category_excluded_from_request() -> None:
    # Use model_validate (JSON-parse path) so exclude=True drops the field on
    # parse — bypassing via **kwargs would set it directly on the instance.
    req = PredictRequestV3.model_validate(_valid_payload(luxury_category="High"))
    assert req.luxury_category is None


def test_predict_request_v3_dump_excludes_luxury_category() -> None:
    req = PredictRequestV3.model_validate(_valid_payload())
    dumped = req.model_dump()
    assert "luxury_category" not in dumped


# C. Response validation -----------------------------------------------------


def _valid_response(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "predicted_price": 14_200_000.0,
        "range_low": 12_800_000.0,
        "range_high": 15_600_000.0,
        "shap_contributions": [{"feature": "built_up_area", "impact": 0.18}],
        "is_outlier_input": False,
        "model_version": "price_model_v1",
        "luxury_category": "High",
    }
    base.update(overrides)
    return base


def test_predict_response_v3_minimal_valid_payload() -> None:
    resp = PredictResponseV3(**_valid_response())
    assert resp.predicted_price == 14_200_000.0
    assert resp.luxury_category == LuxuryCategory.HIGH


def test_predict_response_v3_rejects_negative_price() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictResponseV3(**_valid_response(predicted_price=-1.0))


def test_predict_response_v3_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        PredictResponseV3(**_valid_response(unknown_field="x"))


def test_shap_contribution_accepts_float_impact() -> None:
    ShapContribution(feature="area_sqft", impact=0.18)
    with pytest.raises(pydantic.ValidationError):
        ShapContribution(feature="area_sqft", impact="high")


# D. Boundary rules ----------------------------------------------------------


def test_no_pii_or_contact_fields() -> None:
    pattern = re.compile(r"(contact|dealer|phone|email|photo|url|spid)", re.IGNORECASE)
    for model in (PredictRequestV3, PredictResponseV3):
        for name in model.model_fields.keys():
            assert not pattern.search(name), f"{name!r} contains a banned token"


def test_predict_v3_does_not_import_app_ml_or_models() -> None:
    """Mirror the ast.walk pattern at tests/test_canonical_mapping.py line 463."""
    source = SCHEMA_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                assert top not in {"app", "ml", "models"}, (
                    f"forbidden top-level import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            assert top not in {"app", "ml", "models"}, (
                f"forbidden from-import: {node.module}"
            )


def test_schemas_init_reexports_public_api() -> None:
    from api.schemas import (
        INPUT_FIELD_TYPES_V3 as _TYPES,
    )
    from api.schemas import (  # noqa: PLC0415 — single-statement import per spec
        INPUT_FIELDS_V3 as _FIELDS,
    )
    from api.schemas import (
        AgePossession as _AP,
    )
    from api.schemas import (
        Balcony as _BA,
    )
    from api.schemas import (
        FacingDirection as _FD,
    )
    from api.schemas import (
        FloorCategory as _FC,
    )
    from api.schemas import (
        FurnishingType as _FT,
    )
    from api.schemas import (
        LuxuryCategory as _LC,
    )
    from api.schemas import (
        PredictRequestV3 as _REQ,
    )
    from api.schemas import (
        PredictResponseV3 as _RES,
    )
    from api.schemas import (
        PropertyType as _PT,
    )
    from api.schemas import (
        ShapContribution as _SHAP,
    )
    from api.schemas import (
        TransactType as _TT,
    )
    # Touch each so static analyzers don't complain about unused imports.
    assert _FIELDS and _TYPES and _AP and _BA and _FD and _FC
    assert _FT and _LC and _REQ and _RES and _PT and _SHAP and _TT
