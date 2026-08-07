"""Price Prediction input/output schema v3.

Authority:
        - docs/10-FINALIZED-INPUT-SCHEMA.md (16-field contract, frozen)
        - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-5 (canonical field table) +
          §U-SCHEMA-6 (transact_type routing rule)

Field names are locked to the reference project's contract:
        - ``bedRoom`` stays camelCase (matches
          ``gurgaon_properties_post_feature_selection_v2.csv``).
        - ``servant_room`` / ``store_room`` use snake_case at the API boundary
          (a deliberate one-time exception; the cleaning layer emits these in
          snake_case form too — mapping back to the reference project's
          space-separated form is a Step-08+ training-ingestion concern).

``transact_type`` is a routing key, not a plain model feature (TRD §U-TRD-4):
the FastAPI ``/predict`` handler dispatches to one of two trained pipelines
based on its value before any preprocessing happens.

``luxury_category`` is server-derived from the amenity checklist, never
client-supplied (Rules §10.2). The request model uses ``exclude=True`` so a
client that sends ``{"luxury_category": "High"}`` is silently dropped on
parse — the response echoes back the resolved value.

This module defines the API contract only. The actual model pipeline wiring
into ``api/routers/predict.py`` is a later spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- enums ------------------------------------------------------------------


class TransactType(str, Enum):
    """Sale vs Rent. Strings match the cleaned-dataset raw column values verbatim."""

    SALE = "Sale"
    RENT = "Rent"


class PropertyType(str, Enum):
    """flat / house per docs/10-FINALIZED-INPUT-SCHEMA.md §1 row 1.

    Note: these are API-level contract values (lowercase), distinct from the
    raw facet labels emitted by the cleaning layer (e.g. "Residential
    Apartment"). Value-mapping happens at training ingestion time, not here.
    """

    FLAT = "flat"
    HOUSE = "house"


class Balcony(str, Enum):
    """0 / 1 / 2 / 3 / 3+ — stored as strings to match the reference project's
    categorical/ordinal type (docs/10-FINALIZED-INPUT-SCHEMA.md §1 row 5)."""

    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    THREE_PLUS = "3+"


class AgePossession(str, Enum):
    """Title-Case property-age labels per docs/10-FINALIZED-INPUT-SCHEMA.md §1
    row 6. Note: the raw facet file uses different strings (e.g. "1-5 Year
    Old Property"); the finalized-schema labels are the contract enforced at
    the API boundary."""

    NEW = "New Property"
    RELATIVELY_NEW = "Relatively New"
    MODERATELY_OLD = "Moderately Old"
    OLD = "Old Property"
    UNDER_CONSTRUCTION = "Under Construction"


class FurnishingType(str, Enum):
    """Unfurnished / Semifurnished / Furnished — no hyphen in 'Semifurnished'
    (docs/10-FINALIZED-INPUT-SCHEMA.md §1 row 10). The 0/1/2 ordinal encoding
    is the ColumnTransformer's job at training time (TRD §U-TRD-3), not the
    API's."""

    UNFURNISHED = "Unfurnished"
    SEMIFURNISHED = "Semifurnished"
    FURNISHED = "Furnished"


class LuxuryCategory(str, Enum):
    """Low / Medium / High — server-derived from the amenity checklist, not
    self-reported (Rules §10.2). Appears in ``PredictResponseV3`` as the
    resolved value; excluded from request bodies."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FloorCategory(str, Enum):
    """Low Floor / Mid Floor / High Floor — note trailing ' Floor'."""

    LOW = "Low Floor"
    MID = "Mid Floor"
    HIGH = "High Floor"


class FacingDirection(str, Enum):
    """8 compass values per data/raw/facets/FACING_DIRECTION.csv.

    The Step 11 spec text mentions "9 standard compass values" with an
    UNFURNISHED placeholder, but the facet file has 8 rows and no such
    placeholder. Implement against the 8 facet values.
    """

    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    NORTH_EAST = "North-East"
    NORTH_WEST = "North-West"
    SOUTH_EAST = "South-East"
    SOUTH_WEST = "South-West"


# --- request ----------------------------------------------------------------


class PredictRequestV3(BaseModel):
    """The 16-field input contract for POST /predict (Step 11, frozen).

    Field order in INPUT_FIELDS_V3 below mirrors the spec's pinned order.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    city: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    property_type: PropertyType
    transact_type: TransactType
    bedRoom: int = Field(ge=1, le=15)
    bathroom: int = Field(ge=1, le=15)
    balcony: Balcony
    agePossession: AgePossession
    built_up_area: float = Field(gt=0, le=20000)
    servant_room: bool
    store_room: bool
    furnishing_type: FurnishingType
    floor_category: FloorCategory
    facing: FacingDirection
    amenities: list[str] = Field(default_factory=list)
    # Server-derived, never client-supplied. The pre-validator strips any
    # client-supplied value from the input dict (Rules §10.2 — a client
    # sending {"luxury_category": "High"} is silently ignored, not 422'd).
    luxury_category: LuxuryCategory | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _strip_client_luxury_category(cls, data: object) -> object:
        """Drop a client-supplied ``luxury_category`` before validation.

        ``Field(exclude=True)`` suppresses serialization and JSON schema
        emission, but does not drop the field during ``model_validate``.
        This pre-validator ensures the field is unreachable from the wire.
        """
        if isinstance(data, dict) and "luxury_category" in data:
            data = {k: v for k, v in data.items() if k != "luxury_category"}
        return data

    @model_validator(mode="after")
    def _check_bedroom_bathroom(self) -> PredictRequestV3:
        if self.bedRoom > self.bathroom + 3:
            raise ValueError("bathroom too low for bedroom count")
        return self


# --- response ---------------------------------------------------------------


class ShapContribution(BaseModel):
    """Single SHAP feature contribution. Mirrors docs/05-BACKEND-SCHEMA.md §7."""

    feature: str
    impact: float


class PredictResponseV3(BaseModel):
    """Response shape for POST /predict."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predicted_price: float = Field(ge=0)
    range_low: float = Field(ge=0)
    range_high: float = Field(ge=0)
    shap_contributions: list[ShapContribution] = Field(default_factory=list)
    is_outlier_input: bool
    model_version: str
    luxury_category: LuxuryCategory


# --- constants ---------------------------------------------------------------


INPUT_FIELDS_V3: Final[tuple[str, ...]] = (
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

INPUT_FIELD_TYPES_V3: Final[dict[str, type]] = {
    "property_type": PropertyType,
    "sector": str,
    "city": str,
    "transact_type": TransactType,
    "bedRoom": int,
    "bathroom": int,
    "balcony": Balcony,
    "agePossession": AgePossession,
    "built_up_area": float,
    "servant_room": bool,
    "store_room": bool,
    "furnishing_type": FurnishingType,
    "luxury_category": LuxuryCategory,
    "floor_category": FloorCategory,
    "facing": FacingDirection,
    "amenities": list,
}
