"""Classification input/output schema v1.

Authority:
    - docs/10-FINALIZED-INPUT-SCHEMA.md (16-field contract, frozen)
    - docs/05-BACKEND-SCHEMA.md §U-SCHEMA-10 (POST /classify response contract)
    - docs/02-TRD.md §U-TRD-8 (classification targets)

The request reuses ``PredictRequestV3`` from ``predict_v3.py`` — same 16-field
contract as price prediction. The response leads with ``good_deal_verdict``
per the good-deal-first reframing (Backend Schema §U-SCHEMA-10).

This module defines the API contract only. The actual model pipeline wiring
into ``api/routers/classify.py`` is implemented in Spec 25.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.predict_v3 import (
    LuxuryCategory,
    PredictRequestV3,
    ShapContribution,
)

# --- request ------------------------------------------------------------------

# Same 16-field contract as price prediction — no separate request model needed.
ClassifyRequestV1 = PredictRequestV3

# --- response enums (pinned from Spec 21) -------------------------------------


class VerdictLabel(str, Enum):
    """3-class good_deal_verdict labels (Spec 21).

    Order matches the integer encoding used in training:
    0 = Good Deal, 1 = Fair Deal, 2 = Overpriced.
    """

    GOOD_DEAL = "Good Deal"
    FAIR_DEAL = "Fair Deal"
    OVERPRICED = "Overpriced"


class TierLabel(str, Enum):
    """4-class price_tier labels (Spec 21).

    Order matches the integer encoding used in training:
    0 = Budget, 1 = Mid-Range, 2 = Premium, 3 = Luxury.
    """

    BUDGET = "Budget"
    MID_RANGE = "Mid-Range"
    PREMIUM = "Premium"
    LUXURY = "Luxury"


# --- response -----------------------------------------------------------------


class ClassifyResponseV1(BaseModel):
    """Response shape for POST /classify.

    Field order is significant: ``good_deal_verdict`` leads per
    Backend Schema §U-SCHEMA-10 (good-deal-first reframing).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Primary output: good-deal verdict
    good_deal_verdict: VerdictLabel
    verdict_probabilities: dict[VerdictLabel, float] = Field(
        description="Probabilities for each verdict class; sums to 1.0"
    )

    # Secondary output: affordability tier
    price_tier: TierLabel
    tier_probabilities: dict[TierLabel, float] = Field(
        description="Probabilities for each tier class; sums to 1.0"
    )

    # Explainability (reuses ShapContribution from predict_v3)
    shap_contributions: list[ShapContribution] = Field(default_factory=list)

    # Metadata
    model_version: str
    luxury_category: LuxuryCategory  # server-derived, echoed back


# --- constants ----------------------------------------------------------------

SHAP_TOP_N: Final[int] = 7
"""Top-N SHAP contributions to return (mirrors price model)."""

PRIMARY_TARGET: Final[str] = "good_deal_verdict"
SECONDARY_TARGET: Final[str] = "price_tier"

__all__ = [
    "ClassifyRequestV1",
    "VerdictLabel",
    "TierLabel",
    "ClassifyResponseV1",
    "SHAP_TOP_N",
    "PRIMARY_TARGET",
    "SECONDARY_TARGET",
]