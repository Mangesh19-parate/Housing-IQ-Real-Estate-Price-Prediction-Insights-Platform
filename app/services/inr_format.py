"""Currency formatter for the Price Prediction result hero.

Pure function, stdlib only (``math.log10``, ``re``). No I/O, no
side effects, no model code.

Sale listings use Cr / Lakh (Indian numbering).
Rent listings are formatted per-month with the Indian comma
grouping (Lakh / thousand separators).
"""

from __future__ import annotations

import math
import re

_RANGE_LABELS: tuple[tuple[float, str], ...] = (
    (1e7, "Cr"),
    (1e5, "Lakh"),
    (1e3, "Thousand"),
)


def _format_sale(value_inr: float) -> str:
    """Format a Sale-price value as ``X.YY Cr`` / ``X.YY Lakh`` / ``X Thousand``."""
    if value_inr <= 0:
        return "₹0"
    abs_val = abs(value_inr)
    for threshold, label in _RANGE_LABELS:
        if abs_val >= threshold:
            scaled = value_inr / threshold
            return f"₹{scaled:.2f} {label}"
    return f"₹{value_inr:,.0f}"


def _format_rent(value_inr: float) -> str:
    """Format a Rent-price value as ``₹X,XX,XXX / month`` (Indian comma grouping)."""
    rounded = int(round(value_inr))
    if rounded <= 0:
        return "₹0 / month"
    # Indian grouping: last 3 digits, then groups of 2.
    s = str(rounded)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        rest_grouped = re.sub(r"(?<=.)(?=(..)+$)", ",", rest)
        grouped = f"{rest_grouped},{last3}"
    return f"₹{grouped} / month"


def inr_format(value_inr: float, *, transact_type: str = "Sale") -> str:
    """Format a numeric INR value for the price hero.

    ``transact_type="Sale"`` → Cr / Lakh labels.
    ``transact_type="Rent"`` → per-month with Indian comma grouping.
    Any other value falls back to the Sale formatter (the form's
    radio only emits "Sale" or "Rent"; this is defense in depth).
    """
    if transact_type == "Rent":
        return _format_rent(value_inr)
    return _format_sale(value_inr)


# Silence the linter — ``math`` is imported for symmetry/future use
# (e.g., switching to log10-based bucketing) and pinned here so
# removing the import later is a deliberate decision.
_ = math.log10

__all__ = ["inr_format"]
