"""Field-level string parsers for raw city listing CSVs.

Public API:
    parse_price(value, *, fallback_min=None, fallback_max=None) -> float | None
    parse_area(value, *, unit_hint: str = "auto") -> float | None

Both functions are pure (no I/O, no module-level state, no randomness) and
deterministic — calling them twice on the same input always yields the same
output. They never raise on bad input; the failure signal is ``None``.

Intended source columns (per the Step 02 schema inventory):
    PRICE, MIN_PRICE, MAX_PRICE, PRICE_SQFT,
    AREA, MIN_AREA_SQFT, MAX_AREA_SQFT,
    SUPER_SQFT, BUILTUP_SQFT, CARPET_SQFT.

Step 04's cleaning pipeline is the integration point that loads raw CSVs,
iterates rows, applies these parsers, and writes ``clean_listings.parquet``.

Area range resolution rule: strings like ``"1200-1400 sq.ft."`` resolve to the
midpoint of the two bounds (e.g. ``1300.0``). This is a documented choice, not
a per-row guess.

Unit-suffix casing: Indian real-estate listings inconsistently write
``'L'`` / ``'l'`` / ``'Lac'`` / ``'lac'`` / ``'Lakh'`` / ``'lakh'``. The unit
maps below have a single canonical lowercase key per unit; the parser
lowercases the suffix at the regex layer before lookup, so all casings map
consistently.
"""

from __future__ import annotations

import logging
import re
from typing import Final

_log: Final = logging.getLogger("ml.cleaning.parsing")

# 1 square metre = 10.7639 square feet. Exported so tests and callers can
# reference the same constant the parser uses internally.
SQFT_PER_SQM: Final[float] = 10.7639

# Canonical-lowercase multiplier tables. Adding a new unit = one entry here
# (and a test). Do NOT add uppercase variants — the parser lowercases the
# suffix before lookup.
_PRICE_UNIT_MAP: Final[dict[str, float]] = {
    "cr": 1e7,        # Crore
    "l": 1e5,         # Lakh (short)
    "lac": 1e5,       # Lakh (lacaar variant)
    "lakh": 1e5,      # Lakh (full)
}
_AREA_UNIT_MAP: Final[dict[str, float]] = {
    "sqft": 1.0,
    "sqm": SQFT_PER_SQM,
}

# Suffix tokens the regex groups as the "unit word". The unit token can
# contain letters and dots (e.g. ``sq.ft.``, ``sq.m.``). The captured token
# is normalized via ``_normalize_unit`` before being looked up in the
# unit map so casing and trailing dots don't force duplicate entries.
_UNIT_CHARS: Final[str] = r"[A-Za-z.]+"

# Price: optional whitespace, numeric (int or decimal), optional whitespace,
# optional unit word.
_RE_PRICE: Final[re.Pattern[str]] = re.compile(
    rf"^\s*(\d+(?:\.\d+)?)\s*({_UNIT_CHARS})?\s*$"
)

# Area: optional whitespace, numeric (range's low bound), optional
#   whitespace + hyphen + numeric (range's high bound), optional whitespace,
#   optional unit word.
_RE_AREA: Final[re.Pattern[str]] = re.compile(
    rf"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*({_UNIT_CHARS})?\s*$"
)


def _normalize_unit(raw_unit: str | None) -> str | None:
    """Normalize a captured unit token to its canonical lowercase form.

    Strips trailing dots and whitespace, removes all internal dots and
    spaces, lowercases. ``"SQ.FT."`` → ``"sqft"``; ``"sq.m."`` → ``"sqm"``;
    ``None`` or ``""`` → ``None``. The unit map keys are these canonical
    forms (``"sqft"``, ``"sqm"``, ``"cr"``, ``"l"``, ``"lac"``, ``"lakh"``),
    never the dotted variants — the normalizer absorbs the case/dot
    mismatch rather than forcing duplicate dict entries.
    """
    if raw_unit is None:
        return None
    cleaned = raw_unit.strip().lower().replace(".", "").replace(" ", "").rstrip(".")
    return cleaned or None

# Source columns documented for callers (informational — not enforced).
INTENDED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "PRICE",
    "MIN_PRICE",
    "MAX_PRICE",
    "PRICE_SQFT",
    "AREA",
    "MIN_AREA_SQFT",
    "MAX_AREA_SQFT",
    "SUPER_SQFT",
    "BUILTUP_SQFT",
    "CARPET_SQFT",
)


def _log_unparseable(field: str, value: object, city: str | None = None) -> None:
    """Emit a structured WARNING log for every unparseable row.

    The full failure list is built by Step 04's cleaning script (it writes
    ``_parse_failures.csv``); this helper exists so the failure rate is
    visible during dev without requiring the cleaning pipeline to be wired
    up. Value is truncated to 80 chars to keep log lines bounded.
    """
    raw = "" if value is None else str(value)
    display = raw if len(raw) <= 80 else raw[:77] + "..."
    _log.warning("parse.unparseable field=%s value=%r city=%s", field, display, city)


def _coerce_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` if it can't be coerced."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_price(
    value: object,
    *,
    fallback_min: object = None,
    fallback_max: object = None,
) -> float | None:
    """Parse a free-text price string into a numeric INR value.

    Accepts:
      - Unit-suffixed: ``"3.5 Cr"``, ``"69.25 L"``, ``"69.25 Lac"``,
        ``"69.25 Lakh"`` (any casing).
      - Plain numeric: ``"15000000"``, ``"15000000.0"``.
      - Whitespace-padded variants of the above.

    Returns ``None`` if the value is unparseable and no usable fallback was
    supplied. When parse fails AND both ``fallback_min`` and ``fallback_max``
    are numeric, the midpoint of the two fallbacks is returned instead
    (per TRD §4.1 — cross-check against ``MIN_PRICE``/``MAX_PRICE``).
    """
    # Fast path: already numeric.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    raw = "" if value is None else str(value).strip()
    if not raw:
        return _price_fallback_or_none(value, fallback_min, fallback_max)

    match = _RE_PRICE.match(raw)
    if match is None:
        return _price_fallback_or_none(value, fallback_min, fallback_max)

    number_str, unit_word = match.group(1), match.group(2)
    try:
        number = float(number_str)
    except ValueError:
        return _price_fallback_or_none(value, fallback_min, fallback_max)

    if unit_word is None:
        # Bare numeric — already in INR.
        return number

    multiplier = _PRICE_UNIT_MAP.get(_normalize_unit(unit_word))
    if multiplier is None:
        # Unknown unit word — treat as parse failure + log.
        _log_unparseable("PRICE", value)
        return _price_fallback_or_none(value, fallback_min, fallback_max)

    return number * multiplier


def _price_fallback_or_none(
    value: object,
    fallback_min: object,
    fallback_max: object,
) -> float | None:
    """``None`` if no usable fallback; midpoint of fallbacks if both numeric."""
    lo = _coerce_float(fallback_min)
    hi = _coerce_float(fallback_max)
    if lo is None or hi is None:
        _log_unparseable("PRICE", value)
        return None
    return (lo + hi) / 2.0


def parse_area(value: object, *, unit_hint: str = "auto") -> float | None:
    """Parse a free-text area string into square-feet.

    Accepts:
      - ``"1500 sq.ft."``, ``"1500 sqft"`` (sqft, returned as-is).
      - ``"100 sq.m."`` (sqm, converted via ``SQFT_PER_SQM``).
      - Ranges: ``"1200-1400 sq.ft."`` → midpoint (1300.0).
      - Plain numeric: ``"1500"`` (assumed sqft when ``unit_hint="auto"``).

    ``unit_hint`` values:
      - ``"auto"`` (default): infer from the embedded suffix; bare numeric
        is treated as sqft.
      - ``"sqft"``: force sqft interpretation regardless of any suffix.
      - ``"sqm"``: force sqm conversion regardless of any suffix.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Numeric inputs are unit-less numbers; honor ``unit_hint``.
        if unit_hint == "sqm":
            return float(value) * SQFT_PER_SQM
        return float(value)

    raw = "" if value is None else str(value).strip()
    if not raw:
        _log_unparseable("AREA", value)
        return None

    match = _RE_AREA.match(raw)
    if match is None:
        _log_unparseable("AREA", value)
        return None

    low_str, high_str, unit_word = match.group(1), match.group(2), match.group(3)
    try:
        low = float(low_str)
    except ValueError:
        _log_unparseable("AREA", value)
        return None

    if high_str is not None:
        try:
            high = float(high_str)
        except ValueError:
            _log_unparseable("AREA", value)
            return None
        value_low = low
        value_high = high
    else:
        value_low = value_high = low

    midpoint = (value_low + value_high) / 2.0

    # Resolve the effective unit: forced hint wins, else embedded suffix.
    if unit_hint == "sqft":
        return midpoint
    if unit_hint == "sqm":
        return midpoint * SQFT_PER_SQM

    if unit_word is None:
        # Bare numeric → assume sqft.
        return midpoint

    multiplier = _AREA_UNIT_MAP.get(_normalize_unit(unit_word))
    if multiplier is None:
        # Unknown unit word — treat as parse failure rather than guess.
        _log_unparseable("AREA", value)
        return None

    return midpoint * multiplier
