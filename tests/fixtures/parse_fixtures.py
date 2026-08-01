"""Literal sample strings for ``parse_price`` / ``parse_area`` unit tests.

Pure literals — no real-data dependency, so the test suite stays fast and
CI-friendly. Imported by ``tests/test_parsing.py``.

Each ``(input_str, expected_float)`` pair is one observation of the format
patterns documented in the data-cleaning skill and the Step 03 spec.
"""

from __future__ import annotations

# (input, expected INR) — covers every format observed in the 4 raw city CSVs.
PRICE_SAMPLES: list[tuple[str, float]] = [
    ("3.5 Cr", 35_000_000.0),
    ("3.5 cr", 35_000_000.0),
    ("69.25 L", 6_925_000.0),
    ("69.25 Lac", 6_925_000.0),
    ("69.25 Lakh", 6_925_000.0),
    ("15000000", 15_000_000.0),
    ("15000000.0", 15_000_000.0),
    ("  3.5  Cr  ", 35_000_000.0),
]

# (input, expected sqft) — covers sqft, sq.m., range, plain numeric, whitespace.
AREA_SAMPLES: list[tuple[str, float]] = [
    ("1500 sq.ft.", 1500.0),
    ("1500 sqft", 1500.0),
    ("100 sq.m.", 100 * 10.7639),
    ("1200-1400 sq.ft.", 1300.0),
    ("1500", 1500.0),
    ("  1500  sq.ft.  ", 1500.0),
]

# Strings that should return None from both parsers.
INVALID_STRINGS: list[str] = ["", "call for price", "--", "N/A"]
