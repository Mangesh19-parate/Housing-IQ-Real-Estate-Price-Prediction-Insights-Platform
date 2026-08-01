"""Unit tests for ``ml.cleaning.parsing`` (Step 03).

Test names mirror the spec's "Definition of done" #1 exactly — they're the
contract.

No real-data dependency: every test uses literal strings via
``tests/fixtures/parse_fixtures.py`` or inline literals. The whole suite runs
under ``pytest -m "not realdata"``.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from ml.cleaning.parsing import (
    _AREA_UNIT_MAP,
    _PRICE_UNIT_MAP,
    SQFT_PER_SQM,
    parse_area,
    parse_price,
)
from tests.fixtures.parse_fixtures import AREA_SAMPLES, INVALID_STRINGS, PRICE_SAMPLES

# --- parse_price: format coverage ---------------------------------------------


def test_parse_price_crore_uppercase() -> None:
    assert parse_price("3.5 Cr") == 35_000_000.0


def test_parse_price_crore_lowercase() -> None:
    assert parse_price("3.5 cr") == 35_000_000.0


def test_parse_price_lakh_short() -> None:
    assert parse_price("69.25 L") == 6_925_000.0


def test_parse_price_lakh_full() -> None:
    assert parse_price("69.25 Lac") == 6_925_000.0
    assert parse_price("69.25 Lakh") == 6_925_000.0


def test_parse_price_plain_numeric() -> None:
    assert parse_price("15000000") == 15_000_000.0
    assert parse_price("15000000.0") == 15_000_000.0


def test_parse_price_with_whitespace() -> None:
    assert parse_price("  3.5  Cr  ") == 35_000_000.0


# --- parse_price: failure modes -----------------------------------------------


def test_parse_price_invalid_returns_none() -> None:
    for s in INVALID_STRINGS:
        assert parse_price(s) is None  # noqa: PERF401 — explicit loop is clearer here
    # No exception is raised — bare try/except inside the parser swallows regex failures.


def test_parse_price_unparseable_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="ml.cleaning.parsing")
    parse_price("call for price")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected at least one WARNING-level log record"
    msg = warnings[-1].getMessage()
    assert "PRICE" in msg
    assert "call for price" in msg


# --- parse_price: fallback semantics ------------------------------------------


def test_parse_price_fallback_to_min_max() -> None:
    # Both fallbacks present and numeric → midpoint wins.
    assert parse_price(
        "N/A", fallback_min=14_000_000, fallback_max=16_000_000
    ) == 15_000_000.0
    # Either fallback missing → no fallback available → None.
    assert parse_price("N/A", fallback_min=14_000_000, fallback_max=None) is None
    assert parse_price("N/A", fallback_min=None, fallback_max=16_000_000) is None


def test_parse_price_fallback_min_max_ignore_when_value_parseable() -> None:
    # Parseable value beats fallbacks.
    assert parse_price(
        "3.5 Cr", fallback_min=14_000_000, fallback_max=16_000_000
    ) == 35_000_000.0


# --- parse_area: format coverage ----------------------------------------------


def test_parse_area_sqft() -> None:
    assert parse_area("1500 sq.ft.") == 1500.0
    assert parse_area("1500 sqft") == 1500.0


def test_parse_area_sqm_to_sqft() -> None:
    got = parse_area("100 sq.m.")
    assert got is not None
    assert abs(got - 100 * SQFT_PER_SQM) < 1e-6


def test_parse_area_range_midpoint() -> None:
    assert parse_area("1200-1400 sq.ft.") == 1300.0


def test_parse_area_plain_numeric() -> None:
    assert parse_area("1500") == 1500.0


def test_parse_area_with_whitespace() -> None:
    assert parse_area("  1500  sq.ft.  ") == 1500.0


def test_parse_area_invalid_returns_none() -> None:
    for s in ("", "--"):
        assert parse_area(s) is None  # noqa: PERF401


def test_parse_area_unit_hint_auto() -> None:
    # auto: bare numeric → treated as sqft.
    assert parse_area("1500") == 1500.0
    # sqft hint: forces sqft interpretation even if a suffix is embedded.
    assert parse_area("100 sq.m.", unit_hint="sqft") == 100.0
    # sqm hint: forces sqm conversion even if no suffix.
    got = parse_area("100", unit_hint="sqm")
    assert got is not None
    assert abs(got - 1076.39) < 1e-6


# --- Module-level invariants --------------------------------------------------


def test_price_unit_map_keys_are_canonical_lowercase() -> None:
    keys = set(_PRICE_UNIT_MAP.keys())
    assert keys == {"cr", "l", "lac", "lakh"}
    # No uppercase duplicates accidentally added.
    assert "Cr" not in keys
    assert "L" not in keys
    assert "Lac" not in keys


def test_area_unit_map_includes_sqft_and_sqm() -> None:
    assert _AREA_UNIT_MAP["sqft"] == 1.0
    assert _AREA_UNIT_MAP["sqm"] == SQFT_PER_SQM


def test_sqft_per_sqm_constant_value() -> None:
    assert SQFT_PER_SQM == 10.7639


# --- Idempotency / purity -----------------------------------------------------


def test_parse_price_idempotent() -> None:
    # Calling parse_price on its own output is the same as the result itself —
    # proves no module-level cache leaks state between calls.
    once = parse_price("3.5 Cr")
    assert once is not None
    twice = parse_price(once)
    assert twice == once


def test_parse_area_idempotent() -> None:
    once = parse_area("1500 sq.ft.")
    assert once is not None
    twice = parse_area(once)
    assert twice == once


# --- Source-level invariants (no forbidden imports, no I/O) ------------------


def test_parser_does_not_import_pandas_or_numpy() -> None:
    """Static scan: parser module source must not import pandas/numpy."""
    src_path = Path(__file__).resolve().parent.parent / "ml" / "cleaning" / "parsing.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported.append(f"{module}.{alias.name}" if module else alias.name)
    flat = " ".join(imported).lower()
    assert "pandas" not in flat, f"pandas imported in parsing.py: {imported}"
    assert "numpy" not in flat, f"numpy imported in parsing.py: {imported}"


def test_parser_does_not_touch_data_raw_or_data_processed() -> None:
    """Static scan: parser module source must not open files or reference raw/processed paths."""
    src_path = Path(__file__).resolve().parent.parent / "ml" / "cleaning" / "parsing.py"
    source = src_path.read_text(encoding="utf-8")
    forbidden = [
        "open(",
        'Path("data/raw',
        'Path("data/processed',
        "Path('data/raw",
        "Path('data/processed",
    ]
    for needle in forbidden:
        assert needle not in source, f"parser source contains forbidden literal {needle!r}"


# --- Spot-check that the fixtures themselves are well-formed ------------------


def test_parse_fixtures_price_samples_match() -> None:
    """Cross-check the fixtures module itself — protects against future edits."""
    for raw, expected in PRICE_SAMPLES:
        got = parse_price(raw)
        assert got is not None and abs(got - expected) < 1e-6, (
            f"fixture {raw!r} expected {expected!r}, got {got!r}"
        )


def test_parse_fixtures_area_samples_match() -> None:
    for raw, expected in AREA_SAMPLES:
        got = parse_area(raw)
        assert got is not None and abs(got - expected) < 1e-6, (
            f"fixture {raw!r} expected {expected!r}, got {got!r}"
        )
