"""Unit tests for ``ml.cleaning.facet_decoders`` (Step 04).

Test names mirror the spec's "Definition of done" #1 exactly — they're the
contract. No real-data dependency: every test uses literal ``pd.DataFrame``s
from ``tests/fixtures/facet_decode_fixtures.py`` or inlined literals.

The whole suite runs under ``pytest -m "not realdata"``.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
import pytest

import ml.cleaning.facet_decoders as facet_decoders
from ml.cleaning.facet_decoders import (
    DEFAULT_UNKNOWN_LABEL,
    MULTI_VALUE_DELIMITER,
    MULTI_VALUE_FACETS,
    SINGLE_VALUE_FACETS,
    decode_age,
    decode_amenities,
    decode_facing,
    decode_features,
    decode_floor_num,
    decode_furnish,
    decode_owntype,
    decode_property_type,
)
from tests.fixtures.facet_decode_fixtures import (
    AGE_DF,
    AMENITIES_DF,
    BATHROOM_NUM_DF,
    BEDROOM_NUM_DF,
    BUILDING_ID_DF,
    CITY_DF,
    FACING_DF,
    FEATURES_DF,
    FLOOR_NUM_DF,
    FURNISH_DF,
    LOCALITY_ID_DF,
    OWNTYPE_DF,
    PROPERTY_TYPE_DF,
    SUB_AVAILABILITY_DF,
    TOTAL_FLOOR_DF,
    build_synthetic_facets_dir,
)

# ===========================================================================
# Helpers — `_index` attachment
# ===========================================================================
#
# The decoders expect facet_df to carry an "_index" dict (built by
# load_facet_frames). The per-decoder tests construct their own DataFrames,
# so they have to attach an index before calling the decoder. This helper
# does that and returns the DataFrame (mutated in place; mirrors the way
# load_facet_frames mutates the DataFrame it returns).


def _attach_index(facet_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a per-facet ``_index`` dict the same way load_facet_frames does.

    Stored on ``df.attrs`` so it's separate from the tabular data the
    DataFrame represents (pandas-canonical place for metadata).
    """
    index: dict[str, str] = {}
    for _, row in facet_df.iterrows():
        if pd.isna(row["id"]):
            continue
        index[str(row["id"])] = str(row["label"])
    facet_df.attrs["_index"] = index
    return facet_df


# Pre-attach indices for the fixture DataFrames at import time so test
# bodies stay clean. This is exactly what load_facet_frames does at load.
for _df in (
    FURNISH_DF,
    FACING_DF,
    AGE_DF,
    PROPERTY_TYPE_DF,
    OWNTYPE_DF,
    LOCALITY_ID_DF,
    BUILDING_ID_DF,
    BATHROOM_NUM_DF,
    TOTAL_FLOOR_DF,
    SUB_AVAILABILITY_DF,
    CITY_DF,
    BEDROOM_NUM_DF,
    FLOOR_NUM_DF,
    AMENITIES_DF,
    FEATURES_DF,
):
    _attach_index(_df)


# ===========================================================================
# Single-value decode: known / unknown / NaN
# ===========================================================================


def test_decode_furnish_known_id() -> None:
    assert decode_furnish(4, FURNISH_DF) == "Semifurnished"


def test_decode_furnish_unknown_id_returns_unknown() -> None:
    # Per spec: unknown-but-present IDs return DEFAULT_UNKNOWN_LABEL,
    # not None, not an exception.
    assert decode_furnish(999, FURNISH_DF) == DEFAULT_UNKNOWN_LABEL


def test_decode_furnish_nan_input_returns_none() -> None:
    assert decode_furnish(pd.NA, FURNISH_DF) is None
    assert decode_furnish(float("nan"), FURNISH_DF) is None


# ===========================================================================
# Per-facet known-id coverage (one test per spec DoD #1 item)
# ===========================================================================


def test_decode_facing_known_id() -> None:
    assert decode_facing(1, FACING_DF) == "North"


def test_decode_age_known_id() -> None:
    assert decode_age(2, AGE_DF) == "Relatively New"


def test_decode_property_type_known_id() -> None:
    assert decode_property_type(1, PROPERTY_TYPE_DF) == "Residential Apartment"


def test_decode_owntype_known_id() -> None:
    assert decode_owntype(1, OWNTYPE_DF) == "Freehold"


# ===========================================================================
# FLOOR_NUM special cases (string codes + above-max fallback)
# ===========================================================================


def test_decode_floor_num_integer_known() -> None:
    # FLOOR_NUM_DF includes "1", "5", "10", "50" as string-typed ids.
    assert decode_floor_num(1, FLOOR_NUM_DF) == "1"
    assert decode_floor_num(50, FLOOR_NUM_DF) == "50"


def test_decode_floor_num_string_code_basement() -> None:
    assert decode_floor_num("B", FLOOR_NUM_DF) == "Basement"


def test_decode_floor_num_string_code_multi_storied() -> None:
    assert decode_floor_num("M", FLOOR_NUM_DF) == "Multi-Storied"


def test_decode_floor_num_above_max_returns_above_max() -> None:
    # FLOOR_NUM_DF's max integer row is "50"; 95 must hit the domain-rule
    # fallback and return FLOOR_NUM_ABOVE_MAX_LABEL ("95+").
    assert decode_floor_num(95, FLOOR_NUM_DF) == "95+"


def test_decode_floor_num_unknown_string_returns_unknown() -> None:
    # Alphabetic code not in the facet ("X") is genuinely unknown — must
    # NOT trigger the "above-max" branch (that's integer-only).
    assert decode_floor_num("X", FLOOR_NUM_DF) == DEFAULT_UNKNOWN_LABEL


# ===========================================================================
# Multi-value decode
# ===========================================================================


def test_decode_features_comma_separated_list() -> None:
    labels = decode_features("12,23,33", FEATURES_DF)
    assert labels == ["Lift", "Power Back-up", "Feng Shui Compliant"]


def test_decode_features_nan_returns_empty_list() -> None:
    assert decode_features(pd.NA, FEATURES_DF) == []


def test_decode_features_unknown_ids_dropped_silently() -> None:
    # 999 is not in FEATURES_DF (it has 12/23/33). The decoded list must
    # contain only the two known IDs, not "unknown" and not 999.
    labels = decode_features("12,999,23", FEATURES_DF)
    assert labels == ["Lift", "Power Back-up"]


def test_decode_features_whitespace_stripped() -> None:
    assert decode_features("12, 23 , 33", FEATURES_DF) == decode_features(
        "12,23,33", FEATURES_DF
    )


def test_decode_features_malformed_returns_empty() -> None:
    # Non-numeric, non-comma input fails the _RE_MULTI_VALUE.fullmatch
    # pre-validation; the decoder logs a warning and returns [].
    assert decode_features("not-an-id-list", FEATURES_DF) == []


def test_decode_amenities_known_list() -> None:
    labels = decode_amenities("20,21,23", AMENITIES_DF)
    assert labels == ["Club House", "Swimming Pool", "Power Back-up"]


def test_decode_amenities_drop_unknown() -> None:
    labels = decode_amenities("20,999,21", AMENITIES_DF)
    assert labels == ["Club House", "Swimming Pool"]


# ===========================================================================
# Driver + module-level invariants
# ===========================================================================


def test_load_facet_frames_returns_fifteen_entries(tmp_path: Path) -> None:
    build_synthetic_facets_dir(tmp_path)
    frames = facet_decoders.load_facet_frames(tmp_path / "raw" / "facets")
    # Plus the decode_stats sidecar key, so total = 16 keys.
    assert len(frames) == 15
    expected = {
        "AGE", "AMENITIES", "BATHROOM_NUM", "BEDROOM_NUM", "BUILDING_ID",
        "CITY", "FACING_DIRECTION", "FEATURES", "FLOOR_NUM", "FURNISH",
        "LOCALITY_ID", "OWNERSHIP_TYPE", "PROPERTY_TYPE",
        "SUB_AVAILABILITY", "TOTAL_FLOOR",
    }
    assert set(frames.keys()) == expected
    for name, df in frames.items():
        assert not df.empty
        assert isinstance(df.attrs.get("_index"), dict)
        assert df.attrs["_index"]


def test_load_facet_frames_does_not_modify_files(tmp_path: Path) -> None:
    import ml.cleaning.ingest as ingest

    build_synthetic_facets_dir(tmp_path)
    data_dir = tmp_path
    before = ingest._snapshot_raw_files(data_dir)
    facet_decoders.load_facet_frames(tmp_path / "raw" / "facets")
    after = ingest._snapshot_raw_files(data_dir)
    assert before == after, "data/raw/ was modified during load_facet_frames"


def test_load_facet_frames_normalizes_id_keys(tmp_path: Path) -> None:
    """Lock in the join-key normalization rule.

    Whatever shape the facet CSV's id column arrives in (int, zero-padded
    string, alphabetic string), ``decode_furnish`` must resolve both int
    and string inputs to the same label.
    """
    facets_dir = tmp_path / "raw" / "facets"
    facets_dir.mkdir(parents=True, exist_ok=True)
    # Mixed-shape fixture: one int id, one zero-padded string id, both
    # pointing at the same label. After normalization, "4" and "04" must
    # both resolve to "Furnished".
    pd.DataFrame(
        [
            {"id": 4, "label": "Furnished"},
            {"id": "04", "label": "FurnishedZeroPadded"},
            {"id": "B", "label": "Basement"},
        ]
    ).to_csv(facets_dir / "FURNISH.csv", index=False)
    # Stub the other 14 facets with one row each so load_facet_frames is
    # happy.
    for name in (
        "AGE", "AMENITIES", "BATHROOM_NUM", "BEDROOM_NUM", "BUILDING_ID",
        "CITY", "FACING_DIRECTION", "FEATURES", "FLOOR_NUM", "LOCALITY_ID",
        "OWNERSHIP_TYPE", "PROPERTY_TYPE", "SUB_AVAILABILITY", "TOTAL_FLOOR",
    ):
        pd.DataFrame([{"id": 1, "label": "x"}]).to_csv(
            facets_dir / f"{name}.csv", index=False
        )
    frames = facet_decoders.load_facet_frames(facets_dir)
    # Int and string-typed lookups for the same id must both work.
    assert decode_furnish(4, frames["FURNISH"]) == "Furnished"
    assert decode_furnish("4", frames["FURNISH"]) == "Furnished"


def test_decode_idempotent() -> None:
    # Calling twice on the same input gives the same answer; no module-
    # level cache leaks state between calls.
    once = decode_furnish(4, FURNISH_DF)
    twice = decode_furnish(4, FURNISH_DF)
    assert once == twice == "Semifurnished"


# ===========================================================================
# Source-level invariants (AST scans)
# ===========================================================================


def test_decoder_does_not_import_app_or_api() -> None:
    """Static scan: decoder module must not import anything from app.* or api.*."""
    src_path = (
        Path(__file__).resolve().parent.parent
        / "ml"
        / "cleaning"
        / "facet_decoders.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported.append(
                    f"{module}.{alias.name}" if module else alias.name
                )
    flat = " ".join(imported).lower()
    # Strip leading "ml.cleaning.parsing._log_unparseable" — that's an
    # allowed internal dependency (Step 03 helper). The rule is no
    # app.* or api.* (the Flask / FastAPI HTTP layers).
    assert not any(
        imp.startswith("app") or imp.startswith("api")
        for imp in flat.split()
    ), f"forbidden import from app/api: {imported}"


def test_decoder_does_not_touch_data_raw_or_data_processed() -> None:
    """Static scan: decoder source must not contain path literals to raw/processed."""
    src_path = (
        Path(__file__).resolve().parent.parent
        / "ml"
        / "cleaning"
        / "facet_decoders.py"
    )
    source = src_path.read_text(encoding="utf-8")
    forbidden = [
        "open(",
        'Path("data/raw',
        'Path("data/processed',
        "Path('data/raw",
        "Path('data/processed",
    ]
    for needle in forbidden:
        assert needle not in source, (
            f"decoder source contains forbidden literal {needle!r}"
        )


# ===========================================================================
# Logging contract
# ===========================================================================


def test_decode_emits_log_warning_for_unknown_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_log_unparseable logs under ``ml.cleaning.parsing`` — not under the
    decoder's own logger — because we reuse the Step 03 helper for log-
    line consistency. Tests must filter caplog by that logger name.
    """
    caplog.set_level(logging.WARNING, logger="ml.cleaning.parsing")
    decode_furnish(999, FURNISH_DF)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected at least one WARNING-level log record"
    msg = warnings[-1].getMessage()
    # Field tag is the canonical lowercase name, NOT the raw uppercase column.
    assert "furnish" in msg
    assert "FURNISH" not in msg
    # Value is truncated to 80 chars but "999" fits.
    assert "999" in msg


def test_decode_per_facet_log_field_name_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every facet's log field tag must be its canonical lowercase name."""
    caplog.set_level(logging.WARNING, logger="ml.cleaning.parsing")
    decode_amenities("20,999", AMENITIES_DF)  # 999 is unknown
    decode_features("12,999", FEATURES_DF)  # 999 is unknown
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("amenities" in m for m in msgs)
    assert any("features" in m for m in msgs)
    # And no uppercase raw-column names leaked through.
    for m in msgs:
        assert "AMENITIES" not in m
        assert "FEATURES" not in m


# ===========================================================================
# Module-level constant contracts
# ===========================================================================


def test_single_value_facets_count_is_thirteen() -> None:
    assert len(SINGLE_VALUE_FACETS) == 13


def test_multi_value_facets_is_features_and_amenities() -> None:
    assert MULTI_VALUE_FACETS == ("FEATURES", "AMENITIES")


def test_default_unknown_label_is_unknown() -> None:
    assert DEFAULT_UNKNOWN_LABEL == "unknown"


def test_multi_value_delimiter_is_comma() -> None:
    assert MULTI_VALUE_DELIMITER == ","
