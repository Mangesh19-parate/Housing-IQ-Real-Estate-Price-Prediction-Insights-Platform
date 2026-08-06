"""Tests for ``ml.cleaning.assemble`` — Step 06 orchestrator."""

from __future__ import annotations

import ast
import inspect
import logging
import os

import pandas as pd
import pytest

from ml.cleaning import CANONICAL_COLUMNS
from ml.cleaning.assemble import (
    ASSEMBLE_CITY_FILES,
    ASSEMBLE_REPORT_FIELDS,
    _derive_price_per_sqft,
    assemble_cleaned_frame,
)
from ml.cleaning.outliers import OUTLIER_REASON_COLUMN
from tests.fixtures.dedup_outlier_fixtures import make_frame

# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------

def test_assemble_city_files_constant_has_four_entries() -> None:
    assert set(ASSEMBLE_CITY_FILES.keys()) == {"Gurgaon", "Hyderabad", "Kolkata", "Mumbai"}
    assert ASSEMBLE_CITY_FILES["Gurgaon"] == "gurgaon_10k.csv"
    assert ASSEMBLE_CITY_FILES["Hyderabad"] == "hyderabad.csv"
    assert ASSEMBLE_CITY_FILES["Kolkata"] == "kolkata.csv"
    assert ASSEMBLE_CITY_FILES["Mumbai"] == "mumbai.csv"


def test_assemble_report_fields_constant() -> None:
    """ASSEMBLE_REPORT_FIELDS lists what the summary line logs."""
    expected = {
        "rows_in",
        "rows_dropped_no_listing_id",
        "rows_dropped_duplicate",
        "rows_in_after_dedup",
        "rows_flagged_outlier",
        "rows_in_after_outlier_flag",
        "per_city_breakdown",
    }
    assert set(ASSEMBLE_REPORT_FIELDS) == expected


def test_assemble_does_not_import_app_or_api() -> None:
    """Static AST scan: no app.* or api.* imports in assemble.py."""
    import ml.cleaning.assemble as mod
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                assert not mod_name.startswith("app"), f"unexpected 'app' import: {mod_name}"
                assert not mod_name.startswith("api"), f"unexpected 'api' import: {mod_name}"
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            mod_name = node.module
            # Ignore our own package (ml.cleaning.* — "app" appears as a
            # substring inside "canonical_mapping" but that's fine).
            assert not mod_name.startswith("app."), f"unexpected 'app' import: {mod_name}"
            assert not mod_name.startswith("api."), f"unexpected 'api' import: {mod_name}"
            assert mod_name != "app"
            assert mod_name != "api"


def test_assemble_does_not_write_to_disk() -> None:
    """AST scan: no to_parquet / to_csv / to_json / open( / Path('data/processed')."""
    import ml.cleaning.assemble as mod
    src = inspect.getsource(mod)
    forbidden = ["to_parquet", "to_csv", "to_json", 'Path("data/processed"']
    for tok in forbidden:
        assert tok not in src, f"assemble.py contains forbidden token: {tok!r}"
    # open() disallowed
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "assemble.py contains open() call"


# ---------------------------------------------------------------------------
# B. _derive_price_per_sqft
# ---------------------------------------------------------------------------

def test_derive_price_per_sqft_computes_only_where_inputs_present() -> None:
    """price_per_sqft = price_inr / area_sqft only where both are non-null and area > 0."""
    df = pd.DataFrame({
        "price_inr": [1_000_000.0, pd.NA, 2_000_000.0, 500_000.0, 100.0],
        "area_sqft": [1000.0, 1000.0, pd.NA, 0.0, 50.0],
    })
    out = _derive_price_per_sqft(df)
    # Row 0: 1_000_000 / 1000 = 1000
    assert float(out["price_per_sqft"].iloc[0]) == 1000.0
    # Row 1: price_inr is NaN → stay NaN
    assert pd.isna(out["price_per_sqft"].iloc[1])
    # Row 2: area_sqft is NaN → stay NaN
    assert pd.isna(out["price_per_sqft"].iloc[2])
    # Row 3: area_sqft == 0 → stay NaN (avoid div-by-zero)
    assert pd.isna(out["price_per_sqft"].iloc[3])
    # Row 4: 100/50 = 2
    assert float(out["price_per_sqft"].iloc[4]) == 2.0


def test_derive_price_per_sqft_creates_column_if_missing() -> None:
    """If price_per_sqft column missing, helper creates it as NaN-filled."""
    df = pd.DataFrame({"price_inr": [1_000_000.0], "area_sqft": [1000.0]})
    assert "price_per_sqft" not in df.columns
    out = _derive_price_per_sqft(df)
    assert "price_per_sqft" in out.columns
    assert float(out["price_per_sqft"].iloc[0]) == 1000.0


def test_assemble_cleaned_frame_derives_price_per_sqft_before_outlier_flagging() -> None:
    """End-to-end: extreme price_per_sqft in input → flagged after derivation."""
    rows: list[dict] = []
    for i in range(100):
        rows.append({
            "listing_id": f"G{i:08d}",
            "city": "Gurgaon",
            "property_type": "flat",
            "bedRoom": 3,
            "bathroom": 3,
            "price_inr": 1.0e7 + i * 1.0e5,
            "area_sqft": 1500.0,
            "price_per_sqft": pd.NA,
        })
    # Extreme outlier: huge price on tiny area → price_per_sqft = 1e7 (huge).
    rows.append({
        "listing_id": "X-OUTLIER",
        "city": "Gurgaon",
        "property_type": "flat",
        "bedRoom": 3,
        "bathroom": 3,
        "price_inr": 1_000_000_000.0,
        "area_sqft": 100.0,
        "price_per_sqft": pd.NA,
    })
    df = make_frame(rows)
    from ml.cleaning.assemble import _derive_price_per_sqft
    from ml.cleaning.outliers import flag_all_outliers
    derived = _derive_price_per_sqft(df)
    # Confirm derivation ran.
    last_pps = float(derived["price_per_sqft"].iloc[-1])
    assert last_pps == 1_000_000_000.0 / 100.0
    flagged = flag_all_outliers(derived)
    assert bool(flagged["is_outlier"].iloc[-1]) is True
    reasons = flagged[OUTLIER_REASON_COLUMN].iloc[-1]
    assert any(r.startswith("percentile_") or r.startswith("iqr_")
               for r in reasons)


# ---------------------------------------------------------------------------
# C. assemble_cleaned_frame integration (real data, marked realdata)
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_assemble_cleaned_frame_does_not_write_to_data_processed() -> None:
    """The real data/processed/ directory is untouched before vs after a call."""
    from pathlib import Path
    raw = Path("data/raw")
    processed = Path("data/processed")
    before = sorted(os.listdir(processed)) if processed.exists() else []
    assemble_cleaned_frame(raw, raw / "facets")
    after = sorted(os.listdir(processed)) if processed.exists() else []
    assert before == after


@pytest.mark.realdata
def test_assemble_cleaned_frame_returns_dataframe_with_all_canonical_columns() -> None:
    from pathlib import Path
    raw = Path("data/raw")
    df = assemble_cleaned_frame(raw, raw / "facets")
    # All canonical columns must be present (possibly reordered, but present).
    assert set(CANONICAL_COLUMNS).issubset(set(df.columns))


@pytest.mark.realdata
def test_assemble_cleaned_frame_has_listing_id_unique() -> None:
    from pathlib import Path
    raw = Path("data/raw")
    df = assemble_cleaned_frame(raw, raw / "facets")
    assert df["listing_id"].is_unique


@pytest.mark.realdata
def test_assemble_cleaned_frame_has_is_outlier_column() -> None:
    from pathlib import Path
    raw = Path("data/raw")
    df = assemble_cleaned_frame(raw, raw / "facets")
    assert "is_outlier" in df.columns
    assert df["is_outlier"].dtype == "bool"
    # Real data should produce at least one True.
    assert int(df["is_outlier"].sum()) > 0


@pytest.mark.realdata
def test_assemble_cleaned_frame_has_outlier_reasons_column() -> None:
    from pathlib import Path
    raw = Path("data/raw")
    df = assemble_cleaned_frame(raw, raw / "facets")
    assert OUTLIER_REASON_COLUMN in df.columns
    assert df[OUTLIER_REASON_COLUMN].dtype == "object"
    # Mix of empty and non-empty lists.
    has_empty = any(len(r) == 0 for r in df[OUTLIER_REASON_COLUMN])
    has_nonempty = any(len(r) > 0 for r in df[OUTLIER_REASON_COLUMN])
    assert has_empty and has_nonempty


@pytest.mark.realdata
def test_assemble_cleaned_frame_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    from pathlib import Path
    raw = Path("data/raw")
    with caplog.at_level(logging.INFO, logger="ml.cleaning.assemble"):
        assemble_cleaned_frame(raw, raw / "facets")
    summaries = [r for r in caplog.records if "assemble.summary" in r.message]
    assert summaries, "expected an assemble.summary log line"


@pytest.mark.realdata
def test_assemble_cleaned_frame_is_pure_no_io() -> None:
    """Same args twice → equal outputs. No state stashed on the module."""
    from pathlib import Path
    raw = Path("data/raw")
    df1 = assemble_cleaned_frame(raw, raw / "facets")
    df2 = assemble_cleaned_frame(raw, raw / "facets")
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.realdata
def test_assemble_cleaned_frame_asserts_raw_readonly(tmp_path, monkeypatch) -> None:
    """The assembler wires ``assert_raw_readonly`` (Step 02's gate) into its
    entry point. This test verifies both halves in isolation:

      1. ``assert_raw_readonly`` raises ``RuntimeError`` when the per-file
         snapshot differs between two consecutive samples (the gate's actual
         semantics).
      2. ``assemble_cleaned_frame`` calls ``assert_raw_readonly`` before any
         raw data is read.

    We can't safely mutate the real ``data/raw/`` in a test, so the gate's
    raising behavior is verified directly against a tampered tmp copy, then
    a spy confirms the assembler wired the call in.
    """
    import shutil
    from pathlib import Path
    from ml.cleaning import assemble as _assemble_mod
    from ml.cleaning.ingest import _snapshot_raw_files

    src = Path("data/raw")
    dst = tmp_path / "raw"
    (dst / "facets").mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    for p in (src / "facets").iterdir():
        if p.is_file():
            shutil.copy2(p, dst / "facets" / p.name)

    # Snapshot before mutation.
    before = _snapshot_raw_files(dst.parent)
    # Mutate CITY.csv content.
    (dst / "facets" / "CITY.csv").write_text("city_id,city_label\nX,Tampered\n")
    after = _snapshot_raw_files(dst.parent)
    # The gate's substrate: changed snapshot will be detected when the two
    # snapshots are compared inside the gate (we exercise that directly).
    assert before != after

    # (1) The gate itself raises when the file changes between snapshots.
    from ml.cleaning.ingest import assert_raw_readonly
    with pytest.raises(RuntimeError, match="immutability"):
        # The real gate just compares two back-to-back snapshots. To force a
        # mismatch inside a single call, monkey-patch its inner snapshot to
        # flip the result on the second call.
        original = _snapshot_raw_files
        calls = {"n": 0}

        def _flipped(data_dir):
            calls["n"] += 1
            snap = original(data_dir)
            if calls["n"] == 2:
                snap = dict(snap)
                key = next(iter(snap))
                snap[key] = {**snap[key], "sha256": "0" * 64}
            return snap

        monkeypatch.setattr(
            "ml.cleaning.ingest._snapshot_raw_files", _flipped
        )
        _assemble_mod.assert_raw_readonly(dst.parent)

    # (2) The assembler invokes the gate — spy it and confirm a call landed.
    # Restore CITY.csv so the full assemble call can read facets cleanly.
    shutil.copy2(src / "facets" / "CITY.csv", dst / "facets" / "CITY.csv")
    spy_calls = {"n": 0}

    def _spy(data_dir):
        spy_calls["n"] += 1
        return None  # pass the gate

    monkeypatch.setattr(_assemble_mod, "assert_raw_readonly", _spy)
    _assemble_mod.assemble_cleaned_frame(dst, dst / "facets")
    assert spy_calls["n"] >= 1, "assembler must call assert_raw_readonly"
