"""Tests for ``ml.evaluation.protocol`` — pinned constants + split enforcement."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from ml.evaluation import protocol as P
from ml.evaluation.splits import protocol_split

# ---------------------------------------------------------------------------
# Constants — pinned
# ---------------------------------------------------------------------------


def test_protocol_version_is_pinned() -> None:
    assert P.PROTOCOL_VERSION == "1.0.0"


def test_split_ratios_match_trd_section_10() -> None:
    assert P.SPLIT_RATIOS == {"train": 0.70, "val": 0.15, "test": 0.15}


def test_random_state_is_42() -> None:
    assert P.RANDOM_STATE == 42


def test_metric_names_match_rules_section_2_1() -> None:
    assert P.METRIC_NAMES == ("r2", "mae", "rmse", "mape")


def test_protocol_doc_path_pinned() -> None:
    assert P.PROTOCOL_DOC_PATH == "docs/02-TRD.md"


def test_protocol_thresholds_keys_pinned() -> None:
    assert set(P.protocol_thresholds) == {
        "r2_min",
        "r2_stretch",
        "mae_pct_within_15_at_least",
        "p95_latency_ms_max",
        "rent_min_rows",
    }
    assert P.protocol_thresholds["r2_min"] == 0.80
    assert P.protocol_thresholds["r2_stretch"] == 0.85
    assert P.protocol_thresholds["mae_pct_within_15_at_least"] == 0.70
    assert P.protocol_thresholds["p95_latency_ms_max"] == 300.0


# ---------------------------------------------------------------------------
# protocol_split — enforcement
# ---------------------------------------------------------------------------


def _make_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = pd.DataFrame({"city": ["Gurgaon", "Hyderabad", "Mumbai", "Kolkata"]})
    cities = []
    for i in range(n):
        cities.append(rng.iloc[i % 4]["city"])
    return pd.DataFrame(
        {
            "city": cities,
            "price": [50_000_000.0 + (i * 1000.0) for i in range(n)],
            "bedRoom": [(i % 5) + 1 for i in range(n)],
        }
    )


def test_protocol_split_returns_three_frames() -> None:
    df = _make_df()
    train, val, test = protocol_split(df, target="price")
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(val) + len(test) == len(df)


def test_protocol_split_enforces_ratios_within_one_row() -> None:
    n = 1000
    df = _make_df(n=n)
    train, val, test = protocol_split(df, target="price")
    assert abs(len(train) - int(round(0.70 * n))) <= 1
    assert abs(len(val) - int(round(0.15 * n))) <= 1
    assert abs(len(test) - int(round(0.15 * n))) <= 1


def test_protocol_split_asserts_random_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = _make_df()
    with caplog.at_level(logging.ERROR, logger="ml.evaluation.splits"):
        with pytest.raises(ValueError, match="random_state"):
            protocol_split(df, target="price", random_state=7)


def test_protocol_split_is_deterministic() -> None:
    df1 = _make_df()
    df2 = _make_df()
    a_train, a_val, a_test = protocol_split(df1, target="price")
    b_train, b_val, b_test = protocol_split(df2, target="price")
    assert (a_train["price"].to_numpy() == b_train["price"].to_numpy()).all()
    assert (a_val["price"].to_numpy() == b_val["price"].to_numpy()).all()
    assert (a_test["price"].to_numpy() == b_test["price"].to_numpy()).all()
