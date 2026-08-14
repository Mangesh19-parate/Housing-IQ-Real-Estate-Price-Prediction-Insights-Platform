"""Tests for ml.training.selection (Spec 13 Phase A)."""

from __future__ import annotations

import pytest

from ml.training.selection import select_winner


def _candidate(rmse: float, mae: float) -> dict:
    return {
        "train": {"r2": 0.9, "mae": mae * 0.5, "rmse": rmse * 0.5, "mape": 5.0},
        "val":   {"r2": 0.8, "mae": mae,        "rmse": rmse,        "mape": 8.0},
        "test":  {"r2": 0.7, "mae": mae * 1.1,  "rmse": rmse * 1.1,  "mape": 9.0},
    }


def test_select_winner_returns_lowest_val_rmse():
    candidates = {
        "a": _candidate(rmse=100.0, mae=80.0),
        "b": _candidate(rmse=80.0,  mae=90.0),  # best rmse
        "c": _candidate(rmse=120.0, mae=70.0),
    }
    assert select_winner(candidates) == "b"


def test_select_winner_tie_breaks_on_val_mae():
    candidates = {
        "a": _candidate(rmse=80.0, mae=100.0),  # tie on rmse, higher mae
        "b": _candidate(rmse=80.0, mae=70.0),   # tie on rmse, lower mae wins
        "c": _candidate(rmse=90.0, mae=50.0),
    }
    assert select_winner(candidates) == "b"


def test_select_winner_raises_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        select_winner({})


def test_select_winner_accepts_custom_primary_metric():
    candidates = {
        "a": {"val": {"r2": 0.7, "mae": 100.0, "rmse": 100.0, "mape": 5.0}},
        "b": {"val": {"r2": 0.9, "mae": 100.0, "rmse": 100.0, "mape": 3.0}},
    }
    assert select_winner(candidates, primary_metric="mape") == "b"
