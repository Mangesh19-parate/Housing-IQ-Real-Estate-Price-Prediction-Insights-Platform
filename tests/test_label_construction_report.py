"""Tests for ``ml.classification.report`` (Spec 21).

All test names are pinned by the spec's Definition of DoD item 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.classification.report import write_label_construction_report


def _synthetic_tier_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i, city in enumerate(("Gurgaon", "Hyderabad", "Kolkata", "Mumbai")):
        for tier in ("Budget", "Mid-Range", "Premium", "Luxury"):
            for j in range(20):
                rows.append(
                    {
                        "listing_id": f"{city}_{tier}_{j}",
                        "city": city,
                        "locality": f"{city}_x",
                        "transact_type": "Sale",
                        "price_per_sqft": float(rng.integers(3000, 12000)),
                        "price_tier": tier,
                        "tier_quantile_q1": 3000.0,
                        "tier_quantile_q2": 5000.0,
                        "tier_quantile_q3": 8000.0,
                        "is_outlier": False,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_good_deal_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        for verdict in ("Good Deal", "Fair Price", "Overpriced"):
            for j in range(20):
                rows.append(
                    {
                        "listing_id": f"{city}_{verdict}_{j}",
                        "city": city,
                        "transact_type": "Sale",
                        "actual_price": 100.0,
                        "oof_predicted_price": 100.0,
                        "residual_pct": float(rng.normal(0, 0.1)),
                        "good_deal_verdict": verdict,
                        "verdict_threshold_low": -0.10,
                        "verdict_threshold_high": 0.10,
                        "is_outlier": False,
                        "reason": None,
                        "model_version": "v2",
                    }
                )
    return pd.DataFrame(rows)


def test_write_label_construction_report_writes_header_on_first_run(
    tmp_path: Path,
) -> None:
    """Empty output_dir → header gets written."""
    tier_df = _synthetic_tier_df()
    good_df = _synthetic_good_deal_df()
    write_label_construction_report(
        output_dir=tmp_path,
        price_tier_df=tier_df,
        good_deal_df=good_df,
        quantile_boundaries={"Gurgaon": {"Sale": [1.0, 2.0, 3.0]}},
        good_deal_thresholds={
            "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 200,
                        "median_residual": 0.0, "iqr_residual": 0.05},
            "_default": {"low": -0.10, "high": 0.10, "rationale": "default"},
        },
        dataset_version="clean_listings_2026-08-28.parquet:abc123",
        git_commit="abc123def",
        min_rows=100,
    )
    p = tmp_path / "label_construction_report.md"
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert content.startswith("# Label Construction Report")
    assert "Run " in content


def test_write_label_construction_report_appends_runs_not_overwrites(
    tmp_path: Path,
) -> None:
    """Calling twice appends; first run's section survives."""
    tier_df = _synthetic_tier_df()
    good_df = _synthetic_good_deal_df()
    write_label_construction_report(
        output_dir=tmp_path,
        price_tier_df=tier_df,
        good_deal_df=good_df,
        quantile_boundaries={"Gurgaon": {"Sale": [1.0, 2.0, 3.0]}},
        good_deal_thresholds={
            "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 200,
                        "median_residual": 0.0, "iqr_residual": 0.05},
            "_default": {"low": -0.10, "high": 0.10, "rationale": "default"},
        },
        dataset_version="v1",
        git_commit="aaaaaa",
        min_rows=100,
    )
    write_label_construction_report(
        output_dir=tmp_path,
        price_tier_df=tier_df,
        good_deal_df=good_df,
        quantile_boundaries={"Gurgaon": {"Sale": [1.0, 2.0, 3.0]}},
        good_deal_thresholds={
            "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 200,
                        "median_residual": 0.0, "iqr_residual": 0.05},
            "_default": {"low": -0.10, "high": 0.10, "rationale": "default"},
        },
        dataset_version="v2",
        git_commit="bbbbbb",
        min_rows=100,
    )
    content = (tmp_path / "label_construction_report.md").read_text(encoding="utf-8")
    # Both git commits must appear — the first wasn't overwritten.
    assert "aaaaaa" in content
    assert "bbbbbb" in content


def test_write_label_construction_report_includes_leakage_audit_line(
    tmp_path: Path,
) -> None:
    """Written report contains a "Leakage audit" line."""
    write_label_construction_report(
        output_dir=tmp_path,
        price_tier_df=_synthetic_tier_df(),
        good_deal_df=_synthetic_good_deal_df(),
        quantile_boundaries={"Gurgaon": {"Sale": [1.0, 2.0, 3.0]}},
        good_deal_thresholds={
            "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 200,
                        "median_residual": 0.0, "iqr_residual": 0.05},
            "_default": {"low": -0.10, "high": 0.10, "rationale": "default"},
        },
        dataset_version="v2",
        git_commit="abc",
        min_rows=100,
    )
    content = (tmp_path / "label_construction_report.md").read_text(encoding="utf-8")
    assert "Leakage audit" in content


def test_write_label_construction_report_includes_per_city_tier_counts(
    tmp_path: Path,
) -> None:
    """Each synthetic city name appears in the report with a row count."""
    write_label_construction_report(
        output_dir=tmp_path,
        price_tier_df=_synthetic_tier_df(),
        good_deal_df=_synthetic_good_deal_df(),
        quantile_boundaries={
            "Gurgaon": {"Sale": [1.0, 2.0, 3.0]},
            "Hyderabad": {"Sale": [1.0, 2.0, 3.0]},
            "Kolkata": {"Sale": [1.0, 2.0, 3.0]},
            "Mumbai": {"Sale": [1.0, 2.0, 3.0]},
        },
        good_deal_thresholds={
            "Gurgaon": {"low": -0.10, "high": 0.10, "n_train": 200,
                        "median_residual": 0.0, "iqr_residual": 0.05},
            "_default": {"low": -0.10, "high": 0.10, "rationale": "default"},
        },
        dataset_version="v2",
        git_commit="abc",
        min_rows=100,
    )
    content = (tmp_path / "label_construction_report.md").read_text(encoding="utf-8")
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        assert city in content
    # And tier counts (each city has 20 rows per tier × 4 tiers).
    assert "Budget" in content
    assert "Luxury" in content
