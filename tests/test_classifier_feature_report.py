"""Tests for ``ml.classification.feature_report`` (Spec 22)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.classification.feature_report import (
    write_feature_frame_artifact,
    write_feature_frame_report,
)

# ----- synthetic fixture helpers -----


def _make_feature_frame(n_per_city: int = 20) -> pd.DataFrame:
    """Build a small synthetic classifier feature frame."""
    rng = np.random.default_rng(123)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    rows = []
    for city in cities:
        for _ in range(n_per_city):
            is_outlier = bool(rng.random() < 0.10)
            rows.append(
                {
                    "listing_id": f"F{rng.integers(1_000_000):06d}",
                    "city": city,
                    "split": rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]),
                    "is_outlier": is_outlier,
                    "property_type": "flat",
                    "sector": "Sector 1",
                    "bedRoom": 3,
                    "bathroom": 2,
                    "balcony": "2",
                    "agePossession": "Relatively New",
                    "built_up_area": float(rng.uniform(800, 3000)),
                    "servant_room": False,
                    "store_room": False,
                    "furnishing_type": "Semifurnished",
                    "luxury_category": "Medium",
                    "floor_category": "Mid Floor",
                    "facing": "North",
                    "locality_listing_count": int(rng.integers(10, 200)),
                }
            )
    return pd.DataFrame(rows)


def _make_tier_labels(n_per_city: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    tiers = ["budget", "mid", "premium", "luxury"]
    rows = []
    for city in cities:
        for _ in range(n_per_city):
            rows.append(
                {
                    "listing_id": f"F{rng.integers(1_000_000):06d}",
                    "city": city,
                    "price_tier": rng.choice(tiers),
                }
            )
    return pd.DataFrame(rows)


def _make_good_deal_labels(n_per_city: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    verdicts = ["good_deal", "fair_deal", "overpriced"]
    rows = []
    for city in cities:
        for _ in range(n_per_city):
            rows.append(
                {
                    "listing_id": f"F{rng.integers(1_000_000):06d}",
                    "city": city,
                    "good_deal_verdict": rng.choice(verdicts),
                }
            )
    return pd.DataFrame(rows)


# ----- tests -----


def test_write_feature_frame_artifact_writes_parquet(tmp_path: Path) -> None:
    df = _make_feature_frame()
    out_path = write_feature_frame_artifact(df, tmp_path)
    assert out_path.exists()
    assert out_path.name == "feature_frame.parquet"
    # Round-trip
    back = pd.read_parquet(out_path)
    assert len(back) == len(df)
    assert list(back.columns) == list(df.columns)


def test_write_feature_frame_report_creates_header_on_first_call(
    tmp_path: Path,
) -> None:
    df = _make_feature_frame()
    tier = _make_tier_labels()
    good = _make_good_deal_labels()

    report_path = write_feature_frame_report(
        output_dir=tmp_path,
        feature_frame=df,
        price_tier_labels=tier,
        good_deal_labels=good,
        dataset_version="clean_listings.parquet:abc123",
        git_commit="def456",
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Feature-Frame Construction Report" in content
    assert "## Run" in content
    assert "dataset_version" in content
    assert "git_commit" in content
    assert "total rows:" in content
    assert "retained locality column: `locality_listing_count`" in content
    assert "### Kept columns" in content
    assert "### Dropped columns (leakage ban)" in content
    assert "### Per-city non-outlier row counts" in content
    assert "### Per-(city × price_tier) row counts" in content
    assert "### Per-(city × good_deal_verdict) row counts" in content
    assert "Leakage audit:" in content


def test_write_feature_frame_report_appends_on_subsequent_calls(
    tmp_path: Path,
) -> None:
    df = _make_feature_frame()
    tier = _make_tier_labels()
    good = _make_good_deal_labels()

    # First call
    write_feature_frame_report(
        output_dir=tmp_path,
        feature_frame=df,
        price_tier_labels=tier,
        good_deal_labels=good,
        dataset_version="v1",
        git_commit="a1",
    )
    # Second call
    write_feature_frame_report(
        output_dir=tmp_path,
        feature_frame=df,
        price_tier_labels=tier,
        good_deal_labels=good,
        dataset_version="v2",
        git_commit="b2",
    )

    content = (tmp_path / "feature_frame_report.md").read_text(encoding="utf-8")
    # Header should appear only once
    assert content.count("# Feature-Frame Construction Report") == 1
    # But two Run sections
    assert content.count("## Run") == 2
    assert "dataset_version: `v1`" in content
    assert "dataset_version: `v2`" in content


def test_write_feature_frame_report_handles_empty_label_frames(
    tmp_path: Path,
) -> None:
    df = _make_feature_frame()
    # Empty label frames
    tier = pd.DataFrame(columns=["listing_id", "city", "price_tier"])
    good = pd.DataFrame(columns=["listing_id", "city", "good_deal_verdict"])

    report_path = write_feature_frame_report(
        output_dir=tmp_path,
        feature_frame=df,
        price_tier_labels=tier,
        good_deal_labels=good,
        dataset_version="v1",
        git_commit="c3",
    )

    content = report_path.read_text(encoding="utf-8")
    assert "(no price_tier rows)" in content
    assert "(no good_deal rows)" in content
