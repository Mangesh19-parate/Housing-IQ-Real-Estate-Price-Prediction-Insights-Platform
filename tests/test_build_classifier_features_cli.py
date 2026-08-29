"""CLI tests for ``scripts/build_classifier_features.py`` (Spec 22)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Path to the CLI under test
_CLI = Path(__file__).resolve().parent.parent / "scripts" / "build_classifier_features.py"

# Reuse the PII regex from Spec 21's CLI test.
# Full regex with lookaround boundaries.
_CONTACT_FIELD_REGEX = r"(?:[\'\"\s=:])(contact|dealer|phone|email|photo|url|spid)(?:[\'\"\s,)}]|$)"


def _make_clean_listings_parquet(tmp_path: Path) -> Path:
    """Write a synthetic clean_listings.parquet with correct column names."""
    rng = np.random.default_rng(42)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    localities = ["Sector 1", "Sector 2", "Bandra", "Andheri", "Salt Lake", "Jubilee Hills"]
    rows = []
    idx = 0
    for city in cities:
        for _ in range(50):
            for _ in range(2):
                locality = localities[idx % len(localities)]
                idx += 1
                price_inr = float(rng.uniform(50_000_000.0, 200_000_000.0))
                area = float(rng.uniform(800.0, 3000.0))
                price_per_sqft = price_inr / area
                is_outlier = bool(rng.random() < 0.10)
                rows.append(
                    {
                        "listing_id": f"L{idx:06d}",
                        "city": city,
                        "locality": locality,
                        "transact_type": "Sale" if idx % 2 == 0 else "Rent",
                        "price_inr": price_inr,
                        "price_per_sqft": price_per_sqft,
                        "is_outlier": is_outlier,
                        "property_type": "flat",
                        "sector": locality,
                        "bedRoom": 3,
                        "bathroom": 2,
                        "balcony": "2",
                        "agePossession": "Relatively New",
                        "built_up_area": area,
                        "servant_room": False,
                        "store_room": False,
                        "furnishing_type": "Semifurnished",
                        "luxury_category": "Medium",
                        "floor_category": "Mid Floor",
                        "facing": "North",
                        "amenities": ["Clubhouse", "Swimming Pool"],
                        "amenities_list": ["Clubhouse", "Swimming Pool"],
                        "features_list": ["Power Backup"],
                        "floor_num": 5.0,
                        "total_floor": 10.0,
                        "was_missing_bedRoom": False,
                    }
                )
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "clean_listings.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _make_tier_labels_parquet(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    tiers = ["budget", "mid", "premium", "luxury"]
    rows = []
    for city in cities:
        for _ in range(50):
            rows.append(
                {
                    "listing_id": f"L{rng.integers(1_000_000):06d}",
                    "city": city,
                    "price_tier": rng.choice(tiers),
                }
            )
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "price_tier_labels.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _make_good_deal_labels_parquet(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    verdicts = ["good_deal", "fair_deal", "overpriced"]
    rows = []
    for city in cities:
        for _ in range(50):
            rows.append(
                {
                    "listing_id": f"L{rng.integers(1_000_000):06d}",
                    "city": city,
                    "good_deal_verdict": rng.choice(verdicts),
                }
            )
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "good_deal_labels.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _run_cli(
    processed_dir: Path, output_dir: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run the CLI as a subprocess."""
    cmd = [
        sys.executable,
        str(_CLI),
        "--processed-dir",
        str(processed_dir),
        "--output-dir",
        str(output_dir),
    ]
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=merged_env,
        cwd=processed_dir.parent,  # repo root
    )


def test_cli_returns_zero_on_success() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        processed_dir = tmp / "processed"
        output_dir = tmp / "classifier"
        processed_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        _make_clean_listings_parquet(processed_dir)
        _make_tier_labels_parquet(output_dir)
        _make_good_deal_labels_parquet(output_dir)

        result = _run_cli(processed_dir, output_dir)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "[OK]" in result.stdout
        assert "feature_frame rows=" in result.stdout


def test_cli_output_parquet_contains_no_price_derived_columns() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        processed_dir = tmp / "processed"
        output_dir = tmp / "classifier"
        processed_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        _make_clean_listings_parquet(processed_dir)
        _make_tier_labels_parquet(output_dir)
        _make_good_deal_labels_parquet(output_dir)

        result = _run_cli(processed_dir, output_dir)
        assert result.returncode == 0

        feature_path = output_dir / "feature_frame.parquet"
        assert feature_path.exists()

        # Read schema via pyarrow - this is the authoritative leakage check
        schema_names = set(pq.read_schema(feature_path).names)

        # Import the pinned leakage set
        sys.path.insert(0, str(_CLI.parent.parent))
        from ml.classification.features import DROPPED_FOR_LEAKAGE

        leaked = DROPPED_FOR_LEAKAGE & schema_names
        assert leaked == set(), f"Leakage columns found in artifact: {leaked}"

        # Also verify the retained column is present
        from ml.classification.features import RETAINED_LOCALITY_COLUMN

        assert RETAINED_LOCALITY_COLUMN in schema_names


def test_cli_stdout_contains_no_pii_like_terms() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        processed_dir = tmp / "processed"
        output_dir = tmp / "classifier"
        processed_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        _make_clean_listings_parquet(processed_dir)
        _make_tier_labels_parquet(output_dir)
        _make_good_deal_labels_parquet(output_dir)

        result = _run_cli(processed_dir, output_dir)
        assert result.returncode == 0

        combined = (result.stdout + "\n" + result.stderr).lower()
        matches = re.findall(_CONTACT_FIELD_REGEX, combined)
        # The regex captures the keyword in group 1
        pii_keywords = {"contact", "dealer", "phone", "email", "photo", "url", "spid"}
        pii_hits = [m for m in matches if m in pii_keywords]
        assert pii_hits == [], f"PII-like terms leaked in CLI output: {pii_hits}"


def test_cli_fails_gracefully_when_input_missing() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        processed_dir = tmp / "processed"
        output_dir = tmp / "classifier"
        processed_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        # Do NOT create clean_listings.parquet
        _make_tier_labels_parquet(output_dir)
        _make_good_deal_labels_parquet(output_dir)

        result = _run_cli(processed_dir, output_dir)
        assert result.returncode == 1
        assert "[FAIL]" in result.stdout
        assert "clean_listings_missing" in result.stdout
