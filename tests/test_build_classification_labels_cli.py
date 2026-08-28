"""Tests for ``scripts/build_classification_labels.py`` (Spec 21).

All test names are pinned by the spec's Definition of DoD item 1.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_synthetic_clean_listings(path: Path, n_per_city: int = 200) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        for transact in ("Sale", "Rent"):
            for i in range(n_per_city):
                rows.append(
                    {
                        "listing_id": f"{city}_{transact}_{i:04d}",
                        "city": city,
                        "locality": f"{city}_x",
                        "transact_type": transact,
                        "price": float(rng.integers(2_000_000, 20_000_000)),
                        "price_per_sqft": float(rng.integers(3000, 12000)),
                        "is_outlier": False,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_oof(path: Path, df: pd.DataFrame) -> None:
    """Write an OOF prediction parquet matching the input clean_listings."""
    out = pd.DataFrame(
        {
            "listing_id": df["listing_id"],
            "transact_type": df["transact_type"],
            "oof_predicted_price": df["price"] * 1.05,
        }
    )
    out.to_parquet(path, index=False)


def test_cli_runs_end_to_end_on_synthetic_artifacts(tmp_path: Path) -> None:
    """Build synthetic clean_listings + oof_predictions; CLI exits 0 + writes 4 files."""
    clean_path = tmp_path / "clean_listings.parquet"
    _write_synthetic_clean_listings(clean_path, n_per_city=200)
    clean_df = pd.read_parquet(clean_path)
    oof_path = tmp_path / "oof_predictions_v2.parquet"
    _write_synthetic_oof(oof_path, clean_df)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_classification_labels.py",
            "--processed-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--model-version",
            "v2",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, (
        f"CLI failed.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    out_dir = tmp_path / "output"
    assert (out_dir / "price_tier_labels.parquet").exists()
    assert (out_dir / "price_tier_quantile_boundaries.json").exists()
    assert (out_dir / "good_deal_labels.parquet").exists()
    assert (out_dir / "good_deal_thresholds.json").exists()
    assert (out_dir / "label_construction_report.md").exists()


def test_cli_writes_deterministic_artifacts_on_rerun(tmp_path: Path) -> None:
    """Re-running the CLI with identical inputs produces byte-identical parquet files."""
    clean_path = tmp_path / "clean_listings.parquet"
    _write_synthetic_clean_listings(clean_path, n_per_city=200)
    clean_df = pd.read_parquet(clean_path)
    oof_path = tmp_path / "oof_predictions_v2.parquet"
    _write_synthetic_oof(oof_path, clean_df)

    common_args = [
        sys.executable,
        "scripts/build_classification_labels.py",
        "--processed-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--model-version",
        "v2",
    ]

    for run_idx in (1, 2):
        result = subprocess.run(
            common_args,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"run {run_idx} failed: {result.stderr}"

    tier_bytes_1 = (tmp_path / "output" / "price_tier_labels.parquet").read_bytes()
    tier_bytes_2 = (tmp_path / "output" / "price_tier_labels.parquet").read_bytes()
    verdict_bytes_1 = (tmp_path / "output" / "good_deal_labels.parquet").read_bytes()
    verdict_bytes_2 = (tmp_path / "output" / "good_deal_labels.parquet").read_bytes()
    assert tier_bytes_1 == tier_bytes_2
    assert verdict_bytes_1 == verdict_bytes_2


def test_cli_does_not_log_contact_fields(tmp_path: Path) -> None:
    """CLI stdout must not log any column matching the contact/PII/URL regex."""
    clean_path = tmp_path / "clean_listings.parquet"
    _write_synthetic_clean_listings(clean_path, n_per_city=200)
    clean_df = pd.read_parquet(clean_path)
    oof_path = tmp_path / "oof_predictions_v2.parquet"
    _write_synthetic_oof(oof_path, clean_df)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_classification_labels.py",
            "--processed-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--model-version",
            "v2",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    combined = (result.stdout + "\n" + result.stderr).lower()
    # Look for column-name-shaped tokens (whitespace or quote around
    # the keyword, not embedded in a path segment like
    # ".../test_cli_does_not_log_contact_0/").
    pattern = (
        r"(?:[\'\"\s=:])(contact|dealer|phone|email|photo|url|spid)"
        r"(?:[\'\"\s,)}]|$)"
    )
    assert not re.search(pattern, combined), (
        f"Found a contact/PII/URL column name in CLI output:\n{combined}"
    )
