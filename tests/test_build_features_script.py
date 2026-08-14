"""End-to-end test for scripts/build_features.py (Phase 5)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "build_features.py"
_PYTHON = sys.executable


def _make_synthetic_parquet(path: Path, n: int = 30) -> None:
    """Write a minimal synthetic ``clean_listings.parquet`` to ``path``."""
    rng = np.random.default_rng(42)
    cities = ["Gurgaon", "Hyderabad", "Mumbai", "Kolkata"]
    rows = []
    for i in range(n):
        city = cities[i % len(cities)]
        rows.append(
            {
                "property_type": "flat",
                "sector": f"sector_{i % 5}",
                "city": city,
                "locality": f"L_{i % 4}",
                "transact_type": "Sale",
                "bedRoom": int(rng.integers(1, 5)),
                "bathroom": int(rng.integers(1, 5)),
                "balcony": "2",
                "balconies": 2,
                "agePossession": "Relatively New",
                "age_bucket": "Relatively New",
                "built_up_area": float(rng.uniform(500, 3000)),
                "area_sqft": float(rng.uniform(500, 3000)),
                "servant_room": False,
                "store_room": False,
                "furnish": "Semifurnished",
                "furnishing_type": "Semifurnished",
                "facing": "North",
                "floor_num": int(rng.integers(1, 20)),
                "floor_category": "Mid Floor",
                "luxury_category": "Medium",
                "total_floor": 20,
                "price_inr": float(rng.uniform(5_000_000, 50_000_000)),
                "price_per_sqft": float(rng.uniform(5000, 25000)),
                "floor_ratio": 0.5,
                "features_list": ["F1"],
                "amenities_list": ["Swimming Pool", "Club House"],
                "n_amenities": 2,
                "n_features": 1,
                "is_outlier": False,
                "was_missing_bedRoom": False,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_features_script_runs_end_to_end_on_synthetic_parquet(
    tmp_path: Path,
) -> None:
    """Run the full script against a synthetic Parquet and verify artifacts."""
    # 1. Write a synthetic Parquet.
    parquet_path = tmp_path / "clean_listings.parquet"
    _make_synthetic_parquet(parquet_path, n=40)

    # 2. Run the script with HOUSINGIQ_PROCESSED_DIR set to tmp_path
    #    and a private artifact dir.
    artifact_dir = tmp_path / "models"
    env = {
        **__import__("os").environ,
        "HOUSINGIQ_PROCESSED_DIR": str(tmp_path),
    }
    proc = subprocess.run(
        [
            _PYTHON,
            str(_SCRIPT),
            "--parquet",
            str(parquet_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"build_features.py exited {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # 3. Assert the four expected artifacts land:
    #    - feature_pipeline_v1.pkl
    #    - feature_list_v1.json
    #    - feature_selection_report.md (also produced by the report script,
    #      but the script logs INFO messages about both)
    assert (artifact_dir / "feature_pipeline_v1.pkl").exists()
    assert (artifact_dir / "feature_list_v1.json").exists()
    with open(artifact_dir / "feature_list_v1.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "feature_names" in payload
    assert "engineered_feature_recipe" in payload
    assert len(payload["engineered_feature_recipe"]) == 11


def test_build_features_script_handles_missing_parquet(tmp_path: Path) -> None:
    """Missing Parquet yields a nonzero exit code."""
    proc = subprocess.run(
        [_PYTHON, str(_SCRIPT), "--parquet", str(tmp_path / "nope.parquet")],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode != 0
