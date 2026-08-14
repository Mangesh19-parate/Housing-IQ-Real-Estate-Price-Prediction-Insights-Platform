"""Integration tests for scripts/train_price_model.py (Spec 13 Phase C).

Each test:
    1. Materializes a tiny synthetic ``clean_listings.parquet`` in
       ``tmp_path`` that satisfies the 16-field input contract + has
       ``is_outlier`` + ``locality`` + Sale/Rent rows.
    2. Runs Step 12's ``build_features.py`` (with
       ``HOUSINGIQ_ARTIFACT_DIR`` pointed at ``tmp_path/models``) to
       produce ``feature_pipeline_v1.pkl`` + ``feature_list_v1.json``.
    3. Runs the training script with the same env-var override +
       ``HOUSINGIQ_PROCESSED_DIR`` at ``tmp_path/data/processed``.
    4. Asserts the expected artifacts land + the expected rows in
       ``model_registry.csv``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Repo root = parent of tests/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_FEATURES = _REPO_ROOT / "scripts" / "build_features.py"
_TRAIN = _REPO_ROOT / "scripts" / "train_price_model.py"


# ---------------------------------------------------------------------------
# Synthetic-data fixture
# ---------------------------------------------------------------------------


def _make_synthetic_parquet(
    parquet_path: Path,
    n_sale: int = 520,
    n_rent: int = 520,
    seed: int = 0,
) -> None:
    """Write a minimal-but-valid synthetic clean_listings.parquet.

    ``n_sale``/``n_rent`` defaults to 520 -- enough rows to clear
    ``RENT_MIN_ROWS = 500`` so both pipelines train. Tests that
    specifically want the skip path override ``n_rent=30`` etc.

    The frame MUST have every column in ``INPUT_FIELDS_V3`` plus
    ``is_outlier``, ``locality``, ``price_inr``, ``amenities_list``,
    ``features_list``. Values are intentionally simple -- the script
    only needs to fit + score, not predict well.
    """
    rng = __import__("numpy").random.default_rng(seed)
    rows = []

    for ttype, n in (("Sale", n_sale), ("Rent", n_rent)):
        for i in range(n):
            city = "Gurgaon" if i % 2 == 0 else "Hyderabad"
            sector = f"sector {i % 5 + 1}"
            locality = f"locality_{i % 3}"
            row = {
                # INPUT_FIELDS_V3 (16 fields):
                "property_type": "flat" if i % 2 == 0 else "house",
                "sector": sector,
                "city": city,
                "transact_type": ttype,
                "bedRoom": int(rng.integers(1, 5)),
                "bathroom": int(rng.integers(1, 4)),
                "balcony": ["0", "1", "2", "3", "3+"][i % 5],
                "agePossession": [
                    "New Property",
                    "Relatively New",
                    "Moderately Old",
                    "Old Property",
                    "Under Construction",
                ][i % 5],
                "built_up_area": float(rng.uniform(500, 3000)),
                "servant_room": bool(i % 3 == 0),
                "store_room": bool(i % 4 == 0),
                "furnishing_type": ["Unfurnished", "Semifurnished", "Furnished"][
                    int(rng.integers(0, 3))
                ],
                "luxury_category": ["Low", "Medium", "High"][i % 3],
                "floor_category": [
                    "Low Floor",
                    "Mid Floor",
                    "High Floor",
                ][i % 3],
                "facing": ["North", "South", "East", "West"][i % 4],
                "amenities": [],  # cleaning layer would expand to list
                # Extra columns the cleaning layer emits + this script needs:
                "amenities_list": [],
                "features_list": [],
                "locality": locality,
                "price_inr": float(
                    rng.uniform(5_000_000, 50_000_000)
                    if ttype == "Sale"
                    else rng.uniform(10_000, 100_000)
                ),
                "is_outlier": False,
                # ``price_per_sqft`` is what LocalityAggregator.fit reads
                # directly -- populated by the cleaning layer's
                # ``_derive_price_per_sqft`` step.
                "price_per_sqft": 0.0,  # filled in below
                # ``floor_num``/``total_floor`` derive ``floor_ratio``;
                # without them ``derive_row_features`` emits ``pd.NA``
                # and StandardScaler errors out on partial_fit.
                "floor_num": float(rng.integers(1, 20)),
                "total_floor": float(rng.integers(5, 30)),
                # Optional Step 07 leftovers -- present but harmless.
                "outlier_reasons": [],
            }
            row["price_per_sqft"] = row["price_inr"] / max(row["built_up_area"], 1.0)
            rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def _run_script(script: Path, env: dict, cwd: Path) -> subprocess.CompletedProcess:
    """Invoke ``script`` with ``env`` (which augments the current env)."""
    import os
    full_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Per-test scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def script_env(tmp_path, request):
    """Set up tmp_path layout + env vars + pre-materialized feature artifacts.

    Tests can override the row counts via indirect parameterization:

        @pytest.mark.parametrize("script_env", [{"n_rent": 30}], indirect=True)
        def test_xxx(script_env): ...

    Returns a dict with paths + the build_features subprocess result so
    tests can assert the prerequisite ran cleanly.
    """
    overrides = getattr(request, "param", {}) or {}
    processed_dir = tmp_path / "data" / "processed"
    artifact_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    parquet = processed_dir / "clean_listings.parquet"
    _make_synthetic_parquet(
        parquet,
        n_sale=overrides.get("n_sale", 520),
        n_rent=overrides.get("n_rent", 520),
    )

    env = {
        "HOUSINGIQ_PROCESSED_DIR": str(processed_dir),
        "HOUSINGIQ_ARTIFACT_DIR": str(artifact_dir),
        "HOUSINGIQ_REPORT_PATH": str(processed_dir / "feature_selection_report.md"),
        "HOUSINGIQ_REGISTRY_CSV_PATH": str(processed_dir.parent / "model_registry.csv"),
    }

    # Materialize Step 12 artifacts first.
    build_res = _run_script(_BUILD_FEATURES, env, cwd=_REPO_ROOT)
    assert build_res.returncode == 0, (
        f"build_features.py failed:\nSTDOUT:\n{build_res.stdout}\n"
        f"STDERR:\n{build_res.stderr}"
    )
    assert (artifact_dir / "feature_pipeline_v1.pkl").exists()
    assert (artifact_dir / "feature_list_v1.json").exists()

    return {
        "env": env,
        "processed_dir": processed_dir,
        "artifact_dir": artifact_dir,
        "parquet": parquet,
        "build_stdout": build_res.stdout,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_price_model_script_runs_end_to_end(script_env):
    env = script_env["env"]
    artifact_dir = script_env["artifact_dir"]
    res = _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    assert res.returncode == 0, (
        f"train_price_model.py failed:\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}"
    )

    # Both pipelines trained.
    assert (artifact_dir / "price_model_sale_v1.pkl").exists()
    assert (artifact_dir / "price_model_rent_v1.pkl").exists()
    # Metrics JSON parses.
    metrics = json.loads((artifact_dir / "metrics_v1.json").read_text())
    assert set(metrics["sale"].keys()) >= {"candidates", "chosen_model"}
    assert set(metrics["rent"].keys()) >= {"candidates", "chosen_model"}


def test_metrics_v1_json_contains_all_six_candidates(script_env):
    env = script_env["env"]
    artifact_dir = script_env["artifact_dir"]
    _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    metrics = json.loads((artifact_dir / "metrics_v1.json").read_text())
    assert set(metrics["sale"]["candidates"]) == {
        "linear", "ridge", "lasso", "random_forest",
        "gradient_boosting", "xgboost",
    }


def test_metrics_v1_json_has_git_commit(script_env):
    import subprocess
    env = script_env["env"]
    artifact_dir = script_env["artifact_dir"]
    _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    metrics = json.loads((artifact_dir / "metrics_v1.json").read_text())
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT
    ).decode().strip()[:12]
    assert metrics["git_commit"] == head_sha


def test_train_price_model_script_is_idempotent_on_rerun(script_env):
    """Re-running leaves the registry CSV without duplicates."""
    env = script_env["env"]
    registry = Path(env["HOUSINGIQ_REGISTRY_CSV_PATH"])

    _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    rows_after_first = len(registry.read_text(encoding="utf-8").splitlines())

    _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    rows_after_second = len(registry.read_text(encoding="utf-8").splitlines())

    # First run: 1 header + 2 data rows = 3 lines.
    # Second run with same git commit: still 3 lines (idempotent).
    assert rows_after_first == rows_after_second, (
        f"registry grew on rerun: {rows_after_first} -> {rows_after_second}"
    )


def test_train_price_model_script_appends_report(script_env):
    env = script_env["env"]
    report_path = Path(env["HOUSINGIQ_REPORT_PATH"])
    _run_script(_TRAIN, env, cwd=_REPO_ROOT)

    # The report should now contain Round 2 + Round 3 markers.
    text = report_path.read_text(encoding="utf-8")
    assert "## Round 2" in text and "Tree-based" in text
    assert "## Round 3" in text and "SHAP" in text
    assert "## Final feature list & rationale" in text


@pytest.mark.parametrize("script_env", [{"n_rent": 30}], indirect=True)
def test_train_price_model_script_skips_rent_when_too_small(script_env):
    """When Rent subset has fewer than RENT_MIN_ROWS rows, no .pkl is written.

    The Sale side still trains (520 > 500), so the sale .pkl + registry row
    land normally; the rent pipeline reports ``skipped=True`` in the metrics.
    """
    env = script_env["env"]
    artifact_dir = script_env["artifact_dir"]
    res = _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    assert res.returncode == 0, res.stderr

    assert (artifact_dir / "price_model_sale_v1.pkl").exists()
    assert not (artifact_dir / "price_model_rent_v1.pkl").exists()

    metrics = json.loads((artifact_dir / "metrics_v1.json").read_text())
    assert "candidates" in metrics["sale"]
    assert metrics["rent"].get("skipped") is True
    # Sale registry row was appended; rent was not.
    registry = Path(env["HOUSINGIQ_REGISTRY_CSV_PATH"]).read_text(encoding="utf-8")
    assert "price_model_sale" in registry
    assert "price_model_rent" not in registry


def test_training_script_does_not_log_contact_fields(script_env):
    """Rules sec. 1.1: no contact/dealer/phone/email/photo/url/spid in logs."""
    env = script_env["env"]
    res = _run_script(_TRAIN, env, cwd=_REPO_ROOT)
    combined = res.stdout + "\n" + res.stderr
    matches = re.findall(
        r"(?i)(contact|dealer|phone|email|photo|url|spid)", combined
    )
    assert not matches, f"contact-field leak in logs: {matches[:5]}..."
