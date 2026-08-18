"""Tests for ``scripts/evaluate_price_model.py`` — CLI subprocess runs.

Mirrors the ``script_env`` indirect-fixture pattern from
``tests/test_train_price_model_script.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "evaluate_price_model.py"
)
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_parquet(path: Path, n_sale: int = 600, n_rent: int = 600) -> None:
    """Build a tiny ``clean_listings.parquet`` matching the schema the
    gate expects (same shape as ``tests/test_gate.py``'s helper).
    """
    import pandas as pd

    rows: list[dict] = []

    def _row(i: int, city: str, transact: str, price: float, area: float) -> dict:
        return {
            "city": city,
            "property_type": "flat",
            "transact_type": transact,
            "bedRoom": 3,
            "bathroom": 3,
            "balcony": "2",
            "agePossession": "Relatively New",
            "built_up_area": area,
            "servant_room": 0,
            "store_room": 0,
            "furnishing_type": "Semifurnished",
            "luxury_category": "Medium",
            "floor_category": "Mid Floor",
            "facing": "North",
            "n_amenities": 1,
            "n_features": 0,
            "floor_ratio": 0.5,
            "age_bucket_ord": 1,
            "bath_bed_ratio": 1.0,
            "area_per_bedroom": area / 3,
            "top_amenities_count": 1,
            "locality_avg_price_sqft": 4000.0,
            "locality_listing_count": 100,
            "locality_smoothed_price": 5_900_000.0,
            "locality": "sector 84",
            "price_per_sqft": price / area,
            "price_inr": price,
            "price": price,
            "is_outlier": False,
        }

    cities = ["Gurgaon", "Hyderabad", "Mumbai", "Kolkata"]
    for i in range(n_sale):
        city = cities[i % len(cities)]
        area = 1200.0 + (i % 12) * 50.0
        price = 4000.0 * area
        rows.append(_row(i, city, "Sale", price, area))
    if n_rent > 0:
        for i in range(n_rent):
            city = cities[i % len(cities)]
            area = 1200.0 + (i % 12) * 50.0
            price = 25.0 * area
            rows.append(_row(i, city, "Rent", price, area))
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _build_synthetic_model(
    parquet_path: Path, models_dir: Path, fit_quality: str = "perfect"
) -> None:
    """Fit a tiny synthetic preprocessor + LinearRegression matching the
    fixture in ``tests/test_gate.py`` so the CLI exit code can be
    asserted deterministically.
    """
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from ml.features.locality_aggregator import LocalityAggregator
    from ml.features.preprocessor import fit_preprocessor

    df = __import__("pandas").read_parquet(parquet_path)
    agg = LocalityAggregator().fit(df)

    model_cols = [
        "city",
        "property_type",
        "transact_type",
        "bedRoom",
        "bathroom",
        "balcony",
        "agePossession",
        "built_up_area",
        "servant_room",
        "store_room",
        "furnishing_type",
        "luxury_category",
        "floor_category",
        "facing",
        "n_amenities",
        "n_features",
        "floor_ratio",
        "age_bucket_ord",
        "bath_bed_ratio",
        "area_per_bedroom",
        "top_amenities_count",
        "locality_avg_price_sqft",
        "locality_listing_count",
        "locality_smoothed_price",
        "locality",
        "price_per_sqft",
        "price_inr",
    ]

    sale_df = df[df["transact_type"] == "Sale"].reset_index(drop=True)
    if len(sale_df) < 100:
        sale_df = df
    train_df = sale_df[model_cols + ["price"]]
    preprocessor = fit_preprocessor(train_df, locality_aggregator=agg)

    train_y = sale_df["price"].astype(float).to_numpy().copy()
    if fit_quality == "noise":
        rng = np.random.default_rng(7)
        train_y = rng.uniform(low=1e6, high=1e8, size=len(sale_df))

    X = sale_df[model_cols]
    y_log = np.log1p(train_y)
    pipeline = Pipeline([("pre", preprocessor), ("est", LinearRegression())])
    pipeline.fit(X, y_log)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "price_model_sale_v1.pkl")
    joblib.dump((preprocessor, None), models_dir / "feature_pipeline_v1.pkl")
    (models_dir / "feature_list_v1.json").write_text(
        '{"version": "v1", "feature_names": []}', encoding="utf-8"
    )


@pytest.fixture
def script_env(tmp_path: Path) -> dict[str, str]:
    """Synthetic parquet + synthetic ``v1`` model artifact in ``tmp_path``.

    Returns the env vars the CLI expects (processed dir + models dir).
    """
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)

    models_dir = tmp_path / "models"
    _build_synthetic_model(parquet, models_dir)

    return {
        "HOUSINGIQ_PROCESSED_DIR": str(tmp_path / "data" / "processed"),
        "HOUSINGIQ_ARTIFACT_DIR": str(models_dir),
        "HOUSINGIQ_REPORT_PATH": str(
            tmp_path / "data" / "processed" / "feature_selection_report.md"
        ),
        "PYTHONPATH": str(_REPO_ROOT),
    }


def _run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """Run the CLI in a fresh subprocess and capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=_REPO_ROOT,
        env={**env, **__import__("os").environ},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------


_CONTACT_FIELD_REGEX = re.compile(
    r"(contact|dealer|phone|email|photo|url|spid)",
    re.IGNORECASE,
)


def test_cli_exits_zero_when_model_passes_thresholds(
    script_env: dict[str, str],
) -> None:
    """A perfect-fit synthetic model clears the gate → exit 0."""
    proc = _run_script(script_env, "--version", "v1", "--transact-type", "sale")
    assert proc.returncode == 0, (
        f"unexpected exit code {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_cli_emits_pass_fail_summary_per_transact_type(
    script_env: dict[str, str],
) -> None:
    """Stdout must include a ``[PASS|FAIL] sale_v1 ...`` line."""
    proc = _run_script(script_env, "--version", "v1", "--transact-type", "sale")
    assert proc.returncode == 0
    assert "[PASS]" in proc.stdout or "[FAIL]" in proc.stdout
    assert "sale_v1" in proc.stdout
    assert "R²=" in proc.stdout


def test_cli_exits_one_when_model_artifact_missing(
    tmp_path: Path,
) -> None:
    """If the artifact doesn't exist, the CLI must exit 1 — not crash."""
    empty_models = tmp_path / "models"
    empty_models.mkdir(parents=True, exist_ok=True)
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)

    proc = _run_script(
        {
            "HOUSINGIQ_PROCESSED_DIR": str(tmp_path / "data" / "processed"),
            "HOUSINGIQ_ARTIFACT_DIR": str(empty_models),
            "PYTHONPATH": str(_REPO_ROOT),
        },
        "--version",
        "v1",
        "--transact-type",
        "sale",
    )
    assert proc.returncode == 1
    assert "model_not_found" in proc.stdout


def test_cli_does_not_leak_pii_field_names(
    script_env: dict[str, str],
) -> None:
    """Rules §1.1: stdout must never contain PII/contact column names."""
    proc = _run_script(
        script_env, "--version", "v1", "--transact-type", "sale"
    )
    assert proc.returncode == 0
    # Strip the absolute paths the CLI logs (so the pytest-generated
    # tmpdir names — which can contain substrings like ``test_cli_*``
    # — don't trip the regex). Only the body of the log messages is
    # checked.
    import re as _re
    path_stripped = _re.sub(r"[A-Za-z]:\\[^\s]+", "<PATH>", proc.stdout)
    path_stripped += "\n" + _re.sub(r"[A-Za-z]:\\[^\s]+", "<PATH>", proc.stderr)
    assert not _CONTACT_FIELD_REGEX.search(path_stripped), (
        f"contact-field regex matched in CLI output:\n{path_stripped}"
    )
