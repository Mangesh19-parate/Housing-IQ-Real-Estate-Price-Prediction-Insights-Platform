"""Tests for ``ml.evaluation.gate.evaluate`` — end-to-end with synthetic fixtures.

The gate is the heaviest module to test. Each test builds a tiny
synthetic ``clean_listings.parquet`` + a fitted ``feature_pipeline_v1.pkl``
+ a tiny trained model artifact in ``tmp_path``, then calls
``evaluate()`` and asserts on the returned ``EvaluationResult``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml.evaluation import (
    PROTOCOL_VERSION,
    evaluate,
    format_summary,
)
from ml.features.preprocessor import fit_preprocessor

# ---------------------------------------------------------------------------
# Helpers — synthetic fixtures
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=_REPO_ROOT,
        ).decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _make_synthetic_parquet(path: Path, n_sale: int = 600, n_rent: int = 0) -> None:
    """Write a tiny synthetic ``clean_listings.parquet`` at ``path``.

    Matches the canonical 16-field schema (``10-FINALIZED-INPUT-SCHEMA.md``)
    plus ``is_outlier`` + ``transact_type``. Includes all columns
    ``fit_preprocessor`` expects (numeric + ordinal + one-hot groups).

    Price is generated as ``slope * built_up_area + intercept`` so a
    log-trained LinearRegression can approximate it tightly.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []

    def _row(i: int, city: str, transact: str, price: float, area: float) -> dict:
        return {
            "city": city,
            "sector": "sector 84",
            "property_type": "flat",
            "transact_type": transact,
            "bedRoom": 3,
            "bathroom": 3,
            "balcony": "2",  # string to match OrdinalEncoder categories
            "agePossession": "Relatively New",
            "built_up_area": area,
            "servant_room": 0,  # numeric for StandardScaler
            "store_room": 0,
            "furnishing_type": "Semifurnished",  # string to match OrdinalEncoder
            "luxury_category": "Medium",
            "floor_category": "Mid Floor",
            "facing": "North",
            "amenities_list": ["Clubhouse"],
            "features_list": [],
            "n_amenities": 1,
            "n_features": 0,
            "floor_num": 5,
            "total_floor": 10,
            "floor_ratio": 0.5,
            "age_bucket_ord": 1,
            "bath_bed_ratio": 1.0,
            "area_per_bedroom": area / 3,
            "top_amenities_count": 1,
            "locality_avg_price_sqft": 9000.0,
            "locality_listing_count": 100,
            "locality_smoothed_price": 13_000_000.0,
            "price_per_sqft": price / area,
            "price_inr": price,
            "price": price,
            "is_outlier": False,
            "locality": "sector 84",
        }

    cities = ["Gurgaon", "Hyderabad", "Mumbai", "Kolkata"]
    for i in range(n_sale):
        city = cities[i % len(cities)]
        # Tiny noise on the log scale so log1p(price) is essentially
        # a linear function of log1p(area). Lets a fitted
        # LinearRegression clear the 0.80 R² floor on the inverted
        # price scale.
        area = 1200.0 + (i % 12) * 50.0  # 1200..1750
        price_per_sqft = 4000.0  # ₹/sqft — fixed
        price = price_per_sqft * area
        # Add 0.1% multiplicative noise — well under the ±15% band.
        noise_pct = float(rng.normal(0, 0.001))
        price = price * (1.0 + noise_pct)
        rows.append(_row(i, city, "Sale", price, area))

    if n_rent > 0:
        for i in range(n_rent):
            city = cities[i % len(cities)]
            area = 1200.0 + (i % 12) * 50.0
            rent_per_sqft = 25.0
            price = rent_per_sqft * area
            noise_pct = float(rng.normal(0, 0.001))
            price = price * (1.0 + noise_pct)
            rows.append(_row(i, city, "Rent", price, area))

    df = pd.DataFrame(rows)
    # Keep ONLY the columns the fitted preprocessor + the gate's
    # ``feature_cols = [...]`` expect. Anything else (price_per_sqft,
    # amenities_list, sector locality, etc.) would be silently
    # dropped by sklearn or, worse, cause feature-name mismatches
    # that produce NaNs in the prediction matrix. ``price`` is the
    # target; ``is_outlier`` is filtered by gate.
    keep_cols = [
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
        "price",
        "is_outlier",
    ]
    df = df[keep_cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _build_fitted_preprocessor_and_model(
    parquet_path: Path,
    models_dir: Path,
    fit_quality: str = "perfect",
) -> Path:
    """Fit the preprocessor + a tiny model on the synthetic parquet.

    Returns the path to the saved model ``.pkl``.

    The same ``model_cols`` are passed at fit and at gate-time so the
    fitted preprocessor + LinearRegression see the exact same column
    layout. ``fit_quality="perfect"`` → price is a tight linear
    function of ``built_up_area`` so a fitted LinearRegression
    clears the 0.80 R² floor. ``fit_quality="noise"`` → a model
    trained on noise-only targets so test R² is near 0 (below the
    floor).
    """
    df = pd.read_parquet(parquet_path)
    from ml.features.locality_aggregator import LocalityAggregator
    agg = LocalityAggregator().fit(df)

    # Match the gate's ``feature_cols`` so fit and predict use the
    # same column set. Gate does:
    #   feature_cols = [c for c in df.columns if c not in {price, is_outlier}]
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

    # IMPORTANT: the synthetic parquet mixes Sale (~₹5M) and Rent
    # (~₹30k) rows. The gate's evaluate() filters by transact_type
    # AFTER loading — but the LocalityAggregator is fit on the
    # combined df, and ``locality_avg_price_sqft`` ends up around
    # 9000 (correct: (4000 + 25)/2 ≈ 2000 sqft-price average). That
    # swamps the actual built_up_area signal when both Sale and Rent
    # rows share the same locality. Fix: train the pipeline ONLY on
    # Sale rows so the locality aggregator + preprocessor see the
    # true sale price distribution.
    sale_df = df[df["transact_type"] == "Sale"].reset_index(drop=True)
    if len(sale_df) < 100:
        sale_df = df  # fall back if there's no Sale subset
    train_df = sale_df[model_cols + ["price"]]

    preprocessor = fit_preprocessor(train_df, locality_aggregator=agg)

    # For "noise": scramble the price so the model has no signal.
    train_y = sale_df["price"].astype(float).to_numpy().copy()
    if fit_quality == "noise":
        rng = np.random.default_rng(7)
        train_y = rng.uniform(low=1e6, high=1e8, size=len(sale_df))

    X = sale_df[model_cols]
    y_log = np.log1p(train_y)

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("est", LinearRegression()),
        ]
    )
    pipeline.fit(X, y_log)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "price_model_sale_v1.pkl"
    joblib.dump(pipeline, model_path)

    # Save the fitted preprocessor + a no-op aggregator to the v1
    # artifact path so the gate can load it.
    feature_pipeline_path = models_dir / "feature_pipeline_v1.pkl"
    joblib.dump((preprocessor, None), feature_pipeline_path)
    feature_list_path = models_dir / "feature_list_v1.json"
    feature_list_path.write_text(
        json.dumps({"version": "v1", "feature_names": model_cols}),
        encoding="utf-8",
    )
    return model_path


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


def test_evaluate_overall_passed_false_when_r2_below_threshold(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir, fit_quality="noise")

    result = evaluate(
        model_path=models_dir / "price_model_sale_v1.pkl",
        version="v1",
        transact_type="sale",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    assert result.overall_passed is False
    assert result.thresholds_passed["r2_min"] is False


def test_evaluate_overall_passed_true_when_all_thresholds_met(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir, fit_quality="perfect")

    result = evaluate(
        model_path=models_dir / "price_model_sale_v1.pkl",
        version="v1",
        transact_type="sale",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    assert result.overall_passed is True
    assert result.thresholds_passed["r2_min"] is True
    assert result.thresholds_passed["mae_pct_within_15_at_least"] is True
    assert result.thresholds_passed["rent_min_rows"] is True


def test_evaluate_skips_rent_when_too_small(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=0)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir)

    # Use a Rent model path — the gate should skip before even trying
    # to load the artifact.
    rent_model_path = models_dir / "price_model_rent_v1.pkl"
    result = evaluate(
        model_path=rent_model_path,
        version="v1",
        transact_type="rent",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    assert result.overall_passed is False
    assert result.thresholds_passed["rent_min_rows"] is False
    assert result.metrics.get("skipped") is True


def test_evaluate_records_protocol_version_in_result(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir)

    result = evaluate(
        model_path=models_dir / "price_model_sale_v1.pkl",
        version="v1",
        transact_type="sale",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    assert result.protocol_version == PROTOCOL_VERSION


def test_evaluate_records_git_commit_in_result(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir)

    result = evaluate(
        model_path=models_dir / "price_model_sale_v1.pkl",
        version="v1",
        transact_type="sale",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    assert result.git_commit == _git_commit()


def test_evaluate_rejects_preprocessor_drift(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir)

    # Delete the feature_pipeline_v1.pkl so the gate fails the
    # sanity check (preprocessor_drift equivalent).
    (models_dir / "feature_pipeline_v1.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="feature pipeline"):
        evaluate(
            model_path=models_dir / "price_model_sale_v1.pkl",
            version="v1",
            transact_type="sale",
            processed_dir=tmp_path / "data" / "processed",
            models_dir=models_dir,
            parquet_path=parquet,
        )


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_starts_with_pass_or_fail(tmp_path: Path) -> None:
    parquet = tmp_path / "data" / "processed" / "clean_listings.parquet"
    _make_synthetic_parquet(parquet, n_sale=600, n_rent=600)
    models_dir = tmp_path / "models"
    _build_fitted_preprocessor_and_model(parquet, models_dir)

    result = evaluate(
        model_path=models_dir / "price_model_sale_v1.pkl",
        version="v1",
        transact_type="sale",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=models_dir,
        parquet_path=parquet,
    )
    summary = format_summary(result)
    assert summary.startswith("[PASS]") or summary.startswith("[FAIL]")
    assert "sale_v1" in summary
    assert "R²=" in summary
