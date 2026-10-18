"""``ml.analytics.cache_builders`` — Pure builder functions for the 13 analytics chart tiles.

Each function takes the full clean listings DataFrame and a city filter
("ALL", "Gurgaon", "Hyderabad", "Kolkata", "Mumbai") and returns a
JSON-serializable dict with the pre-shaped data structure expected by the
frontend Chart.js/Leaflet tiles.

All builders:
- Exclude outlier rows (is_outlier == True) from computations
- Return consistent structure: {labels: [...], series: [...]} or {markers: [...], bounds: [...]}
- Include metadata: source_parquet, source_parquet_hash, built_at, row_count, outlier_count
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from ml.cleaning.writers import CLEAN_LISTINGS_PARQUET_PATH

_LOG = logging.getLogger("ml.analytics.cache_builders")

# City slice keys — must match frontend filter values exactly
CITY_SLICES: Final[tuple[str, ...]] = ("ALL", "Gurgaon", "Hyderabad", "Kolkata", "Mumbai")

# Max markers for spatial map (clustered) — keep under Leaflet's comfortable limit
MAX_SPATIAL_MARKERS: Final[int] = 2000

# Max localities for price distribution box plot (top-N by volume)
MAX_LOCALITIES_DIST: Final[int] = 20

# Max points for scatter plots (downsample if needed)
MAX_SCATTER_POINTS: Final[int] = 500

# Correlation matrix: only numeric engineered features
CORRELATION_FEATURES: Final[list[str]] = [
    "price_inr",
    "area_sqft",
    "bedRoom",
    "bathroom",
    "balcony",
    "price_per_sqft",
    "floor_ratio",
    "n_amenities",
    "n_features",
    "floor_num",
    "total_floor",
]


def _get_source_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Compute standard metadata block for all cache files."""
    # Compute hash of source parquet for cache invalidation
    parquet_bytes = CLEAN_LISTINGS_PARQUET_PATH.read_bytes()
    parquet_hash = hashlib.sha256(parquet_bytes).hexdigest()

    total_rows = len(df)
    outlier_count = int(df["is_outlier"].sum()) if "is_outlier" in df.columns else 0

    return {
        "source_parquet": CLEAN_LISTINGS_PARQUET_PATH.name,
        "source_parquet_hash": f"sha256:{parquet_hash}",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "row_count": total_rows,
        "outlier_count": outlier_count,
    }


def _slice_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return DataFrame filtered by city, or full frame for 'ALL'."""
    if city == "ALL":
        return df
    return df[df["city"] == city]


def _downsample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Random downsample if frame exceeds max_rows."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude outliers and ensure numeric columns are properly typed."""
    base = df[~df["is_outlier"]].copy()

    # Ensure numeric columns are numeric (some may be string from facet decoding)
    numeric_cols = ["bedRoom", "bathroom", "balcony", "price_inr", "area_sqft",
                    "price_per_sqft", "floor_ratio", "n_amenities", "n_features",
                    "floor_num", "total_floor", "latitude", "longitude"]
    for col in numeric_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    return base


# ---------------------------------------------------------------------------
# Chart 01: Spatial Map — Leaflet clustered markers colored by price/sqft
# ---------------------------------------------------------------------------

def build_chart_01_spatial_map(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Spatial analysis — listing map with clustered markers.

    Returns: {markers: [{lat, lng, price_per_sqft, price_inr, locality, bhk}], bounds: [[min_lat, min_lng], [max_lat, max_lng]]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["latitude", "longitude", "price_per_sqft"])

    if base.empty:
        return {"markers": [], "bounds": [[0, 0], [0, 0]]}

    base = _downsample(base, MAX_SPATIAL_MARKERS)

    markers = []
    for _, row in base.iterrows():
        markers.append({
            "lat": float(row["latitude"]),
            "lng": float(row["longitude"]),
            "price_per_sqft": float(row["price_per_sqft"]),
            "price_inr": float(row["price_inr"]),
            "locality": str(row.get("locality", "Unknown")),
            "bhk": int(row["bedRoom"]) if pd.notna(row["bedRoom"]) else 0,
        })

    lats = [m["lat"] for m in markers]
    lngs = [m["lng"] for m in markers]
    bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]

    return {"markers": markers, "bounds": bounds}


# ---------------------------------------------------------------------------
# Chart 02: Price Distribution — Box plot by top-N localities
# ---------------------------------------------------------------------------

def build_chart_02_price_distribution(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Price distribution across sectors/localities — Box plot.

    Returns: {labels: [locality1, locality2, ...], series: [[prices_for_loc1], [prices_for_loc2], ...]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "locality"])

    if base.empty:
        return {"labels": [], "series": []}

    # Top-N localities by listing count
    top_localities = (
        base.groupby("locality").size()
        .sort_values(ascending=False)
        .head(MAX_LOCALITIES_DIST)
        .index.tolist()
    )

    labels = []
    series = []
    for loc in top_localities:
        loc_prices = base[base["locality"] == loc]["price_inr"].dropna().tolist()
        if len(loc_prices) >= 3:  # Need at least 3 for a meaningful box plot
            labels.append(str(loc))
            series.append([float(p) for p in loc_prices])

    return {"labels": labels, "series": series}


# ---------------------------------------------------------------------------
# Chart 03: Price vs Area — Scatter plot with trend line
# ---------------------------------------------------------------------------

def build_chart_03_price_vs_area(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Price vs. Area analysis — Scatter plot + trend line.

    Returns: {x: [area_sqft], y: [price_inr], trend: {slope, intercept, r_squared}}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["area_sqft", "price_inr"])
    base = base[(base["area_sqft"] > 0) & (base["price_inr"] > 0)]

    if base.empty:
        return {"x": [], "y": [], "trend": {"slope": 0, "intercept": 0, "r_squared": 0}}

    base = _downsample(base, MAX_SCATTER_POINTS)

    x = base["area_sqft"].astype(float).tolist()
    y = base["price_inr"].astype(float).tolist()

    # Linear regression for trend line
    x_arr = np.array(x)
    y_arr = np.array(y)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_pred = slope * x_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "x": x,
        "y": y,
        "trend": {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}
    }


# ---------------------------------------------------------------------------
# Chart 04: BHK Distribution — Pie/Donut chart
# ---------------------------------------------------------------------------

def build_chart_04_bhk_distribution(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Number of rooms (BHK) distribution — Pie/Donut chart.

    Returns: {labels: ["1 BHK", "2 BHK", ...], series: [count1, count2, ...]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["bedRoom"])

    if base.empty:
        return {"labels": [], "series": []}

    # Convert to numeric and round
    base["bedRoom_num"] = pd.to_numeric(base["bedRoom"], errors="coerce").round().astype("Int64")
    base = base.dropna(subset=["bedRoom_num"])

    counts = base["bedRoom_num"].value_counts().sort_index()
    labels = [f"{int(b)} BHK" for b in counts.index]
    series = counts.values.astype(int).tolist()

    return {"labels": labels, "series": series}


# ---------------------------------------------------------------------------
# Chart 05: Word Cloud — Top features from descriptions/amenities
# ---------------------------------------------------------------------------

def build_chart_05_word_cloud(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Top-feature word cloud — Word frequency data.

    Returns: {words: [{text, weight, source: "description"|"amenity"|"feature"}]}
    """
    from collections import Counter
    import re

    base = _slice_by_city(df, city)

    # Collect text from multiple sources
    words: Counter = Counter()

    # From description_clean
    if "description_clean" in base.columns:
        for desc in base["description_clean"].dropna():
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", str(desc).lower())
            # Filter common stopwords
            stopwords = {"the", "and", "for", "with", "are", "has", "this", "that",
                         "from", "have", "been", "will", "your", "more", "area",
                         "near", "very", "good", "best", "new", "old", "big", "small"}
            words.update(t for t in tokens if t not in stopwords)

    # From amenities_list
    if "amenities_list" in base.columns:
        for amenities in base["amenities_list"].dropna():
            if isinstance(amenities, list):
                for amenity in amenities:
                    words[str(amenity).lower().replace(" ", "_")] += 5  # Weight amenities higher

    # From features_list
    if "features_list" in base.columns:
        for features in base["features_list"].dropna():
            if isinstance(features, list):
                for feature in features:
                    words[str(feature).lower().replace(" ", "_")] += 3

    # Top 100 words
    top_words = words.most_common(100)
    max_weight = max((w for _, w in top_words), default=1)

    word_data = [
        {"text": word, "weight": weight / max_weight, "source": "mixed"}
        for word, weight in top_words
    ]

    return {"words": word_data}


# ---------------------------------------------------------------------------
# Chart 06: Correlation Heatmap — Numeric engineered features
# ---------------------------------------------------------------------------

def build_chart_06_correlation_heatmap(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Correlation heatmap — Pairwise correlations of numeric features.

    Returns: {features: [feature_names], matrix: [[corr_values]]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))

    # Select only available correlation features
    available = [f for f in CORRELATION_FEATURES if f in base.columns]
    base = base[available].dropna()

    if base.empty or len(available) < 2:
        return {"features": [], "matrix": []}

    corr_matrix = base.corr(method="pearson")
    # Round for cleaner JSON
    matrix = corr_matrix.round(3).values.tolist()
    features = corr_matrix.columns.tolist()

    return {"features": features, "matrix": matrix}


# ---------------------------------------------------------------------------
# Chart 07: City Price Comparison — Box/violin plot across 4 cities
# ---------------------------------------------------------------------------

def build_chart_07_city_price_comparison(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """City-wise price comparison — Box/violin plot.

    Returns: {labels: [city1, city2, ...], series: [[prices_city1], [prices_city2], ...]}
    """
    # For city comparison, always use ALL cities but filter display if specific city selected
    base = _prepare_base_df(df if city == "ALL" else _slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "city"])

    if base.empty:
        return {"labels": [], "series": []}

    cities_order = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
    labels = []
    series = []

    for c in cities_order:
        c_data = base[base["city"] == c]
        if city != "ALL" and c != city:
            continue
        prices = c_data["price_inr"].dropna().tolist()
        if len(prices) >= 3:
            labels.append(c)
            series.append([float(p) for p in prices])

    return {"labels": labels, "series": series}


# ---------------------------------------------------------------------------
# Chart 08: Price per Sqft by Locality Heatmap — Choropleth-style or ranked bar
# ---------------------------------------------------------------------------

def build_chart_08_price_sqft_locality_heatmap(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Price-per-sqft by locality heatmap — Ranked horizontal bar.

    Returns: {labels: [locality1, locality2, ...], series: [median_price_per_sqft]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_per_sqft", "locality"])

    if base.empty:
        return {"labels": [], "series": []}

    # Median price/sqft per locality, top 30 by volume
    locality_stats = base.groupby("locality").agg(
        median_ppsqft=("price_per_sqft", "median"),
        count=("price_per_sqft", "size")
    ).sort_values("count", ascending=False).head(30)

    labels = locality_stats.index.tolist()
    series = locality_stats["median_ppsqft"].round(0).astype(int).tolist()

    return {"labels": labels, "series": series}


# ---------------------------------------------------------------------------
# Chart 09: Furnish vs Price — Box plot (Furnished/Semi/Unfurnished)
# ---------------------------------------------------------------------------

def build_chart_09_furnish_vs_price(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Furnishing status vs. price — Box plot.

    Returns: {labels: ["Furnished", "Semi-Furnished", "Unfurnished"], series: [[prices], ...]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "furnishing_type"])

    if base.empty:
        return {"labels": [], "series": []}

    # Standardize furnishing categories
    furnish_order = ["Furnished", "Semifurnished", "Unfurnished"]
    labels = []
    series = []

    for f in furnish_order:
        f_data = base[base["furnishing_type"] == f]["price_inr"].dropna()
        if len(f_data) >= 3:
            labels.append(f)
            series.append([float(p) for p in f_data])

    return {"labels": labels, "series": series}


# ---------------------------------------------------------------------------
# Chart 10: Age vs Price Trend — Line/box plot across age buckets
# ---------------------------------------------------------------------------

def build_chart_10_age_price_trend(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Property age vs. price trend — Line chart with median per age bucket.

    Returns: {labels: [age_bucket1, age_bucket2, ...], series: [median_price], counts: [n_listings]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "agePossession"])

    if base.empty:
        return {"labels": [], "series": [], "counts": []}

    # Define age bucket order
    age_order = ["New Property", "Under Construction", "Relatively New", "Moderately Old", "Old Property"]

    labels = []
    series = []
    counts = []

    for age in age_order:
        age_data = base[base["agePossession"] == age]["price_inr"].dropna()
        if len(age_data) >= 3:
            labels.append(age)
            series.append(float(age_data.median()))
            counts.append(int(len(age_data)))

    return {"labels": labels, "series": series, "counts": counts}


# ---------------------------------------------------------------------------
# Chart 11: Amenity Count vs Price — Scatter + regression trend
# ---------------------------------------------------------------------------

def build_chart_11_amenity_count_vs_price(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Amenity count vs. price — Scatter + regression trend line.

    Returns: {x: [n_amenities], y: [price_inr], trend: {slope, intercept, r_squared}}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["n_amenities", "price_inr"])

    if base.empty:
        return {"x": [], "y": [], "trend": {"slope": 0, "intercept": 0, "r_squared": 0}}

    base = _downsample(base, MAX_SCATTER_POINTS)

    x = base["n_amenities"].astype(float).tolist()
    y = base["price_inr"].astype(float).tolist()

    x_arr = np.array(x)
    y_arr = np.array(y)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_pred = slope * x_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "x": x,
        "y": y,
        "trend": {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}
    }


# ---------------------------------------------------------------------------
# Chart 12: Floor Price Premium — Line chart (avg price by floor bucket)
# ---------------------------------------------------------------------------

def build_chart_12_floor_price_premium(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Floor number vs. price premium — Line chart.

    Returns: {labels: [floor_category1, ...], series: [median_price], counts: [n_listings]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "floor_category"])

    if base.empty:
        return {"labels": [], "series": [], "counts": []}

    floor_order = ["Low Floor", "Mid Floor", "High Floor"]

    labels = []
    series = []
    counts = []

    for floor in floor_order:
        f_data = base[base["floor_category"] == floor]["price_inr"].dropna()
        if len(f_data) >= 3:
            labels.append(floor)
            series.append(float(f_data.median()))
            counts.append(int(len(f_data)))

    return {"labels": labels, "series": series, "counts": counts}


# ---------------------------------------------------------------------------
# Chart 13: Top Builders by Avg Price — Horizontal bar / treemap
# ---------------------------------------------------------------------------

def build_chart_13_top_builders_price(df: pd.DataFrame, city: str) -> dict[str, Any]:
    """Top builders/societies by avg price — Horizontal bar.

    Returns: {labels: [builder1, builder2, ...], series: [avg_price], counts: [n_listings]}
    """
    base = _prepare_base_df(_slice_by_city(df, city))
    base = base.dropna(subset=["price_inr", "building_name"])

    if base.empty:
        return {"labels": [], "series": [], "counts": []}

    # Filter builders with at least 3 listings
    builder_stats = base.groupby("building_name").agg(
        avg_price=("price_inr", "mean"),
        median_price=("price_inr", "median"),
        count=("price_inr", "size")
    ).query("count >= 3").sort_values("median_price", ascending=False).head(15)

    labels = builder_stats.index.tolist()
    series = builder_stats["median_price"].round(0).astype(int).tolist()
    counts = builder_stats["count"].astype(int).tolist()

    return {"labels": labels, "series": series, "counts": counts}


# ---------------------------------------------------------------------------
# Main builder dispatcher
# ---------------------------------------------------------------------------

BUILDER_MAP: Final[dict[str, callable]] = {
    "chart_01_spatial_map": build_chart_01_spatial_map,
    "chart_02_price_distribution": build_chart_02_price_distribution,
    "chart_03_price_vs_area": build_chart_03_price_vs_area,
    "chart_04_bhk_distribution": build_chart_04_bhk_distribution,
    "chart_05_word_cloud": build_chart_05_word_cloud,
    "chart_06_correlation_heatmap": build_chart_06_correlation_heatmap,
    "chart_07_city_price_comparison": build_chart_07_city_price_comparison,
    "chart_08_price_sqft_locality_heatmap": build_chart_08_price_sqft_locality_heatmap,
    "chart_09_furnish_vs_price": build_chart_09_furnish_vs_price,
    "chart_10_age_price_trend": build_chart_10_age_price_trend,
    "chart_11_amenity_count_vs_price": build_chart_11_amenity_count_vs_price,
    "chart_12_floor_price_premium": build_chart_12_floor_price_premium,
    "chart_13_top_builders_price": build_chart_13_top_builders_price,
}

CHART_TITLES: Final[dict[str, str]] = {
    "chart_01_spatial_map": "Listing Map by Price per Sqft",
    "chart_02_price_distribution": "Price Distribution Across Localities",
    "chart_03_price_vs_area": "Price vs. Area Analysis",
    "chart_04_bhk_distribution": "BHK Distribution",
    "chart_05_word_cloud": "Top Feature Word Cloud",
    "chart_06_correlation_heatmap": "Feature Correlation Heatmap",
    "chart_07_city_price_comparison": "City-wise Price Comparison",
    "chart_08_price_sqft_locality_heatmap": "Price per Sqft by Locality",
    "chart_09_furnish_vs_price": "Furnishing Status vs. Price",
    "chart_10_age_price_trend": "Property Age vs. Price Trend",
    "chart_11_amenity_count_vs_price": "Amenity Count vs. Price",
    "chart_12_floor_price_premium": "Floor Category vs. Price Premium",
    "chart_13_top_builders_price": "Top Builders by Median Price",
}

CHART_TYPES: Final[dict[str, str]] = {
    "chart_01_spatial_map": "map",
    "chart_02_price_distribution": "box",
    "chart_03_price_vs_area": "scatter",
    "chart_04_bhk_distribution": "pie",
    "chart_05_word_cloud": "wordcloud",
    "chart_06_correlation_heatmap": "heatmap",
    "chart_07_city_price_comparison": "box",
    "chart_08_price_sqft_locality_heatmap": "bar",
    "chart_09_furnish_vs_price": "box",
    "chart_10_age_price_trend": "line",
    "chart_11_amenity_count_vs_price": "scatter",
    "chart_12_floor_price_premium": "line",
    "chart_13_top_builders_price": "bar",
}


def build_all_cache_files(
    df: pd.DataFrame,
    output_dir: Path,
    city_slices: tuple[str, ...] = CITY_SLICES,
) -> list[Path]:
    """Build all 13 cache JSON files for all city slices.

    Args:
        df: Full clean_listings DataFrame
        output_dir: Directory to write JSON files
        city_slices: Tuple of city keys to generate slices for

    Returns:
        List of written file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    metadata = _get_source_metadata(df)

    for chart_id, builder_fn in BUILDER_MAP.items():
        city_slices_data = {}
        for city in city_slices:
            _LOG.info("Building %s for city=%s", chart_id, city)
            city_data = builder_fn(df, city)
            city_slices_data[city] = city_data

        cache_obj = {
            "chart_id": chart_id,
            "title": CHART_TITLES.get(chart_id, chart_id),
            "chart_type": CHART_TYPES.get(chart_id, "unknown"),
            "city_slices": city_slices_data,
            "metadata": metadata,
        }

        output_path = output_dir / f"{chart_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(cache_obj, f, ensure_ascii=False, separators=(",", ":"))

        written_files.append(output_path)
        _LOG.info("Wrote %s (%d bytes)", output_path, output_path.stat().st_size)

    return written_files