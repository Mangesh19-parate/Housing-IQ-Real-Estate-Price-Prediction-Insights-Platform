"""``api.schemas.analytics`` — Pydantic models for Analytics cache API.

Defines the response contract for GET /api/analytics/<chart_id>?city=<city>.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsMetadata(BaseModel):
    """Metadata about the cache file source and build."""

    model_config = ConfigDict(extra="forbid")

    source_parquet: str = Field(..., description="Source Parquet filename")
    source_parquet_hash: str = Field(..., description="SHA256 hash of source file")
    built_at: str = Field(..., description="ISO 8601 build timestamp")
    row_count: int = Field(..., description="Total rows in source (pre-outlier-filter)")
    outlier_count: int = Field(..., description="Rows flagged as outliers")


class SpatialMapSlice(BaseModel):
    """City slice for spatial map chart."""

    model_config = ConfigDict(extra="forbid")

    markers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {lat, lng, price_per_sqft, price_inr, locality, bhk}"
    )
    bounds: list[list[float]] = Field(
        default_factory=list,
        description="[[min_lat, min_lng], [max_lat, max_lng]]"
    )


class ChartDataSlice(BaseModel):
    """Generic city slice for chart-based tiles (box, scatter, line, bar, pie, heatmap, wordcloud)."""

    model_config = ConfigDict(extra="forbid")

    labels: list[Any] = Field(default_factory=list, description="Axis labels / categories")
    series: list[Any] = Field(default_factory=list, description="Data series (values per label)")
    counts: list[int] = Field(default_factory=list, description="Optional: listing counts per label")
    trend: dict[str, float] | None = Field(default=None, description="Regression trend {slope, intercept, r_squared}")
    x: list[float] = Field(default_factory=list, description="Scatter X values")
    y: list[float] = Field(default_factory=list, description="Scatter Y values")
    words: list[dict[str, Any]] = Field(default_factory=list, description="Word cloud entries")
    features: list[str] = Field(default_factory=list, description="Correlation feature names")
    matrix: list[list[float]] = Field(default_factory=list, description="Correlation matrix")


class AnalyticsCacheResponse(BaseModel):
    """Full response for GET /api/analytics/<chart_id>?city=<city>.

    Contains pre-shaped data for all 5 city slices plus metadata.
    """

    model_config = ConfigDict(extra="forbid")

    chart_id: str = Field(..., description="Chart identifier (e.g., chart_01_spatial_map)")
    title: str = Field(..., description="Human-readable chart title")
    chart_type: str = Field(..., description="Chart type: map, box, scatter, line, bar, pie, heatmap, wordcloud")
    city_slices: dict[str, dict[str, Any]] = Field(
        ...,
        description="Pre-shaped data for each city slice (ALL, Gurgaon, Hyderabad, Kolkata, Mumbai)"
    )
    metadata: AnalyticsMetadata = Field(..., description="Cache build metadata")


# Chart ID to title/type mapping for validation
CHART_ID_TO_TITLE: dict[str, str] = {
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

CHART_ID_TO_TYPE: dict[str, str] = {
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

VALID_CHART_IDS: frozenset[str] = frozenset(CHART_ID_TO_TITLE.keys())
VALID_CITIES: frozenset[str] = frozenset(["ALL", "Gurgaon", "Hyderabad", "Kolkata", "Mumbai"])