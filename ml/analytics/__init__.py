"""ml.analytics — Analytics cache builders.

Exports builder functions for all 13 precomputed chart tiles.
Each builder is a pure function: DataFrame + city -> dict (JSON-serializable).
"""

from __future__ import annotations

from ml.analytics.cache_builders import (
    build_all_cache_files,
    build_chart_01_spatial_map,
    build_chart_02_price_distribution,
    build_chart_03_price_vs_area,
    build_chart_04_bhk_distribution,
    build_chart_05_word_cloud,
    build_chart_06_correlation_heatmap,
    build_chart_07_city_price_comparison,
    build_chart_08_price_sqft_locality_heatmap,
    build_chart_09_furnish_vs_price,
    build_chart_10_age_price_trend,
    build_chart_11_amenity_count_vs_price,
    build_chart_12_floor_price_premium,
    build_chart_13_top_builders_price,
)

__all__ = [
    "build_all_cache_files",
    "build_chart_01_spatial_map",
    "build_chart_02_price_distribution",
    "build_chart_03_price_vs_area",
    "build_chart_04_bhk_distribution",
    "build_chart_05_word_cloud",
    "build_chart_06_correlation_heatmap",
    "build_chart_07_city_price_comparison",
    "build_chart_08_price_sqft_locality_heatmap",
    "build_chart_09_furnish_vs_price",
    "build_chart_10_age_price_trend",
    "build_chart_11_amenity_count_vs_price",
    "build_chart_12_floor_price_premium",
    "build_chart_13_top_builders_price",
]