"""Tests for ``ml.cleaning.imputation`` — Step 07 missing-value imputation."""

from __future__ import annotations

import ast
import inspect
import logging

import pandas as pd

from ml.cleaning.imputation import (
    IMPUTATION_CATEGORICAL_LOW,
    IMPUTATION_DROP_THRESHOLD,
    IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS,
    IMPUTATION_NUMERIC_LOW,
    MISSINGNESS_HIGH_THRESHOLD,
    MISSINGNESS_LOW_THRESHOLD,
    MISSINGNESS_MEDIUM_THRESHOLD,
    add_was_missing_flags,
    classify_missingness_tiers,
    drop_high_missing_columns,
    impute_high_tier,
    impute_low_tier,
    impute_medium_tier,
    impute_missing_values,
)

# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------


def test_missingness_threshold_constants_match_trd() -> None:
    """TRD §5: 0.05 / 0.40 / 0.70 (drop)."""
    assert MISSINGNESS_LOW_THRESHOLD == 0.05
    assert MISSINGNESS_MEDIUM_THRESHOLD == 0.40
    assert MISSINGNESS_HIGH_THRESHOLD == 0.70
    assert IMPUTATION_DROP_THRESHOLD == 0.70


def test_imputation_low_tier_constants_match_spec() -> None:
    """Numeric + categorical candidates pinned at module top."""
    assert IMPUTATION_NUMERIC_LOW == (
        "balconies",
        "floor_num_int",
        "total_floor",
        "area_sqft",
    )
    assert IMPUTATION_CATEGORICAL_LOW == (
        "furnish",
        "facing",
        "ownership_type",
        "age_bucket",
        "property_type",
    )
    assert IMPUTATION_GROUPWISE_MEDIUM_TIER_COLUMNS == (
        "price_inr",
        "price_per_sqft",
        "bedrooms",
        "bathrooms",
    )


# ---------------------------------------------------------------------------
# B. classify_missingness_tiers
# ---------------------------------------------------------------------------


def _tier_frame(n: int = 100) -> pd.DataFrame:
    """Helper: a frame with one numeric col whose missingness = n% (n out of 100)."""
    return pd.DataFrame(
        {"tier_col": [None if i < n else 1.0 for i in range(100)]}
    )


def test_classify_missingness_tiers_returns_four_keys() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None], "b": [None, None, None]})
    result = classify_missingness_tiers(df)
    assert set(result.keys()) == {"low", "medium", "high", "drop"}
    assert all(isinstance(v, list) for v in result.values())


def test_classify_missingness_tiers_low_under_5pct() -> None:
    df = _tier_frame(n=4)  # 4% missing → low (< 5)
    result = classify_missingness_tiers(df)
    assert "tier_col" in result["low"]


def test_classify_missingness_tiers_medium_between_5_and_40pct() -> None:
    df = _tier_frame(n=20)
    result = classify_missingness_tiers(df)
    assert "tier_col" in result["medium"]


def test_classify_missingness_tiers_high_between_40_and_70pct() -> None:
    df = _tier_frame(n=50)
    result = classify_missingness_tiers(df)
    assert "tier_col" in result["high"]


def test_classify_missingness_tiers_drop_above_70pct() -> None:
    df = _tier_frame(n=80)
    result = classify_missingness_tiers(df)
    assert "tier_col" in result["drop"]


def test_classify_missingness_tiers_uses_input_frame_not_imputed() -> None:
    """60% input missingness is classified `high` even though a fillna
    would zero it. The classifier operates on the input frame.
    """
    df = _tier_frame(n=60)
    result = classify_missingness_tiers(df)
    assert "tier_col" in result["high"]


# ---------------------------------------------------------------------------
# C. impute_low_tier
# ---------------------------------------------------------------------------


def test_impute_low_tier_numeric_uses_global_median() -> None:
    """A numeric column with <5% NaN is filled with the global median."""
    rows = []
    for i in range(96):
        rows.append({"listing_id": f"G{i:05d}", "city": "Gurgaon",
                     "property_type": "flat", "balconies": float(i % 5)})
    for i in range(4):
        rows.append({"listing_id": f"MISS{i}", "city": "Gurgaon",
                     "property_type": "flat", "balconies": pd.NA})
    df = pd.DataFrame(rows)
    # 4 NaN / 100 rows = 4% → low tier.
    tiers = classify_missingness_tiers(df)
    assert "balconies" in tiers["low"], f"Got tiers={tiers}"
    out = impute_low_tier(df)
    assert bool(out["balconies"].isna().any()) is False
    expected_median = float(df["balconies"].dropna().median())
    for i in range(96, 100):
        assert float(out["balconies"].iloc[i]) == expected_median


def test_impute_low_tier_categorical_uses_global_mode() -> None:
    """A categorical column with <5% NaN is filled with the global mode."""
    rows = []
    for i in range(96):
        rows.append({"listing_id": f"G{i:05d}", "city": "Gurgaon",
                     "property_type": "flat",
                     "furnish": "furnished" if i % 2 == 0 else "semi"})
    for i in range(4):
        rows.append({"listing_id": f"MISS{i}", "city": "Gurgaon",
                     "property_type": "flat", "furnish": pd.NA})
    df = pd.DataFrame(rows)
    tiers = classify_missingness_tiers(df)
    assert "furnish" in tiers["low"], f"Got tiers={tiers}"
    out = impute_low_tier(df)
    assert bool(out["furnish"].isna().any()) is False
    assert str(out["furnish"].iloc[96]) == "furnished"


def test_impute_low_tier_numeric_coerces_object_dtype_with_stray_strings() -> None:
    """A numeric low-tier column that ships as ``object`` dtype (due to
    stray string values like ``"G"`` for ground floor) is coerced via
    ``pd.to_numeric(..., errors="coerce")`` before computing the median.
    Regression: real-data ``total_floor`` column has dtype=object because
    ~1% of rows say ``"G"``; without coerce the median throws.
    """
    rows = []
    for i in range(96):
        rows.append({"listing_id": f"G{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "total_floor": str(5 + i % 10)})
    for i in range(4):
        rows.append({"listing_id": f"MISS{i}", "city": "Gurgaon",
                     "property_type": "flat", "total_floor": pd.NA})
    df = pd.DataFrame(rows)
    # The column is string/Object dtype because the values are strings.
    assert df["total_floor"].dtype != float
    tiers = classify_missingness_tiers(df)
    assert "total_floor" in tiers["low"]
    out = impute_low_tier(df)
    # All NaNs filled — no exception.
    assert bool(out["total_floor"].isna().any()) is False


def test_impute_low_tier_no_op_for_columns_not_in_tier() -> None:
    """`bedrooms` is a medium-tier column; impute_low_tier does not touch it."""
    rows = []
    for i in range(50):
        rows.append({"listing_id": f"G{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "bedrooms": 2})
    for i in range(50):
        rows.append({"listing_id": f"M{i:04d}", "city": "Mumbai",
                     "property_type": "flat", "bedrooms": pd.NA})  # 50% missing
    df = pd.DataFrame(rows)
    out = impute_low_tier(df)
    assert bool(out["bedrooms"].isna().any()) is True


# ---------------------------------------------------------------------------
# D. impute_medium_tier
# ---------------------------------------------------------------------------


def test_impute_medium_tier_uses_groupwise_median() -> None:
    """Group median wins over global median when they disagree.

    Construct: a GroupA cluster where the global median would be 1.0 (taken
    over GroupB's heavy 1.0 mass) but the group-A median is 300. After
    impute_medium_tier, the GroupA NaNs should be filled with 300.
    """
    rows = []
    # GroupA: 5x 100 + 5x 500 = 10 non-null. GroupA median = 300.
    for i in range(5):
        rows.append({"listing_id": f"A{i:04d}", "city": "GroupA", "locality": "L1",
                     "property_type": "flat", "price_inr": 100.0})
    for i in range(5):
        rows.append({"listing_id": f"A5{i:04d}", "city": "GroupA", "locality": "L1",
                     "property_type": "flat", "price_inr": 500.0})
    # GroupB: 90 rows of 1.0 (so global median = 1.0, GroupA median = 300).
    for i in range(90):
        rows.append({"listing_id": f"B{i:04d}", "city": "GroupB", "locality": "L1",
                     "property_type": "flat", "price_inr": 1.0})
    # 20 additional GroupA NaNs to land overall in 5-40% tier.
    for i in range(20):
        rows.append({"listing_id": f"AN{i:04d}", "city": "GroupA", "locality": "L1",
                     "property_type": "flat", "price_inr": pd.NA})
    df = pd.DataFrame(rows)
    # 120 rows total, 20 NaN → 16.7% → medium tier.
    tiers = classify_missingness_tiers(df)
    assert "price_inr" in tiers["medium"], f"Got tiers={tiers}"
    out = impute_medium_tier(df)
    # GroupA NaNs (positions 100-119) should be 300, NOT 1.
    for pos in range(100, 120):
        actual = float(out["price_inr"].iloc[pos])
        assert actual == 300.0, f"position {pos}: expected 300.0, got {actual}"


def test_impute_medium_tier_falls_back_to_global_when_group_empty() -> None:
    """A (city, locality, property_type) group with zero non-null values
    falls back to the global median.
    """
    rows = []
    # GroupA: 5 non-null + 5 NaN.
    for i in range(5):
        rows.append({"listing_id": f"A{i}", "city": "GroupA", "locality": "L1",
                     "property_type": "flat", "price_inr": 100.0})
    for i in range(5):
        rows.append({"listing_id": f"AN{i}", "city": "GroupA", "locality": "L1",
                     "property_type": "flat", "price_inr": pd.NA})
    # GroupB: 95 non-null = 200.
    for i in range(95):
        rows.append({"listing_id": f"B{i}", "city": "GroupB", "locality": "L1",
                     "property_type": "flat", "price_inr": 200.0})
    # GroupC: 5 NaN, 0 non-null → empty group → fallback.
    for i in range(5):
        rows.append({"listing_id": f"CN{i}", "city": "GroupC", "locality": "L1",
                     "property_type": "flat", "price_inr": pd.NA})
    df = pd.DataFrame(rows)
    # 110 rows total, 15 NaN → ~13.6% → medium tier.
    out = impute_medium_tier(df)
    group_c_filled = out.loc[df["city"] == "GroupC", "price_inr"]
    # Global non-null = [100]*5 + [200]*95 → median = 200.
    assert float(group_c_filled.iloc[0]) == 200.0
    assert bool(out["price_inr"].isna().any()) is False


# ---------------------------------------------------------------------------
# E. impute_high_tier
# ---------------------------------------------------------------------------


def test_impute_high_tier_categorical_filled_with_unknown() -> None:
    """A 40-70% missing string column is filled with literal "Unknown"."""
    rows = []
    for i in range(45):
        rows.append({"listing_id": f"A{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "color": "red"})
    for i in range(55):
        rows.append({"listing_id": f"B{i:04d}", "city": "Mumbai",
                     "property_type": "flat", "color": pd.NA})
    df = pd.DataFrame(rows)
    out = impute_high_tier(df)
    assert bool(out["color"].isna().any()) is False
    for i in range(45, 100):
        assert str(out["color"].iloc[i]) == "Unknown"


def test_impute_high_tier_numeric_left_nan_with_flag() -> None:
    """A 40-70% missing numeric column is left NaN (flag carries signal)."""
    rows = []
    for i in range(45):
        rows.append({"listing_id": f"A{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "score": float(i)})
    for i in range(55):
        rows.append({"listing_id": f"B{i:04d}", "city": "Mumbai",
                     "property_type": "flat", "score": pd.NA})
    df = pd.DataFrame(rows)
    out = impute_high_tier(df)
    assert bool(out["score"].isna().any()) is True


# ---------------------------------------------------------------------------
# F. drop_high_missing_columns
# ---------------------------------------------------------------------------


def test_drop_high_missing_columns_drops_above_70pct() -> None:
    df = pd.DataFrame(
        {"good": [1.0, 2.0, 3.0, 4.0, 5.0],
         "bad": [None, None, None, None, 1.0]}  # 80% missing → drop
    )
    out, dropped = drop_high_missing_columns(df)
    assert "bad" not in out.columns
    assert "good" in out.columns
    assert "bad" in dropped


def test_drop_high_missing_columns_logs_dropped(caplog) -> None:
    df = pd.DataFrame(
        {"good": [1.0, 2.0, 3.0, 4.0, 5.0],
         "bad": [None, None, None, None, 1.0]}
    )
    with caplog.at_level(logging.INFO, logger="ml.cleaning.imputation"):
        drop_high_missing_columns(df)
    assert any("bad" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# G. add_was_missing_flags
# ---------------------------------------------------------------------------


def test_add_was_missing_flags_creates_one_flag_per_imputed_column() -> None:
    df = pd.DataFrame(
        {"a": [1.0, 2.0, None], "b": ["x", None, "y"], "c": [None, None, None]}
    )
    out = add_was_missing_flags(df, ("a", "b", "c"))
    assert "was_missing_a" in out.columns
    assert "was_missing_b" in out.columns
    assert "was_missing_c" in out.columns


def test_add_was_missing_flags_are_set_before_imputation() -> None:
    """Running add_was_missing_flags then impute_low_tier leaves the
    flag `True` for the row whose value was filled.
    """
    rows = []
    for i in range(95):
        rows.append({"listing_id": f"G{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "balconies": float(i % 5)})
    rows.append({"listing_id": "MISS", "city": "Gurgaon",
                 "property_type": "flat", "balconies": pd.NA})
    df = pd.DataFrame(rows)
    flagged = add_was_missing_flags(df, ("balconies",))
    assert bool(flagged["was_missing_balconies"].iloc[95]) is True
    filled = impute_low_tier(flagged)
    assert bool(filled["was_missing_balconies"].iloc[95]) is True
    assert pd.notna(filled["balconies"].iloc[95])


def test_add_was_missing_flags_does_not_create_flag_for_column_without_nans() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [None, 2.0, 3.0]})
    out = add_was_missing_flags(df, ("a", "b"))
    assert "was_missing_a" not in out.columns
    assert "was_missing_b" in out.columns


# ---------------------------------------------------------------------------
# H. impute_missing_values (top-level)
# ---------------------------------------------------------------------------


def test_impute_missing_values_is_idempotent() -> None:
    rows = []
    for i in range(100):
        rows.append({"listing_id": f"G{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "balconies": float(i % 5),
                     "furnish": "furnished" if i % 2 == 0 else "semi"})
    df = pd.DataFrame(rows)
    once = impute_missing_values(df)
    twice = impute_missing_values(once)
    pd.testing.assert_frame_equal(once, twice)


def test_impute_missing_values_logs_summary(caplog) -> None:
    rows = []
    for i in range(100):
        rows.append({"listing_id": f"G{i:04d}", "city": "Gurgaon",
                     "property_type": "flat", "balconies": float(i % 5)})
    df = pd.DataFrame(rows)
    with caplog.at_level(logging.INFO, logger="ml.cleaning.imputation"):
        impute_missing_values(df)
    summary_records = [r for r in caplog.records if "impute.summary" in r.message]
    assert len(summary_records) >= 1
    msg = summary_records[-1].message
    assert "dropped=" in msg
    assert "flag_cols=" in msg
    assert "nans_before=" in msg
    assert "nans_after=" in msg


# ---------------------------------------------------------------------------
# I. AST scan: impute must not write to disk
# ---------------------------------------------------------------------------


def test_impute_does_not_write_to_disk() -> None:
    """imputation.py source contains no to_parquet / to_csv / to_json / open(."""
    import ml.cleaning.imputation as mod
    src = inspect.getsource(mod)
    forbidden = ["to_parquet", "to_csv", "to_json"]
    for tok in forbidden:
        assert tok not in src, f"imputation.py contains forbidden token: {tok!r}"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "imputation.py contains open() call"
