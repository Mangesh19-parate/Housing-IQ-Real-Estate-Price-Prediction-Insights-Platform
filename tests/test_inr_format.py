"""Pinned cases for the price-hero currency formatter."""

from app.services.inr_format import inr_format


def test_inr_format_sale_cr():
    assert inr_format(14_200_000, transact_type="Sale") == "₹1.42 Cr"


def test_inr_format_sale_lakh():
    assert inr_format(450_000, transact_type="Sale") == "₹4.50 Lakh"


def test_inr_format_sale_thousand():
    assert inr_format(7_500, transact_type="Sale") == "₹7.50 Thousand"


def test_inr_format_sale_zero():
    assert inr_format(0, transact_type="Sale") == "₹0"


def test_inr_format_rent_indian_grouping():
    assert inr_format(42000, transact_type="Rent") == "₹42,000 / month"


def test_inr_format_rent_high_value():
    assert inr_format(1_50_000, transact_type="Rent") == "₹1,50,000 / month"


def test_inr_format_rent_zero():
    assert inr_format(0, transact_type="Rent") == "₹0 / month"


def test_inr_format_unknown_transact_falls_back_to_sale():
    assert inr_format(14_200_000, transact_type="Other") == "₹1.42 Cr"
