"""Flask ``/classify`` route tests.

Pins the GET form rendering and the POST forward-to-FastAPI
behavior. Monkeypatches the ``ClassifyClient`` to avoid network
I/O — per Spec 26's "graceful degradation" rule, a stuck FastAPI
must not freeze the UI; we exercise that path explicitly.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from api.schemas.classify_v1 import (
    ClassifyResponseV1,
    LuxuryCategory,
    ShapContribution,
    TierLabel,
    VerdictLabel,
)
from app.app import create_app
from app.services import ClassifyUnavailable

# ---------- helpers ----------

_VALID_FORM = {
    "city": "Gurgaon",
    "sector": "Sector 84 Gurgaon",
    "property_type": "flat",
    "transact_type": "Sale",
    "bedRoom": "3",
    "bathroom": "3",
    "balcony": "2",
    "agePossession": "Relatively New",
    "built_up_area": "1450",
    "furnishing_type": "Semifurnished",
    "floor_category": "Mid Floor",
    "facing": "North",
}


def _canned_classify_response(
    *,
    verdict: VerdictLabel = VerdictLabel.GOOD_DEAL,
    tier: TierLabel = TierLabel.MID_RANGE,
) -> ClassifyResponseV1:
    return ClassifyResponseV1(
        good_deal_verdict=verdict,
        verdict_probabilities={
            VerdictLabel.GOOD_DEAL: 0.65,
            VerdictLabel.FAIR_DEAL: 0.25,
            VerdictLabel.OVERPRICED: 0.10,
        },
        price_tier=tier,
        tier_probabilities={
            TierLabel.BUDGET: 0.10,
            TierLabel.MID_RANGE: 0.55,
            TierLabel.PREMIUM: 0.25,
            TierLabel.LUXURY: 0.10,
        },
        shap_contributions=[
            ShapContribution(feature="num__built_up_area", impact=0.18),
            ShapContribution(feature="num__sector_smoothed_price", impact=-0.12),
        ],
        model_version="v1",
        luxury_category=LuxuryCategory.MEDIUM,
    )


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Flask app with a temp DB and a stubbed ClassifyClient."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    from app.database.db import init_db
    init_db(db_path=str(db_file))

    return create_app()


@pytest.fixture
def client(app, monkeypatch):
    """Flask test client. The ClassifyClient inside ``app.py`` is stubbed
    so no network I/O happens during tests.
    """
    from app import app as app_module

    mock = MagicMock()
    canned = _canned_classify_response()
    mock.post_classify.return_value = canned
    mock.get_localities.return_value = [
        "Sector 84 Gurgaon", "Sector 81 Gurgaon",
    ]
    monkeypatch.setattr(app_module, "_get_classify_client", lambda: mock)
    return app.test_client(), mock


# ---------- GET tests ----------


def test_classify_get_renders_form(client):
    flask_client, _ = client
    resp = flask_client.get("/classify")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Heading per UI/UX §U-UX-10
    assert "Is this a good deal?" in html
    # Button text per UI/UX §U-UX-10
    assert "Check Price Tier" in html
    # All 15 visible input fields (luxury_category is server-derived).
    for field in (
        "city", "sector", "property_type", "transact_type",
        "bedRoom", "bathroom", "balcony", "agePossession",
        "built_up_area", "servant_room", "store_room",
        "furnishing_type", "floor_category", "facing", "amenities",
    ):
        assert f'name="{field}"' in html, f"missing field: {field}"
    # No raw luxury_category dropdown — server-derived.
    assert 'name="luxury_category"' not in html


def test_classify_get_injects_localities_by_city(client):
    flask_client, _ = client
    resp = flask_client.get("/classify")
    html = resp.get_data(as_text=True)
    # The dependent-dropdown script receives a JSON blob keyed by city.
    assert "Sector 84 Gurgaon" in html
    # Each known city is rendered as a <select> option.
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        assert f'value="{city}"' in html


# ---------- POST happy path ----------


def test_classify_post_forwards_to_fastapi_and_renders_result(client):
    flask_client, mock = client
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # VerdictBadge rendered with icon and text
    assert "Good Deal" in html
    assert "✅" in html
    # AffordabilityChip rendered
    assert "Mid-Range" in html
    # Model version shown
    assert "model: v1" in html
    # Luxury category shown
    assert "luxury: Medium" in html
    # ClassifyClient received the request.
    mock.post_classify.assert_called_once()
    sent = mock.post_classify.call_args[0][0]
    assert sent.bedRoom == 3
    assert sent.transact_type.value == "Sale"


def test_classify_post_returns_unavailable_state_when_fastapi_down(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.side_effect = ClassifyUnavailable("simulated")
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "temporarily unavailable" in html.lower()


# ---------- POST validation paths ----------


def test_classify_post_returns_400_on_missing_field(client):
    flask_client, _ = client
    bad = {**_VALID_FORM}
    bad.pop("built_up_area")
    resp = flask_client.post("/classify", data=bad, follow_redirects=False)
    # Validation fail → flash + redirect to GET form (302).
    assert resp.status_code == 302
    assert "/classify" in resp.headers["Location"]


def test_classify_post_returns_400_on_invalid_bedroom(client):
    flask_client, _ = client
    bad = {**_VALID_FORM, "bedRoom": "20"}
    resp = flask_client.post("/classify", data=bad, follow_redirects=False)
    assert resp.status_code == 302


def test_classify_post_returns_400_on_bedroom_bathroom_violation(client):
    flask_client, _ = client
    bad = {**_VALID_FORM, "bedRoom": "5", "bathroom": "1"}
    resp = flask_client.post("/classify", data=bad, follow_redirects=False)
    assert resp.status_code == 302


# ---------- Verdict variants ----------


def test_classify_post_renders_fair_deal_verdict(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = _canned_classify_response(
        verdict=VerdictLabel.FAIR_DEAL,
    )
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Fair Deal" in html
    assert "⚖️" in html
    # VerdictBadge CSS class should be fair-deal
    assert 'verdict-badge--fair-deal' in html


def test_classify_post_renders_overpriced_verdict(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = _canned_classify_response(
        verdict=VerdictLabel.OVERPRICED,
    )
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Overpriced" in html
    assert "⚠️" in html
    # VerdictBadge CSS class should be overpriced
    assert 'verdict-badge--overpriced' in html


# ---------- Tier variants ----------


def test_classify_post_renders_all_four_tier_chips(client, monkeypatch):
    """Each tier gets its own AffordabilityChip CSS class."""
    flask_client, mock = client
    for tier in (TierLabel.BUDGET, TierLabel.MID_RANGE, TierLabel.PREMIUM, TierLabel.LUXURY):
        mock.post_classify.return_value = _canned_classify_response(tier=tier)
        resp = flask_client.post("/classify", data=_VALID_FORM)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        # The chip class uses kebab-case (budget, mid-range, premium, luxury)
        expected_class = f'affordability-chip--{tier.value.lower().replace("-", "-")}'
        assert expected_class in html, f"missing chip class for {tier.value}: {expected_class}"


# ---------- SHAP chart on result page ----------


def test_classify_post_renders_shap_chart_for_each_contribution(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = _canned_classify_response()
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Placeholder gone, real chart canvas present.
    assert 'id="shap-chart"' in html
    assert 'data-rows="' in html
    # At least one human label is rendered in the chart's data-rows.
    assert "Built-up Area (sqft)" in html
    assert "Sector Average Price (smoothed)" in html


def test_classify_post_renders_direction_summary_line(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = ClassifyResponseV1(
        good_deal_verdict=VerdictLabel.GOOD_DEAL,
        verdict_probabilities={
            VerdictLabel.GOOD_DEAL: 0.65,
            VerdictLabel.FAIR_DEAL: 0.25,
            VerdictLabel.OVERPRICED: 0.10,
        },
        price_tier=TierLabel.MID_RANGE,
        tier_probabilities={
            TierLabel.BUDGET: 0.10,
            TierLabel.MID_RANGE: 0.55,
            TierLabel.PREMIUM: 0.25,
            TierLabel.LUXURY: 0.10,
        },
        shap_contributions=[
            ShapContribution(feature="num__built_up_area", impact=0.20),
            ShapContribution(feature="num__sector_smoothed_price", impact=-0.12),
            ShapContribution(feature="ord__furnishing_type", impact=0.05),
            ShapContribution(feature="num__age_bucket_ord", impact=-0.03),
        ],
        model_version="v1",
        luxury_category=LuxuryCategory.MEDIUM,
    )
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # 2 up, 2 down → summary reflects exactly that count.
    # The summary text says "pushed toward Good Deal" / "pushed toward Overpriced"
    assert "<strong>2</strong> factor" in html
    assert "pushed toward Good Deal" in html
    assert "pushed toward Overpriced" in html


def test_classify_post_renders_chart_even_when_shap_empty(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = ClassifyResponseV1(
        good_deal_verdict=VerdictLabel.GOOD_DEAL,
        verdict_probabilities={
            VerdictLabel.GOOD_DEAL: 0.65,
            VerdictLabel.FAIR_DEAL: 0.25,
            VerdictLabel.OVERPRICED: 0.10,
        },
        price_tier=TierLabel.MID_RANGE,
        tier_probabilities={
            TierLabel.BUDGET: 0.10,
            TierLabel.MID_RANGE: 0.55,
            TierLabel.PREMIUM: 0.25,
            TierLabel.LUXURY: 0.10,
        },
        shap_contributions=[],
        model_version="v1",
        luxury_category=LuxuryCategory.MEDIUM,
    )
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # No chart canvas, no SHAP list — but the section still renders
    # the friendly inline empty-state.
    assert 'id="shap-chart"' not in html
    assert "No contribution breakdown available for this classification." in html


# ---------- Tier probability 4-bar chart ----------


def test_classify_post_renders_tier_probability_chart(client, monkeypatch):
    flask_client, mock = client
    mock.post_classify.return_value = _canned_classify_response()
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Tier probability chart canvas
    assert 'id="tier-chart"' in html
    assert 'data-probs="' in html
    # All four tiers present in the data-probs JSON
    assert "Budget" in html
    assert "Mid-Range" in html
    assert "Premium" in html
    assert "Luxury" in html
    # Accessible text fallback list
    assert 'class="tier-text-list" hidden' in html


# ---------- Layering rule ----------


def test_classify_post_does_not_import_ml_or_models(client):
    """Rules §5.1: Flask never imports model code."""
    from pathlib import Path

    repo = Path(__file__).parent.parent
    flask_files = [
        repo / "app" / "app.py",
        repo / "app" / "services" / "classify_client.py",
        repo / "app" / "services" / "inr_format.py",
        repo / "app" / "services" / "shap_format.py",
    ]
    allowed = {"ml.explainability.labels"}
    forbidden: list[str] = []
    for path in flask_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed and (
                        alias.name.startswith("ml.")
                        or alias.name.split(".")[0] == "models"
                    ):
                        forbidden.append(f"{path.name}: {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module not in allowed and (
                    node.module.startswith("ml.")
                    or node.module.split(".")[0] == "models"
                ):
                    forbidden.append(f"{path.name}: {node.module}")
    assert forbidden == [], f"Flask imports forbidden modules: {forbidden}"


# ---------- CTA to recommender ----------


def test_classify_result_has_cta_to_recommender(client):
    flask_client, mock = client
    mock.post_classify.return_value = _canned_classify_response()
    resp = flask_client.post("/classify", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'class="cta-recommend"' in html
    assert 'href="/recommend"' in html or 'href="http://localhost/recommend"' in html
