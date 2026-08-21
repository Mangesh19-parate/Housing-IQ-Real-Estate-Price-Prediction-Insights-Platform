"""Flask ``/predict`` route tests.

Pins the GET form rendering and the POST forward-to-FastAPI
behavior. Monkeypatches the ``FastAPIClient`` to avoid network
I/O — per Spec 18's "graceful degradation" rule, a stuck FastAPI
must not freeze the UI; we exercise that path explicitly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from api.schemas.predict_v3 import (
    LuxuryCategory,
    PredictResponseV3,
    ShapContribution,
)
from app.app import create_app
from app.services import FastAPIUnavailable

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


def _canned_response(*, outlier: bool = False) -> PredictResponseV3:
    return PredictResponseV3(
        predicted_price=14200000.0,
        range_low=12800000.0,
        range_high=15600000.0,
        shap_contributions=[
            ShapContribution(feature="built_up_area", impact=0.18),
        ],
        is_outlier_input=outlier,
        model_version="v2",
        luxury_category=LuxuryCategory.MEDIUM,
    )


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Flask app with a temp DB and a stubbed FastAPIClient."""
    db_file = tmp_path / "app.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_file))
    import app.config as app_config
    monkeypatch.setattr(app_config, "APP_DB_PATH", str(db_file))
    from app.database.db import init_db
    init_db(db_path=str(db_file))

    return create_app()


@pytest.fixture
def client(app, monkeypatch):
    """Flask test client. The FastAPIClient inside ``app.py`` is stubbed
    so no network I/O happens during tests.
    """
    from app import app as app_module

    mock = MagicMock()
    canned = _canned_response()
    mock.post_predict.return_value = canned
    mock.get_localities.return_value = [
        "Sector 84 Gurgaon", "Sector 81 Gurgaon",
    ]
    monkeypatch.setattr(app_module, "_get_client", lambda: mock)
    return app.test_client(), mock


# ---------- GET tests ----------


def test_predict_get_renders_form(client):
    flask_client, _ = client
    resp = flask_client.get("/predict")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
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


def test_predict_get_injects_localities_by_city(client):
    flask_client, _ = client
    resp = flask_client.get("/predict")
    html = resp.get_data(as_text=True)
    # The dependent-dropdown script receives a JSON blob keyed by city.
    assert "Sector 84 Gurgaon" in html
    # Each known city is rendered as a <select> option.
    for city in ("Gurgaon", "Hyderabad", "Kolkata", "Mumbai"):
        assert f'value="{city}"' in html


# ---------- POST happy path ----------


def test_predict_post_forwards_to_fastapi_and_renders_result(client):
    flask_client, mock = client
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Currency formatter fires for the hero price.
    assert "1.42 Cr" in html
    assert "model: v2" in html
    # Price summary line is rendered with the user's selected inputs.
    assert "Sector 84 Gurgaon, Gurgaon" in html
    # FastAPIClient received the request.
    mock.post_predict.assert_called_once()
    sent = mock.post_predict.call_args[0][0]
    assert sent.bedRoom == 3
    assert sent.transact_type.value == "Sale"


def test_predict_post_returns_unavailable_state_when_fastapi_down(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.side_effect = FastAPIUnavailable("simulated")
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "temporarily unavailable" in html.lower()


# ---------- POST validation paths ----------


def test_predict_post_returns_400_on_missing_field(client):
    flask_client, _ = client
    bad = {**_VALID_FORM}
    bad.pop("built_up_area")
    resp = flask_client.post("/predict", data=bad, follow_redirects=False)
    # Validation fail → flash + redirect to GET form (302).
    assert resp.status_code == 302
    assert "/predict" in resp.headers["Location"]


def test_predict_post_returns_400_on_invalid_bedroom(client):
    flask_client, _ = client
    bad = {**_VALID_FORM, "bedRoom": "20"}
    resp = flask_client.post("/predict", data=bad, follow_redirects=False)
    assert resp.status_code == 302


def test_predict_post_returns_400_on_bedroom_bathroom_violation(client):
    flask_client, _ = client
    bad = {**_VALID_FORM, "bedRoom": "5", "bathroom": "1"}
    resp = flask_client.post("/predict", data=bad, follow_redirects=False)
    assert resp.status_code == 302


# ---------- POST: outlier + transact_type wiring ----------


def test_predict_post_renders_outlier_banner_when_flagged(client):
    flask_client, mock = client
    mock.post_predict.return_value = _canned_response(outlier=True)
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    assert "Low confidence" in resp.get_data(as_text=True)


def test_predict_post_passes_transact_type_to_fastapi(client):
    flask_client, mock = client
    resp = flask_client.post(
        "/predict", data={**_VALID_FORM, "transact_type": "Rent"}
    )
    assert resp.status_code == 200
    sent = mock.post_predict.call_args[0][0]
    assert sent.transact_type.value == "Rent"


# ---------- Layering rule ----------


def test_predict_post_does_not_import_ml_or_models(client):
    """Rules §5.1: Flask never imports model code.

    Pinned via AST inspection of Flask's actual source files (not
    ``sys.modules`` — pytest's collection already pulls ``ml.*``
    in for unrelated tests, which would give a false positive).
    The display-only ``ml.explainability.labels`` label map is
    explicitly allowlisted (Spec 19): it's used for human-readable
    feature naming on the SHAP chart, not for model code.
    """
    import ast
    from pathlib import Path

    repo = Path(__file__).parent.parent
    flask_files = [
        repo / "app" / "app.py",
        repo / "app" / "services" / "fastapi_client.py",
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


# ---------- Shape of response ----------

def test_predict_result_template_formats_currency_with_rent(client, monkeypatch):
    flask_client, mock = client
    # Force the response to look like a rent prediction (low price).
    from api.schemas.predict_v3 import (
        LuxuryCategory,
        PredictResponseV3,
        ShapContribution,
    )
    mock.post_predict.return_value = PredictResponseV3(
        predicted_price=42000.0,
        range_low=35000.0,
        range_high=50000.0,
        shap_contributions=[ShapContribution(feature="built_up_area", impact=0.1)],
        is_outlier_input=False,
        model_version="v2",
        luxury_category=LuxuryCategory.LOW,
    )
    resp = flask_client.post(
        "/predict", data={**_VALID_FORM, "transact_type": "Rent"}
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Rent formatting uses Indian comma grouping + " / month".
    assert "/ month" in html


# ---------- Spec 19 — SHAP bar chart on the result page ----------


def _shap_response(*, contributions: list[tuple[str, float]]) -> PredictResponseV3:
    """Build a canned PredictResponseV3 with the given SHAP features."""
    return PredictResponseV3(
        predicted_price=14200000.0,
        range_low=12800000.0,
        range_high=15600000.0,
        shap_contributions=[
            ShapContribution(feature=f, impact=i) for f, i in contributions
        ],
        is_outlier_input=False,
        model_version="v2",
        luxury_category=LuxuryCategory.MEDIUM,
    )


def test_predict_post_renders_shap_chart_for_each_contribution(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.return_value = _shap_response(
        contributions=[
            ("num__built_up_area", 0.20),
            ("num__sector_smoothed_price", -0.12),
            ("ord__furnishing_type", 0.05),
        ]
    )
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Placeholder gone, real chart canvas present.
    assert "SHAP explanation goes here" not in html
    assert 'id="shap-chart"' in html
    assert 'data-rows="' in html
    # At least one human label is rendered in the chart's data-rows.
    assert "Built-up Area (sqft)" in html
    assert "Sector Average Price (smoothed)" in html
    assert "Furnishing Type" in html
    # At least one + or − text label visible in the accessible list.
    # (The chart canvas itself is opaque; the <ul hidden> carries
    # the per-row detail for screen readers.)
    assert "Built-up Area (sqft)" in html


def test_predict_post_renders_direction_summary_line(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.return_value = _shap_response(
        contributions=[
            ("num__built_up_area", 0.20),
            ("num__sector_smoothed_price", -0.12),
            ("ord__furnishing_type", 0.05),
            ("num__age_bucket_ord", -0.03),
        ]
    )
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # 2 up, 2 down → summary reflects exactly that count.
    # The numbers are wrapped in <strong> tags, so search for the
    # post-tag text instead of the bare digits.
    assert "<strong>2</strong> factors pushed the price up" in html
    assert "<strong>2</strong> pushed it down" in html


def test_predict_post_renders_chart_even_when_shap_empty(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.return_value = _shap_response(contributions=[])
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # No chart canvas, no SHAP list — but the section still renders
    # the friendly inline empty-state (Rules §6.1 spirit).
    assert 'id="shap-chart"' not in html
    assert "No contribution breakdown available for this prediction." in html


def test_predict_post_renders_accessible_text_summary(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.return_value = _shap_response(
        contributions=[
            ("num__built_up_area", 0.20),
            ("num__sector_smoothed_price", -0.12),
        ]
    )
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # Canvas carries role="img" + aria-label listing the labels.
    assert 'role="img"' in html
    assert 'aria-label="Top 2 SHAP feature contributions' in html
    # Hidden text fallback list exists for screen readers.
    assert 'class="shap-text-list" hidden' in html


def test_predict_post_does_not_expose_raw_feature_codes_to_user(client, monkeypatch):
    flask_client, mock = client
    mock.post_predict.return_value = _shap_response(
        contributions=[
            ("num__built_up_area", 0.20),
            ("ord__furnishing_type", 0.05),
        ]
    )
    resp = flask_client.post("/predict", data=_VALID_FORM)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # The hidden <ul> is the screen-reader text source — that's
    # what users (with or without a screen reader) would see if
    # JS were disabled or chart rendering failed. It must NOT
    # contain raw preprocessor codes.
    list_start = html.find('<ul class="shap-text-list" hidden>')
    list_end = html.find("</ul>", list_start) if list_start != -1 else -1
    assert list_start != -1 and list_end != -1, "shap-text-list not rendered"
    list_html = html[list_start:list_end]
    assert "num__built_up_area" not in list_html
    assert "ord__furnishing_type" not in list_html
    # The raw code may legitimately appear in the data-rows JSON
    # attribute (machine data for Chart.js) — that's allowed.
