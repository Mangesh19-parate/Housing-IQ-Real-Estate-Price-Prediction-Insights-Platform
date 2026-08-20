"""Flask web app — pages, forms, rendering ONLY.

Per CLAUDE.md: Flask never imports model code or touches ``.pkl`` files.
Inference goes through the FastAPI service over HTTP (see ``api/``).

Spec 18 wires the ``/predict`` form to forward directly to FastAPI's
``POST /predict`` (Spec 17) via ``app.services.FastAPIClient``. The
Flask route does no model code (Rules §5.1), no separate validation
(Pydantic handles it on the API side, Rules §5.1 + §5.2), and no DB
writes (the FastAPI side handles ``prediction_log`` per Spec 17).
``luxury_category`` is server-derived from the amenity checklist
(Rules §10.2) — the Flask side never assigns a category.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for
from pydantic import ValidationError

from app.config import FASTAPI_BASE_URL, FLASK_DEBUG, FLASK_SECRET_KEY
from app.database.db import init_db
from app.services import FastAPIClient, FastAPIUnavailable, inr_format
from app.services.fastapi_client import KNOWN_CITIES

# Initialized on first request — flag prevents init_db() running twice per process.
_db_initialized = False

# Module-level lazy singleton for the FastAPI HTTP client.
_fastapi_client: FastAPIClient | None = None


def _get_client() -> FastAPIClient:
    """Process-wide FastAPI client; first call instantiates, later calls reuse."""
    global _fastapi_client
    if _fastapi_client is None:
        _fastapi_client = FastAPIClient(FASTAPI_BASE_URL)
    return _fastapi_client


def _enum_options() -> dict[str, list[str]]:
    """Return each Pydantic enum's allowed string values for the form dropdowns.

    Pulled directly from the Spec 11 schema modules — never
    hardcoded duplicates (keeps the form in lockstep with API
    validation).
    """
    from api.schemas.predict_v3 import (
        AgePossession,
        Balcony,
        FacingDirection,
        FloorCategory,
        FurnishingType,
        PropertyType,
        TransactType,
    )

    return {
        "property_types": [e.value for e in PropertyType],
        "transact_types": [e.value for e in TransactType],
        "balconies": [e.value for e in Balcony],
        "age_possessions": [e.value for e in AgePossession],
        "furnishing_types": [e.value for e in FurnishingType],
        "floor_categories": [e.value for e in FloorCategory],
        "facing_directions": [e.value for e in FacingDirection],
    }


def create_app() -> Flask:
    """Application factory. Tests and ``__main__`` both go through here."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = FLASK_SECRET_KEY
    app.jinja_env.filters["inr_format"] = inr_format

    @app.before_request
    def _ensure_db() -> None:
        """Idempotent — creates the 4 operational tables on first request only."""
        global _db_initialized
        if not _db_initialized:
            init_db()
            _db_initialized = True

    @app.route("/")
    def landing() -> str:
        """Landing page: city quick-filter + module cards."""
        cities = list(KNOWN_CITIES)
        modules = [
            ("predict", "Price Prediction"),
            ("classify", "Affordability & Investment Tier"),
            ("analytics", "Analytics"),
            ("recommend", "Recommender"),
            ("insights", "Market Insights"),
            ("map", "Map Explorer"),
        ]
        return render_template(
            "landing.html",
            cities=cities,
            modules=modules,
        )

    @app.route("/predict", methods=["GET"], endpoint="predict")
    def predict_get() -> str:
        """Render the 16-field Price Prediction form.

        Per UI/UX §U-UX-6, the field order is locked to the
        finalized 16-field contract; per Rules §10.2,
        ``luxury_category`` is server-derived so it has no
        dropdown — the form collects a guided 3-checkbox
        finish checklist that forwards as part of ``amenities``.
        """
        client = _get_client()
        localities_by_city = {
            city: client.get_localities(city) for city in KNOWN_CITIES
        }
        return render_template(
            "predict.html",
            cities=list(KNOWN_CITIES),
            localities_by_city=localities_by_city,
            enum_options=_enum_options(),
        )

    @app.route("/predict", methods=["POST"])
    def predict_post() -> Any:
        """Forward the submitted form to FastAPI and render the result.

        Validates the form into ``PredictRequestV3`` first (Pydantic
        catches bad input before we burn a network call); on a
        validation failure we flash a friendly message and redirect
        back to the form (Rules §5.2 spirit). On FastAPI failure we
        render the result page with ``unavailable=True`` so the user
        sees the friendly empty state instead of a 500.
        """
        from api.schemas.predict_v3 import PredictRequestV3

        form = request.form
        # Luxury finish checklist → append to amenities list (per
        # user-confirmed lazy wiring). The server still resolves
        # ``luxury_category`` from the amenities count (Spec 17).
        amenities = list(form.getlist("amenities"))
        for flag in form.getlist("luxury_finish"):
            amenities.append(f"finish:{flag}")

        payload: dict[str, Any] = {
            "city": form.get("city", "").strip(),
            "sector": form.get("sector", "").strip(),
            "property_type": form.get("property_type", "").strip(),
            "transact_type": form.get("transact_type", "").strip(),
            "bedRoom": int(form.get("bedRoom", 0)),
            "bathroom": int(form.get("bathroom", 0)),
            "balcony": form.get("balcony", "").strip(),
            "agePossession": form.get("agePossession", "").strip(),
            "built_up_area": float(form.get("built_up_area", 0)),
            "servant_room": form.get("servant_room") == "1",
            "store_room": form.get("store_room") == "1",
            "furnishing_type": form.get("furnishing_type", "").strip(),
            "floor_category": form.get("floor_category", "").strip(),
            "facing": form.get("facing", "").strip(),
            "amenities": amenities,
        }

        try:
            request_obj = PredictRequestV3.model_validate(payload)
        except ValidationError as exc:
            # First error message is enough for the user — Pydantic
            # errors are verbose for engineers but unfriendly for
            # end-users.
            first = exc.errors()[0] if exc.errors() else {"msg": "Invalid input"}
            flash(f"Please check your input: {first.get('msg', 'Invalid input')}",
                  "error")
            return redirect(url_for("predict_get"))

        client = _get_client()
        try:
            response = client.post_predict(request_obj)
        except FastAPIUnavailable:
            return render_template(
                "predict_result.html",
                unavailable=True,
                cities=list(KNOWN_CITIES),
            )

        return render_template(
            "predict_result.html",
            unavailable=False,
            response=response,
            city=request_obj.city,
            sector=request_obj.sector,
            bedRoom=request_obj.bedRoom,
            built_up_area=request_obj.built_up_area,
            transact_type=request_obj.transact_type,
            cities=list(KNOWN_CITIES),
        )

    @app.errorhandler(404)
    def _not_found(_err):
        return ("Not found", 404)

    @app.errorhandler(500)
    def _server_error(_err):
        return ("Server error", 500)

    # ---- module page stubs (per CLAUDE.md route table) ----
    # Each just renders a placeholder so `url_for()` builds the link
    # in the shared nav. The real pages are filled in by follow-on
    # specs (Day 39+ in the Implementation Plan).
    def _stub(module: str):
        def handler() -> str:
            return render_template("base.html"), 200 if False else (
                "<section class='hero'><h1>{} coming soon</h1></section>".format(module),
                200,
            )
        handler.__name__ = f"_stub_{module}"
        return handler

    for _endpoint, _label in [
        ("classify", "Classify"),
        ("analytics", "Analytics"),
        ("recommend", "Recommender"),
        ("insights", "Insights"),
        ("map_explorer", "Map"),
    ]:
        app.add_url_rule(
            f"/{_endpoint}",
            endpoint=_endpoint,
            view_func=_stub(_label),
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=FLASK_DEBUG)
