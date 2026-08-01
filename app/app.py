"""Flask web app — pages, forms, rendering ONLY.

Per CLAUDE.md: Flask never imports model code or touches ``.pkl`` files.
Inference goes through the FastAPI service over HTTP (see ``api/``).
"""

from __future__ import annotations

from flask import Flask, render_template

from app.config import FLASK_DEBUG, FLASK_SECRET_KEY
from app.database.db import init_db

# Initialized on first request — flag prevents init_db() running twice per process.
_db_initialized = False


def create_app() -> Flask:
    """Application factory. Tests and ``__main__`` both go through here."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = FLASK_SECRET_KEY

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
        cities = ["Gurgaon", "Hyderabad", "Kolkata", "Mumbai"]
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

    @app.errorhandler(404)
    def _not_found(_err):
        return ("Not found", 404)

    @app.errorhandler(500)
    def _server_error(_err):
        return ("Server error", 500)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=FLASK_DEBUG)
