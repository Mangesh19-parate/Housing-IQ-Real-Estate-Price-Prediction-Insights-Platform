"""Flask-side runtime configuration.

Reads environment variables via python-dotenv; falls back to sensible local-dev
defaults. No model code, no FastAPI imports — kept deliberately separate from
``api/config.py`` so Flask can never accidentally touch the inference service.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Search starts at this file's parent
# and walks up; on a real checkout the .env lives at the repo root.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

APP_DB_PATH = os.environ.get("APP_DB_PATH", "data/app.db")
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-only-not-a-real-secret")
