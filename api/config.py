"""FastAPI-side runtime configuration.

Reads environment variables via python-dotenv; falls back to sensible local-dev
defaults. No Flask imports — kept deliberately separate from ``app/config.py``
so FastAPI can never accidentally render a Jinja template.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

# APP_DB_PATH is shared with Flask so both services log to the same SQLite file.
APP_DB_PATH = os.environ.get("APP_DB_PATH", "data/app.db")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
ANALYTICS_CACHE_DIR = os.environ.get("ANALYTICS_CACHE_DIR", "data/processed/analytics_cache")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
