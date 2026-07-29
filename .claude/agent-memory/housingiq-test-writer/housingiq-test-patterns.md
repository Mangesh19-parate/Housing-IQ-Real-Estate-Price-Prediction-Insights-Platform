# Reusable test fixtures & naming conventions

## Fixtures (once added to conftest.py, keep this list current)
- `client` — Flask `app.test_client()`, fresh app context per test.
- `api_client` — FastAPI `TestClient` wrapping the `api/main.py` app.
- `tmp_db` — throws away a fresh SQLite file per test, `PRAGMA foreign_keys
  = ON` applied.
- `seeded_listings` — loads a small (10-20 row) fixture set of cleaned
  listings across all 4 cities, for analytics/recommender tests.
- `client_with_logged_in_user` — wraps `client` with a pre-created test user
  and an active session.

## Naming
- Files: `tests/test_<feature>.py`
- Functions: `test_<behavior>_<condition>_<expected_result>`
  e.g. `test_predict_endpoint_missing_bedroom_returns_422`
