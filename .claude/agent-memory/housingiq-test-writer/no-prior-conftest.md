# Fixtures known NOT to exist yet in conftest.py

Check this before assuming a fixture is available — it's easy to reference
a fixture "because it sounds like it should exist" and get a confusing
collection error instead of a clear signal.

- `postgres_db` — not implemented, SQLite-only for now.
- `mock_shap_explainer` — not implemented; SHAP tests currently load the
  real (small) test-fixture model directly.
- `authenticated_admin_client` — no admin role exists in the schema yet;
  don't write tests assuming one.
