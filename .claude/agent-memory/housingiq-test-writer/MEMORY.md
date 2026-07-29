# housingiq-test-writer — Memory

## Project-specific conventions (do not re-derive)
- Price prediction tests must assert on the **original ₹ scale**, not
  log-price — a passing test on log scale that fails on ₹ scale has happened
  before; always convert back before asserting.
- Classification tests must never assert against a specific price value as
  input feature — price-derived fields are excluded from that model's inputs.
- `client_with_logged_in_user` fixture already handles session cookies —
  don't hand-roll login in individual tests.
- SHAP-related tests should assert on the *shape* and *sign* of explanation
  output, not exact float values (SHAP values shift slightly between library
  versions).

## Known gaps as of last run
- No Postgres-backed test fixture exists yet — all DB tests run against
  SQLite. If a spec explicitly targets Postgres-only behavior, flag this
  gap rather than assuming a fixture exists.
- Analytics cache tests currently mock the cache file directly rather than
  running the real `/build-analytics-cache` job end-to-end — acceptable for
  unit tests, but flag if a spec needs an integration-level test instead.

## Specs validated against so far
(Update this list every time you write tests for a new spec.)
- None yet — this project is freshly scaffolded.
