You are creating a test user and an authenticated session for manual QA.

User input: $ARGUMENTS (optional: email, city preference)

## Step 1 - Create the user
Insert a user row via `app/database/db.py` helpers (never raw SQL inline),
with a hashed password (werkzeug), matching the registration spec's rules.

## Step 2 - Start a session
Log the user in via the app's real login flow (do not bypass hashing/session
logic) so the resulting session matches production behavior exactly.

## Step 3 - Report
Print the test user's email and a note that the password is a fixed dev-only
value (never reuse this seeding path against a prod DB).
