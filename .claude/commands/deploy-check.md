You are running pre-deployment checks for FastAPI and/or Flask.

## Step 1 - Config check
Confirm required environment variables (DB URL, model version, API base URL)
are documented and not hardcoded anywhere in the diff.

## Step 2 - Health checks
Confirm both services expose a health-check endpoint and that it currently
returns healthy in the target environment.

## Step 3 - Model version check
Confirm the model version referenced by the API matches the latest
evaluated-and-approved artifact in `models/` (per `housingiq-ml-evaluator`
memory) - flag if it's stale.

## Step 4 - Report
Print a go/no-go summary with any blocking issues listed first.
