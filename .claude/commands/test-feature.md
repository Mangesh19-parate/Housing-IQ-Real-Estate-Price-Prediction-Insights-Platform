You are running the test suite scoped to a single feature/spec.

User input: $ARGUMENTS (spec number, slug, or test file path)

## Step 1 - Resolve scope
Map $ARGUMENTS to the relevant spec and its test file(s) under `tests/`.
If no test file exists yet, tell the user to invoke the
`housingiq-test-writer` agent first.

## Step 2 - Run
```
pytest tests/test_<feature>.py -v
```
For FastAPI-only features, also run the relevant `api/` test module.

## Step 3 - Summarize
Report pass/fail counts, and for every failure:
- The assertion that failed
- The spec line it was validating
- A one-line hypothesis for the cause (do not attempt a fix unless asked)

## Step 4 - Update tracker
If all tests pass and this closes out a spec's Definition of Done, remind
the user to run `/update-tracker`.
