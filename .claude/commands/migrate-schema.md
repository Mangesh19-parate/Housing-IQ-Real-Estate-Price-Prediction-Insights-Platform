You are applying a schema change to the application database.

User input: $ARGUMENTS (description of the schema change)

## Step 1 - Write the migration
Write a versioned migration script (never a manual ad hoc `ALTER TABLE`
run directly against a live DB).

## Step 2 - Test against a copy
Apply the migration to a copy of the dev DB first; verify existing queries
still work.

## Step 3 - Apply
Apply to the real dev DB once verified.

## Step 4 - Update docs
Update `05-BACKEND-SCHEMA.md` to reflect the new schema.

## Step 5 - Report
Print the migration file path and a summary of what changed.
