# housingiq-security-reviewer — Memory

## Non-negotiable rules (from 08-RULES.md)
- No dealer/agent contact fields, phone-like fields, or raw photo/media URLs
  in any UI, API response, or export.
- Raw CSVs under `data/raw/` are immutable.
- All SQL parameterized, no string interpolation.
- Every derived table/cache states its computation date and source dataset
  version.
- Outliers are flagged, never deleted.

## Review history
(Append a one-line entry per review: date, branch, headline finding.)
