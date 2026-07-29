# housingiq-quality-reviewer — Memory

## Binding conventions (from CLAUDE.md)
- FastAPI = inference only. Flask = pages/session/auth only. Any PR that
  blurs this line is a BLOCKING finding.
- All internal links use `url_for()`.
- DB access only through `app/database/db.py` helpers.

## Known accepted deviations
- None recorded yet — this project is freshly scaffolded.

## Review history
(Append a one-line entry per review: date, branch, headline finding.)
