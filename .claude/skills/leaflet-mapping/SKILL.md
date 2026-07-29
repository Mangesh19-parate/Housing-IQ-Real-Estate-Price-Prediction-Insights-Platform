# Skill: Leaflet Map Integration

**Trigger:** Building the map/spatial view (locality markers, map-based property explorer).

## Use this skill when
- Adding or editing the Leaflet map component
- Wiring locality/geocoded coordinates onto the map

## Key conventions (binding for this project)
- Leaflet.js via CDN, no other mapping library
- Never plot raw dealer/agent location data — only locality-level or listing-level (post-privacy-scrub) coordinates, per Rules doc §1.1
- Marker clustering required once a locality has more than ~30 points, to avoid an unreadable map

## Workflow
1. Fetch geocoded locality/listing points from the backend (never geocode client-side per request)
2. Render markers with clustering; wire click → card popup using the shared card component

## Gotchas / things that have bitten us before
- Geocoding lookups should be cached (spec `48-locality-geocoding-lookup`) — don't call an external geocoder per page load

## Cross-references
Consult `CLAUDE.md`, the relevant `.claude/specs/*.md`, and the original docs (`02-TRD.md`, `05-BACKEND-SCHEMA.md`, `08-RULES.md`) before deviating from this skill.
