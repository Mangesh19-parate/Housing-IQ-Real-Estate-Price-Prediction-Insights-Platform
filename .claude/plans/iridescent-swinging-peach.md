# Implementation Plan — Step 11: Price Prediction Input Schema v3 (Pydantic Contracts)

## Context

Step 11 codifies the frozen 16-field input contract from `docs/10-FINALIZED-INPUT-SCHEMA.md` as Pydantic v2 models in `api/schemas/predict_v3.py`, plus a single source-of-truth `INPUT_FIELDS_V3` tuple (column order + types) that future training and serving code will both import. The contract is locked, so pinning it as code now — while Steps 06–10 (dedup, Parquet, log-transform, feature engineering, classification tier) are still landing — prevents the training script and the FastAPI `/predict` route from drifting on field names or typing.

Step 11 defines schemas only — no routes, no DB writes, no model artifacts. The existing `api/routers/predict.py` stub stays a stub; wiring is a later spec.

**Pre-implementation research findings** (verified against the codebase):

- `api/schemas/__init__.py` exists but is **0 bytes** — no collision risk.
- `api/schemas/predict_v3.py` does not exist — greenfield.
- **No Pydantic models anywhere** in `api/`, `app/`, `ml/`, or `models/`. The schema layer is fully greenfield.
- `pydantic==2.8.2` + `fastapi==0.111.1` already pinned in `requirements.txt` — no new dependencies.
- `pytest.ini` has `pythonpath = .`, so `from api.schemas import ...` works in tests without manipulation.
- `tests/conftest.py` fixtures (`tmp_clean_db`, `app_client`, `api_client`) are **not used** — schema tests are pure in-memory Pydantic validation.
- Ruff: line-length 100, target py310, rules `E`, `F`, `W`, `I`.
- Tests are function-based (not class-based); `tests/test_canonical_mapping.py` is the reference style. The `ast.walk` import-scan pattern at line 463 of that file is the house-style way to check forbidden imports.

**Two intentional deviations from spec prose** (the implementer must know these):

1. **`FacingDirection` has 8 members, not 9.** The spec text says "9 standard compass values"; `data/raw/facets/FACING_DIRECTION.csv` has exactly 8 (North, South, East, West, North-East, North-West, South-East, South-West). No UNFURNISHED placeholder. Implement with the 8 facet values. Note in the test file's module docstring as "8 directions per FACING_DIRECTION.csv, not 9 as spec prose".
2. **`test_predict_v3_does_not_import_app_ml_or_models`** uses `ast.walk`, not `sys.modules` introspection. The spec literal mentions `sys.modules`; the existing house pattern is `ast.parse`-based. The `ast` approach is more reliable (no double-import, no module-cache pollution). End-state is identical.

## Files to change

| Path | Action |
|---|---|
| `api/schemas/predict_v3.py` | **create** |
| `api/schemas/__init__.py` | **modify** (fill 0-byte file with re-exports) |
| `tests/test_predict_v3_schema.py` | **create** |
| `.claude/specs/11-price-prediction-input-schema-v3.md` | already exists (spec, reference only) |

**No changes** to: `requirements.txt`, `api/main.py`, `api/routers/*`, `api/services/*`, `app/`, `ml/`, `data/`, `models/`, `notebooks/`, `migrations/`, `tests/conftest.py`, `pytest.ini`, `pyproject.toml`, `CLAUDE.md`.

## Implementation steps (in order)

### Step 1 — `api/schemas/predict_v3.py`

**Imports** (exact list — the `ast.walk` test will fail on anything else):
```python
from __future__ import annotations
from enum import Enum
from typing import Final
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

**Module docstring** names `docs/10-FINALIZED-INPUT-SCHEMA.md` and `docs/05-BACKEND-SCHEMA.md` §U-SCHEMA-5 as the schema authority. Documents: camelCase `bedRoom`, snake_case `servant_room`/`store_room` at the API boundary (deliberate, per Step 05); `transact_type` is a routing key not a model feature (TRD §U-TRD-4); `luxury_category` is server-derived per Rules §10.2; this spec is API contract only — no model wiring.

**Section ordering inside the module:**
1. docstring
2. imports
3. 8 string enums (in spec order)
4. `PredictRequestV3`
5. `ShapContribution`
6. `PredictResponseV3`
7. `INPUT_FIELDS_V3` and `INPUT_FIELD_TYPES_V3`

**The 8 enums** — all `(str, Enum)`. Member names UPPER_SNAKE, `.value` is the verbatim string per `docs/10-FINALIZED-INPUT-SCHEMA.md`:

| Enum | Values |
|---|---|
| `TransactType` | `SALE = "Sale"`, `RENT = "Rent"` (case-sensitive) |
| `PropertyType` | `FLAT = "flat"`, `HOUSE = "house"` (lowercase per doc row 1; NOT facet labels) |
| `Balcony` | `ZERO = "0"`, `ONE = "1"`, `TWO = "2"`, `THREE = "3"`, `THREE_PLUS = "3+"` (strings, not ints) |
| `AgePossession` | 5 members: `New Property`, `Relatively New`, `Moderately Old`, `Old Property`, `Under Construction` (Title Case) |
| `FurnishingType` | `UNFURNISHED = "Unfurnished"`, `SEMIFURNISHED = "Semifurnished"` (no hyphen), `FURNISHED = "Furnished"` |
| `LuxuryCategory` | `LOW = "Low"`, `MEDIUM = "Medium"`, `HIGH = "High"` |
| `FloorCategory` | `LOW = "Low Floor"`, `MID = "Mid Floor"`, `HIGH = "High Floor"` (trailing " Floor") |
| `FacingDirection` | **8 members**: North, South, East, West, North-East, North-West, South-East, South-West |

**`PredictRequestV3`** — 16 fields in spec order, `extra="forbid"` + `str_strip_whitespace=True`:

| # | name | type | constraint |
|---|---|---|---|
| 1 | `city` | `str` | `min_length=1` |
| 2 | `sector` | `str` | `min_length=1` |
| 3 | `property_type` | `PropertyType` | — |
| 4 | `transact_type` | `TransactType` | — |
| 5 | `bedRoom` | `int` | `ge=1, le=15` |
| 6 | `bathroom` | `int` | `ge=1, le=15` |
| 7 | `balcony` | `Balcony` | — |
| 8 | `agePossession` | `AgePossession` | — |
| 9 | `built_up_area` | `float` | `gt=0, le=20000` |
| 10 | `servant_room` | `bool` | — |
| 11 | `store_room` | `bool` | — |
| 12 | `furnishing_type` | `FurnishingType` | — |
| 13 | `floor_category` | `FloorCategory` | — |
| 14 | `facing` | `FacingDirection` | — |
| 15 | `amenities` | `list[str]` | `Field(default_factory=list)` |
| 16 | `luxury_category` | `LuxuryCategory \| None` | `Field(default=None, exclude=True)` |

`@model_validator(mode="after")` cross-field rule: `bedRoom <= bathroom + 3`, raises `ValueError("bathroom too low for bedroom count")` on violation. Returns `self` on success.

**`ShapContribution`** — `feature: str`, `impact: float`. Plain `BaseModel` (spec didn't mandate `extra="forbid"` here; if added, the test still passes).

**`PredictResponseV3`** — 7 fields, `extra="forbid"`:
- `predicted_price: float` (`ge=0`)
- `range_low: float` (`ge=0`)
- `range_high: float` (`ge=0`)
- `shap_contributions: list[ShapContribution]` (`default_factory=list`)
- `is_outlier_input: bool`
- `model_version: str`
- `luxury_category: LuxuryCategory` (required — server-resolved)

**Constants at module bottom:**
```python
INPUT_FIELDS_V3: Final[tuple[str, ...]] = (
    "property_type", "sector", "city", "transact_type",
    "bedRoom", "bathroom", "balcony", "agePossession",
    "built_up_area", "servant_room", "store_room",
    "furnishing_type", "luxury_category", "floor_category",
    "facing", "amenities",
)

INPUT_FIELD_TYPES_V3: Final[dict[str, type]] = {
    "property_type": PropertyType, "sector": str, "city": str,
    "transact_type": TransactType, "bedRoom": int, "bathroom": int,
    "balcony": Balcony, "agePossession": AgePossession,
    "built_up_area": float, "servant_room": bool, "store_room": bool,
    "furnishing_type": FurnishingType, "luxury_category": LuxuryCategory,
    "floor_category": FloorCategory, "facing": FacingDirection,
    "amenities": list,
}
```

### Step 2 — `api/schemas/__init__.py`

Replace the empty file with re-exports for `PredictRequestV3`, `PredictResponseV3`, `ShapContribution`, `INPUT_FIELDS_V3`, `INPUT_FIELD_TYPES_V3`, and the 8 enums. Alphabetized import block + matching `__all__`. 11 names total.

### Step 3 — `tests/test_predict_v3_schema.py`

Function-based, follows `tests/test_canonical_mapping.py` style: `from __future__ import annotations` at top, `# A. Constants`-style section headers. Self-contained (no fixtures).

**Imports:**
```python
from __future__ import annotations
import ast
import re
from pathlib import Path
import pydantic
import pytest
from api.schemas import (
    INPUT_FIELDS_V3, INPUT_FIELD_TYPES_V3,
    AgePossession, Balcony, FacingDirection, FloorCategory, FurnishingType,
    LuxuryCategory, PredictRequestV3, PredictResponseV3, PropertyType,
    ShapContribution, TransactType,
)
```

**Section A — Constants (6 tests):**
- `test_input_fields_v3_has_exactly_sixteen_entries` — `len == 16`
- `test_input_fields_v3_is_tuple` — `isinstance(..., tuple)`
- `test_input_fields_v3_order_matches_reference_project` — equality with `EXPECTED_16_TUPLE` literal at module top
- `test_input_fields_v3_names_match_input_schema_doc` — reads `docs/10-FINALIZED-INPUT-SCHEMA.md` via `Path(__file__).resolve().parent.parent / "docs" / "10-FINALIZED-INPUT-SCHEMA.md"`; asserts each name appears in the doc text. No `realdata` marker (reads markdown, not `data/raw/`).
- `test_input_field_types_v3_covers_all_input_fields` — keys equality with `INPUT_FIELDS_V3`
- `test_enums_have_expected_string_values` — each enum's `.value`s match the documented list (8 members for `FacingDirection` — note the deviation)

**Section B — Request validation (15 tests):**
- `test_predict_request_v3_minimal_valid_payload` — all 15 visible fields, parses
- `test_predict_request_v3_rejects_missing_required_field` — drop `built_up_area`, raises
- `test_predict_request_v3_rejects_extra_fields` — `"unknown_field": "x"`, raises
- `test_predict_request_v3_rejects_bedroom_zero` — `bedRoom=0`, raises
- `test_predict_request_v3_rejects_bedroom_over_15` — `bedRoom=16`, raises
- `test_predict_request_v3_rejects_negative_area` — `built_up_area=-100`, raises
- `test_predict_request_v3_rejects_area_over_20000` — `built_up_area=50000`, raises
- `test_predict_request_v3_bedroom_bathroom_sanity_check` — `bedRoom=5, bathroom=1`, raises with `"bathroom"` in message
- `test_predict_request_v3_balcony_accepts_three_plus` — `balcony="3+"`, parses
- `test_predict_request_v3_transact_type_enum_values` — `"Sale"`/`"Rent"` parse; `"sale"` raises
- `test_predict_request_v3_strips_string_whitespace` — `" Gurgaon "` → `"Gurgaon"`
- `test_predict_request_v3_amenities_defaults_to_empty_list` — omit → `[]`
- `test_predict_request_v3_amenities_accepts_list_of_strings` — 2-elem list
- `test_predict_request_v3_luxury_category_excluded_from_request` — `"luxury_category": "High"` parses but `obj.luxury_category is None`
- `test_predict_request_v3_dump_excludes_luxury_category` — `.model_dump()` keys lack `"luxury_category"`

**Section C — Response validation (4 tests):**
- `test_predict_response_v3_minimal_valid_payload` — 7 fields, parses
- `test_predict_response_v3_rejects_negative_price` — `predicted_price=-1`, raises
- `test_predict_response_v3_rejects_extra_fields` — extra key raises
- `test_shap_contribution_accepts_float_impact` — `impact=0.18` parses; `impact="high"` raises

**Section D — Boundary rules (3 tests):**
- `test_no_pii_or_contact_fields` — regex `r"(contact|dealer|phone|email|photo|url|spid)"` with `re.IGNORECASE`; iterate `PredictRequestV3.model_fields.keys()` and `PredictResponseV3.model_fields.keys()`; assert no banned token substring in any name
- `test_predict_v3_does_not_import_app_ml_or_models` — `ast.parse(Path("api/schemas/predict_v3.py").read_text())`, walk imports, fail on any `app.*`, `ml.*`, or `models.*`. Mirror the pattern at `tests/test_canonical_mapping.py` line 463.
- `test_schemas_init_reexports_public_api` — single import line covering all 11 names, no `ImportError`

Total: **28 test functions** across 4 sections.

## Pitfalls to flag

- **`Field(default_factory=list)`**, not `Field(default=[])` — mutable default trap.
- **`@model_validator(mode="after")`** imported from top-level `pydantic`, not `pydantic.validator`. Method returns `self` on success, raises `ValueError` on failure.
- **`Field(default=None, exclude=True)`** — both flags together. `exclude=True` drops on parse AND dump; `default=None` makes it optional. Single declaration covers both `test_predict_request_v3_luxury_category_excluded_from_request` and `test_predict_request_v3_dump_excludes_luxury_category`.
- **`from __future__ import annotations`** is required in `predict_v3.py` because the validator signature references the class itself as a string `"PredictRequestV3"` under PEP 563 deferred evaluation.
- **Don't confuse `PropertyType` enum values with facet labels.** API uses `flat`/`house`; the raw facet CSV uses `"Residential Apartment"` etc. The API enum is the contract, not the cleaning-layer labels.
- **Don't confuse `amenities` (API) with `amenities_list` (canonical DataFrame column).** The Step 11 wire-format field is `amenities` per spec line 103.
- **`test_no_pii_or_contact_fields`** is safe with `re.IGNORECASE` — none of the 16+7 field names contains any banned token. Sanity-checked: `amenities`, `floor_category`, `luxury_category`, `predicted_price`, `range_low`, `shap_contributions`, `model_version` all pass.

## Existing functions/utilities reused

- **Test pattern**: `tests/test_canonical_mapping.py` (function-based, `from __future__ import annotations`, `# A. Constants` headers, `ast.walk` import-scan at line 463).
- **No production code reused** — the schema layer is greenfield. The only cross-reference is the **contract authority** in `docs/10-FINALIZED-INPUT-SCHEMA.md` and `docs/05-BACKEND-SCHEMA.md` §U-SCHEMA-5.

## Verification

Run each from repo root `C:\Users\HP\OneDrive\Desktop\Housing predictor`. All must pass before the PR is ready.

```bash
# 1. Targeted test suite (DoD #1) — 28 tests, all pass
python -m pytest tests/test_predict_v3_schema.py -v

# 2. Full non-realdata suite (DoD #2) — no regressions in Steps 01–08
python -m pytest -m "not realdata"

# 3. Ruff clean (DoD #3) — zero issues across the 3 changed files
ruff check api/schemas/predict_v3.py tests/test_predict_v3_schema.py api/schemas/__init__.py

# 4. Smoke-test one-liner from DoD #4 — prints "16" + a valid JSON dump with no luxury_category key
python -c "from api.schemas import PredictRequestV3, INPUT_FIELDS_V3; print(len(INPUT_FIELDS_V3)); req = PredictRequestV3(city='Gurgaon', sector='Sector 84', property_type='flat', transact_type='Sale', bedRoom=3, bathroom=3, balcony='2', agePossession='Relatively New', built_up_area=1450, servant_room=True, store_room=False, furnishing_type='Semifurnished', floor_category='Mid Floor', facing='North'); print(req.model_dump_json())"

# 5. git status — DoD #5: only 4 paths changed
git status
# Expect:
#   modified:   api/schemas/__init__.py
#   new file:   api/schemas/predict_v3.py
#   new file:   tests/test_predict_v3_schema.py
#   new file:   .claude/specs/11-price-prediction-input-schema-v3.md

# 6. CLAUDE.md untouched (DoD #6)
git diff CLAUDE.md   # expect: nothing

# 7. requirements.txt untouched (no new deps)
git diff requirements.txt   # expect: nothing
```

## Out of scope (deferred)

- Wiring `PredictRequestV3`/`PredictResponseV3` into `api/routers/predict.py` — that FastAPI route wiring is a later spec (after the model pipeline lands per the Implementation Plan's Day 27 / Week 4).
- Mapping `amenities` (API) back to `amenities_list` (canonical) at training ingestion time — Step-08+ concern, not this spec's.
- Reconciling `agePossession` values (facet-decoded vs API enum) — known tension, surfaces in feature engineering, not here.
- The 7th-day tracker update — DoD #7 marks this optional; skip unless the user requests it.