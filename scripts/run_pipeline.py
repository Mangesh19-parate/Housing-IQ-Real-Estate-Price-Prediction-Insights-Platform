"""Pipeline entry point — TRD §13.

Each pipeline stage is wired in by the spec that owns it. Current stages:

- ingest_raw  (Step 02 — raw data ingestion and schema inventory)
- facet_decoders / canonical_mapping (Steps 04 / 05 — schema mapping)
- (cleaning, EDA, training, etc. — added by later specs)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``import scripts.ingest_raw`` works whether this file is invoked
# as ``python scripts/run_pipeline.py`` (where ``scripts/`` is just a
# directory in cwd) or via pytest with ``pythonpath = .`` (where it
# imports as a package). The scripts/ directory has an __init__.py so it
# is a real package; this just needs the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.cleaning import (  # noqa: E402
    assemble,  # noqa: F401  (Step 06 dedup + outlier flagging orchestrator)
    canonical_mapping,  # noqa: F401  (Step 05 per-city canonical schema mapping)
    facet_decoders,  # noqa: F401  (Step 04 decoders; Step 05 wires per-row)
    pipeline as _clean_pipeline,  # noqa: F401  (Step 07 impute + Parquet writer orchestrator)
)
from scripts import (  # noqa: E402,F401  (parse_check is a placeholder; Step 04 wires the real run)
    ingest_raw,
    parse_check,
)


def main() -> None:
    rc = ingest_raw.main()
    if rc != 0:
        sys.exit(rc)
    print("ingest done — see data/processed/raw_inventory.json")


if __name__ == "__main__":
    main()
