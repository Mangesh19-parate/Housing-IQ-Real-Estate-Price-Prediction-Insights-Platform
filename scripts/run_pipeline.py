"""Pipeline entry point — TRD §13.

Each pipeline stage is wired in by the spec that owns it. Current stages:

- ingest_raw  (Step 02 — raw data ingestion and schema inventory)
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

from scripts import ingest_raw  # noqa: E402  (path-adjusted import)


def main() -> None:
    rc = ingest_raw.main()
    if rc != 0:
        sys.exit(rc)
    print("ingest done — see data/processed/raw_inventory.json")


if __name__ == "__main__":
    main()
