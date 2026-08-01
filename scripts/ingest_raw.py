"""CLI wrapper around ``ml.cleaning.ingest.run_ingestion``.

Usage::

    python scripts/ingest_raw.py [--data-dir data] [--output-dir data/processed]

Prints a one-line summary per city, then a total. Exits non-zero on any
empty city or unreadable facet file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when invoked as ``python scripts/ingest_raw.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.cleaning import ingest  # noqa: E402  (path-adjusted import)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run raw data ingestion pipeline stage.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data dir (containing data/raw/). Default: data/",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write the 6 output artifacts. Default: data/processed/",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = ingest.run_ingestion(args.data_dir, args.output_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    # Read back the inventory to print a per-city summary.
    inv_path = args.output_dir / ingest.OUTPUT_FILES["raw_inventory"]
    coverage_path = args.output_dir / ingest.OUTPUT_FILES["coverage"]
    schema_map_path = args.output_dir / ingest.OUTPUT_FILES["schema_map"]

    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    coverage_rows = coverage_path.read_text(encoding="utf-8").splitlines()
    schema_map = json.loads(schema_map_path.read_text(encoding="utf-8"))

    # Map city → mean join match rate across that city's coverage rows.
    n_rows = max(len(coverage_rows) - 1, 1)
    by_city: dict[str, list[float]] = {}
    import csv as _csv

    with coverage_path.open(encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                rate = float(row.get("join_match_rate", 0.0) or 0.0)
            except ValueError:
                rate = 0.0
            by_city.setdefault(row["city"], []).append(rate)

    print(f"source_version: {result['source_version']}")
    for city, info in inventory["cities"].items():
        cols = info["columns"]
        pii_count = sum(1 for c in cols if c["pii_risk"])
        rates = by_city.get(city, [])
        avg_rate = round((sum(rates) / len(rates)) if rates else 0.0, 3)
        print(
            f"{city}: {info['rows']} rows, {info['total_columns']} cols, "
            f"{pii_count} pii-flagged, {avg_rate:.0%} avg facet coverage"
        )
    pending = len(schema_map.get("pending_review", []))
    print(f"total: {len(inventory['cities'])} cities, {n_rows} coverage rows, "
          f"{pending} columns pending review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
