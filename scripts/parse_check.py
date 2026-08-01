"""Placeholder parse run — owned by Step 04 (canonical cleaning).

Step 03 ships the pure-function parsers (``ml.cleaning.parsing``) and their
unit tests. Step 04 wires the actual ``parse_price()`` / ``parse_area()`` run
over the raw CSVs and writes ``clean_listings.parquet``.

This module exists only so ``scripts/run_pipeline.py`` can import it as a
no-op wiring placeholder — the import must resolve without error to keep the
single-entry-point contract from TRD §13 working end-to-end. The real entry
point (``main()`` below) is intentionally a no-op until Step 04 lands.
"""

from __future__ import annotations


def main() -> int:
    """No-op entry point. Step 04 replaces this body with the real parse run."""
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
