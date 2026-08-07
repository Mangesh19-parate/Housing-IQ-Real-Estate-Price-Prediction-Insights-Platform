"""CLI wrapper around ``migrations.runner.migrate``.

Usage::

    python scripts/init_db.py [--db-path PATH] [--print-version]

Defaults to ``APP_DB_PATH`` from ``app.config``. Prints ``applied 001_initial``
on the first run, ``already up to date`` on subsequent runs, exits 0.
Exits non-zero on any migration failure with the failed filename and the
exception class on stderr (no full traceback spam).

Mirrors the CLI pattern of ``scripts/ingest_raw.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when invoked as ``python scripts/init_db.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import APP_DB_PATH  # noqa: E402  (path-adjusted import)
from migrations.runner import current_version, migrate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pending migrations against the HousingIQ application DB.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--db-path",
        type=Path,
        default=Path(APP_DB_PATH),
        help="Path to the SQLite DB file. Defaults to APP_DB_PATH from app.config.",
    )
    group.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Postgres connection URL (e.g. postgresql://user:pass@host/db). "
             "If given, overrides --db-path.",
    )
    p.add_argument(
        "--print-version",
        action="store_true",
        help="Print the current highest applied version and exit (no migrations run).",
    )
    return p.parse_args()


def _resolve_target(args: argparse.Namespace) -> str | None:
    return args.db_url if args.db_url is not None else str(args.db_path)


def main() -> int:
    args = _parse_args()
    target = _resolve_target(args)

    if args.print_version:
        version = current_version(target)
        print(version if version is not None else "(no migrations applied)")
        return 0

    try:
        applied = migrate(db_path_or_url=target, source="cli")
    except Exception as exc:
        print(f"ERROR: migration failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    if applied:
        for record in applied:
            print(f"applied {record.version}")
    else:
        print("already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
