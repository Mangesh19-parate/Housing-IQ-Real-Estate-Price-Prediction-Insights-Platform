"""CLI gate for the price model evaluation protocol (Spec 15).

Usage::

    python scripts/evaluate_price_model.py \\
        --version v2 \\
        --transact-type sale \\
        --transact-type rent \\
        [--fastapi-url http://localhost:8000] \\
        [--processed-dir data/processed] \\
        [--models-dir models] \\
        [--report-path data/processed/feature_selection_report.md]

For each ``--transact-type`` flag, the gate:
    1. Calls ``ml.evaluation.gate.evaluate(...)`` to score the model
       against the pinned protocol.
    2. Calls ``ml.evaluation.report.write_evaluation_report`` to
       persist a versioned JSON report.
    3. Calls ``ml.evaluation.report.append_protocol_section`` to
       append a "Protocol Certification" section to the feature
       selection report.
    4. Prints a one-line ``[PASS|FAIL]`` summary.

Exit code: ``0`` iff every ``transact_type``'s ``overall_passed ==
True``, else ``1``. The pipeline calls this with ``check=False`` —
a FAIL is a signal for review, not a reason to abort unrelated
stages.

No new pip packages — stdlib argparse + the ``ml.evaluation`` package
already in scope.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force UTF-8 stdout/stderr on Windows (cp1252 default) — the
# ``≥`` and ``±`` glyphs in ``format_summary`` + ``append_protocol_section``
# would otherwise crash with ``UnicodeEncodeError``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml.evaluation import (  # noqa: E402
    evaluate,
    format_summary,
)
from ml.evaluation.report import (  # noqa: E402
    append_protocol_section,
    write_evaluation_report,
)

logger = logging.getLogger("evaluate_price_model")

#: Pinned regex for contact/PII/URL column names. Mirrors the
#: pinned literal in ``scripts/train_price_model.py:79``. The CLI's
#: stdout must never log any column matching this regex (Rules §1.1).
CONTACT_FIELD_REGEX: str = r"(contact|dealer|phone|email|photo|url|spid)"


def _git_commit() -> str:
    """12-char git short SHA; ``"unknown"`` if not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained price model against the pinned protocol "
            "(TRD §10 / Rules §2.1). Exits 0 iff the model clears every "
            "threshold; 1 otherwise."
        )
    )
    p.add_argument(
        "--version",
        required=True,
        choices=["v1", "v2"],
        help="Model version to certify (filename suffix, e.g. 'v2').",
    )
    p.add_argument(
        "--transact-type",
        choices=["sale", "rent"],
        action="append",
        required=True,
        help=(
            "Transact type to certify. Repeatable — both --transact-type "
            "sale and --transact-type rent are typical."
        ),
    )
    p.add_argument(
        "--fastapi-url",
        default=None,
        help=(
            "If set, POST /predict against this URL to measure p95 "
            "latency. Optional; offline runs leave latency_p95_ms=null."
        ),
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_PROCESSED_DIR",
                str(_REPO_ROOT / "data" / "processed"),
            )
        ),
        help="Directory containing clean_listings.parquet.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_ARTIFACT_DIR",
                str(_REPO_ROOT / "models"),
            )
        ),
        help="Directory containing price_model_*_v{n}.pkl artifacts.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_REPORT_PATH",
                str(
                    _REPO_ROOT
                    / "data"
                    / "processed"
                    / "feature_selection_report.md"
                ),
            )
        ),
        help="Path to the feature selection report markdown file.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    rc = 0
    for transact_type in args.transact_type:
        model_path = (
            args.models_dir
            / f"price_model_{transact_type}_{args.version}.pkl"
        )
        if not model_path.exists():
            logger.error(
                "Model artifact not found at %s — run the training "
                "script first.",
                model_path,
            )
            print(
                f"[FAIL] {transact_type}_{args.version} "
                f"model_not_found at {model_path}"
            )
            rc = 1
            continue

        try:
            result = evaluate(
                model_path=model_path,
                version=args.version,
                transact_type=transact_type,
                processed_dir=args.processed_dir,
                models_dir=args.models_dir,
                parquet_path=args.processed_dir / "clean_listings.parquet",
                fastapi_url=args.fastapi_url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "evaluate() raised for %s_%s: %s",
                transact_type,
                args.version,
                exc,
            )
            print(
                f"[FAIL] {transact_type}_{args.version} "
                f"evaluate_raised: {exc}"
            )
            rc = 1
            continue

        # Persist artifacts
        try:
            write_evaluation_report(result, args.models_dir)
            append_protocol_section(result, args.report_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to persist report for %s_%s: %s",
                transact_type,
                args.version,
                exc,
            )

        # One-line summary for stdout
        print(format_summary(result))

        if not result.overall_passed:
            rc = 1

    if rc != 0:
        logger.warning(
            "Gate FAILED for at least one transact_type (git=%s).",
            _git_commit(),
        )
    else:
        logger.info(
            "Gate PASSED for all %d transact_type(s) (git=%s).",
            len(args.transact_type),
            _git_commit(),
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
