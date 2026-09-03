"""CLI gate for the classification model evaluation protocol (Spec 24).

Usage::

    python scripts/evaluate_classifiers.py \
        --version v1 \
        --target good_deal \
        --target price_tier \
        [--processed-dir data/processed] \
        [--models-dir models] \
        [--feature-frame data/processed/classifier/feature_frame.parquet] \
        [--labels-dir data/processed/classifier] \
        [--report-path models/classification_training_report.md]

For each ``--target`` flag, the gate:
    1. Calls ``ml.evaluation.classification.gate.evaluate(...)`` to score the model
       against the pinned protocol.
    2. Calls ``ml.evaluation.classification.report.write_evaluation_report`` to
       persist a versioned JSON report.
    3. Calls ``ml.evaluation.classification.report.append_protocol_section`` to
       append a "Protocol Certification" section to the classification
       training report.
    4. Prints a one-line ``[PASS|FAIL]`` summary.

Exit code: ``0`` iff every ``target``'s ``overall_passed == True``, else ``1``.
The pipeline calls this with ``check=False`` — a FAIL is a signal for review,
not a reason to abort unrelated stages.

No new pip packages — stdlib argparse + the ``ml.evaluation.classification`` package
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

from ml.evaluation.classification import (  # noqa: E402
    evaluate,
    format_summary,
)
from ml.evaluation.classification.report import (  # noqa: E402
    append_protocol_section,
    write_evaluation_report,
)

logger = logging.getLogger("evaluate_classifiers")

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
            "Evaluate trained classification models against the pinned protocol "
            "(TRD §U-TRD-8 / Rules §2.1). Exits 0 iff the models clear every "
            "threshold; 1 otherwise."
        )
    )
    p.add_argument(
        "--version",
        required=True,
        help="Model version to certify (filename suffix, e.g. 'v1').",
    )
    p.add_argument(
        "--target",
        choices=["good_deal", "price_tier"],
        action="append",
        required=True,
        help=(
            "Target to certify. Repeatable — both --target good_deal and "
            "--target price_tier are typical."
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
        help="Directory containing classifier/feature_frame.parquet.",
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
        help="Directory containing classifier_*_v{n}.pkl artifacts.",
    )
    p.add_argument(
        "--feature-frame",
        type=Path,
        default=None,
        help="Path to feature_frame.parquet (default: processed-dir/classifier/feature_frame.parquet).",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory containing good_deal_labels.parquet and price_tier_labels.parquet (default: processed-dir/classifier).",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_REPORT_PATH",
                str(_REPO_ROOT / "models" / "classification_training_report.md"),
            )
        ),
        help="Path to the classification training report markdown file.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    rc = 0
    for target in args.target:
        version_stripped = args.version.lstrip("v")
        model_path = args.models_dir / f"{target}_classifier_v{version_stripped}.pkl"
        if not model_path.exists():
            logger.error(
                "Model artifact not found at %s — run the training script first.",
                model_path,
            )
            print(f"[FAIL] {target}_{args.version} model_not_found at {model_path}")
            rc = 1
            continue

        try:
            result = evaluate(
                model_path=model_path,
                version=args.version,
                target=target,
                processed_dir=args.processed_dir,
                models_dir=args.models_dir,
                feature_frame_path=args.feature_frame,
                labels_dir=args.labels_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "evaluate() raised for %s_%s: %s",
                target,
                args.version,
                exc,
            )
            print(f"[FAIL] {target}_{args.version} evaluate_raised: {exc}")
            rc = 1
            continue

        # Persist artifacts
        try:
            write_evaluation_report(result, args.models_dir)
            append_protocol_section(result, args.report_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to persist report for %s_%s: %s",
                target,
                args.version,
                exc,
            )

        # One-line summary for stdout
        print(format_summary(result))

        if not result.overall_passed:
            rc = 1

    if rc != 0:
        logger.warning(
            "Gate FAILED for at least one target (git=%s).",
            _git_commit(),
        )
    else:
        logger.info(
            "Gate PASSED for all %d target(s) (git=%s).",
            len(args.target),
            _git_commit(),
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())