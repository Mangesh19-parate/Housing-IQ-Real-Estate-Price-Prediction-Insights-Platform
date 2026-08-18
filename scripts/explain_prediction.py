"""CLI for per-prediction SHAP explanations (Spec 16).

Usage::

    python scripts/explain_prediction.py \\
        --version v2 \\
        --transact-type sale \\
        --input-json '{"num__bedRoom": 2, "num__bathroom": 2, "num__built_up_area": 1200, ...}' \\
        [--models-dir models] \\
        [--report-path data/processed/feature_selection_report.md]

The CLI:
    1. Loads the SHAP explainer artifact ``shap_explainer_{transact_type}_{version}.pkl``
       from ``--models-dir``.
    2. Loads the label map from ``feature_label_map_{version}.json``.
    3. Transforms the input JSON via the pipeline[:-1] (all steps except the tree estimator).
    4. Calls ``explain_one()`` to produce per-feature ``ShapContribution`` objects.
    5. Prints a markdown table of the top-7 contributions and exits 0.
       If the explainer artifact is missing, exits 1 with an error message.

Exit code: ``0`` iff a valid explanation was produced, else ``1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.explainability.explainer import (  # noqa: E402
    load_explainer,  # noqa: F401
    EXPLAINER_VERSION,  # noqa: F401
)
from ml.explainability.contributions import explain_one  # noqa: F401
from ml.explainability.labels import load_label_map_from_disk  # noqa: F401
from ml.pipeline import create_pipeline  # noqa: F401 - minimal pipeline loader

logger = logging.getLogger("explain_prediction")

#: Regex mirroring Rules §1.1 contact/PII/URL column filter.
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
            "Generate a per-prediction SHAP explanation for a single listing. "
            "Loads the explainer artifact and produces a markdown table of "
            "top-7 feature contributions."
        )
    )
    p.add_argument(
        "--version",
        required=True,
        choices=["v1", "v2"],
        help="Model version to use (e.g. 'v2').",
    )
    p.add_argument(
        "--transact-type",
        choices=["sale", "rent"],
        required=True,
        help="Transact type for the explainer artifact.",
    )
    p.add_argument(
        "--input-json",
        required=True,
        help=(
            "JSON string of the canonical 12 input features as accepted "
            "by the price model. Keys must be internal names (e.g. "
            '"num__bedRoom", "num__built_up_area"). '
            "See ``10-FINALIZED-INPUT-SCHEMA.md`` for the full contract."
        ),
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
        help="Directory containing ``shap_explainer_{transact_type}_v{n}.pkl`` artifacts.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            os.environ.get(
                "HOUSINGIQ_REPORT_PATH",
                str(_REPO_ROOT / "data" / "processed" / "feature_selection_report.md"),
            )
        ),
        help="Not used in this CLI; kept for parity with the build CLI.",
    )
    return p.parse_args()
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    # Load explainer artifact
    explainer_path = (
        args.models_dir
        / f"shap_explainer_{args.transact_type}_{args.version}.pkl"
    )
    if not explainer_path.exists():
        logger.error(
            "SHAP explainer artifact not found at %s", explainer_path
        )
        print(
            f"[FAIL] Explainer not found at {explainer_path}. "
            "Run scripts/build_shap_explainer.py first."
        )
        return 1

    try:
        import joblib

        explainer = joblib.load(explainer_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load explainer from %s: %s", explainer_path, exc)
        print(f"[FAIL] Failed to load explainer: {exc}")
        return 1

    # Load label map
    label_map_path = args.models_dir / f"feature_label_map_{args.version}.json"
    label_map = load_label_map_from_disk(label_map_path)
    logger.info("Loaded label map from %s", label_map_path)

    # Parse input JSON
    try:
        input_dict = json.loads(args.input_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON input: %s", exc)
        print(f"[FAIL] Invalid JSON input: {exc}")
        return 1

    # Validate no contact/PII fields in input
    import re
    for key in input_dict:
        if re.search(CONTACT_FIELD_REGEX, key, re.IGNORECASE):
            logger.error("Input contains contact/PII field: %s", key)
            print(f"[FAIL] Input contains disallowed field: {key}")
            return 1

    # Transform input via pipeline[:-1] (preprocessor steps only)
    try:
        from sklearn.pipeline import Pipeline
        import numpy as np

        # Load the price model pipeline to get the preprocessor
        price_model_path = (
            args.models_dir / f"price_model_{args.transact_type}_{args.version}.pkl"
        )
        if not price_model_path.exists():
            logger.error("Price model artifact not found at %s", price_model_path)
            print(f"[FAIL] Price model not found at {price_model_path}")
            return 1

        price_model = joblib.load(price_model_path)
        if isinstance(price_model, Pipeline):
            # Transform using all steps except the last (tree estimator)
            preprocessor = price_model.named_steps["preproc"]
            feature_names_in = list(price_model.named_steps["preproc"].get_feature_names_out())
            # Convert input dict to array
            # Build feature vector in the correct order
            input_array = np.zeros((1, len(feature_names_in)))
            for i, name in enumerate(feature_names_in):
                if name in input_dict:
                    input_array[0, i] = float(input_dict[name])
            # Transform with preprocessor
            transformed = preprocessor.transform(input_array)
        else:
            # Not a pipeline - just use the input as features
            feature_names_in = list(input_dict.keys())
            transformed = np.array([list(input_dict.values())], dtype=float)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to transform input: %s", exc)
        print(f"[FAIL] Failed to transform input: {exc}")
        return 1

    # Call explain_one
    try:
        # The transformed features need feature_names matching what the explainer expects
        # For now, use the preprocessor feature names
        feature_names = feature_names_in if isinstance(price_model, Pipeline) else list(input_dict.keys())

        contributions = explain_one(
            explainer=explainer,
            request_features=transformed,
            feature_names=feature_names,
            label_map=label_map,
            top_n=7,
        )

        # Print markdown table
        print("| Rank | Feature | Label | Impact | Direction |")
        print("|------|---------|-------|--------|-----------|")
        for rank, c in enumerate(contributions, start=1):
            print(f"| {rank} | `{c.feature}` | {c.label} | {c.impact:.6f} | {c.direction} |")

        # Also print direction breakdown
        from ml.explainability.contributions import direction_breakdown
        breakdown = direction_breakdown(contributions)
        print(f"\nDirection breakdown: {breakdown['up']} up, {breakdown['down']} down")

        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate explanation: %s", exc)
        print(f"[FAIL] Failed to generate explanation: {exc}")
        return 1
if __name__ == "__main__":
    sys.exit(main())