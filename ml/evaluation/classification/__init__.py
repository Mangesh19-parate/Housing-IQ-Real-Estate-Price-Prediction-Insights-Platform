"""Public API for the classification-model evaluation gate (Spec 24).

Re-exports the four pinned protocol symbols so callers can write
``from ml.evaluation.classification import PROTOCOL_VERSION, evaluate,
ClassificationEvaluationResult, CLASSIFIER_THRESHOLDS`` without touching the
submodule layout.
"""

from ml.evaluation.classification import gate, protocol, report, scoring, splits
from ml.evaluation.classification.gate import (
    ClassificationEvaluationResult,
    evaluate,
    format_summary,
)
from ml.evaluation.classification.protocol import (
    CLASSIFIER_THRESHOLDS,
    METRIC_NAMES,
    PROTOCOL_DOC_PATH,
    PROTOCOL_VERSION,
    RANDOM_STATE,
    SPLIT_RATIOS,
)
from ml.evaluation.classification.report import (
    append_protocol_section,
    write_evaluation_report,
)
from ml.evaluation.classification.scoring import (
    confusion_matrix_dict,
    per_class_metrics,
    score_classifier,
)
from ml.evaluation.classification.splits import protocol_split

#: Semver-pinned package version. Bumped on any intentional change to
#: the gate's behavior; surfaced in every ``ClassificationEvaluationResult``.
__version__: str = "1.0.0"

__all__ = [
    # protocol constants
    "CLASSIFIER_THRESHOLDS",
    "METRIC_NAMES",
    "PROTOCOL_DOC_PATH",
    "PROTOCOL_VERSION",
    "RANDOM_STATE",
    "SPLIT_RATIOS",
    # gate
    "ClassificationEvaluationResult",
    "evaluate",
    "format_summary",
    # report writer
    "append_protocol_section",
    "write_evaluation_report",
    # scoring
    "confusion_matrix_dict",
    "per_class_metrics",
    "score_classifier",
    # splits
    "protocol_split",
    # submodules (for tests / advanced callers)
    "gate",
    "protocol",
    "report",
    "scoring",
    "splits",
    # version
    "__version__",
]