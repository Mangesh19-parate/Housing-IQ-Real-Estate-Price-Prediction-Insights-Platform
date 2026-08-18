"""Public API for the price-model evaluation gate (Spec 15).

Re-exports the four pinned protocol symbols so callers can write
``from ml.evaluation import PROTOCOL_VERSION, evaluate,
EvaluationResult, protocol_thresholds`` without touching the
submodule layout.
"""

from ml.evaluation import gate, protocol, report, scoring, splits
from ml.evaluation.gate import EvaluationResult, evaluate, format_summary
from ml.evaluation.protocol import (
    METRIC_NAMES,
    PROTOCOL_DOC_PATH,
    PROTOCOL_VERSION,
    RANDOM_STATE,
    SPLIT_RATIOS,
    protocol_thresholds,
)
from ml.evaluation.report import append_protocol_section, write_evaluation_report
from ml.evaluation.scoring import score_predictions, within_tolerance_pct
from ml.evaluation.splits import protocol_split

#: Semver-pinned package version. Bumped on any intentional change to
#: the gate's behavior; surfaced in every ``EvaluationResult``.
__version__: str = "1.0.0"

__all__ = [
    # protocol constants
    "METRIC_NAMES",
    "PROTOCOL_DOC_PATH",
    "PROTOCOL_VERSION",
    "RANDOM_STATE",
    "SPLIT_RATIOS",
    "protocol_thresholds",
    # gate
    "EvaluationResult",
    "evaluate",
    "format_summary",
    # report writer
    "append_protocol_section",
    "write_evaluation_report",
    # scoring
    "score_predictions",
    "within_tolerance_pct",
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
