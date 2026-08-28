"""Public API for the Classification target builder (Spec 21).

Re-exports the five pinned public symbols so callers can write
``from ml.classification import build_price_tier_labels,
build_good_deal_labels, calibrate_good_deal_thresholds, TIER_LABELS,
VERDICT_LABELS`` without touching the submodule layout.
"""

from ml.classification import report, thresholds, tiers, verdicts
from ml.classification.report import (
    write_good_deal_artifacts,
    write_label_construction_report,
    write_price_tier_artifacts,
)
from ml.classification.thresholds import (
    MIN_CITY_TRAIN_ROWS,
    calibrate_good_deal_thresholds,
)
from ml.classification.tiers import (
    QUANTILE_CUTPOINTS,
    TIER_LABELS,
    build_price_tier_labels,
    compute_per_city_quantile_boundaries,
)
from ml.classification.verdicts import (
    DEFAULT_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD_LOW,
    VERDICT_LABELS,
    build_good_deal_labels,
)

#: Semver-pinned package version. Bumped on any intentional change to
#: the label-construction behavior; surfaced in the report's run
#: section so a reviewer can audit drift over time.
__version__: str = "1.0.0"

__all__ = [
    # constants
    "DEFAULT_THRESHOLD_HIGH",
    "DEFAULT_THRESHOLD_LOW",
    "MIN_CITY_TRAIN_ROWS",
    "QUANTILE_CUTPOINTS",
    "TIER_LABELS",
    "VERDICT_LABELS",
    # tier builder
    "build_price_tier_labels",
    "compute_per_city_quantile_boundaries",
    # verdict builder
    "build_good_deal_labels",
    # threshold calibrator
    "calibrate_good_deal_thresholds",
    # report writer
    "write_good_deal_artifacts",
    "write_label_construction_report",
    "write_price_tier_artifacts",
    # submodules
    "report",
    "thresholds",
    "tiers",
    "verdicts",
    # version
    "__version__",
]
