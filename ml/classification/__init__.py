"""Public API for the Classification module (Specs 21, 22, 23).

Re-exports the pinned public symbols so callers can write
``from ml.classification import ...`` without touching the submodule layout.
"""

from ml.classification import (
    feature_report,
    features,
    report,
    thresholds,
    tiers,
    training,      # Spec 23
    training_report,  # Spec 23
    verdicts,
)
from ml.classification.feature_report import (
    write_feature_frame_artifact,
    write_feature_frame_report,
)
from ml.classification.features import (
    DROPPED_FOR_LEAKAGE,
    RETAINED_LOCALITY_COLUMN,
    build_classifier_feature_frame,
)
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
from ml.classification.training import (  # Spec 23
    GOOD_DEAL_LABELS,
    PRICE_TIER_LABELS,
    evaluate_classifier,
    train_good_deal_classifier,
    train_price_tier_classifier,
)
from ml.classification.training_report import (  # Spec 23
    write_training_report,
)
from ml.classification.verdicts import (
    DEFAULT_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD_LOW,
    VERDICT_LABELS,
    build_good_deal_labels,
)

#: Semver-pinned package version. Bumped on any intentional change to
#: the label-construction behavior (1.1.0 → 1.2.0 added Spec 23's
#: training + report writer); surfaced in the report's run section so a
#: reviewer can audit drift over time.
__version__: str = "1.2.0"

__all__ = [
    # constants
    "DEFAULT_THRESHOLD_HIGH",
    "DEFAULT_THRESHOLD_LOW",
    "DROPPED_FOR_LEAKAGE",
    "MIN_CITY_TRAIN_ROWS",
    "QUANTILE_CUTPOINTS",
    "RETAINED_LOCALITY_COLUMN",
    "TIER_LABELS",
    "VERDICT_LABELS",
    # Spec 23 constants
    "GOOD_DEAL_LABELS",
    "PRICE_TIER_LABELS",
    # tier builder
    "build_price_tier_labels",
    "compute_per_city_quantile_boundaries",
    # verdict builder
    "build_good_deal_labels",
    # threshold calibrator
    "calibrate_good_deal_thresholds",
    # feature-frame builder (Spec 22)
    "build_classifier_feature_frame",
    # Spec 23 training
    "evaluate_classifier",
    "train_good_deal_classifier",
    "train_price_tier_classifier",
    # report writers (Spec 21)
    "write_good_deal_artifacts",
    "write_label_construction_report",
    "write_price_tier_artifacts",
    # report writers (Spec 22)
    "write_feature_frame_artifact",
    "write_feature_frame_report",
    # report writer (Spec 23)
    "write_training_report",
    # submodules
    "feature_report",
    "features",
    "report",
    "thresholds",
    "tiers",
    "training",
    "training_report",
    "verdicts",
    # version
    "__version__",
]
