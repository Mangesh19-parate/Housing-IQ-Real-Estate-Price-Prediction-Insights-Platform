"""SHAP explainability layer for the price model (Spec 16).

Sub-modules:
    - labels         feature-label map (human-readable UI names)
    - explainer      TreeExplainer factory + joblib persistence
    - contributions  per-prediction helper consumed by /predict
    - summary        global SHAP summary + report section writer

Public API consumed by the FastAPI route layer (follow-on spec):
    - load_explainer(...)
    - explain_one(...)
    - direction_breakdown(...)
    - FEATURE_LABEL_MAP_V2()

Public API consumed by scripts/build_shap_explainer.py:
    - build_explainer(...)
    - save_explainer(...)
    - build_label_map(...)
    - save_label_map(...)
    - global_summary(...)
    - write_summary_section(...)
    - label_map_hash(...)
"""

from __future__ import annotations

__version__ = "1.0.0"

from ml.explainability.contributions import (
    Direction,
    ShapContribution,
    direction_breakdown,
    explain_one,
)
from ml.explainability.explainer import (
    EXPLAINER_VERSION,
    SHAP_TOP_N,
    build_explainer,
    load_explainer,
    save_explainer,
)
from ml.explainability.labels import (
    FEATURE_LABEL_MAP_V2,
    build_label_map,
    label_map_hash,
    load_label_map_from_disk,
    resolve_label,
    save_label_map,
)
from ml.explainability.summary import (
    GLOBAL_N_SAMPLES,
    TOP_K_SUMMARY,
    global_summary,
    write_summary_section,
)

__all__ = [
    "__version__",
    # contributions
    "ShapContribution",
    "Direction",
    "explain_one",
    "direction_breakdown",
    # explainer
    "EXPLAINER_VERSION",
    "SHAP_TOP_N",
    "build_explainer",
    "save_explainer",
    "load_explainer",
    # labels
    "FEATURE_LABEL_MAP_V2",
    "build_label_map",
    "load_label_map_from_disk",
    "resolve_label",
    "save_label_map",
    "label_map_hash",
    # summary
    "GLOBAL_N_SAMPLES",
    "TOP_K_SUMMARY",
    "global_summary",
    "write_summary_section",
]
