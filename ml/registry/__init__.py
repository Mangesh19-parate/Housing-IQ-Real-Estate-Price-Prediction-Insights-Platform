"""Model registry (Spec 20) — single source of truth for "which model is live?".

Re-exports the public surface from the three modules below. Keep this
list minimal — only helpers that get called from outside ``ml/registry``
belong here. ``_row_to_dict`` and the module-private functions stay
un-exported so callers don't grow dependencies on internals.
"""

from __future__ import annotations

from ml.registry.feature_hash import compute_feature_hash
from ml.registry.naming import artifact_path, metrics_path, next_version
from ml.registry.registry import (
    get_active,
    get_active_artifact,
    list_models,
    register_model,
    set_active,
)

__all__ = [
    "artifact_path",
    "compute_feature_hash",
    "get_active",
    "get_active_artifact",
    "list_models",
    "metrics_path",
    "next_version",
    "register_model",
    "set_active",
]
