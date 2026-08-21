"""Stable feature-list fingerprint for the model registry (Spec 20).

Picked SHA-256 over the SHA-1 used in ``scripts/train_price_model_v2.py``
because SHA-1 has known collision weaknesses and we want a hash that
will outlive the project's lifetime. The sorted-join normalisation
makes the fingerprint independent of caller-side column ordering —
two versions trained on the same features produce the same hash even
if one script happened to emit them in a different order.

Truncated to 16 hex chars (64 bits) — long enough to be unique within
a single project's feature space, short enough to scan in logs and
column displays. Raise the truncation if a future project scales
beyond ~10^19 versions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

_HASH_HEX_LEN: int = 16


def compute_feature_hash(features: Iterable[str]) -> str:
    """Return a 16-char hex digest over the sorted, deduped feature list.

    Empty input is a valid input → produces a stable hash for the empty
    case (no features). Callers should not interpret that as "missing".
    """
    normalized = "\n".join(sorted(set(features)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:_HASH_HEX_LEN]


__all__ = ["compute_feature_hash"]
