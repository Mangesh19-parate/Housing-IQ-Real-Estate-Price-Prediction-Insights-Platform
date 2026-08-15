"""Artifact persistence for the price baseline (Spec 13).

Writes:
    ``models/price_model_{transact_type.lower()}_v{n}.pkl`` — joblib
        dump of the full ``sklearn.Pipeline`` (preprocessor + estimator).
    ``models/metrics_v{n}.json`` — full per-candidate metrics payload.
    ``data/model_registry.csv`` — append-only row per trained
        ``(model_name, version)`` triple (Backend Schema §U-SCHEMA-13).

All three writers honour the ``HOUSINGIQ_ARTIFACT_DIR`` env var so
tests can redirect to ``tmp_path`` without monkey-patching the
production default. ``append_model_registry`` is idempotent on the
``(model_name, version, git_commit)`` triple so re-running the
training script produces no duplicate rows (Rules §2.5).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

#: v1 baseline version (Rules §2.5). Pinned at module level — a future
#: retrain bumps to ``"v2"``; old artifacts are never overwritten in place.
PRICE_MODEL_VERSION_V1: str = "v1"
#: v2 boosted-tree version (Spec 14).
PRICE_MODEL_VERSION_V2: str = "v2"

#: Output directory for ``price_model_*.pkl`` and ``metrics_*.json``.
#: Override at test time via the ``HOUSINGIQ_ARTIFACT_DIR`` env var.
ARTIFACT_DIR: Path = Path(os.environ.get("HOUSINGIQ_ARTIFACT_DIR", "models"))

#: CSV registry path (Backend Schema §U-SCHEMA-13). Independent of
#: ``ARTIFACT_DIR`` so the file lives under the processed-data tree
#: where Step 07/12 also write.
REGISTRY_CSV_PATH: Path = Path(
    os.environ.get(
        "HOUSINGIQ_REGISTRY_CSV_PATH", "data/model_registry.csv"
    )
)

#: Column order for the model registry CSV — exact match against
#: Backend Schema §U-SCHEMA-13.
MODEL_REGISTRY_FIELDS: tuple[str, ...] = (
    "model_name",
    "version",
    "training_dataset_version",
    "git_commit",
    "training_date",
    "rmse",
    "mae",
    "r2",
    "hyperparameters",
    "feature_hash",
)


def save_price_model(
    pipeline,
    transact_type: str,
    version: str = PRICE_MODEL_VERSION_V1,
    artifact_dir: Path | str | None = None,
) -> Path:
    """Joblib-dump ``pipeline`` to ``price_model_{transact}_v{n}.pkl``.

    Filename rules per Rules §2.5: versioned, never overwritten in
    place. The preprocessor bundled inside ``pipeline`` is the fitted
    Step 12 preprocessor — it is NOT refit here (Rules §2.4).
    """
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"price_model_{transact_type.lower()}_{version}.pkl"
    joblib.dump(pipeline, path)
    logger.info("Wrote %s", path)
    return path


def save_metrics(
    payload: dict,
    version: str = PRICE_MODEL_VERSION_V1,
    artifact_dir: Path | str | None = None,
) -> Path:
    """JSON-dump ``payload`` to ``metrics_v{n}.json`` with deterministic key order.

    Uses ``default=str`` to handle ``numpy`` scalars + ``datetime``
    without forcing the caller to coerce. ``sort_keys=True`` keeps the
    output diff-friendly across runs.
    """
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{version}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)
    logger.info("Wrote %s", path)
    return path


def append_model_registry(
    row: dict,
    csv_path: Path | str | None = None,
) -> bool:
    """Append one row to the model registry CSV.

    Returns ``True`` if the row was appended, ``False`` if the
    ``(model_name, version, git_commit)`` triple was already present.
    Header is written on first call (file doesn't exist).
    """
    csv_path = Path(csv_path) if csv_path else REGISTRY_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()

    sig = (
        str(row.get("model_name", "")),
        str(row.get("version", "")),
        str(row.get("git_commit", "")),
    )

    existing: set[tuple[str, str, str]] = set()
    if not is_new:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add(
                    (r["model_name"], r["version"], r["git_commit"])
                )

    if sig in existing:
        logger.info(
            "Registry row already present (%s) — skipping append.", sig
        )
        return False

    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODEL_REGISTRY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MODEL_REGISTRY_FIELDS})
    logger.info("Appended registry row %s -> %s", sig, csv_path)
    return True


def load_metrics(
    version: str = PRICE_MODEL_VERSION_V1,
    artifact_dir: Path | str | None = None,
) -> dict:
    """Read a previously-written ``metrics_{version}.json`` back as a dict.

    Used by ``vs_v1_metrics`` (Spec 14) to compare the v2 winner against
    the v1 baseline numbers that were written by the Step 13 script.
    Returns an empty dict when the file is missing — the caller decides
    whether to surface this as a WARNING (the v1 metrics may legitimately
    not exist on a fresh checkout).
    """
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    path = out_dir / f"metrics_{version}.json"
    if not path.exists():
        logger.warning(
            "load_metrics: no metrics file at %s — returning {{}}.", path
        )
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "ARTIFACT_DIR",
    "MODEL_REGISTRY_FIELDS",
    "PRICE_MODEL_VERSION_V1",
    "PRICE_MODEL_VERSION_V2",
    "REGISTRY_CSV_PATH",
    "append_model_registry",
    "load_metrics",
    "save_metrics",
    "save_price_model",
]
