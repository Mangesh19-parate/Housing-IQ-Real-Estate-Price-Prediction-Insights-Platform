"""Artifact-path + version helpers for the model registry (Spec 20).

Pure-path module — no DB I/O. The path conventions mirror
``ml/training/persistence.py:78,97`` so a ``.pkl`` written by
``save_price_model`` resolves through ``artifact_path(model_name, version)``
to the exact same ``Path`` the registry row records.

Ponytail notes:
- ``next_version`` scans the ``models/`` directory only (the on-disk
  truth). It does NOT cross-check the registry — the registry indexes
  what's been registered, the directory indexes what exists. Add a
  registry-scan variant when the two views actually drift.
- ``_MODEL_FILE_RE`` pins the ``{name}_v{n}.pkl`` filename shape so
  accidental files (``metrics_v2.json``, ``feature_pipeline_v1.pkl``)
  don't get parsed as version markers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Env var matches ``ml/training/persistence.py:ARTIFACT_DIR``.
_ARTIFACT_DIR_ENV: str = "HOUSINGIQ_ARTIFACT_DIR"
_DEFAULT_ARTIFACT_DIR: str = "models"

#: Captures ``{name}_v{n}.pkl``. ``name`` excludes underscores so a
#: suffix like ``price_model_sale_v2`` parses as ``name=price_model_sale``
#: and ``v=v2`` — the train script's actual filenames.
_MODEL_FILE_RE = re.compile(r"^(?P<name>.+)_v(?P<n>\d+)\.pkl$")


def _artifact_dir(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override)
    raw = os.environ.get(_ARTIFACT_DIR_ENV, _DEFAULT_ARTIFACT_DIR)
    return Path(raw)


def artifact_path(
    model_name: str,
    version: str,
    artifact_dir: Path | str | None = None,
) -> Path:
    """Return ``{artifact_dir}/{model_name}_{version}.pkl``.

    Mirrors ``save_price_model`` at ``ml/training/persistence.py:78``.
    """
    return _artifact_dir(artifact_dir) / f"{model_name}_{version}.pkl"


def metrics_path(
    version: str,
    artifact_dir: Path | str | None = None,
) -> Path:
    """Return ``{artifact_dir}/metrics_{version}.json``.

    Metrics files are per-version, not per-model — matches
    ``save_metrics`` at ``ml/training/persistence.py:97``.
    """
    return _artifact_dir(artifact_dir) / f"metrics_{version}.json"


def next_version(
    model_name: str,
    artifact_dir: Path | str | None = None,
) -> str:
    """Return ``"v{max+1}"`` (or ``"v1"`` when no ``{name}_v*.pkl`` exists).

    Parses trailing ``_v{n}.pkl`` filenames via ``_MODEL_FILE_RE`` so a
    stray ``metrics_v2.json`` is ignored. On ``FileNotFoundError`` (the
    default ``models/`` directory doesn't exist on a fresh checkout),
    returns ``"v1"`` — the first version is always the safe default.
    """
    d = _artifact_dir(artifact_dir)
    try:
        max_n = 0
        for entry in d.iterdir():
            m = _MODEL_FILE_RE.match(entry.name)
            if not m or m.group("name") != model_name:
                continue
            max_n = max(max_n, int(m.group("n")))
    except FileNotFoundError:
        return "v1"
    return f"v{max_n + 1}" if max_n else "v1"


__all__ = ["artifact_path", "metrics_path", "next_version"]
