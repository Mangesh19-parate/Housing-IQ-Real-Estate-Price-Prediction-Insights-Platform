"""One-shot backfill for the model_registry SQLite table (Spec 20).

Scans ``models/`` for ``price_model_{sale,rent}_v{n}.pkl`` artifacts and
registers any that aren't already in the registry. Idempotent — safe to
re-run after every training run. No-op on a fresh checkout (prints
"no artifacts to register" and exits 0).

Usage::

    python scripts/seed_registry.py

The newest version per ``model_name`` (highest ``v{n}`` found on disk)
is marked active. The CSV at ``data/model_registry.csv`` is left alone
— Spec 14 still writes it for legacy consumers.

Ponytail notes:
- We don't try to recompute the feature hash from the saved pipeline
  (would require joblib-loading every .pkl just to read its column
  list). Instead, we hash an empty feature list and record that —
  this means a seeded row carries ``feature_hash=""`` instead of the
  real fingerprint. Trade-off: avoids loading multi-MB pickles just
  to backfill. If a real feature_hash matters for seeded rows, the
  next persistence refactor should record it at save time, not here.
- Re-registration is a no-op per ``register_model()``'s contract, so
  re-running with a real ``metrics_v2.json`` already on disk just
  re-reads the metrics.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable when invoked as ``python scripts/seed_registry.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.database.db import init_db  # noqa: E402
from ml.registry import artifact_path, metrics_path  # noqa: E402
from ml.training import list_models, register_model, set_active  # noqa: E402

#: Filename shape we know how to seed. Mirrors ``ml/training/persistence.py:78``.
_KNOWN_MODEL_NAMES: tuple[str, ...] = (
    "price_model_sale",
    "price_model_rent",
)
_VERSION_RE = re.compile(r"^price_model_(?P<transact>sale|rent)_v(?P<n>\d+)\.pkl$")


def _git_commit() -> str:
    """Return the current HEAD SHA, or ``"unknown"`` if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _latest_version(models_dir: Path, model_name: str) -> str | None:
    """Return ``"v{max_n}"`` for the given ``model_name``, or ``None``."""
    pattern = re.compile(rf"^{re.escape(model_name)}_v(?P<n>\d+)\.pkl$")
    max_n = 0
    found = False
    if not models_dir.exists():
        return None
    for entry in models_dir.iterdir():
        m = pattern.match(entry.name)
        if not m:
            continue
        found = True
        max_n = max(max_n, int(m.group("n")))
    return f"v{max_n}" if found else None


def _seed_one(
    *,
    model_name: str,
    version: str,
    models_dir: Path,
    git_sha: str,
) -> bool:
    """Register one ``(model_name, version)`` if its .pkl exists. Returns registered?"""
    pkl = artifact_path(model_name, version, artifact_dir=models_dir)
    if not pkl.exists():
        return False

    metrics_file = metrics_path(version, artifact_dir=models_dir)
    metrics: dict = {}
    if metrics_file.exists():
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            transact = model_name.rsplit("_", 1)[-1]
            transact_block = payload.get(transact, {}) or {}
            chosen = transact_block.get("chosen_metrics", {}) or {}
            test_block = chosen.get("test", {}) or {}
            metrics = {
                "rmse": test_block.get("rmse"),
                "mae": test_block.get("mae"),
                "r2": test_block.get("r2"),
            }
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  warn: could not read {metrics_file.name}: {exc}", file=sys.stderr)
            metrics = {}

    register_model(
        model_name=model_name,
        version=version,
        training_dataset_version="clean_listings.parquet",
        git_commit=git_sha,
        training_date=datetime.now(timezone.utc),
        artifact_path=str(pkl),
        hyperparameters={"source": "seed_registry"},
        feature_hash="",
        metrics=metrics,
    )
    return True


def _print_summary() -> None:
    rows = list_models()
    if not rows:
        print("(registry is empty)")
        return
    print(f"{'model_name':<22} {'version':<6} {'active':<6} {'training_date':<32} artifact_path")
    print("-" * 110)
    for row in rows:
        active = "yes" if row.get("is_active") else "no"
        print(
            f"{row.get('model_name', ''):<22} "
            f"{row.get('version', ''):<6} "
            f"{active:<6} "
            f"{str(row.get('training_date', '')):<32} "
            f"{row.get('artifact_path', '')}"
        )


def main() -> int:
    models_dir = Path("models")
    print(f"seed_registry: scanning {models_dir.resolve()}")
    init_db()
    git_sha = _git_commit()

    registered_count = 0
    latest_per_model: dict[str, str] = {}
    for model_name in _KNOWN_MODEL_NAMES:
        latest = _latest_version(models_dir, model_name)
        if latest is None:
            print(f"  {model_name}: no artifacts on disk, skipping")
            continue
        latest_per_model[model_name] = latest
        # Register every version we find on disk so the registry reflects history.
        pattern = re.compile(rf"^{re.escape(model_name)}_v(?P<n>\d+)\.pkl$")
        if models_dir.exists():
            for entry in sorted(models_dir.iterdir()):
                m = pattern.match(entry.name)
                if not m:
                    continue
                version = f"v{m.group('n')}"
                if _seed_one(
                    model_name=model_name,
                    version=version,
                    models_dir=models_dir,
                    git_sha=git_sha,
                ):
                    registered_count += 1

    for model_name, latest in latest_per_model.items():
        try:
            set_active(model_name, latest)
            print(f"  activated {model_name} -> {latest}")
        except AssertionError as exc:
            print(f"  warn: could not activate {model_name}@{latest}: {exc}", file=sys.stderr)

    print()
    if registered_count == 0:
        print("no artifacts to register (fresh checkout)")
    else:
        print(f"registered {registered_count} artifact(s)")
    print()
    print("Registry summary:")
    _print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
