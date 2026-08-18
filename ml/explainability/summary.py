"""Global SHAP summary over a held-out sample of test rows.

Used by the build CLI to emit the "SHAP Explainability Artifact v{n}"
section appended to ``data/processed/feature_selection_report.md``.

Ponytail: one ``rng.choice`` + one ``shap_values`` call + one
``abs().mean()`` comprehension. No matplotlib (the spec's "summary"
is a markdown table — the existing ``generate-shap-report`` command
already covers the PNG path).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import shap

logger = logging.getLogger(__name__)


GLOBAL_N_SAMPLES = 200  # ponytail: pinned default; matches the spec's body
TOP_K_SUMMARY = 10      # markdown table row count


def global_summary(
    explainer: shap.TreeExplainer,
    X_background: np.ndarray,
    feature_names: list[str],
    n_samples: int = GLOBAL_N_SAMPLES,
    random_state: int = 42,
) -> dict[str, float]:
    """Return ``mean |SHAP value|`` per feature over a random sample.

    Sampling is deterministic via ``np.random.default_rng(random_state)``
    (Rules §5.4). When ``len(X_background) < n_samples`` we take all
    rows without replacement; never raises on small samples (pinned
    by ``test_global_summary_uses_pinned_n_samples``).
    """
    if X_background is None or len(X_background) == 0:
        logger.warning("global_summary called with empty background; returning {}")
        return {name: 0.0 for name in feature_names}

    rng = np.random.default_rng(random_state)
    k = min(n_samples, len(X_background))
    idx = rng.choice(len(X_background), size=k, replace=False)
    X_sample = X_background[idx]

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.asarray(shap_values, dtype=float)

    return {name: float(np.abs(arr[:, i]).mean()) for i, name in enumerate(feature_names)}


def write_summary_section(
    summary: dict[str, float],
    version: str,
    out_path: Path | str,
    top_k: int = TOP_K_SUMMARY,
    transact_type: str = "",
) -> None:
    """Append a markdown section to ``feature_selection_report.md``.

    Section header is pinned by
    ``test_write_summary_section_includes_section_header``. File is
    opened in append mode (Rules §2.5 + spec's "append-only" rule);
    prior content is never overwritten.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(summary.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    header = f"## SHAP Explainability Artifact {version}"
    if transact_type:
        header += f" — {transact_type}"

    lines = [
        "",
        header,
        "",
        f"Top-{top_k} features by `mean |SHAP value|` over {GLOBAL_N_SAMPLES} "
        f"randomly-sampled test rows (`random_state=42`).",
        "",
        "| Rank | Feature | Human-readable Label | mean &#124;SHAP&#124; |",
        "|------|---------|----------------------|---------------------|",
    ]
    for rank, (name, value) in enumerate(ranked, start=1):
        # Lazy label resolve — module-level import to avoid a cycle
        # (labels is imported by contributions, contributions imports labels).
        from ml.explainability.labels import _STATIC_LABEL_MAP
        label = _STATIC_LABEL_MAP.get(name, name)
        lines.append(f"| {rank} | `{name}` | {label} | {value:.6f} |")
    lines.append("")

    with out_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("appended SHAP summary section -> %s", out_path)


__all__ = ["GLOBAL_N_SAMPLES", "TOP_K_SUMMARY", "global_summary", "write_summary_section"]


# ponytail: the summary table is markdown, not a PNG plot. The
# generate-shap-report command already covers the visual path; this
# is the programmatic counterpart the build CLI needs.
