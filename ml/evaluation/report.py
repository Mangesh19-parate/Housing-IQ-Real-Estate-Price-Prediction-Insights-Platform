"""Report writer for the evaluation gate (Spec 15).

Two functions:
    - :func:`write_evaluation_report` — JSON-dump the
      :class:`EvaluationResult` to a versioned filename. Re-run with
      the same ``evaluated_at`` writes the same content; re-run with
      a different timestamp writes a timestamp-suffixed sibling
      (Rules §2.5: never overwrite in place).
    - :func:`append_protocol_section` — append a "Protocol
      Certification" section to
      ``data/processed/feature_selection_report.md``. Append-only —
      never overwrites prior content.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _result_to_payload(result) -> dict:
    """Convert an :class:`EvaluationResult` to a JSON-safe dict."""
    return {
        "version": result.version,
        "transact_type": result.transact_type,
        "protocol_version": result.protocol_version,
        "dataset_version": result.dataset_version,
        "git_commit": result.git_commit,
        "split_sizes": dict(result.split_sizes),
        "metrics": _stringify_floats(result.metrics),
        "per_city_test": _stringify_floats(result.per_city_test),
        "within_tol_15_pct": result.within_tol_15_pct,
        "latency_p95_ms": result.latency_p95_ms,
        "thresholds_passed": dict(result.thresholds_passed),
        "overall_passed": result.overall_passed,
        "evaluated_at": result.evaluated_at,
        "evaluator_version": result.evaluator_version,
    }


def _stringify_floats(obj):
    """Recursively round floats to 6 decimal places for diff-friendly JSON."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _stringify_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_floats(v) for v in obj]
    return obj


def write_evaluation_report(result, out_dir: Path | str) -> Path:
    """Write ``evaluation_report_{version}_{transact_type}.json``.

    Filename rules per Rules §2.5: versioned, never overwritten in
    place. If a file with the same ``evaluated_at`` already exists
    with the same payload, the same file is rewritten idempotently.
    If ``evaluated_at`` differs from an existing file at the canonical
    name, the new file is suffixed with ``_rerun_<safe_timestamp>``
    so the original is preserved.

    Returns the written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / (
        f"evaluation_report_{result.version}_{result.transact_type}.json"
    )

    payload = _result_to_payload(result)

    # Idempotency check: if a file already exists at ``canonical`` and
    # its ``evaluated_at`` matches the result's, rewrite it in place
    # (same content). Otherwise, suffix the new file with the
    # timestamp so the original is preserved (Rules §2.5).
    if canonical.exists():
        try:
            with open(canonical, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("evaluated_at") == result.evaluated_at:
                with open(canonical, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, sort_keys=True)
                logger.info("Wrote %s (idempotent re-run)", canonical)
                return canonical
        except (OSError, json.JSONDecodeError):
            # Fall through to timestamp-suffix path on any read/parse
            # error — never silently overwrite a corrupt file.
            pass
        safe_ts = result.evaluated_at.replace(":", "-").replace(".", "-")
        path = out_dir / (
            f"evaluation_report_{result.version}_{result.transact_type}"
            f"_rerun_{safe_ts}.json"
        )
    else:
        path = canonical

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("Wrote %s", path)
    return path


def append_protocol_section(result, report_path: Path | str) -> None:
    """Append a "Protocol Certification" section to ``report_path``.

    Opens the file in append mode — prior content (Specs 12/13/14's
    Round 1/2/3 + v2 sections) is preserved verbatim. If the file
    does not exist, it is created (Rules §1.3 — every derived
    artifact states its source; the first section header includes
    this fact).
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    section = _render_section(result)
    with open(report_path, "a", encoding="utf-8") as fh:
        if not report_path.exists() or report_path.stat().st_size == 0:
            fh.write(
                "# Feature Selection Report — Protocol Certification "
                "Appendices\n\n"
            )
            fh.write(
                "This file is appended to by `scripts/evaluate_price_model.py` "
                "after the v1/v2 training scripts land their artifacts.\n\n"
            )
        else:
            fh.write("\n")
        fh.write(section)
    logger.info("Appended protocol section to %s", report_path)


def _render_section(result) -> str:
    """Render the markdown section body for ``result``."""
    tag = "PASS" if result.overall_passed else "FAIL"
    test = result.metrics.get("test", {}) if isinstance(result.metrics, dict) else {}
    city_lines: list[str] = []
    for city, m in sorted(result.per_city_test.items()):
        r2 = m.get("r2", 0.0)
        city_lines.append(f"| {city} | {r2:.4f} |")

    thresholds_lines: list[str] = []
    for k, passed in sorted(result.thresholds_passed.items()):
        mark = "✅" if passed else "❌"
        thresholds_lines.append(f"| {k} | {mark} |")

    def _format_latency(latency: float | None) -> str:
        if latency is None:
            return "n/a (offline)"
        return f"{latency:.1f} ms"

    city_table = (
        "\n".join(city_lines)
        if city_lines
        else "| _no per-city rows_ | _n/a_ |"
    )
    thresholds_table = "\n".join(thresholds_lines)

    return (
        f"## Protocol Certification — {result.transact_type}_{result.version} "
        f"({tag})\n\n"
        f"- evaluated_at: `{result.evaluated_at}`\n"
        f"- protocol_version: `{result.protocol_version}`\n"
        f"- evaluator_version: `{result.evaluator_version}`\n"
        f"- dataset_version: `{result.dataset_version}`\n"
        f"- git_commit: `{result.git_commit}`\n"
        f"- split_sizes: {result.split_sizes}\n"
        f"- test R²: {test.get('r2', 0.0):.4f} "
        f"(≥ {0.80} required)\n"
        f"- test MAE: ₹{test.get('mae', 0.0):.0f}\n"
        f"- test RMSE: {test.get('rmse', 0.0):.0f}\n"
        f"- test MAPE: {test.get('mape', 0.0):.2f}%\n"
        f"- within ±15%: {result.within_tol_15_pct:.1%} "
        f"(≥ 70% required)\n"
        f"- latency p95: "
        f"{_format_latency(result.latency_p95_ms)}\n"
        f"\n### Thresholds\n\n"
        f"| Threshold | Passed |\n|---|---| \n"
        f"{thresholds_table}\n"
        f"\n### Per-city test R²\n\n"
        f"| City | R² |\n|---|---|\n"
        f"{city_table}\n"
    )


__all__ = ["write_evaluation_report", "append_protocol_section"]
