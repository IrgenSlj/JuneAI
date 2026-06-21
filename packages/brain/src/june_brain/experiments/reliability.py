"""Reliability suite — run a task N times and report run-to-run variance.

The CLEAR experiment (ADR 0018) found the hand-written loop fast at equal
efficacy but with higher run-to-run variance on some tasks (recall cv 75.6%).
S5's provider-native tool calling is the planned fix; this module is the
instrument that measures whether it worked. ``cv_percent`` is the coefficient
of variation (stddev / mean, as a percentage) — lower is more reliable.

Pure and dependency-free so the math is unit-tested without a live model; the
``tools/reliability_harness.py`` CLI drives real tasks against Ollama.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def cv_percent(values: list[float]) -> float:
    """Coefficient of variation as a percentage. 0.0 for <2 values or mean 0."""
    clean = [float(v) for v in values]
    if len(clean) < 2:
        return 0.0
    mean = statistics.fmean(clean)
    if mean == 0:
        return 0.0
    return (statistics.pstdev(clean) / abs(mean)) * 100.0


@dataclass
class TaskReport:
    """Aggregate of one task run ``runs`` times."""

    label: str
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def add(self, metric: str, value: float) -> None:
        self.metrics.setdefault(metric, []).append(float(value))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for metric, values in self.metrics.items():
            out[metric] = {
                "mean": statistics.fmean(values) if values else 0.0,
                "cv_percent": cv_percent(values),
                "n": float(len(values)),
            }
        return out


def run_task(
    label: str,
    task: Callable[[], dict[str, float]],
    *,
    runs: int = 10,
) -> TaskReport:
    """Run ``task`` ``runs`` times, collecting each call's metric dict.

    ``task`` returns a flat ``{metric: value}`` dict per run (e.g.
    ``{"latency_ms": 1200, "recalled": 1.0}``).
    """
    report = TaskReport(label=label)
    for _ in range(max(1, runs)):
        result = task()
        for metric, value in result.items():
            report.add(metric, value)
    return report


def render_reports(reports: list[TaskReport]) -> str:
    """Render reports as a plain-text table (for the CLI and docs)."""
    lines = [f"{'task':<20}{'metric':<16}{'mean':>12}{'cv%':>10}{'n':>5}"]
    for rep in reports:
        for metric, stats in rep.summary().items():
            lines.append(
                f"{rep.label:<20}{metric:<16}{stats['mean']:>12.2f}"
                f"{stats['cv_percent']:>10.1f}{int(stats['n']):>5}"
            )
    return "\n".join(lines)


def reports_to_dict(reports: list[TaskReport]) -> dict[str, Any]:
    """Machine-readable summary for recording alongside the baseline."""
    return {rep.label: rep.summary() for rep in reports}
