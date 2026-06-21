#!/usr/bin/env python
"""Reliability harness — measure run-to-run variance of the hand-written loop.

Runs representative tasks (recall, multi-step tool use, compaction) N times each
against the live local model and reports the coefficient of variation (cv%) per
metric. The S5 target is recall cv < 25% (baseline 75.6%, ADR 0018/0020).

Requires a running Ollama with the chat model pulled. Record the printed numbers
in docs/experiments/baseline-2026-06.md next to the baseline.

Usage:
    packages/brain/.venv/bin/python tools/reliability_harness.py [--runs 10]
"""

from __future__ import annotations

import argparse
import asyncio
import time


async def _run() -> int:
    from june_brain.experiments.reliability import TaskReport, render_reports
    from june_brain.experiments.reliability import run_task as _run_task  # noqa: F401
    from june_brain.loop.engine import get_loop
    from june_brain.loop.interface import SessionState
    from june_brain.providers.base import Message

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    loop = get_loop()

    async def _one_turn(text: str) -> dict[str, float]:
        session = SessionState(user_id="reliability-bench", messages=[])
        t0 = time.monotonic()
        result = await loop.run_turn(session, Message(role="user", content=text))
        latency = (time.monotonic() - t0) * 1000.0
        return {
            "latency_ms": latency,
            "tool_calls": float(len(result.tool_calls)),
            "memories_recalled": float(result.provenance.memories_recalled),
            "output_tokens": float(result.tokens.output_tokens),
        }

    tasks = {
        "recall": "What did I tell you about my goals last week?",
        "multi_step": "Search the web for today's date, then summarize it.",
        "compaction": "Tell me a long story about a city, in several paragraphs.",
    }

    reports: list[TaskReport] = []
    for label, prompt in tasks.items():
        report = TaskReport(label=label)
        for _ in range(max(1, args.runs)):
            metrics = await _one_turn(prompt)
            for metric, value in metrics.items():
                report.add(metric, value)
        reports.append(report)

    print(render_reports(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
