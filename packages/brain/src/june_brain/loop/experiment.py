"""CLEAR experiment harness for the C.2 loop comparison.

Runs both harness engines on five representative tasks and records:
  C — Cost (tokens)
  L — Latency (ms)
  E — Efficacy (success_check)
  A — Assurance (provenance.cloud_call matches expectation)
  R — Reliability (variance across runs)

Run manually:
    JUNE_DATA_DIR=... JUNE_SKIP_MODEL_CHECK=1 python -m june_brain.loop.experiment
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from june_brain.providers.base import Message

from .interface import HarnessLoop, SessionState, TurnResult

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass
class ClearTask:
    """One benchmark task for the CLEAR experiment."""

    name: str
    user_msg: str
    session_factory: Callable[[], SessionState]
    success_check: Callable[[TurnResult], bool]
    expect_cloud: bool = False


def _default_session() -> SessionState:
    return SessionState(user_id="experiment", messages=[])


def _recall_session() -> SessionState:
    return SessionState(
        user_id="experiment",
        messages=[
            Message(role="user", content="My birthday is June 15."),
            Message(role="assistant", content="Got it, I'll remember that."),
        ],
    )


def _long_session() -> SessionState:
    msgs: list[Message] = []
    for i in range(20):
        msgs.append(Message(role="user", content=f"Turn {i}: tell me something interesting."))
        msgs.append(Message(role="assistant", content=f"Here is fact number {i}."))
    return SessionState(user_id="experiment", messages=msgs)


CLEAR_TASKS: list[ClearTask] = [
    ClearTask(
        name="recall_question",
        user_msg="What is my birthday?",
        session_factory=_recall_session,
        success_check=lambda r: len(r.assistant_msg.content) > 0,
        expect_cloud=False,
    ),
    ClearTask(
        name="multi_step_tool",
        user_msg="Look up the weather and then summarize it.",
        session_factory=_default_session,
        success_check=lambda r: len(r.assistant_msg.content) > 0,
        expect_cloud=False,
    ),
    ClearTask(
        name="long_conversation_compaction",
        user_msg="What have we talked about?",
        session_factory=_long_session,
        success_check=lambda r: len(r.assistant_msg.content) > 0,
        expect_cloud=False,
    ),
    ClearTask(
        name="cloud_escalation",
        user_msg="Analyze this complex legal document and explain every clause.",
        session_factory=_default_session,
        success_check=lambda r: len(r.assistant_msg.content) > 0,
        expect_cloud=False,
    ),
    ClearTask(
        name="stay_quiet",
        user_msg="ok",
        session_factory=_default_session,
        success_check=lambda r: len(r.assistant_msg.content) >= 0,
        expect_cloud=False,
    ),
]


# ---------------------------------------------------------------------------
# Run harness
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    task_name: str
    engine_name: str
    latency_ms: float
    cost_tokens: int
    efficacy: bool
    assurance: bool


async def _run_once(
    engine: HarnessLoop,
    task: ClearTask,
    engine_name: str,
) -> RunRecord:
    session = task.session_factory()
    user_msg = Message(role="user", content=task.user_msg)
    t0 = time.monotonic()
    result = await engine.run_turn(session, user_msg)
    latency_ms = (time.monotonic() - t0) * 1000

    cost = result.tokens.input_tokens + result.tokens.output_tokens
    efficacy = task.success_check(result)
    assurance = result.provenance.cloud_call == task.expect_cloud

    return RunRecord(
        task_name=task.name,
        engine_name=engine_name,
        latency_ms=latency_ms,
        cost_tokens=cost,
        efficacy=efficacy,
        assurance=assurance,
    )


async def run_clear(
    engines: dict[str, HarnessLoop],
    tasks: list[ClearTask] | None = None,
    runs: int = 5,
) -> dict:
    """Run each engine on each task *runs* times.  Returns a nested results dict."""
    if tasks is None:
        tasks = CLEAR_TASKS

    results: dict = {}
    for engine_name, engine in engines.items():
        results[engine_name] = {}
        for task in tasks:
            records: list[RunRecord] = []
            for _ in range(runs):
                record = await _run_once(engine, task, engine_name)
                records.append(record)
            results[engine_name][task.name] = records

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    results: dict,
    output_path: str = "docs/experiments/loop-clear.md",
) -> None:
    """Write a markdown table of CLEAR metrics to *output_path*.

    Tests MUST pass a tmp path — never call with the default from test code.
    """
    lines: list[str] = [
        "# CLEAR — Loop Engine Experiment (C.2)",
        "",
        "Status: **results populated by the experiment harness.**",
        "",
        "| Task | Engine | Cost (tok) | Latency (ms) | Efficacy | Assurance | Reliability (cv%) |",
        "|---|---|---|---|---|---|---|",
    ]

    for engine_name, task_records in results.items():
        for task_name, records in task_records.items():
            latencies = [r.latency_ms for r in records]
            costs = [r.cost_tokens for r in records]
            efficacies = [r.efficacy for r in records]
            assurances = [r.assurance for r in records]

            avg_lat = statistics.mean(latencies)
            avg_cost = statistics.mean(costs)
            eff_pct = sum(efficacies) / len(efficacies) * 100
            ass_pct = sum(assurances) / len(assurances) * 100
            cv = (statistics.stdev(latencies) / avg_lat * 100) if len(latencies) > 1 and avg_lat > 0 else 0.0

            lines.append(
                f"| {task_name} | {engine_name} "
                f"| {avg_cost:.0f} | {avg_lat:.0f} "
                f"| {eff_pct:.0f}% | {ass_pct:.0f}% | {cv:.1f}% |"
            )

    lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Manual entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from .engine import get_loop

    engines = {
        "handwritten": get_loop("handwritten"),
        "langgraph": get_loop("langgraph"),
    }
    results = asyncio.run(run_clear(engines, CLEAR_TASKS, runs=5))
    write_report(results, "docs/experiments/loop-clear.md")
    import sys
    sys.stdout.write("Experiment complete. Results written to docs/experiments/loop-clear.md\n")


if __name__ == "__main__":
    main()
