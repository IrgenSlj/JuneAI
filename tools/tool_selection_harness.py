#!/usr/bin/env python
"""Tool-selection accuracy harness — D.5d.

Runs the corpus in ``june_brain.experiments.tool_selection`` against the live
local model through the real loop, and reports how often June reaches for the
right tool, the wrong one, or one it should not have reached for at all.

Writes go to a throwaway data dir (``JUNE_DATA_DIR``), so `remember` and
`forget` really execute — measuring the path the user gets — without touching
the real store. Set ``--data-dir`` to keep it.

Usage:
    packages/brain/.venv/bin/python tools/tool_selection_harness.py
    packages/brain/.venv/bin/python tools/tool_selection_harness.py --repeat 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile


def _seed_memories(user_id: str) -> None:
    """Store the memories the `forget` cases refer to.

    Same fixture error the promises had, found the same way: without these the
    corpus asks June to forget things that were never stored, so a model that
    answers "I have nothing about your old address" is behaving correctly and is
    scored as a miss. That measures the fixture, not the model.
    """
    from june_brain.memory import MemoryManager

    manager = MemoryManager(user_id)
    for text in (
        "The user's old address is 14 Oak Street.",
        "The user works at Acme as a systems engineer.",
        "The user's coffee order is a flat white.",
        "The user's ex-partner is called Sam.",
        "The user's sister is called Mira.",
    ):
        manager.write({"kind": "fact", "fields": {"text": text}}, source="tool:remember")


def _seed_promises(user_id: str) -> None:
    from june_brain.tasks.store import TasksStore

    store = TasksStore(user_id=user_id)
    for goal in (
        "Renew the passport",
        "Book the flight to Lisbon",
        "File the tax return",
        "Find a dentist and book a check-up",
    ):
        store.create(goal=goal)


async def _run(args: argparse.Namespace) -> int:
    from june_brain.experiments.tool_selection import (
        CORPUS,
        SelectionReport,
        SelectionResult,
        render_report,
    )
    from june_brain.loop.engine import get_loop
    from june_brain.loop.interface import SessionState
    from june_brain.providers.base import Message

    # Seed the state the corpus refers to. Without it the
    # store is empty, and a model that answers "the passport renewal is done"
    # by calling `list_promises` first is not wrong — it cannot name a promise
    # id it has never seen. Benchmarking that as a selection failure would
    # measure the fixture, not the model.
    _seed_promises("tool-selection-bench")
    _seed_memories("tool-selection-bench")

    loop = get_loop()
    report = SelectionReport()

    for _ in range(args.repeat):
        for case in CORPUS:
            session = SessionState(user_id="tool-selection-bench", messages=[])
            try:
                result = await loop.run_turn(
                    session, Message(role="user", content=case.utterance)
                )
                calls = list(result.tool_calls)
            except Exception as exc:  # noqa: BLE001 - a failed turn is a data point
                print(f"  ! turn failed for {case.utterance!r}: {exc}")
                calls = []
            called = calls[0].name if calls else None
            report.add(
                SelectionResult(
                    case=case,
                    called=called,
                    raw_called=called,
                    all_called=tuple(c.name for c in calls),
                )
            )
            mark = "ok " if report.results[-1].correct else "MISS"
            print(f"  {mark} {case.expected or '-none-':<16} <- {case.utterance[:52]}")

    print()
    print(render_report(report))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "summary": report.summary(),
                    "confusions": report.confusions(),
                    # Every call of every turn, so a later reader can tell a
                    # two-step answer from a stall without re-running an hour
                    # of local inference.
                    "cases": [
                        {
                            "utterance": r.case.utterance,
                            "expected": r.case.expected,
                            "called": r.called,
                            "all_called": list(r.all_called),
                            "correct": r.correct,
                            "reached": r.reached,
                        }
                        for r in report.results
                    ],
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="passes over the corpus")
    parser.add_argument("--data-dir", default="", help="persist writes here instead of a temp dir")
    parser.add_argument("--json", default="", help="also write results to this path")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["JUNE_DATA_DIR"] = args.data_dir
        return asyncio.run(_run(args))
    with tempfile.TemporaryDirectory(prefix="june-tool-selection-") as tmp:
        os.environ["JUNE_DATA_DIR"] = tmp
        return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
