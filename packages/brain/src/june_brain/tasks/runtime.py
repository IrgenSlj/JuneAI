"""Minimal task runtime — runs a task's goal through the hand-written loop and
records the trace as TaskStep entries.

This is the *vessel-filling* version of the runtime. Rather than reinvent a
planner-executor loop, it streams the goal through ``june_brain.loop`` (the one
engine, ADR 0018) as if it were a chat turn. Every tool call the loop makes
becomes one TaskStep; the streamed assistant text becomes a final step labeled
"Final response".

A richer multi-step planner with intermediate user approvals comes later. The
contract that matters now: the user clicks Start, June actually does
something, and every step is visible in /tasks with its tool, args, and
result.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from ..providers.base import Message
from .models import Task, TaskStatus, TaskStep, TaskStepStatus
from .store import TasksStore

logger = logging.getLogger(__name__)

_MAX_STEPS = 25  # Safety: a runaway loop should stop here.


@dataclass(frozen=True)
class _TraceOutcome:
    blocked_reason: str | None = None
    next_action: str | None = None
    final_deliverable: str | None = None


class TaskRuntime:
    """Executes one task by streaming its goal through the hand-written loop."""

    def __init__(
        self,
        store: TasksStore,
        *,
        loop_factory: Any | None = None,
    ) -> None:
        self.store = store
        self._loop_factory = loop_factory

    async def execute(self, task_id: str) -> Task:
        """Run the task to completion. Raises if the task is missing."""
        task = self.store.get(task_id)
        if task is None:
            raise KeyError(f"task '{task_id}' not found")

        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
        }:
            return task

        self.store.set_status(task_id, TaskStatus.RUNNING)
        try:
            events = await self._stream_events(task)
        except Exception as exc:  # noqa: BLE001
            # Honour a cancel even when the loop raised — a cancelled task that
            # errored after the cancel is still cancelled, not failed.
            if self._was_cancelled(task_id):
                logger.info("task %s ended in cancelled state after stream error", task_id)
                refreshed = self.store.get(task_id)
                assert refreshed is not None
                return refreshed
            self.store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
            logger.exception("task %s failed during loop stream", task_id)
            raise

        try:
            outcome = self._record_trace(task_id, events)
            # Cooperative cancel: if the user PATCHed status=cancelled while
            # the loop was running, preserve that state instead of forcing
            # the row to COMPLETED. The trace we just recorded still lands so
            # the user can see how far the task got before being stopped.
            if self._was_cancelled(task_id):
                logger.info("task %s cancelled mid-run; preserving cancelled state", task_id)
            elif outcome.blocked_reason:
                self.store.set_blocked(
                    task_id,
                    reason=outcome.blocked_reason,
                    next_action=(
                        outcome.next_action
                        or "Change the privacy dial, then retry this promise."
                    ),
                    final_deliverable=outcome.final_deliverable,
                )
            else:
                self.store.set_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    final_deliverable=outcome.final_deliverable,
                )
        except Exception as exc:  # noqa: BLE001
            self.store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
            logger.exception("task %s failed during trace recording", task_id)
            raise

        refreshed = self.store.get(task_id)
        assert refreshed is not None
        return refreshed

    def _was_cancelled(self, task_id: str) -> bool:
        """Re-read the task to see whether the user cancelled it during the run."""
        current = self.store.get(task_id)
        return current is not None and current.status == TaskStatus.CANCELLED

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_loop(self) -> Any:
        if self._loop_factory is not None:
            return self._loop_factory()
        from ..loop.engine import get_loop

        return get_loop()

    async def _stream_events(self, task: Task) -> list[Any]:
        """Drive the loop and collect every StreamEvent it emits."""
        from ..loop.interface import SessionState

        loop = self._resolve_loop()
        session = SessionState(
            user_id=task.user_id,
            messages=[],
            skill="default",
            approved_tools=set(task.approved_tools),
        )
        user_msg = Message(role="user", content=task.goal)
        events: list[Any] = []
        async for ev in loop.stream_turn(session, user_msg):
            events.append(ev)
        return events

    def _record_trace(self, task_id: str, events: list[Any]) -> _TraceOutcome:
        """Translate StreamEvents into TaskStep entries.

        The loop emits a ``tool_call`` event when it requests a tool and a
        ``tool_result`` event when that tool returns; they arrive in dispatch
        order, so we pair them FIFO. ``token`` events accumulate into the final
        assistant response.
        """
        steps_added = 0
        blocked_reason: str | None = None
        next_action: str | None = None
        blocked_step_added = False
        pending: list[dict[str, Any]] = []
        final_parts: list[str] = []

        for ev in events:
            etype = getattr(ev, "type", "")
            if etype == "tool_call":
                pending.append(
                    {
                        "name": str(getattr(ev, "tool_name", "") or "unknown"),
                        "args": dict(getattr(ev, "tool_args", None) or {}),
                    }
                )
            elif etype == "tool_result":
                if steps_added >= _MAX_STEPS:
                    continue
                call = pending.pop(0) if pending else {
                    "name": str(getattr(ev, "tool_name", "") or "tool"),
                    "args": None,
                }
                self.store.append_step(
                    task_id,
                    TaskStep(
                        description=f"Called {call['name']}",
                        tool_name=call["name"],
                        tool_args=call["args"],
                        tool_result=getattr(ev, "tool_result", "") or "",
                        status=TaskStepStatus.COMPLETED,
                        finished_at=self._now(),
                    ),
                )
                steps_added += 1
            elif etype == "tool_blocked":
                tool_name = str(getattr(ev, "tool_name", "") or "tool")
                detail = str(getattr(ev, "detail", "") or "").strip()
                # Two block flavours share this event: the guard's approval gate
                # (a consequential action June won't take without an explicit OK)
                # and the Local-only egress block (a networked tool the privacy
                # dial forbids). They need different next actions — approve the
                # one action vs. change the dial — so the promise must not
                # conflate them.
                if getattr(ev, "needs_approval", False):
                    reason_tail = f" ({detail})" if detail else ""
                    blocked_reason = (
                        f"{tool_name} needs your approval before it can run{reason_tail}."
                    )
                    next_action = "Approve this action, then retry the promise."
                    waiting_desc = f"Waiting for your approval: {tool_name}"
                else:
                    blocked_reason = f"{tool_name} is blocked by local-only mode."
                    next_action = (
                        "Switch to Private-by-default if this networked tool is acceptable, "
                        "then retry the promise."
                    )
                    waiting_desc = (
                        f"Waiting for user approval: {tool_name} is blocked by local-only mode"
                    )
                if steps_added >= _MAX_STEPS and blocked_step_added:
                    continue
                self.store.append_step(
                    task_id,
                    TaskStep(
                        description=waiting_desc,
                        tool_name=tool_name,
                        tool_args=dict(getattr(ev, "tool_args", None) or {}),
                        tool_result=detail or None,
                        status=TaskStepStatus.PENDING,
                    ),
                )
                steps_added += 1
                blocked_step_added = True
            elif etype == "token":
                final_parts.append(getattr(ev, "content", "") or "")

        # Drain any tool_calls that were requested without a matching result —
        # they were proposed but never executed; record as skipped steps so the
        # user can see what was attempted.
        for call in pending:
            if steps_added >= _MAX_STEPS:
                break
            self.store.append_step(
                task_id,
                TaskStep(
                    description=f"Proposed {call['name']} (no result)",
                    tool_name=call["name"],
                    tool_args=call["args"],
                    status=TaskStepStatus.SKIPPED,
                    finished_at=self._now(),
                ),
            )
            steps_added += 1

        final_text = "".join(final_parts).strip()
        if final_text:
            self.store.append_step(
                task_id,
                TaskStep(
                    description="Final response",
                    tool_result=final_text,
                    status=TaskStepStatus.COMPLETED,
                    finished_at=self._now(),
                ),
            )

        return _TraceOutcome(
            blocked_reason=blocked_reason,
            next_action=next_action,
            final_deliverable=final_text or None,
        )

    @staticmethod
    def _now() -> str:
        from datetime import datetime

        return datetime.now(UTC).isoformat()


def execute_task_in_background(store: TasksStore, task_id: str) -> None:
    """Synchronous wrapper for triggering a runtime from a FastAPI BackgroundTask.

    BackgroundTasks runs in a worker thread, so a fresh event loop is fine here.
    Errors are logged but never raised — the task's status already captures the
    failure.
    """
    runtime = TaskRuntime(store)
    try:
        asyncio.run(runtime.execute(task_id))
    except Exception:  # noqa: BLE001
        logger.exception("background task execution failed for %s", task_id)
