"""Minimal task runtime — runs a task's goal through the existing chat agent
and records the trace as TaskStep entries.

This is the *vessel-filling* version of the runtime. Rather than reinvent a
planner-executor loop, it asks the LangGraph agent in ``june_brain.graph`` to
handle the goal as if it were a chat message. Every tool call the agent makes
becomes one TaskStep; the final assistant message becomes a final step labeled
"response". The runtime owns provenance — each step records which model
handled the chat-side call when known.

A richer multi-step planner with intermediate user approvals comes later. The
contract that matters now: the user clicks Start, June actually does
something, and every step is visible in /tasks with its tool, args, and
result.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from .models import Task, TaskStatus, TaskStep, TaskStepStatus
from .store import TasksStore

logger = logging.getLogger(__name__)

_MAX_STEPS = 25  # Safety: a runaway agent should stop here.


class TaskRuntime:
    """Executes one task by feeding its goal to the chat agent."""

    def __init__(
        self,
        store: TasksStore,
        *,
        agent_factory: Optional[Any] = None,
        message_factory: Optional[Any] = None,
    ) -> None:
        self.store = store
        self._agent_factory = agent_factory
        self._message_factory = message_factory

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
            agent = self._resolve_agent()
            HumanMessage = self._resolve_message_class()
            input_state = {"messages": [HumanMessage(content=task.goal)]}
            result = await self._invoke(agent, input_state)
        except Exception as exc:  # noqa: BLE001
            # Honour a cancel even when the agent invoke raised — a cancelled
            # task that errored after the cancel is still cancelled, not failed.
            if self._was_cancelled(task_id):
                logger.info("task %s ended in cancelled state after invoke error", task_id)
                refreshed = self.store.get(task_id)
                assert refreshed is not None
                return refreshed
            self.store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
            logger.exception("task %s failed during agent invoke", task_id)
            raise

        try:
            self._record_trace(task_id, result)
            # Cooperative cancel: if the user PATCHed status=cancelled while
            # the agent was running, preserve that state instead of forcing
            # the row to COMPLETED. The trace we just recorded still lands so
            # the user can see how far the task got before being stopped.
            if self._was_cancelled(task_id):
                logger.info("task %s cancelled mid-run; preserving cancelled state", task_id)
            else:
                self.store.set_status(task_id, TaskStatus.COMPLETED)
        except Exception as exc:  # noqa: BLE001
            self.store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
            logger.exception("task %s failed during trace recording", task_id)
            raise

        refreshed = self.store.get(task_id)
        assert refreshed is not None
        return refreshed

    def _was_cancelled(self, task_id: str) -> bool:
        """Re-read the task to see whether the user cancelled it during invoke."""
        current = self.store.get(task_id)
        return current is not None and current.status == TaskStatus.CANCELLED

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_agent(self) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory()
        from .. import graph as brain_graph

        return brain_graph.get_or_create_agent()

    def _resolve_message_class(self) -> Any:
        if self._message_factory is not None:
            return self._message_factory
        from langchain_core.messages import HumanMessage  # type: ignore[import-untyped]

        return HumanMessage

    async def _invoke(self, agent: Any, input_state: dict[str, Any]) -> dict[str, Any]:
        """Call the agent with whichever of ainvoke/invoke it supports."""
        if hasattr(agent, "ainvoke"):
            return await agent.ainvoke(input_state)
        if hasattr(agent, "invoke"):
            return await asyncio.to_thread(agent.invoke, input_state)
        raise TypeError("agent does not expose invoke/ainvoke")

    def _record_trace(self, task_id: str, result: Any) -> None:
        """Walk the resulting message list and record each tool call as a step."""
        messages = self._extract_messages(result)
        steps_added = 0
        pending_calls: dict[str, dict[str, Any]] = {}

        for message in messages:
            tool_calls = self._extract_tool_calls(message)
            if tool_calls:
                for call in tool_calls:
                    call_id = str(call.get("id") or call.get("name") or steps_added)
                    pending_calls[call_id] = {
                        "name": str(call.get("name") or "unknown"),
                        "args": call.get("args") or {},
                    }
                continue

            tool_result = self._extract_tool_result(message)
            if tool_result is not None:
                call_id = str(tool_result.get("tool_call_id") or "")
                pending = pending_calls.pop(call_id, None)
                tool_name = pending["name"] if pending else str(
                    tool_result.get("name") or "tool"
                )
                tool_args = pending["args"] if pending else None
                self.store.append_step(
                    task_id,
                    TaskStep(
                        description=f"Called {tool_name}",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=tool_result.get("content"),
                        status=TaskStepStatus.COMPLETED,
                        finished_at=self._now(),
                    ),
                )
                steps_added += 1
                if steps_added >= _MAX_STEPS:
                    break

        # Drain any tool_calls that the agent issued without a matching result —
        # they were proposed but never executed; record as failed steps so the
        # user can see what was attempted.
        for call_id, pending in pending_calls.items():
            if steps_added >= _MAX_STEPS:
                break
            self.store.append_step(
                task_id,
                TaskStep(
                    description=f"Proposed {pending['name']} (no result)",
                    tool_name=pending["name"],
                    tool_args=pending["args"],
                    status=TaskStepStatus.SKIPPED,
                    finished_at=self._now(),
                ),
            )
            steps_added += 1

        final_text = self._extract_final_assistant_text(messages)
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

    @staticmethod
    def _extract_messages(result: Any) -> list[Any]:
        if isinstance(result, dict):
            messages = result.get("messages")
            if isinstance(messages, list):
                return messages
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return [
                {
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None),
                }
                for tc in tool_calls
            ]
        return []

    @staticmethod
    def _extract_tool_result(message: Any) -> Optional[dict[str, Any]]:
        msg_type = getattr(message, "type", None) or message.__class__.__name__
        if msg_type not in ("tool", "ToolMessage"):
            return None
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except (TypeError, ValueError):
                content = str(content)
        return {
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None),
            "content": content,
        }

    @staticmethod
    def _extract_final_assistant_text(messages: list[Any]) -> str:
        for message in reversed(messages):
            msg_type = getattr(message, "type", None) or message.__class__.__name__
            if msg_type in ("ai", "AIMessage"):
                content = getattr(message, "content", "")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") == "text":
                            parts.append(str(chunk.get("text", "")))
                        elif isinstance(chunk, str):
                            parts.append(chunk)
                    text = "\n".join(p for p in parts if p).strip()
                    if text:
                        return text
        return ""

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


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
