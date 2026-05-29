"""Default-seam builders for HandwrittenLoop.

Keeps handwritten.py clean by isolating the "production wiring" — the code
that connects the loop to graph helpers, memory recall, the tool registry, and
the difficulty router.  Every function here returns a plain callable so the
loop stays fully injectable for tests.
"""

from __future__ import annotations

from typing import Any

from june_brain.providers.base import Message

from .interface import SessionState, ToolCall

# ---------------------------------------------------------------------------
# Recall wiring
# ---------------------------------------------------------------------------


def make_recall_fn(
    recall_with_hits_fn: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Return (recall_callable, shared_state_dict).

    The shared_state_dict is mutated by each call so the loop can read back
    ``memories_recalled`` and ``recall_hits`` after assemble_context runs.
    """
    # lazy import to avoid circular dependency (graph.py is heavy)
    if recall_with_hits_fn is None:
        from june_brain.graph import _recall_with_hits  # type: ignore[attr-defined]

        recall_with_hits_fn = _recall_with_hits

    state: dict[str, Any] = {"memories_recalled": 0, "recall_hits": []}

    def recall(session: object, user_msg: Message) -> list[Message]:
        try:
            user_id = getattr(session, "user_id", "") or ""
            content = user_msg.content or ""
            block, hits = recall_with_hits_fn(user_id, content, k=5)
        except Exception:
            block, hits = "", []
        state["memories_recalled"] = len(hits)
        state["recall_hits"] = hits
        if block:
            return [Message(role="system", content=block)]
        return []

    return recall, state


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------


def make_extract_tool_calls_fn() -> Any:
    """Return a callable that parses model text into ToolCall list."""

    def extract(result: Any) -> list[ToolCall]:
        # lazy import to avoid heavy graph import at module load
        try:
            from june_brain.graph import (  # type: ignore[attr-defined]
                _coerce_tool_calls,
                _extract_json_payload,
            )
        except Exception:
            return []

        text = getattr(result, "text", "") or ""
        try:
            payload = _extract_json_payload(text)
            if payload is None:
                return []
            pairs = _coerce_tool_calls(payload)
            return [ToolCall(name=name, args=args) for name, args in pairs]
        except Exception:
            return []

    return extract


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def make_dispatch_fn(dispatched_names: list[str]) -> Any:
    """Return an async callable that executes ToolCalls via the tool registry.

    ``dispatched_names`` is a mutable list owned by the caller; this function
    appends each successfully-dispatched tool name so the loop can report
    ``skills_called`` in provenance.
    """

    # Build the tool map once at construction time (lazy import)
    _tool_map: dict[str, Any] | None = None

    def _get_tool_map() -> dict[str, Any]:
        nonlocal _tool_map
        if _tool_map is None:
            try:
                from june_brain.config import resolve_runtime_config  # type: ignore[attr-defined]
                from june_brain.graph import _select_tools_for_runtime  # type: ignore[attr-defined]

                runtime = resolve_runtime_config()
                tools = _select_tools_for_runtime(runtime)
                _tool_map = {t.name: t for t in tools}
            except Exception:
                _tool_map = {}
        return _tool_map

    async def dispatch(tool_calls: list[ToolCall], session: SessionState) -> list[Message]:
        tool_map = _get_tool_map()
        observations: list[Message] = []
        for tc in tool_calls:
            tool = tool_map.get(tc.name)
            if tool is None:
                observations.append(
                    Message(
                        role="tool",
                        content=f"Error: unknown tool '{tc.name}'.",
                    )
                )
                continue
            try:
                result = tool.invoke(tc.args)
                content = str(result)[:4000]
                dispatched_names.append(tc.name)
            except Exception as exc:  # noqa: BLE001
                content = f"Error invoking tool '{tc.name}': {exc}"
            observations.append(Message(role="tool", content=content))
        return observations

    return dispatch
