"""Shared, engine-neutral helpers for the harness loop.

These functions used to live in ``june_brain.graph`` (the LangGraph agent), but
they are pure utilities that the hand-written loop's wiring depends on:
JSON-from-prose tool-call extraction, runtime tool selection, and per-turn
memory recall. Relocating them here lets the LangGraph engine be deleted
without dragging the hand-written path down with it.

No LangChain/LangGraph imports belong in this module.
"""

from __future__ import annotations

import json
import logging
import re
from html import unescape
from json import JSONDecodeError
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text + JSON extraction (local models emit tool calls as prose JSON)
# ---------------------------------------------------------------------------


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _strip_internal_thoughts(text: str) -> str:
    """Remove Gemma thought-channel markup before display or reuse."""
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_json_payload(text: str) -> Any | None:
    """Extract the outermost JSON object or array from free-form model text."""
    cleaned = unescape(_strip_internal_thoughts(_strip_code_fence(text))).strip()
    if not cleaned:
        return None

    pairs = []
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        pairs.append((object_start, object_end))
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        pairs.append((array_start, array_end))
    if not pairs:
        return None

    for start, end in sorted(pairs, key=lambda item: item[0]):
        candidate = cleaned[start : end + 1]
        # strategy 1: direct parse
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            pass
        # strategy 2: collapse whitespace
        normalized = candidate.replace("\n", " ").replace("\t", " ")
        try:
            return json.loads(normalized)
        except JSONDecodeError:
            pass
        # strategy 3: replace single quotes with double quotes (common local-model mistake)
        single_to_double = re.sub(r"(?<!\\)'", '"', normalized)
        try:
            return json.loads(single_to_double)
        except JSONDecodeError:
            pass
        # strategy 4: unescape HTML entities and retry
        unescaped = unescape(normalized)
        try:
            return json.loads(unescaped)
        except JSONDecodeError:
            continue

    # Debug, and shape only. This function runs on *every* turn to test whether
    # the model emitted a tool call, so "no JSON here" is the ordinary outcome
    # for an ordinary prose answer — and a prose answer containing a markdown
    # list reaches this line, because `[` and `]` look like an array. Logging it
    # at WARNING made the normal case the loudest thing in the log.
    #
    # The raw text is not logged. It is the model's answer to the user, so it
    # carries whatever was recalled into that turn; writing 120 characters of it
    # into a log file on every bracketed reply put user memories in the log for
    # a non-event. Same rule the ledger already follows for injection reports:
    # shape, never content.
    logger.debug("_extract_json_payload: no JSON payload in %d chars of text", len(text))
    return None


def _normalize_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Correct common local-model tool formatting mistakes.

    Delegates to the data-driven table in ``tool_aliases.py``.
    """
    from ..tool_aliases import resolve_tool_call

    return resolve_tool_call(name, args)


def _coerce_tool_calls(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Convert model-emitted JSON into normalized tool calls."""
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_calls"), list):
            items = payload["tool_calls"]
        elif isinstance(payload.get("calls"), list):
            items = payload["calls"]
        elif isinstance(payload.get("tools"), list):
            items = payload["tools"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    normalized_calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function_block = item.get("function")
        if isinstance(function_block, dict):
            name = str(
                item.get("name")
                or item.get("tool")
                or function_block.get("name")
                or ""
            ).strip()
            parameters = (
                item.get("parameters")
                or item.get("args")
                or item.get("arguments")
                or item.get("input")
                or function_block.get("arguments")
                or {}
            )
        else:
            name = str(item.get("name") or item.get("tool") or "").strip()
            parameters = (
                item.get("parameters")
                or item.get("args")
                or item.get("arguments")
                or item.get("input")
                or {}
            )
        if not name and isinstance(item.get("tool_name"), str):
            name = item["tool_name"].strip()
        if isinstance(parameters, str):
            parsed = _extract_json_payload(parameters)
            parameters = parsed if isinstance(parsed, dict) else {}
        if name and isinstance(parameters, dict):
            normalized_calls.append((name, parameters))
    return normalized_calls


# ---------------------------------------------------------------------------
# Recall — fresh per-turn fan-out across the three memory stores
# ---------------------------------------------------------------------------


def _normalize_recall_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Make a recall hit's ``ref`` match the prefix scheme used in /memory.

    ``MemoryManager.recall`` returns vector hits with bare ``fact_id`` and
    graph hits with bare ``node_id``; the memory snapshot route emits
    those as ``semantic:<id>`` and ``node:<id>``. The prefix rule itself lives
    in ``memory.recall.prefixed_ref`` so the UI, the feedback table and the
    ``forget`` tool cannot drift apart; this only reshapes the hit around it.
    """
    from ..memory.recall import prefixed_ref

    source = hit.get("source")
    kind = hit.get("kind", "")
    return {
        "ref": prefixed_ref(hit),
        "text": hit.get("text", ""),
        "source": str(source or ""),
        "kind": str(kind or ""),
        "score": hit.get("score"),
        "feedback": str(hit.get("feedback") or ""),
    }


def _recall_with_hits(
    user_id: str, query: str, k: int = 5
) -> tuple[str, list[dict[str, Any]]]:
    """Fresh per-turn fan-out across the three memory stores.

    Returns the formatted prompt block AND the raw hits, so the caller can
    both inject the block into the system prompt and stream the hits to
    the UI for a "memories used" disclosure.

    Not cached because the query changes every turn and the individual
    lookups are cheap (vector search ~tens of ms, graph + sqlite keyword
    scans ~ms). If recall raises we swallow and return empties so a broken
    memory store never takes down the chat loop.
    """
    from ..memory import MemoryManager

    query = (query or "").strip()
    if not query:
        return "", []
    try:
        manager = MemoryManager(user_id)
        hits = manager.recall(query, k=k)
    except Exception:
        logger.exception("recall block failed for user=%s", user_id)
        return "", []
    return manager.format_for_prompt(hits), [_normalize_recall_hit(h) for h in hits]


# ---------------------------------------------------------------------------
# Tool selection — choose the tool set sized for the active runtime
# ---------------------------------------------------------------------------


def _select_tools_for_runtime(runtime: Any) -> list[Any]:
    """Choose a tool set sized for the active runtime.

    Gemma 4 (local, small) runs with the trimmed tool schema to keep prompts
    lean. Gemini (cloud) gets the full tool set. Enabled MCP skills append
    their tools on top, so toggling a skill on the /skills page adds or
    removes its capabilities on the next agent build.

    Native tools win on name collision — the skill is shadowed until the
    native copy is removed. That matters while we're mid-migration: we do
    not want two implementations of e.g. ``log_water`` competing for the
    same call.
    """
    from ..skills import load_skill_tools
    from ..tools import JUNE_TOOLS, JUNE_TOOLS_GEMMA, RETIRED_TOOL_NAMES

    if getattr(runtime, "preset_key", "") == "gemma":
        native = JUNE_TOOLS_GEMMA
    else:
        native = JUNE_TOOLS

    native_names = {getattr(t, "name", "") for t in native}
    try:
        skill_tools = []
        for t in load_skill_tools():
            if t.name in native_names:
                continue
            if t.name in RETIRED_TOOL_NAMES:
                # Shadowing alone is not enough: it only holds while a native
                # tool of the same name exists, so deleting one hands the name
                # to whatever skill declares it. A retired name stays retired.
                logger.warning(
                    "skill tool %r ignored: the name was retired with the v1 "
                    "domain layer", t.name,
                )
                continue
            skill_tools.append(t)
    except Exception:
        logger.exception("Failed to load skill tools")
        skill_tools = []

    return list(native) + skill_tools
