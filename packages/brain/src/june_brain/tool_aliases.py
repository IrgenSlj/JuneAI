"""Data-driven tool alias and parameter normalizer table.

Each tool has:
- ``aliases`` — list of old/alternate names that map to the canonical name
- ``param_map`` — mapping from non-canonical param names to canonical ones
- ``normalizer`` — optional callable that reshapes the entire args dict

This replaces the 200-line if/elif chain in ``graph.py:_normalize_tool_call``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Normalizer = Callable[[dict[str, Any]], dict[str, Any] | tuple[str, dict[str, Any]]]


class ToolAlias:
    """Configuration for a single tool's alias resolution and parameter normalization."""

    def __init__(
        self,
        *,
        aliases: list[str] | None = None,
        param_map: dict[str, str] | None = None,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.aliases = aliases or []
        self.param_map = param_map or {}
        self.normalizer = normalizer


# ---------------------------------------------------------------------------
# Normalizer helpers
# ---------------------------------------------------------------------------


def _first_of(args: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty, non-None value for any of the given keys."""
    for key in keys:
        val = args.get(key)
        if val is not None and val != "":
            return val
    return ""


def _best_of(args: dict[str, Any], canonical: str, alt_keys: list[str]) -> Any:
    """Like ``_first_of`` but takes a list."""
    return _first_of(args, canonical, *alt_keys)


def _merge_optional(args: dict[str, Any], field: str, *alt_keys: str) -> Any:
    """Prefer the canonical field, fall back to alternatives."""
    return _first_of(args, field, *alt_keys)


# ---------------------------------------------------------------------------
# Individual normalizer functions for tools with complex mapping
# ---------------------------------------------------------------------------


def _normalize_save_calendar_item(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _best_of(args, "title", ["event", "name"]),
        "date": _best_of(args, "date", ["day", "when"]),
        "details": _best_of(args, "details", ["note", "description"]),
        "time": _best_of(args, "time", ["at"]),
    }


def _normalize_track_goal(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _best_of(args, "title", ["goal", "name"]),
        "category": _best_of(args, "category", ["area"]) or "personal",
        "target_date": _best_of(args, "target_date", ["deadline", "date"]),
        "next_step": _best_of(args, "next_step", ["next", "action"]),
        "status": args.get("status") or "active",
    }


def _normalize_save_open_loop(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "topic": _best_of(args, "topic", ["title", "name"]),
        "next_step": _best_of(args, "next_step", ["next", "action"]),
        "due_date": _best_of(args, "due_date", ["deadline", "date"]),
        "status": args.get("status") or "open",
    }


def _normalize_save_relationship_profile(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "person": _best_of(args, "person", ["name"]),
        "relationship": _best_of(args, "relationship", ["relation"]),
        "summary": _best_of(args, "summary", ["context", "details"]),
        "user_needs": _best_of(args, "user_needs", ["needs"]),
        "cautions": _best_of(args, "cautions", ["warnings"]),
    }


def _normalize_save_user_preference(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": _best_of(args, "category", ["type"]) or "general",
        "value": _best_of(args, "value", ["preference", "title"]),
        "context": _best_of(args, "context", ["details", "reason"]),
    }


def _normalize_save_favorite_recommendation(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": _best_of(args, "category", ["type"]) or "general",
        "title": _best_of(args, "title", ["name"]),
        "reason": _best_of(args, "reason", ["details"]),
        "creator": _best_of(args, "creator", ["author", "artist"]),
        "status": args.get("status") or "saved",
    }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Heuristic: extract first JSON-like object from ``text``."""
    import json

    idx = text.find("{")
    if idx == -1:
        return None
    text = text[idx:]
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[: i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _normalize_save_journal_entry(args: dict[str, Any]) -> dict[str, Any] | tuple[str, dict[str, Any]]:
    """Extracts JSON payload from the entry string; may reroute to save_calendar_item."""
    entry = args.get("entry", "")
    if not isinstance(entry, str):
        return args
    payload = _extract_json_payload(entry)
    if payload is None:
        return args
    date = str(payload.get("date", "")).strip()
    if not date:
        return args
    title = str(
        payload.get("title")
        or payload.get("event")
        or payload.get("name")
        or "Saved reminder"
    ).strip()
    details = str(payload.get("details") or payload.get("note") or "").strip()
    blob = " ".join(str(value).lower() for value in payload.values())
    if "birthday" in blob and "birthday" not in title.lower():
        title = f"{title} birthday"
    return "save_calendar_item", {
        "title": title,
        "date": date,
        "details": details,
        "source": "conversation",
    }


# ---------------------------------------------------------------------------
# The master table
# ---------------------------------------------------------------------------

TOOL_ALIASES: dict[str, ToolAlias] = {
    "track_goal": ToolAlias(
        aliases=["save_goal", "create_goal", "add_goal"],
        param_map={"goal": "title", "name": "title", "area": "category", "deadline": "target_date", "next": "next_step"},
        normalizer=_normalize_track_goal,
    ),
    "save_calendar_item": ToolAlias(
        aliases=["save_reminder", "add_calendar_item", "create_calendar_item", "save_trip", "save_birthday"],
        normalizer=_normalize_save_calendar_item,
    ),
    "save_user_preference": ToolAlias(
        aliases=["save_preference"],
        param_map={"type": "category", "preference": "value", "reason": "context"},
        normalizer=_normalize_save_user_preference,
    ),
    "save_favorite_recommendation": ToolAlias(
        aliases=["save_favorite", "add_favorite"],
        param_map={"type": "category", "name": "title", "author": "creator", "artist": "creator"},
        normalizer=_normalize_save_favorite_recommendation,
    ),
    "save_open_loop": ToolAlias(
        param_map={"deadline": "due_date", "action": "next_step"},
        normalizer=_normalize_save_open_loop,
    ),
    "save_relationship_profile": ToolAlias(
        param_map={"relation": "relationship", "context": "summary", "needs": "user_needs", "warnings": "cautions"},
        normalizer=_normalize_save_relationship_profile,
    ),
    "save_journal_entry": ToolAlias(
        param_map={"entry": "entry"},
        normalizer=_normalize_save_journal_entry,
    ),
}


def resolve_tool_call(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve aliases and normalize parameters for a tool call.

    Returns ``(canonical_name, normalized_args)``.
    """
    args = dict(args or {})

    for canonical, alias_def in TOOL_ALIASES.items():
        if name == canonical or name in alias_def.aliases:
            name = canonical
            # Apply param_map first (key renames)
            for old, new in alias_def.param_map.items():
                if old in args and new != old:
                    if new not in args or not args[new]:
                        args[new] = args.pop(old)
            # Apply normalizer (full reshape)
            if alias_def.normalizer is not None:
                result = alias_def.normalizer(args)
                # If normalizer returns a (name, args) tuple, it's a tool reroute
                if isinstance(result, tuple):
                    return result
                return name, result
            break

    return name, args
