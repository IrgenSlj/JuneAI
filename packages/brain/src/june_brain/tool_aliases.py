"""Data-driven tool alias and parameter normalizer table.

Each tool has:
- ``aliases`` — list of old/alternate names that map to the canonical name
- ``param_map`` — mapping from non-canonical param names to canonical ones
- ``normalizer`` — optional callable that reshapes the entire args dict

It replaced a 200-line if/elif chain, and D.5a shrank it again: the table exists
because small local models miss on tool names and parameter shapes, so an entry
only earns its lines while the tool it points at is still advertised. After the
v1 domain writers went (ADR 0032) one entry is left, and it points at a name the
calendar *skill* serves rather than a native tool. D.5d re-measures selection
accuracy against the new surface to decide whether even that is still needed.
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


# ---------------------------------------------------------------------------
# The master table
# ---------------------------------------------------------------------------

TOOL_ALIASES: dict[str, ToolAlias] = {
    "save_calendar_item": ToolAlias(
        aliases=["save_reminder", "add_calendar_item", "create_calendar_item", "save_trip", "save_birthday"],
        normalizer=_normalize_save_calendar_item,
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
