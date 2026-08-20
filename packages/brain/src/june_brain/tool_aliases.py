"""Data-driven tool alias and parameter normalizer table.

Each tool has:
- ``aliases`` — list of old/alternate names that map to the canonical name
- ``param_map`` — mapping from non-canonical param names to canonical ones
- ``normalizer`` — optional callable that reshapes the entire args dict

It replaced a 200-line if/elif chain, and D.5a shrank it again: the table exists
because small local models miss on tool names and parameter shapes, so an entry
only earns its lines while the tool it points at is still advertised. After the
v1 domain writers went (ADR 0032) one entry is left, and it points at a name the
calendar *skill* serves rather than a native tool.

**Measured, D.5d (2026-08-20).** Across 288 corpus turns the table never fired
once, and a targeted probe of calendar utterances found the model emitting the
canonical `save_calendar_item` on 7 of 7 calls with canonical parameter names on
all of them — no alias, no `param_map` hit. The reason is not subtle: the tools
block now names 15 tools canonically and the model copies what it is shown,
where the table was built when it was choosing among 54 with v1's odd names.

Kept anyway, on one condition. The measurement says the aliases are unused, not
that they are harmful, and n=7 is thin evidence on which to delete a fallback
that costs nothing when it does not fire. What the probe *did* find was the
normalizer silently dropping arguments, which is fixed below. Revisit deletion
if a later measurement covers more of the calendar path and still sees nothing.
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
    """Fold the model's alternate key names onto the canonical ones.

    Merges rather than rebuilds. It used to `return {...}` a fixed four-key
    dict, which silently dropped every other argument the model supplied —
    measurably `status` and `source` on real calls (D.5d). `source` is the
    provenance tag the memory browser uses to say where a saved item came from,
    so a normalizer whose whole job is repairing the model's arguments was
    quietly discarding valid ones instead.
    """
    out = dict(args)
    out["title"] = _best_of(args, "title", ["event", "name"])
    out["date"] = _best_of(args, "date", ["day", "when"])
    out["details"] = _best_of(args, "details", ["note", "description"])
    out["time"] = _best_of(args, "time", ["at"])
    # The alternates have been folded in; leaving them would hand the tool two
    # spellings of the same field.
    for alt in ("event", "name", "day", "when", "note", "description", "at"):
        out.pop(alt, None)
    return out


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
