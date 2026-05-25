"""Data-driven tool alias and parameter normalizer table.

Each tool has:
- ``aliases`` — list of old/alternate names that map to the canonical name
- ``param_map`` — mapping from non-canonical param names to canonical ones
- ``normalizer`` — optional callable that reshapes the entire args dict

This replaces the 200-line if/elif chain in ``graph.py:_normalize_tool_call``.
"""

from __future__ import annotations

from typing import Any, Callable

Normalizer = Callable[[dict[str, Any]], dict[str, Any]]


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


def _normalize_save_gym_plan(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _best_of(args, "name", ["title", "plan_name"]) or "Gym Plan",
        "schedule": _best_of(args, "schedule", ["split", "routine"]),
        "goal": _best_of(args, "goal", ["focus"]),
        "notes": _best_of(args, "notes", ["details"]),
        "status": args.get("status") or "active",
    }


def _normalize_save_food_program(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _best_of(args, "name", ["title", "program_name"]) or "Food Program",
        "goal": _best_of(args, "goal", ["focus"]),
        "daily_structure": _best_of(args, "daily_structure", ["structure", "meal_structure", "plan"]),
        "notes": _best_of(args, "notes", ["details"]),
        "status": args.get("status") or "active",
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


def _normalize_log_workout_session(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_name": _best_of(args, "plan_name", ["name", "title"]) or "Workout",
        "exercises": _best_of(args, "exercises", ["workout", "details"]),
        "duration_min": args.get("duration_min") or args.get("duration") or 0,
        "notes": args.get("notes") or "",
        "energy_rating": args.get("energy_rating") or args.get("energy") or 0,
    }


def _normalize_log_body_metrics(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "weight_kg": args.get("weight_kg") or args.get("weight") or 0.0,
        "sleep_hours": args.get("sleep_hours") or args.get("sleep") or 0.0,
        "sleep_quality": args.get("sleep_quality") or args.get("sleep_score") or 0,
        "energy": args.get("energy") or args.get("energy_rating") or 0,
        "stress": args.get("stress") or args.get("stress_level") or 0,
        "soreness": args.get("soreness") or args.get("muscle_soreness") or 0,
        "resting_hr": args.get("resting_hr") or args.get("heart_rate") or args.get("rest_hr") or 0,
        "steps": args.get("steps") or args.get("step_count") or 0,
        "notes": args.get("notes") or "",
    }


def _normalize_create_habit(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _best_of(args, "name", ["habit", "title"]),
        "category": args.get("category") or "health",
        "target_days": _best_of(args, "target_days", ["frequency"]) or "daily",
    }


def _normalize_log_habit_completion(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "habit_name": _best_of(args, "habit_name", ["name", "habit"]),
    }


def _normalize_log_nutrition(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "meal": _best_of(args, "meal", ["meal_type"]) or "meal",
        "description": _best_of(args, "description", ["details", "food"]),
        "calories_est": args.get("calories_est") or args.get("calories") or 0,
        "protein_est": args.get("protein_est") or args.get("protein") or 0,
    }


def _normalize_log_water(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "glasses": _best_of(args, "glasses", ["count", "amount"]) or 1,
    }


def _normalize_set_ui_focus(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _best_of(args, "title", ["heading"]) or "Workspace",
        "body": _best_of(args, "body", ["content", "text"]),
        "footer": _best_of(args, "footer", ["notice"]),
    }


def _normalize_set_ui_checklist(args: dict[str, Any]) -> dict[str, Any]:
    items = _best_of(args, "items", ["checklist", "lines"])
    if isinstance(items, list):
        items = "\n".join(str(item) for item in items)
    return {
        "title": _best_of(args, "title", ["heading"]) or "Next steps",
        "items": items,
    }


def _normalize_set_ui_layout(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "layout": _best_of(args, "layout", ["mode"]) or "split",
        "notice": args.get("notice") or "",
    }


def _normalize_set_ui_chapter(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "chapter": _best_of(args, "chapter", ["name", "section"]),
        "notice": args.get("notice") or "",
    }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Heuristic: extract first JSON-like object from ``text``."""
    import json
    import re

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
    "log_workout_session": ToolAlias(
        aliases=["save_workout", "record_workout"],
        param_map={"workout": "exercises", "duration": "duration_min", "energy": "energy_rating"},
        normalizer=_normalize_log_workout_session,
    ),
    "log_nutrition": ToolAlias(
        aliases=["save_meal", "record_meal"],
        param_map={"meal_type": "meal", "food": "description", "calories": "calories_est", "protein": "protein_est"},
        normalizer=_normalize_log_nutrition,
    ),
    "log_water": ToolAlias(
        aliases=["record_water"],
        param_map={"count": "glasses", "amount": "glasses"},
        normalizer=_normalize_log_water,
    ),
    "set_ui_chapter": ToolAlias(
        aliases=["set_chapter", "open_chapter"],
        param_map={"section": "chapter"},
        normalizer=_normalize_set_ui_chapter,
    ),
    "set_ui_focus": ToolAlias(
        aliases=["focus_workspace", "update_workspace"],
        param_map={"heading": "title", "content": "body", "text": "body", "notice": "footer"},
        normalizer=_normalize_set_ui_focus,
    ),
    "save_open_loop": ToolAlias(
        param_map={"deadline": "due_date", "action": "next_step"},
        normalizer=_normalize_save_open_loop,
    ),
    "save_gym_plan": ToolAlias(
        param_map={"split": "schedule", "routine": "schedule", "focus": "goal"},
        normalizer=_normalize_save_gym_plan,
    ),
    "save_food_program": ToolAlias(
        param_map={"structure": "daily_structure", "meal_structure": "daily_structure", "plan": "daily_structure", "focus": "goal"},
        normalizer=_normalize_save_food_program,
    ),
    "save_relationship_profile": ToolAlias(
        param_map={"relation": "relationship", "context": "summary", "needs": "user_needs", "warnings": "cautions"},
        normalizer=_normalize_save_relationship_profile,
    ),
    "log_body_metrics": ToolAlias(
        param_map={"weight": "weight_kg", "sleep": "sleep_hours", "sleep_score": "sleep_quality", "energy_rating": "energy", "stress_level": "stress", "muscle_soreness": "soreness", "heart_rate": "resting_hr", "rest_hr": "resting_hr", "step_count": "steps"},
        normalizer=_normalize_log_body_metrics,
    ),
    "create_habit": ToolAlias(
        param_map={"habit": "name", "frequency": "target_days"},
        normalizer=_normalize_create_habit,
    ),
    "log_habit_completion": ToolAlias(
        param_map={"habit": "habit_name"},
        normalizer=_normalize_log_habit_completion,
    ),
    "set_ui_checklist": ToolAlias(
        param_map={"checklist": "items", "lines": "items"},
        normalizer=_normalize_set_ui_checklist,
    ),
    "set_ui_layout": ToolAlias(
        param_map={"mode": "layout"},
        normalizer=_normalize_set_ui_layout,
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
