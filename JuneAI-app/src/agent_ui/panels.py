"""Reusable rail and dashboard panel helpers for the JuneAI UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from agent.context_intelligence import (
    build_active_commitments_summary,
    build_recovery_readiness_summary,
)
from agent.memory import Memory
from agent.telemetry import get_recent_events

from .chapter_surface import chapter_status_cards, today_memory_summary
from .chapters import CHAPTERS, chapter_items
from .onboarding import (
    FirstRunSummary,
    OnboardingStage,
    first_run_setup_summary,
    workspace_onboarding_plan,
)

_SETUP_TARGETS = {"calendar", "plans", "habits", "body", "family", "birthdays"}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _panel_lines_from_cards(cards: list[dict[str, Any]], limit: int = 4) -> tuple[PanelLine, ...]:
    lines: list[PanelLine] = []
    for card in cards[:limit]:
        title = str(card.get("title", "Item"))
        value = str(card.get("value", ""))
        preview = str(card.get("preview", ""))
        copy = " · ".join(part for part in (value, preview) if part).strip()
        lines.append(PanelLine(title=title, copy=copy or value or preview or ""))
    return tuple(lines)


def _setup_section_items(summary: FirstRunSummary, setup: SetupProgressModel) -> tuple[PanelLine, ...]:
    if setup.is_complete:
        return tuple(PanelLine(title=step, copy="") for step in summary.setup_steps[:3])
    return setup.missing_rows


def _chrome_for_today(summary: FirstRunSummary) -> ChromePlan:
    visible = ("Today", "Workspace")
    hidden = ("Memory", "Debug") if summary.chrome_hint == "minimal" else ("Debug",)
    return ChromePlan(
        mode="today-first",
        primary_surface="Today",
        visible_surfaces=visible,
        hidden_surfaces=hidden,
        rationale="Keep the main page focused on today, with fewer competing surfaces in view.",
        density=summary.chrome_hint,
        surface_budget=2 if summary.chrome_hint == "minimal" else 3,
    )


def _chrome_for_secondary(primary_surface: str, secondary_surface: str, summary: FirstRunSummary) -> ChromePlan:
    return ChromePlan(
        mode="minimal",
        primary_surface=primary_surface,
        visible_surfaces=(primary_surface, secondary_surface),
        hidden_surfaces=tuple(summary.missing_surfaces[:3]),
        rationale="Prefer a single active surface and keep the rest quiet.",
        density=summary.chrome_hint,
        surface_budget=2,
    )


@dataclass(frozen=True)
class PanelMetric:
    """Compact KPI-style data for the rail surfaces."""

    label: str
    value: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelLine:
    """Compact title/copy data for list-style rail items."""

    title: str
    copy: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelSection:
    """Named section for a premium, calmer panel layout."""

    key: str
    title: str
    items: tuple[PanelLine, ...]
    note: str = ""
    density: str = "calm"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChromePlan:
    """Surface plan that describes how much chrome should be visible."""

    mode: str
    primary_surface: str
    visible_surfaces: tuple[str, ...]
    hidden_surfaces: tuple[str, ...]
    rationale: str
    density: str = "calm"
    surface_budget: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SetupProgressModel:
    """Actionable setup metadata for the Today rail."""

    title: str
    caption: str
    headline: str
    recommended_next_action: str
    recommended_next_reason: str
    ready_count: int
    total_count: int
    missing_titles: tuple[str, ...]
    missing_rows: tuple[PanelLine, ...]
    stages: tuple[OnboardingStage, ...]
    setup_steps: tuple[str, ...]
    next_actions: tuple[str, ...]
    has_data: bool
    is_complete: bool
    summary: FirstRunSummary
    chrome: ChromePlan

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TodayPanelModel:
    """Structured data for the Today rail."""

    title: str
    headline: str
    caption: str
    subheadline: str
    kpis: tuple[PanelMetric, ...]
    setup: SetupProgressModel
    today_summary: dict[str, Any]
    readiness_summary: dict[str, Any]
    sections: tuple[PanelSection, ...]
    chrome: ChromePlan
    next_up: tuple[PanelLine, ...]
    priority_stack: tuple[PanelLine, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryPanelModel:
    """Structured data for the Memory rail."""

    title: str
    caption: str
    chrome: ChromePlan
    selected_key: str
    selected_label: str
    kicker_copy: str
    chapter_cards: tuple[dict[str, Any], ...]
    selected_card: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspacePanelModel:
    """Structured data for the Workspace rail."""

    title: str
    caption: str
    chrome: ChromePlan
    focus_title: str
    focus_body: str
    checklist_title: str
    checklist_items: tuple[str, ...]
    notice: str
    onboarding: Any | None
    is_empty: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebugPanelModel:
    """Structured data for the trust rail."""

    title: str
    caption: str
    what_june_saved: tuple[PanelLine, ...]
    recent_assistant_actions: tuple[PanelLine, ...]
    chrome: ChromePlan
    recent_events: tuple[dict[str, Any], ...]
    recent_activity: tuple[str, ...]
    capture_health_counts: dict[str, int]
    recent_saves: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_setup_progress_model(memory: Memory) -> SetupProgressModel:
    """Build actionable setup metadata from chapter completeness."""
    statuses = chapter_status_cards(memory)
    targeted = [status for status in statuses if status["key"] in _SETUP_TARGETS]
    ready_count = sum(1 for status in targeted if status["freshness"] != "empty")
    total_count = len(targeted)
    missing_titles = tuple(status["title"] for status in targeted if status["freshness"] == "empty")
    onboarding = first_run_setup_summary(memory)
    return SetupProgressModel(
        title="Teach June your foundations first",
        caption=(
            f"{ready_count}/{total_count} core surfaces seeded. "
            "The more complete this is, the more useful June becomes across reminders, routines, and follow-up questions."
        ),
        headline=onboarding.headline,
        recommended_next_action=onboarding.recommended_next_action,
        recommended_next_reason=onboarding.recommended_next_reason,
        ready_count=ready_count,
        total_count=total_count,
        missing_titles=missing_titles,
        missing_rows=tuple(PanelLine(title=name, copy="core setup still missing") for name in missing_titles[:4]),
        stages=onboarding.stages,
        setup_steps=onboarding.setup_steps,
        next_actions=onboarding.next_actions,
        has_data=onboarding.has_data,
        is_complete=ready_count >= total_count,
        summary=onboarding,
        chrome=ChromePlan(
            mode=onboarding.chrome_hint,
            primary_surface=onboarding.primary_focus,
            visible_surfaces=tuple(item for item in (onboarding.primary_focus, *onboarding.secondary_focus[:1]) if item),
            hidden_surfaces=tuple(item for item in onboarding.missing_surfaces[:2]),
            rationale="Keep the setup lane tight and let June ask for only the next missing signal.",
            density=onboarding.chrome_hint,
            surface_budget=2 if not onboarding.has_data else 3,
        ),
    )


def _today_kpis(memory: Memory, snapshot: Mapping[str, int]) -> tuple[PanelMetric, ...]:
    commitments = build_active_commitments_summary(memory)
    water_today = memory.get_water_today()
    habits = memory.get_habits()
    habits_done = sum(1 for habit in habits if habit.get("done_today"))
    notifications = memory.get_upcoming_notifications(limit=3)

    return (
        PanelMetric("Agenda", str(len(notifications) or _coerce_int(snapshot.get("calendar_count", 0))), "upcoming"),
        PanelMetric("Load", str(_coerce_int(commitments.get("load_score", 0))), str(commitments.get("load_label", "steady"))),
        PanelMetric("Habits", f"{habits_done}/{len(habits)}" if habits else "0/0", "today"),
        PanelMetric("Water", f"{water_today}/8", "glasses"),
    )


def _panel_line_from_notification(item: Mapping[str, Any]) -> PanelLine:
    prefix = "today" if _coerce_int(item.get("days_until", 0)) == 0 else f"in {item.get('days_until')}d"
    suffix = f"{item.get('kind', '')} · {item.get('when', '')} · {prefix}"
    details = item.get("details")
    if details:
        suffix = f"{suffix} | {details}"
    return PanelLine(title=str(item.get("title", "Reminder")), copy=suffix)


def _priority_stack_rows(memory: Memory, commitments: Mapping[str, Any]) -> tuple[PanelLine, ...]:
    rows: list[PanelLine] = []
    for goal in memory.get_goals(status="active", limit=3):
        suffix = goal.get("next_step") or goal.get("status", "active")
        if goal.get("target_date"):
            suffix = f"{suffix} (target {goal['target_date']})"
        rows.append(PanelLine(title=str(goal.get("title", "Goal")), copy=f"goal · {suffix}"))
    for loop in memory.get_open_loops(status="open", limit=3):
        suffix = loop.get("next_step") or loop.get("status", "open")
        if loop.get("due_date"):
            suffix = f"{suffix} (due {loop['due_date']})"
        rows.append(PanelLine(title=str(loop.get("topic", "Open loop")), copy=f"open loop · {suffix}"))
    for action in commitments.get("next_actions", [])[:2]:
        rows.append(PanelLine(title=str(action), copy="recommended next move"))
    return tuple(rows)


def build_today_panel_model(memory: Memory, snapshot: Mapping[str, int]) -> TodayPanelModel:
    """Build a structured Today rail model from the current memory state."""
    commitments = build_active_commitments_summary(memory)
    readiness = build_recovery_readiness_summary(memory)
    summary = today_memory_summary(memory)
    notifications = memory.get_upcoming_notifications(limit=3)
    next_up = tuple(_panel_line_from_notification(item) for item in notifications)
    setup = build_setup_progress_model(memory)
    card_lines = _panel_lines_from_cards(summary.get("cards", []), limit=4)
    readiness_lines: tuple[PanelLine, ...] = (
        PanelLine(title="Readiness", copy=f"{readiness.get('readiness_label', 'unknown')} · {readiness.get('readiness_score', 0)}/100"),
    )
    if readiness.get("signals"):
        readiness_lines = readiness_lines + tuple(
            PanelLine(title="Signal", copy=str(signal))
            for signal in readiness.get("signals", [])[:2]
        )
    sections = (
        PanelSection(
            key="today",
            title="Today",
            items=card_lines,
            note=summary.get("headline", "No tracked activity yet."),
            density="calm",
        ),
        PanelSection(
            key="readiness",
            title="Recovery",
            items=readiness_lines,
            note=str(readiness.get("recommendations", ["Keep the routine steady."])[0]),
            density="minimal",
        ),
        PanelSection(
            key="setup",
            title="Setup",
            items=_setup_section_items(setup.summary, setup),
            note=setup.headline,
            density="minimal",
        ),
        PanelSection(
            key="priority",
            title="Priority stack",
            items=tuple(next_up[:2]) + tuple(_priority_stack_rows(memory, commitments)[:3]),
            note="Keep the next move visible.",
            density="calm",
        ),
    )

    return TodayPanelModel(
        title="Daily operating view",
        headline=summary.get("headline", "No tracked activity yet."),
        caption="See what matters now, what is slipping, and what June can help you move forward today.",
        subheadline=f"{setup.summary.activation_level.title()} mode · Next: {setup.recommended_next_action}",
        kpis=_today_kpis(memory, snapshot),
        setup=setup,
        today_summary=summary,
        readiness_summary=readiness,
        sections=sections,
        chrome=_chrome_for_today(setup.summary),
        next_up=next_up,
        priority_stack=_priority_stack_rows(memory, commitments),
    )


def build_memory_panel_model(memory: Memory, selected_chapter: str = "") -> MemoryPanelModel:
    """Build a structured Memory rail model with chapter metadata."""
    cards = tuple(chapter_status_cards(memory))
    selected_key = selected_chapter.strip().lower()
    selected_label = dict(CHAPTERS).get(selected_key, "")
    kicker_copy = (
        f"Open: {selected_label}"
        if selected_label
        else "Open a chapter to inspect stored context without leaving the page."
    )
    selected_card = next((card for card in cards if card["key"] == selected_key), None) if selected_key else None
    return MemoryPanelModel(
        title="Areas",
        caption="Persistent memory should feel alive. Open chapters to review what June knows and what needs attention.",
        chrome=_chrome_for_secondary("Memory", selected_label or "Memory", first_run_setup_summary(memory)),
        selected_key=selected_key,
        selected_label=selected_label,
        kicker_copy=kicker_copy,
        chapter_cards=cards,
        selected_card=selected_card,
    )


def build_workspace_panel_model(ui_state: Mapping[str, Any]) -> WorkspacePanelModel:
    """Build a structured Workspace rail model."""
    focus_title = str(ui_state.get("focus_title") or "Workspace")
    focus_body = str(ui_state.get("focus_body") or "")
    checklist_title = str(ui_state.get("checklist_title") or "Next steps")
    checklist_items = tuple(str(item) for item in ui_state.get("checklist_items", []) if str(item))
    notice = str(ui_state.get("notice") or "")
    onboarding = workspace_onboarding_plan(dict(ui_state)) if not (focus_body or checklist_items or notice) else None
    return WorkspacePanelModel(
        title=focus_title,
        caption="This is where June keeps the current frame, checklist, and short-term focus visible.",
        chrome=ChromePlan(
            mode="minimal" if onboarding else "calm",
            primary_surface="Workspace",
            visible_surfaces=("Workspace", "Today"),
            hidden_surfaces=("Debug", "Memory"),
            rationale="Keep the workspace quiet unless the user has something pinned.",
            density="minimal" if onboarding else "calm",
            surface_budget=2 if onboarding else 3,
        ),
        focus_title=focus_title,
        focus_body=focus_body,
        checklist_title=checklist_title,
        checklist_items=checklist_items,
        notice=notice,
        onboarding=onboarding,
        is_empty=onboarding is not None,
    )


def capture_health_counts(memory: Memory) -> dict[str, int]:
    """Count the core capture surfaces used by the debug rail."""
    return {
        "Agenda": len(memory.get_calendar_items(limit=100)),
        "Habits": len(memory.get_habits()),
        "Body": len(memory.get_body_metrics(days=30)),
        "Workout Sessions": len(memory.get_workout_sessions(limit=100)),
        "Nutrition": len(memory.get_nutrition_recent(limit=100)),
        "Water": 1 if memory.get_water_today() else 0,
        "Birthdays": len(chapter_items(memory, "birthdays")),
        "Trips": len(chapter_items(memory, "trips")),
        "Gym": len(memory.get_gym_plans(limit=100)),
        "Food": len(memory.get_food_programs(limit=100)),
        "Plans": len(memory.get_goals(status="", limit=100)) + len(memory.get_open_loops(status="", limit=100)),
        "Dating": len(chapter_items(memory, "dating")),
        "Family": len(chapter_items(memory, "family")),
    }


def recent_activity_lines(activity_log: list[str], limit: int = 10) -> tuple[str, ...]:
    """Return the latest activity entries in display order."""
    return tuple(line for line in activity_log[-limit:] if str(line))


def recent_save_lines(activity_log: list[str], limit: int = 5) -> tuple[str, ...]:
    """Return the recent save-related activity entries."""
    recent_saves = [
        line for line in activity_log[-30:]
        if "save_" in line.lower() or "saved" in line.lower()
    ][-limit:]
    return tuple(recent_saves)


_SAVE_TOOL_LABELS = {
    "save_goal": "Goal",
    "save_open_loop": "Open loop",
    "save_calendar_item": "Calendar item",
    "save_relationship_profile": "Relationship note",
    "save_user_preference": "Preference",
    "save_favorite_recommendation": "Favorite",
    "save_journal_entry": "Journal entry",
    "log_body_metrics": "Body check-in",
    "log_workout_session": "Workout session",
    "log_nutrition": "Meal log",
    "log_water": "Water log",
    "log_habit_completion": "Habit completion",
    "track_goal": "Goal update",
    "create_habit": "Habit setup",
    "set_ui_focus": "Workspace focus",
    "set_ui_checklist": "Workspace checklist",
    "set_ui_layout": "Layout",
    "set_ui_chapter": "Chapter",
}


def _tool_label(tool_name: str) -> str:
    normalized = tool_name.strip().lower()
    if normalized in _SAVE_TOOL_LABELS:
        return _SAVE_TOOL_LABELS[normalized]
    for prefix in ("save_", "log_", "track_", "set_ui_"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    normalized = normalized.replace("_", " ").strip()
    return normalized.title() if normalized else "Saved item"


def _what_june_saved_lines(events: list[dict[str, Any]], limit: int = 5) -> tuple[PanelLine, ...]:
    rows: list[PanelLine] = []
    for event in reversed(events):
        if len(rows) >= limit:
            break
        event_type = str(event.get("event_type", "")).strip().lower()
        if event_type != "tool_call" or str(event.get("status", "")).lower() != "success":
            continue
        payload_raw = event.get("payload")
        payload: dict[str, Any] = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}
        tool_name = str(event.get("name") or payload.get("tool_name") or "").strip()
        if not tool_name.startswith(("save_", "log_", "track_", "set_ui_")):
            continue
        preview = str(payload.get("preview") or "").strip()
        if not preview:
            preview = "Saved to memory."
        title = (
            f"{_tool_label(tool_name)} saved"
            if tool_name.startswith(("save_", "log_", "track_"))
            else f"{_tool_label(tool_name)} updated"
        )
        rows.append(PanelLine(title=title, copy=preview))

    if len(rows) < limit:
        for event in reversed(events):
            if len(rows) >= limit:
                break
            if str(event.get("event_type", "")).strip().lower() != "save_event":
                continue
            payload_raw = event.get("payload")
            save_payload: dict[str, Any] = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}
            kind = str(save_payload.get("kind") or event.get("name") or "").strip()
            if not kind:
                continue
            copy = str(save_payload.get("preview") or event.get("name") or "Saved to memory.").strip()
            rows.append(PanelLine(title=f"{_tool_label(kind)} saved", copy=copy))
    return tuple(rows)


def _assistant_action_lines(events: list[dict[str, Any]], activity_log: list[str], limit: int = 5) -> tuple[PanelLine, ...]:
    rows: list[PanelLine] = []

    for event in reversed(events):
        if len(rows) >= limit:
            break
        event_type = str(event.get("event_type", "")).strip().lower()
        name = str(event.get("name") or "").strip()
        status = str(event.get("status") or "").strip().lower()
        payload_raw = event.get("payload")
        action_payload: dict[str, Any] = dict(payload_raw) if isinstance(payload_raw, Mapping) else {}

        if event_type == "route_selection":
            route = str(event.get("route") or action_payload.get("route") or name or "").strip()
            rows.append(PanelLine(title="Route chosen", copy=route or "June selected a route."))
            continue

        if event_type != "tool_call":
            continue
        if status == "success" and name.startswith(("save_", "log_", "track_", "set_ui_")):
            continue

        preview = str(action_payload.get("preview") or "").strip()
        if status == "requested":
            rows.append(
                PanelLine(
                    title=f"Tool requested: {_tool_label(name)}",
                    copy=preview or "June queued the tool call.",
                )
            )
        elif status == "success":
            rows.append(
                PanelLine(
                    title=f"Tool finished: {_tool_label(name)}",
                    copy=preview or "June completed the tool call.",
                )
            )
        else:
            rows.append(
                PanelLine(
                    title=f"Tool issue: {_tool_label(name)}",
                    copy=preview or "June hit a problem while using the tool.",
                )
            )

    if len(rows) < limit:
        for line in reversed(activity_log):
            if len(rows) >= limit:
                break
            normalized = line.strip()
            if not normalized:
                continue
            prefix, _, detail = normalized.partition("|")
            prefix = prefix.strip().lower()
            detail = detail.strip()
            if prefix in {"route", "auto route"}:
                rows.append(PanelLine(title="Route chosen", copy=detail or normalized))
            elif prefix == "layout":
                rows.append(PanelLine(title="Layout changed", copy=detail or normalized))
            elif prefix == "rail view":
                rows.append(PanelLine(title="Rail view changed", copy=detail or normalized))
            elif prefix == "right rail":
                rows.append(PanelLine(title="Right rail toggled", copy=detail or normalized))
            elif prefix == "daily check-in":
                rows.append(PanelLine(title="Check-in sent", copy=detail or normalized))
            elif prefix == "planning":
                rows.append(PanelLine(title="Planning", copy=detail or normalized))
            elif prefix == "tool request":
                rows.append(PanelLine(title="Tool requested", copy=detail or normalized))
            elif prefix == "response":
                rows.append(PanelLine(title="Response", copy=detail or normalized))
            elif prefix == "node":
                rows.append(PanelLine(title="Agent step", copy=detail or normalized))
            elif prefix == "tool args":
                rows.append(PanelLine(title="Tool arguments", copy=detail or normalized))
            elif prefix == "tool":
                rows.append(PanelLine(title="Tool output", copy=detail or normalized))

    return tuple(rows[:limit])


def build_trust_panel_model(memory: Memory, activity_log: list[str]) -> DebugPanelModel:
    """Build a structured trust rail model."""
    events = get_recent_events(memory, limit=12)
    return DebugPanelModel(
        title="Saved context",
        caption="Review what June stored, what it acted on, and a light health check. The raw diagnostics stay secondary.",
        what_june_saved=_what_june_saved_lines(events, limit=5),
        recent_assistant_actions=_assistant_action_lines(events, activity_log, limit=5),
        chrome=ChromePlan(
            mode="minimal",
            primary_surface="Trust",
            visible_surfaces=("Trust",),
            hidden_surfaces=("Today", "Workspace", "Memory"),
            rationale="Trust should stay calm and lightweight unless the user opens it.",
            density="minimal",
            surface_budget=1,
        ),
        recent_events=tuple(events[-8:]),
        recent_activity=recent_activity_lines(activity_log),
        capture_health_counts=capture_health_counts(memory),
        recent_saves=recent_save_lines(activity_log),
    )


def build_debug_panel_model(memory: Memory, activity_log: list[str]) -> DebugPanelModel:
    """Backward-compatible alias for the trust rail model."""
    return build_trust_panel_model(memory, activity_log)
