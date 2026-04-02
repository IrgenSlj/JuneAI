"""Onboarding and empty-state helpers for the JuneAI UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent.memory import Memory

from .chapter_surface import CHAPTERS, chapter_statuses, chapter_subtitle


@dataclass(frozen=True)
class OnboardingCard:
    """Renderable empty-state/onboarding copy."""

    title: str
    body: str
    tips: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    note: str | None = None
    eyebrow: str | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FirstRunSummary:
    """Compact first-run profile/setup snapshot."""

    headline: str
    has_data: bool
    activation_level: str
    primary_focus: str
    secondary_focus: tuple[str, ...]
    chrome_hint: str
    recommended_next_action: str
    recommended_next_reason: str
    staged_steps: tuple[str, ...]
    stages: tuple["OnboardingStage", ...]
    profile_lines: tuple[str, ...]
    setup_steps: tuple[str, ...]
    missing_surfaces: tuple[str, ...]
    active_surfaces: tuple[str, ...]
    next_actions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnboardingStage:
    """One stage in the first-run onboarding flow."""

    key: str
    title: str
    summary: str
    step: str
    action: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chapter_label(chapter_key: str) -> str:
    return dict(CHAPTERS).get(chapter_key, chapter_key.replace("_", " ").title())


def workspace_onboarding_plan(ui_state: dict[str, Any]) -> OnboardingCard:
    """Build onboarding copy for the empty workspace rail."""
    layout = str(ui_state.get("layout") or "split").lower()
    selected_chapter = str(ui_state.get("selected_chapter") or "").strip()
    checklist_items = [str(item) for item in ui_state.get("checklist_items", []) if str(item)]
    notice = str(ui_state.get("notice") or "").strip()

    tips = (
        "Keep one open loop visible.",
        "Add the next concrete action.",
        "Ask June to summarize the current focus.",
    )
    actions = (
        "Capture a note",
        "Set a next step",
        "Ask June to summarize",
    )
    note = f"Layout: {layout}"
    if selected_chapter:
        note = f"Current chapter: {selected_chapter}"
    if checklist_items:
        note = f"{len(checklist_items)} pinned action(s) already captured."
    elif notice:
        note = notice

    return OnboardingCard(
        title="Nothing pinned yet.",
        body="Use this space to pin the active focus, next step, or note June should keep in mind.",
        tips=tips,
        actions=actions,
        note=note,
        eyebrow="Workspace",
    )


def chapter_onboarding_plan(memory: Memory, chapter_key: str) -> OnboardingCard:
    """Build empty-state copy for a specific chapter."""
    subtitle = chapter_subtitle(chapter_key)
    title = "Nothing saved yet."
    body = f"{subtitle or _chapter_label(chapter_key)} is empty for now."
    tips: tuple[str, ...]
    actions: tuple[str, ...]
    note = f"Ask June to save something for {_chapter_label(chapter_key).lower()}."

    if chapter_key == "body":
        body = "Log sleep, energy, stress, soreness, resting HR, and notes so June can infer patterns."
        tips = (
            "Add one quick body check-in after waking up.",
            "Body data helps June compare recovery, fatigue, and consistency.",
        )
        actions = ("Log body", "Add sleep", "Capture energy")
    elif chapter_key == "plans":
        body = "Save one goal, one open loop, or the next action you want June to keep visible."
        tips = (
            "Ask June to capture something relevant here.",
            "Keep the next concrete step attached to the goal.",
            "Open loops help June protect unfinished work.",
        )
        actions = ("Save a goal", "Add an open loop", "Capture next step")
    elif chapter_key == "habits":
        body = "Add a habit and June will track consistency, streaks, and today's completion."
        tips = (
            "Start with a daily habit you can complete quickly.",
            "Use habits for the routines you want June to notice.",
        )
        actions = ("Create habit", "Log completion", "Add streak")
    elif chapter_key == "calendar":
        body = "Add one event, reminder, or appointment so June can remind you at the right time."
        tips = (
            "Include the date and time when you know them.",
            "Use calendar items for anything time-bound or easy to forget.",
        )
        actions = ("Save event", "Add reminder", "Plan trip")
    elif chapter_key == "water":
        body = "Record today's water count so June can nudge hydration and spot patterns."
        tips = (
            "Small counts are still useful.",
            "Hydration is easiest to improve when it is visible.",
        )
        actions = ("Log water", "Set water goal", "Add reminder")
    elif chapter_key == "nutrition":
        body = "Log a meal and June can help compare energy, protein, and appetite through the day."
        tips = (
            "Add the meal you most want to remember.",
            "A few meal logs are enough to start seeing patterns.",
        )
        actions = ("Log meal", "Add calories", "Track protein")
    elif chapter_key == "workouts":
        body = "Save a workout session so June can remember the training work you already did."
        tips = (
            "Capture the session title, duration, and energy.",
            "Notes about pace or recovery make the memory more useful.",
        )
        actions = ("Log workout", "Add exercises", "Save energy")
    else:
        tips = (
            "Ask June to capture something relevant here.",
            "Save one concrete item to give this chapter a foothold.",
            "June gets more useful as the chapter holds specific examples.",
        )
        actions = ("Save note", "Ask June to capture", "Add reminder")

    return OnboardingCard(
        title=title,
        body=body,
        tips=tips,
        actions=actions,
        note=note,
        eyebrow=_chapter_label(chapter_key),
    )


def notification_onboarding_plan(memory: Memory) -> OnboardingCard:
    """Build empty-state copy for the notifications rail."""
    _ = memory  # Keeps the signature ready for future memory-aware copy.
    return OnboardingCard(
        title="Upcoming",
        body="No upcoming reminders right now.",
        tips=(
            "Calendar items and reminders will surface here as they get closer.",
            "Saving one date is enough to activate this rail.",
        ),
        actions=("Save calendar item", "Set reminder", "Plan trip"),
        note="This space stays lightweight until something is due soon.",
        eyebrow="Onboarding",
    )


def _progress_markers(memory: Memory) -> tuple[list[str], list[str], dict[str, int]]:
    counts = {status.key: status.item_count for status in chapter_statuses(memory)}
    active = [label for key, label in CHAPTERS if counts.get(key, 0)]
    missing = [label for key, label in CHAPTERS if not counts.get(key, 0)]
    return active, missing, counts


def _activation_level(has_data: bool, active_count: int) -> str:
    if not has_data:
        return "quiet"
    if active_count <= 2:
        return "starting"
    if active_count <= 5:
        return "growing"
    return "established"


def _chrome_hint(activation_level: str) -> str:
    if activation_level in {"quiet", "starting"}:
        return "minimal"
    return "calm"


def _stage_rows(
    *,
    has_data: bool,
    primary_focus: str,
    secondary_focus: tuple[str, ...],
    missing_surfaces: tuple[str, ...],
    setup_steps: tuple[str, ...],
    next_actions: tuple[str, ...],
) -> tuple[OnboardingStage, ...]:
    if not has_data:
        return (
            OnboardingStage(
                key="seed",
                title="Seed",
                summary="Add a goal, calendar item, or body check-in so June has a stable anchor.",
                step="Start with one concrete save.",
                action="Save one anchor item",
            ),
            OnboardingStage(
                key="activate",
                title="Activate",
                summary=f"Open {primary_focus.lower()} and give June one more signal to work with.",
                step="Fill the surface most likely to matter today.",
                action=f"Open {primary_focus}",
            ),
            OnboardingStage(
                key="stabilize",
                title="Stabilize",
                summary="Keep one routine, reminder, or body metric visible so June can compare over time.",
                step="Protect continuity with a simple repeatable habit.",
                action="Keep one repeatable signal visible",
            ),
        )
    return (
        OnboardingStage(
            key="confirm",
            title="Confirm",
            summary=f"Keep {primary_focus.lower()} in view and let June use it as the first surface.",
            step=setup_steps[0] if setup_steps else "Keep the active surface visible.",
            action=f"Keep {primary_focus} visible",
        ),
        OnboardingStage(
            key="expand",
            title="Expand",
            summary="Add one adjacent signal, like a calendar item, habit, or body check-in.",
            step=setup_steps[1] if len(setup_steps) > 1 else "Add one adjacent signal.",
            action="Add one adjacent signal",
        ),
        OnboardingStage(
            key="maintain",
            title="Maintain",
            summary=f"Use {secondary_focus[0].lower() if secondary_focus else 'today'} to keep follow-up grounded.",
            step=setup_steps[2] if len(setup_steps) > 2 else "Keep one repeatable signal visible.",
            action="Keep the surface steady",
        ),
    )


def first_run_profile_summary(memory: Memory) -> FirstRunSummary:
    """Summarize what June already knows about the user profile."""
    progress = memory.get_progress_snapshot()
    active, missing, counts = _progress_markers(memory)
    has_data = any(counts.values()) or any(
        progress[key]
        for key in ("mood_count", "journal_count", "goal_count", "calendar_count", "favorite_count")
    )
    activation_level = _activation_level(has_data, len(active))
    primary_focus = active[0] if active else "Today"
    secondary_focus = tuple(active[1:4] or missing[:3])
    headline = "June is starting fresh." if not has_data else f"June already knows {len(active)} chapter(s)."
    profile_lines = (
        f"Goals: {progress['goal_count']}",
        f"Calendar items: {progress['calendar_count']}",
        f"Habits: {progress['habit_count']}",
        f"Body logs: {counts.get('body', 0)}",
        f"Water today: {progress['water_today']}",
    )
    next_actions = (
        "Add one goal",
        "Log a body check-in",
        "Save one calendar item",
    ) if not has_data else tuple(
        f"Fill {label.lower()}"
        for label in missing[:3]
    ) or ("Keep adding useful context",)
    setup_steps = (
        "Tell June what matters most right now.",
        "Capture one concrete item that should not be forgotten.",
        "Add a body, habit, or calendar check-in so June can personalize follow-ups.",
    ) if not has_data else (
        "Keep the most important next step visible.",
        "Use the workspace to pin open loops and reminders.",
        "Log the signals June should compare over time.",
    )
    recommended_next_action = next_actions[0]
    recommended_next_reason = (
        "Start by anchoring one concrete item so June has a reliable context signal."
        if not has_data
        else f"June already has {primary_focus.lower()} context, so the next best move is to fill {recommended_next_action.lower()}."
    )
    stages = _stage_rows(
        has_data=has_data,
        primary_focus=primary_focus,
        secondary_focus=secondary_focus,
        missing_surfaces=tuple(missing),
        setup_steps=setup_steps,
        next_actions=tuple(next_actions),
    )
    staged_steps = tuple(stage.step for stage in stages)
    return FirstRunSummary(
        headline=headline,
        has_data=has_data,
        activation_level=activation_level,
        primary_focus=primary_focus,
        secondary_focus=secondary_focus,
        chrome_hint=_chrome_hint(activation_level),
        recommended_next_action=recommended_next_action,
        recommended_next_reason=recommended_next_reason,
        staged_steps=staged_steps,
        stages=stages,
        profile_lines=profile_lines,
        setup_steps=setup_steps,
        missing_surfaces=tuple(missing),
        active_surfaces=tuple(active),
        next_actions=tuple(next_actions),
    )


def first_run_setup_summary(memory: Memory) -> FirstRunSummary:
    """Summarize the setup steps that will make June immediately useful."""
    summary = first_run_profile_summary(memory)
    if summary.has_data:
        setup_steps = (
            "Keep one active goal visible.",
            "Use calendar items for deadlines and reminders.",
            "Add habits or body metrics when you want trend-aware follow-up.",
        )
        headline = "June can already personalize follow-up from your saved context."
    else:
        setup_steps = (
            "Add one goal so June has something to protect.",
            "Log a body check-in so June can learn your baseline.",
            "Save one calendar item or habit to activate reminders and streaks.",
        )
        headline = "A few starter saves will activate June's memory quickly."
    return FirstRunSummary(
        headline=headline,
        has_data=summary.has_data,
        activation_level=summary.activation_level,
        primary_focus=summary.primary_focus,
        secondary_focus=summary.secondary_focus,
        chrome_hint=summary.chrome_hint,
        recommended_next_action=summary.recommended_next_action,
        recommended_next_reason=summary.recommended_next_reason,
        staged_steps=summary.staged_steps,
        stages=summary.stages,
        profile_lines=summary.profile_lines,
        setup_steps=setup_steps,
        missing_surfaces=summary.missing_surfaces,
        active_surfaces=summary.active_surfaces,
        next_actions=summary.next_actions,
    )
