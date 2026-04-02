"""Detail and focus renderers for memory-backed chapter surfaces."""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import date
from typing import Any

import streamlit as st

from agent.memory import Memory

from .chapters import chapter_subtitle
from .rendering import render_memory_focus, water_dots_html


def _focus_hero(title: str, copy: str) -> None:
    st.markdown(
        '<div class="june-focus-hero">'
        f'<div class="june-title">{html.escape(title)}</div>'
        f'<div class="june-focus-copy">{html.escape(copy)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _mini_stats(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        '<div class="june-mini-card">'
        f'<div class="june-mini-label">{html.escape(label)}</div>'
        f'<div class="june-mini-value">{html.escape(value)}</div>'
        '</div>'
        for label, value in items
    )
    st.markdown('<div class="june-mini-grid">' + cards + '</div>', unsafe_allow_html=True)


def calendar_focus_items(memory: Memory, chapter_key: str) -> list[dict[str, Any]]:
    """Return the filtered calendar items for a calendar-like chapter."""
    items = memory.get_calendar_items(status="", limit=20)
    if chapter_key == "trips":
        return [
            item for item in items
            if any(
                term in " ".join(str(item.get(field, "")).lower() for field in ("title", "details", "source"))
                for term in ("trip", "travel", "flight")
            )
        ]
    if chapter_key == "birthdays":
        return [
            item for item in items
            if "birthday" in " ".join(str(item.get(field, "")).lower() for field in ("title", "details", "source"))
        ]
    return items


def render_calendar_focus(
    memory: Memory,
    chapter_key: str,
    *,
    on_activity: Callable[[str], None],
) -> None:
    """Render calendar-like entries with inline status actions."""
    items = calendar_focus_items(memory, chapter_key)
    if not items:
        st.markdown(render_memory_focus(memory, chapter_key), unsafe_allow_html=True)
        return
    today_iso = date.today().isoformat()
    due_today = sum(1 for item in items if str(item.get("date", "")) == today_iso)
    _focus_hero(
        "Calendar focus",
        "Review time-bound commitments here, update statuses quickly, and keep the next dated item visible.",
    )
    _mini_stats(
        [
            ("Visible", str(len(items))),
            ("Today", str(due_today)),
        ]
    )
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle(chapter_key))}</div>', unsafe_allow_html=True)
    for index, item in enumerate(items):
        label = (
            f"{item.get('date', 'date?')}"
            f"{' ' + item.get('time', '') if item.get('time') else ''}"
            f" · {item.get('title', 'Item')}"
            f" · {item.get('status', 'planned')}"
        )
        with st.expander(label, expanded=index == 0):
            if item.get("details"):
                st.write(item["details"])
            st.caption(f"source: {item.get('source', 'conversation')}")
            cols = st.columns(3, gap="small")
            options = [("planned", "Plan"), ("completed", "Done"), ("cancelled", "Cancel")]
            for col, (status, button_label) in zip(cols, options):
                with col:
                    if st.button(
                        button_label,
                        key=f"{chapter_key}_{index}_{status}",
                        use_container_width=True,
                        disabled=item.get("status", "").lower() == status,
                    ):
                        memory.update_calendar_item_status(
                            title=item["title"],
                            status=status,
                            date=item.get("date", ""),
                            time=item.get("time", ""),
                        )
                        on_activity(f"calendar status | {item['title']} -> {status}")
                        st.rerun()


def render_plan_focus(memory: Memory, *, on_activity: Callable[[str], None]) -> None:
    """Render goals and open loops with status controls."""
    goals = memory.get_goals(status="", limit=20)
    loops = memory.get_open_loops(status="", limit=20)
    _focus_hero(
        "Plans workspace",
        "Keep goals directional, open loops finite, and the next step specific enough that June can follow up well.",
    )
    _mini_stats(
        [
            ("Goals", str(len(goals))),
            ("Open loops", str(len(loops))),
        ]
    )
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("plans"))}</div>', unsafe_allow_html=True)
    if not goals and not loops:
        return
    if goals:
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Goals</div>', unsafe_allow_html=True)
        for index, goal in enumerate(goals):
            with st.expander(
                f"{goal['title']} · {goal.get('status', 'active')} · {goal.get('category', 'personal')}",
                expanded=index == 0,
            ):
                if goal.get("next_step"):
                    st.write(f"Next: {goal['next_step']}")
                if goal.get("target_date"):
                    st.caption(f"Target: {goal['target_date']}")
                cols = st.columns(3, gap="small")
                for col, (status, label) in zip(cols, [("active", "Active"), ("paused", "Pause"), ("completed", "Done")]):
                    with col:
                        if st.button(
                            label,
                            key=f"goal_{index}_{status}",
                            use_container_width=True,
                            disabled=goal.get("status", "").lower() == status,
                        ):
                            memory.update_goal_status(goal["title"], status)
                            on_activity(f"goal status | {goal['title']} -> {status}")
                            st.rerun()
    if loops:
        st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="june-label">Open Loops</div>', unsafe_allow_html=True)
        for index, loop in enumerate(loops):
            with st.expander(
                f"{loop['topic']} · {loop.get('status', 'open')}",
                expanded=not goals and index == 0,
            ):
                if loop.get("next_step"):
                    st.write(f"Next: {loop['next_step']}")
                if loop.get("due_date"):
                    st.caption(f"Due: {loop['due_date']}")
                cols = st.columns(3, gap="small")
                for col, (status, label) in zip(cols, [("open", "Open"), ("resolved", "Resolve"), ("closed", "Close")]):
                    with col:
                        if st.button(
                            label,
                            key=f"loop_{index}_{status}",
                            use_container_width=True,
                            disabled=loop.get("status", "").lower() == status,
                        ):
                            memory.update_open_loop_status(loop["topic"], status)
                            on_activity(f"loop status | {loop['topic']} -> {status}")
                            st.rerun()


def render_habits_focus(memory: Memory, *, on_activity: Callable[[str], None]) -> None:
    """Render tracked habits with inline completion controls."""
    habits = memory.get_habits()
    done_count = sum(1 for habit in habits if habit.get("done_today"))
    _focus_hero(
        "Habits dashboard",
        "Use this view to keep daily repetition light, visible, and easy to complete before the day gets noisy.",
    )
    _mini_stats(
        [
            ("Tracked", str(len(habits))),
            ("Done today", str(done_count)),
        ]
    )
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("habits"))}</div>', unsafe_allow_html=True)
    if not habits:
        return
    st.markdown('<div class="june-panel-divider"></div>', unsafe_allow_html=True)
    for index, habit in enumerate(habits):
        cols = st.columns([1.25, 0.7, 0.55], gap="small")
        with cols[0]:
            st.markdown(f"**{habit['name']}**")
            st.caption(
                f"{habit.get('category', 'health')} · {habit.get('target_days', 'daily')} · "
                f"streak {habit.get('streak', 0)}d"
            )
        with cols[1]:
            status = "done today" if habit.get("done_today") else "pending"
            st.caption(status)
        with cols[2]:
            if st.button(
                "Done",
                key=f"habit_done_{index}",
                use_container_width=True,
                disabled=habit.get("done_today", False),
            ):
                item = memory.log_habit_completion(habit["name"])
                on_activity(f"habit done | {item['name']} | streak {item.get('streak', 0)}")
                st.rerun()


def render_water_focus(
    memory: Memory,
    *,
    water_goal: int,
    on_activity: Callable[[str], None],
) -> None:
    """Render today's hydration controls inline."""
    count = memory.get_water_today()
    _focus_hero(
        "Hydration",
        "Simple visibility matters more than precision here. Keep the count honest and the trend easy to maintain.",
    )
    _mini_stats(
        [
            ("Today", f"{count}/{water_goal}"),
            ("Progress", f"{int((count / water_goal) * 100) if water_goal else 0}%"),
        ]
    )
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("water"))}</div>', unsafe_allow_html=True)
    st.markdown(water_dots_html(count, water_goal), unsafe_allow_html=True)
    cols = st.columns([1, 0.5, 0.5], gap="small")
    with cols[0]:
        st.caption(f"{count}/{water_goal} glasses")
    with cols[1]:
        if st.button("−", key="focus_water_minus", use_container_width=True, disabled=count <= 0):
            memory.set_water(count - 1)
            on_activity("water | decrement")
            st.rerun()
    with cols[2]:
        if st.button("+", key="focus_water_plus", use_container_width=True):
            memory.log_water(1)
            on_activity("water | increment")
            st.rerun()


def recent_body_series(memory: Memory, days: int = 7) -> list[dict[str, Any]]:
    """Return recent body entries in ascending date order."""
    items = memory.get_body_metrics(days=days)
    return sorted(items, key=lambda item: item.get("date", ""))


def body_metric_stats(items: list[dict[str, Any]], key: str) -> tuple[float | None, float | None, float | None]:
    """Return current, delta-vs-previous, and simple average for one body metric."""
    values = [float(item.get(key, 0)) for item in items if item.get(key)]
    if not values:
        return None, None, None
    current = values[-1]
    previous = values[-2] if len(values) > 1 else None
    delta = current - previous if previous is not None else None
    average = sum(values) / len(values)
    return current, delta, average


def _metric_card(label: str, current: float | None, delta: float | None, average: float | None, suffix: str = "") -> str:
    if current is None:
        return (
            '<div class="june-stat-card">'
            f'<div class="june-stat-label">{html.escape(label)}</div>'
            '<div class="june-stat-value">-</div>'
            '<div class="june-item-meta">No data</div>'
            '</div>'
        )
    current_text = f"{current:.1f}{suffix}" if isinstance(current, float) and not current.is_integer() else f"{int(current) if float(current).is_integer() else current}{suffix}"
    if delta is None:
        delta_text = "first entry"
    else:
        sign = "+" if delta > 0 else ""
        delta_value = f"{delta:.1f}" if isinstance(delta, float) and not float(delta).is_integer() else f"{int(delta)}"
        delta_text = f"vs prev {sign}{delta_value}{suffix}"
    avg_text = (
        f"7d avg {average:.1f}{suffix}"
        if average is not None and not float(average).is_integer()
        else (f"7d avg {int(average)}{suffix}" if average is not None else "")
    )
    return (
        '<div class="june-stat-card">'
        f'<div class="june-stat-label">{html.escape(label)}</div>'
        f'<div class="june-stat-value">{html.escape(current_text)}</div>'
        f'<div class="june-item-meta">{html.escape(delta_text)}</div>'
        f'<div class="june-item-meta">{html.escape(avg_text)}</div>'
        '</div>'
    )


def render_body_trend_card(memory: Memory, days: int = 7) -> None:
    """Render compact 7-day trend cards for key body metrics."""
    items = recent_body_series(memory, days=days)
    if not items:
        st.caption("No body trend data yet.")
        return
    cards = []
    for label, key, suffix in [
        ("Sleep", "sleep_hours", "h"),
        ("Energy", "energy", "/5"),
        ("Stress", "stress", "/5"),
        ("Soreness", "soreness", "/5"),
        ("Weight", "weight_kg", "kg"),
    ]:
        current, delta, average = body_metric_stats(items, key)
        cards.append(_metric_card(label, current, delta, average, suffix))
    st.markdown('<div class="june-stat-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def body_snapshot_line(item: dict[str, Any]) -> str:
    """Build a compact one-line summary for a body check-in."""
    parts = []
    if item.get("sleep_hours"):
        parts.append(f"sleep {item['sleep_hours']:.1f}h")
    if item.get("energy"):
        parts.append(f"energy {item['energy']}/5")
    if item.get("stress"):
        parts.append(f"stress {item['stress']}/5")
    if item.get("soreness"):
        parts.append(f"soreness {item['soreness']}/5")
    if item.get("weight_kg"):
        parts.append(f"weight {item['weight_kg']:.1f}kg")
    return " · ".join(parts) if parts else "No body metrics recorded."


def render_body_focus(memory: Memory, *, on_activity: Callable[[str], None]) -> None:
    """Render detailed body metrics with a richer daily log form."""
    today = memory.get_today_body_metrics()
    recent = memory.get_body_metrics(days=7)
    readiness_copy = "No check-in yet today."
    if today:
        readiness_signals = []
        if today.get("sleep_hours"):
            readiness_signals.append(f"sleep {today['sleep_hours']}h")
        if today.get("energy"):
            readiness_signals.append(f"energy {today['energy']}/5")
        if today.get("stress"):
            readiness_signals.append(f"stress {today['stress']}/5")
        readiness_copy = " · ".join(readiness_signals) or readiness_copy
    _focus_hero(
        "Body and recovery",
        "This is the most useful place to teach June how your recovery actually feels, not just what happened on paper.",
    )
    _mini_stats(
        [
            ("Today", "logged" if today else "open"),
            ("7d entries", str(len(recent))),
        ]
    )
    st.markdown(f'<div class="june-subtitle">{html.escape(chapter_subtitle("body"))}</div>', unsafe_allow_html=True)
    st.caption(readiness_copy)

    if today:
        details = []
        if today.get("weight_kg"):
            details.append(f"Weight {today['weight_kg']} kg")
        if today.get("sleep_hours"):
            details.append(f"Sleep {today['sleep_hours']} h")
        if today.get("sleep_quality"):
            details.append(f"Sleep quality {today['sleep_quality']}/5")
        if today.get("energy"):
            details.append(f"Energy {today['energy']}/5")
        if today.get("stress"):
            details.append(f"Stress {today['stress']}/5")
        if today.get("soreness"):
            details.append(f"Soreness {today['soreness']}/5")
        if today.get("resting_hr"):
            details.append(f"Resting HR {today['resting_hr']}")
        if today.get("steps"):
            details.append(f"Steps {today['steps']}")
        st.markdown("**Today**")
        st.caption(" | ".join(details) if details else "No body metrics logged today.")
        if today.get("notes"):
            st.write(today["notes"])
    else:
        st.caption("No body metrics logged today.")

    if recent:
        st.markdown("**7-day trend**")
        render_body_trend_card(memory, days=7)
        st.markdown("**Recent check-ins**")
        st.markdown(render_memory_focus(memory, "body"), unsafe_allow_html=True)

    with st.expander("Log body check-in", expanded=not bool(today)):
        with st.form("body_focus_form", clear_on_submit=False):
            top_left, top_right = st.columns(2, gap="small")
            with top_left:
                weight_kg = st.number_input("Weight kg", min_value=0.0, max_value=300.0, step=0.1, value=float(today.get("weight_kg", 0.0)) if today else 0.0)
                sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, step=0.5, value=float(today.get("sleep_hours", 0.0)) if today else 0.0)
                resting_hr = st.number_input("Resting HR", min_value=0, max_value=240, step=1, value=int(today.get("resting_hr", 0)) if today else 0)
                steps = st.number_input("Steps", min_value=0, max_value=100000, step=500, value=int(today.get("steps", 0)) if today else 0)
            with top_right:
                sleep_quality = st.select_slider("Sleep quality", options=[0, 1, 2, 3, 4, 5], value=int(today.get("sleep_quality", 3)) if today else 3)
                energy = st.select_slider("Energy", options=[0, 1, 2, 3, 4, 5], value=int(today.get("energy", 3)) if today else 3)
                stress = st.select_slider("Stress", options=[0, 1, 2, 3, 4, 5], value=int(today.get("stress", 0)) if today else 0)
                soreness = st.select_slider("Soreness", options=[0, 1, 2, 3, 4, 5], value=int(today.get("soreness", 0)) if today else 0)
            notes = st.text_area("Notes", value=today.get("notes", "") if today else "", placeholder="Recovery notes, pain points, appetite, mood-body link, cycle, illness, etc.")
            if st.form_submit_button("Save body check-in", use_container_width=True):
                memory.log_body_metrics(
                    weight_kg=weight_kg,
                    sleep_hours=sleep_hours,
                    sleep_quality=sleep_quality,
                    energy=energy,
                    stress=stress,
                    soreness=soreness,
                    resting_hr=resting_hr,
                    steps=steps,
                    notes=notes,
                )
                on_activity("body | detailed check-in saved")
                st.rerun()


def render_detail_focus(
    memory: Memory,
    chapter_key: str,
    *,
    water_goal: int,
    on_activity: Callable[[str], None],
) -> None:
    """Render the selected right-panel surface inline on the same page."""
    if chapter_key in {"calendar", "trips", "birthdays"}:
        render_calendar_focus(memory, chapter_key, on_activity=on_activity)
        return
    if chapter_key == "plans":
        render_plan_focus(memory, on_activity=on_activity)
        return
    if chapter_key == "habits":
        render_habits_focus(memory, on_activity=on_activity)
        return
    if chapter_key == "water":
        render_water_focus(memory, water_goal=water_goal, on_activity=on_activity)
        return
    if chapter_key == "body":
        render_body_focus(memory, on_activity=on_activity)
        return
    st.markdown(render_memory_focus(memory, chapter_key), unsafe_allow_html=True)
