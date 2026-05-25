"""Derived context summaries for June's backend intelligence layer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from .memory import Memory

JsonDict = dict[str, Any]

_COMPLETED_STATUSES = {"completed", "done", "canceled", "cancelled", "archived"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    return str(value).strip()


def _parse_date(value: Any) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _active_metric_value(entry: Mapping[str, Any], key: str) -> float | None:
    value = _as_float(entry.get(key), 0.0)
    return value if value > 0 else None


def _recent_average(entries: list[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for entry in entries:
        value = _active_metric_value(entry, key)
        if value is not None:
            values.append(value)
    return _average(values)


def _format_metric(value: float | None, suffix: str = "", precision: int = 1) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.{precision}f}{suffix}"


def _score_recovery(
    body_today: Mapping[str, Any] | None,
    recent_body: list[Mapping[str, Any]],
    workout_today: Mapping[str, Any] | None,
    meals_today: list[Mapping[str, Any]],
    water_today: int,
    habits_done: int,
    habits_total: int,
) -> tuple[int, str, list[str], list[str]]:
    score = 50
    signals: list[str] = []
    recommendations: list[str] = []

    sleep_hours = _active_metric_value(body_today, "sleep_hours") if body_today else None
    energy = _active_metric_value(body_today, "energy") if body_today else None
    stress = _active_metric_value(body_today, "stress") if body_today else None
    soreness = _active_metric_value(body_today, "soreness") if body_today else None
    resting_hr = _active_metric_value(body_today, "resting_hr") if body_today else None

    if sleep_hours is None:
        sleep_hours = _recent_average(recent_body, "sleep_hours")
    if energy is None:
        energy = _recent_average(recent_body, "energy")
    if stress is None:
        stress = _recent_average(recent_body, "stress")
    if soreness is None:
        soreness = _recent_average(recent_body, "soreness")
    if resting_hr is None:
        resting_hr = _recent_average(recent_body, "resting_hr")

    if sleep_hours is not None:
        if sleep_hours >= 7:
            score += 12
            signals.append(f"sleep is solid at {_format_metric(sleep_hours, 'h')}")
        elif sleep_hours < 6:
            score -= 10
            signals.append(f"sleep is short at {_format_metric(sleep_hours, 'h')}")
            recommendations.append("Protect sleep tonight and keep the next load lighter.")

    if energy is not None:
        if energy >= 4:
            score += 10
            signals.append(f"energy is strong at {_format_metric(energy, '/5')}")
        elif energy <= 2:
            score -= 8
            signals.append(f"energy is low at {_format_metric(energy, '/5')}")
            recommendations.append("Keep today simple and avoid stacking extra tasks.")

    if stress is not None:
        if stress <= 2:
            score += 6
            signals.append(f"stress is manageable at {_format_metric(stress, '/5')}")
        elif stress >= 4:
            score -= 10
            signals.append(f"stress is elevated at {_format_metric(stress, '/5')}")
            recommendations.append("Reduce decision load and protect recovery time.")

    if soreness is not None:
        if soreness <= 2:
            score += 5
            signals.append(f"soreness is low at {_format_metric(soreness, '/5')}")
        elif soreness >= 4:
            score -= 8
            signals.append(f"soreness is high at {_format_metric(soreness, '/5')}")
            recommendations.append("Keep training intensity conservative until soreness settles.")

    if water_today >= 8:
        score += 8
        signals.append(f"hydration is strong at {water_today} glasses")
    elif water_today >= 5:
        score += 4
        signals.append(f"hydration is on track at {water_today} glasses")
    elif water_today > 0:
        score -= 4
        signals.append(f"hydration is light at {water_today} glasses")
        recommendations.append("Catch up on water before adding more strain.")
    else:
        recommendations.append("Log water if you have had any today so June can track hydration accurately.")

    if workout_today is not None:
        energy_rating = _as_int(workout_today.get("energy_rating"), 0)
        duration_min = _as_int(workout_today.get("duration_min"), 0)
        if energy_rating >= 4:
            score += 6
            signals.append(f"workout load looked good ({duration_min} min, energy {energy_rating}/5)")
        elif energy_rating <= 2:
            score -= 5
            signals.append(f"workout load felt heavy ({duration_min} min, energy {energy_rating}/5)")
            recommendations.append("Use the next session for recovery-focused work.")

    if habits_total > 0:
        completion_ratio = habits_done / habits_total
        if completion_ratio >= 0.75:
            score += 6
            signals.append(f"habits are steady at {habits_done}/{habits_total} done")
        elif completion_ratio <= 0.25:
            score -= 5
            signals.append(f"habits are slipping at {habits_done}/{habits_total} done")
            recommendations.append("Finish the smallest remaining habit before taking on more.")

    if meals_today:
        calories = sum(_as_int(item.get("calories_est"), 0) for item in meals_today)
        protein = sum(_as_int(item.get("protein_est"), 0) for item in meals_today)
        score += 3
        signals.append(
            f"nutrition is logged: {len(meals_today)} meal(s), ~{calories} kcal, ~{protein}g protein"
        )
        if protein >= 80:
            score += 4
            signals.append("protein intake looks supportive")
        elif protein > 0 and protein < 50:
            score -= 3
            recommendations.append("Consider a more protein-forward meal if recovery matters today.")
    else:
        recommendations.append("Log meals if you want June to reason more precisely about recovery.")

    score = max(0, min(100, score))
    if score >= 75:
        label = "ready"
    elif score >= 55:
        label = "steady"
    elif score >= 40:
        label = "recovering"
    else:
        label = "low"

    if not signals:
        signals.append("No recovery signals logged yet.")
    if not recommendations:
        recommendations.append("Keep the current routine steady.")

    return score, label, signals, recommendations


def build_recovery_readiness_summary(memory: Memory) -> JsonDict:
    """Build a structured readiness snapshot from recovery signals."""
    body_today = memory.get_today_body_metrics()
    recent_body = memory.get_body_metrics(days=7)
    workout_today = memory.get_today_workout()
    meals_today = memory.get_nutrition_today()
    water_today = memory.get_water_today()
    habits = memory.get_habits()
    habits_done = [habit for habit in habits if habit.get("done_today")]
    habits_pending = [habit for habit in habits if not habit.get("done_today")]

    recent_sleep = _recent_average(recent_body, "sleep_hours")
    recent_energy = _recent_average(recent_body, "energy")
    recent_stress = _recent_average(recent_body, "stress")
    recent_soreness = _recent_average(recent_body, "soreness")
    recent_steps = _recent_average(recent_body, "steps")
    recent_weight = _recent_average(recent_body, "weight_kg")
    recent_resting_hr = _recent_average(recent_body, "resting_hr")

    score, label, signals, recommendations = _score_recovery(
        body_today=body_today,
        recent_body=recent_body,
        workout_today=workout_today,
        meals_today=meals_today,
        water_today=water_today,
        habits_done=len(habits_done),
        habits_total=len(habits),
    )

    if body_today is not None:
        body_source = "today"
        body_summary: JsonDict = {
            "date": body_today.get("date", ""),
            "weight_kg": body_today.get("weight_kg", 0.0),
            "sleep_hours": body_today.get("sleep_hours", 0.0),
            "sleep_quality": body_today.get("sleep_quality", 0),
            "energy": body_today.get("energy", 0),
            "stress": body_today.get("stress", 0),
            "soreness": body_today.get("soreness", 0),
            "resting_hr": body_today.get("resting_hr", 0),
            "steps": body_today.get("steps", 0),
            "notes": body_today.get("notes", ""),
        }
    elif recent_body:
        body_source = "7d average"
        body_summary = {
            "date": "",
            "weight_kg": round(recent_weight or 0.0, 1) if recent_weight is not None else 0.0,
            "sleep_hours": round(recent_sleep or 0.0, 1) if recent_sleep is not None else 0.0,
            "sleep_quality": 0,
            "energy": round(recent_energy or 0.0, 1) if recent_energy is not None else 0.0,
            "stress": round(recent_stress or 0.0, 1) if recent_stress is not None else 0.0,
            "soreness": round(recent_soreness or 0.0, 1) if recent_soreness is not None else 0.0,
            "resting_hr": round(recent_resting_hr or 0.0, 1) if recent_resting_hr is not None else 0.0,
            "steps": round(recent_steps or 0.0, 1) if recent_steps is not None else 0.0,
            "notes": "",
        }
    else:
        body_source = "none"
        body_summary = {}

    workout_summary = None
    if workout_today is not None:
        workout_summary = {
            "date": workout_today.get("date", ""),
            "plan_name": workout_today.get("plan_name", ""),
            "duration_min": workout_today.get("duration_min", 0),
            "energy_rating": workout_today.get("energy_rating", 0),
            "exercises": workout_today.get("exercises", ""),
            "notes": workout_today.get("notes", ""),
        }

    nutrition_summary = {
        "meals_logged": len(meals_today),
        "calories_est": sum(_as_int(item.get("calories_est"), 0) for item in meals_today),
        "protein_est": sum(_as_int(item.get("protein_est"), 0) for item in meals_today),
    }

    return {
        "date": date.today().isoformat(),
        "readiness_score": score,
        "readiness_label": label,
        "body_source": body_source,
        "body": body_summary,
        "workout": workout_summary,
        "nutrition": nutrition_summary,
        "water_glasses": water_today,
        "habits_total": len(habits),
        "habits_done": len(habits_done),
        "habits_pending": [habit.get("name", "") for habit in habits_pending],
        "recent_trends": {
            "sleep_hours": round(recent_sleep, 1) if recent_sleep is not None else None,
            "energy": round(recent_energy, 1) if recent_energy is not None else None,
            "stress": round(recent_stress, 1) if recent_stress is not None else None,
            "soreness": round(recent_soreness, 1) if recent_soreness is not None else None,
            "weight_kg": round(recent_weight, 1) if recent_weight is not None else None,
            "resting_hr": round(recent_resting_hr, 1) if recent_resting_hr is not None else None,
            "steps": round(recent_steps, 1) if recent_steps is not None else None,
        },
        "signals": signals,
        "recommendations": recommendations,
    }


def format_recovery_readiness_summary(summary: Mapping[str, Any]) -> str:
    """Render a recovery snapshot as a concise, model-friendly string."""
    lines = [
        f"Recovery readiness for {summary.get('date', '')}:",
        f"- Status: {summary.get('readiness_label', 'unknown')} ({summary.get('readiness_score', 0)}/100)",
    ]

    body = summary.get("body", {})
    if body:
        body_parts = []
        if body.get("sleep_hours"):
            body_parts.append(f"sleep {body['sleep_hours']}h")
        if body.get("sleep_quality"):
            body_parts.append(f"sleep quality {body['sleep_quality']}/5")
        if body.get("energy"):
            body_parts.append(f"energy {body['energy']}/5")
        if body.get("stress"):
            body_parts.append(f"stress {body['stress']}/5")
        if body.get("soreness"):
            body_parts.append(f"soreness {body['soreness']}/5")
        if body.get("weight_kg"):
            body_parts.append(f"weight {body['weight_kg']}kg")
        if body.get("resting_hr"):
            body_parts.append(f"resting HR {body['resting_hr']}")
        if body.get("steps"):
            body_parts.append(f"steps {body['steps']}")
        if body.get("notes"):
            body_parts.append(f"notes: {body['notes']}")
        prefix = "today" if summary.get("body_source") == "today" else "7d average"
        lines.append(f"- Body ({prefix}): " + (", ".join(body_parts) or "no body metrics logged"))
    else:
        lines.append("- Body: no body metrics logged")

    workout = summary.get("workout")
    if workout:
        lines.append(
            f"- Workout: {workout.get('plan_name', 'Workout')} "
            f"({workout.get('duration_min', 0)} min, energy {workout.get('energy_rating', 0)}/5)"
        )
    else:
        lines.append("- Workout: not logged today")

    nutrition = summary.get("nutrition", {})
    lines.append(
        "- Nutrition: "
        f"{nutrition.get('meals_logged', 0)} meal(s), "
        f"~{nutrition.get('calories_est', 0)} kcal, "
        f"~{nutrition.get('protein_est', 0)}g protein"
    )
    lines.append(f"- Water: {summary.get('water_glasses', 0)} glasses")
    lines.append(
        f"- Habits: {summary.get('habits_done', 0)}/{summary.get('habits_total', 0)} done"
    )

    trends = summary.get("recent_trends", {})
    trend_parts = []
    if trends.get("sleep_hours") is not None:
        trend_parts.append(f"sleep avg {trends['sleep_hours']}h")
    if trends.get("energy") is not None:
        trend_parts.append(f"energy avg {trends['energy']}/5")
    if trends.get("stress") is not None:
        trend_parts.append(f"stress avg {trends['stress']}/5")
    if trends.get("soreness") is not None:
        trend_parts.append(f"soreness avg {trends['soreness']}/5")
    if trend_parts:
        lines.append("- 7d trend: " + ", ".join(trend_parts))

    signals = summary.get("signals", [])
    if signals:
        lines.append("- Signals: " + "; ".join(str(signal) for signal in signals))
    recommendations = summary.get("recommendations", [])
    if recommendations:
        lines.append("- Focus: " + "; ".join(str(item) for item in recommendations))

    return "\n".join(lines)


def build_active_commitments_summary(memory: Memory) -> JsonDict:
    """Build a structured summary of active commitments across chapters."""
    today = date.today()
    calendar_items = memory.get_calendar_items(limit=50)
    goals = memory.get_goals(status="active", limit=20)
    open_loops = memory.get_open_loops(status="open", limit=20)
    habits = memory.get_habits()

    upcoming_calendar: list[JsonDict] = []
    active_goal_items: list[JsonDict] = []
    open_loop_items: list[JsonDict] = []
    pending_habit_items: list[JsonDict] = []

    for item in calendar_items:
        parsed_date = _parse_date(item.get("date"))
        if parsed_date is None:
            continue
        if _as_text(item.get("status")).lower() in _COMPLETED_STATUSES:
            continue
        days_until = (parsed_date - today).days
        if days_until <= 14:
            upcoming_calendar.append(
                {
                    "title": item.get("title", ""),
                    "date": item.get("date", ""),
                    "time": item.get("time", ""),
                    "details": item.get("details", ""),
                    "days_until": days_until,
                    "status": item.get("status", ""),
                }
            )

    for goal in goals:
        parsed_date = _parse_date(goal.get("target_date"))
        active_goal_items.append(
            {
                "title": goal.get("title", ""),
                "category": goal.get("category", ""),
                "status": goal.get("status", ""),
                "target_date": goal.get("target_date", ""),
                "next_step": goal.get("next_step", ""),
                "days_until": (parsed_date - today).days if parsed_date is not None else None,
            }
        )

    for loop in open_loops:
        parsed_date = _parse_date(loop.get("due_date"))
        if parsed_date is not None and (parsed_date - today).days > 14:
            continue
        open_loop_items.append(
            {
                "topic": loop.get("topic", ""),
                "status": loop.get("status", ""),
                "due_date": loop.get("due_date", ""),
                "next_step": loop.get("next_step", ""),
                "days_until": (parsed_date - today).days if parsed_date is not None else None,
            }
        )

    for habit in habits:
        if habit.get("done_today"):
            continue
        pending_habit_items.append(
            {
                "name": habit.get("name", ""),
                "category": habit.get("category", ""),
                "target_days": habit.get("target_days", ""),
                "streak": habit.get("streak", 0),
            }
        )

    load_score = min(
        100,
        len(upcoming_calendar) * 4
        + len(active_goal_items) * 3
        + len(open_loop_items) * 3
        + len(pending_habit_items) * 2,
    )
    if load_score <= 8:
        load_label = "light"
    elif load_score <= 18:
        load_label = "moderate"
    elif load_score <= 32:
        load_label = "busy"
    else:
        load_label = "loaded"

    next_actions: list[str] = []
    if upcoming_calendar:
        next_actions.extend(
            f"{item['title']} on {item['date']}"
            + (f" at {item['time']}" if item.get("time") else "")
            for item in upcoming_calendar[:3]
        )
    for goal in active_goal_items:
        if goal.get("next_step"):
            goal_text = f"{goal['title']}: {goal['next_step']}"
            if goal.get("target_date"):
                goal_text += f" (target {goal['target_date']})"
            next_actions.append(goal_text)
    for loop in open_loop_items:
        loop_text = f"{loop['topic']}"
        if loop.get("next_step"):
            loop_text += f": {loop['next_step']}"
        if loop.get("due_date"):
            loop_text += f" (due {loop['due_date']})"
        next_actions.append(loop_text)
    for habit in pending_habit_items[:3]:
        next_actions.append(f"{habit['name']} today")

    signals: list[str] = []
    if upcoming_calendar:
        signals.append(f"{len(upcoming_calendar)} calendar item(s) due within 14 days")
    if active_goal_items:
        signals.append(f"{len(active_goal_items)} active goal(s)")
    if open_loop_items:
        signals.append(f"{len(open_loop_items)} open loop(s)")
    if pending_habit_items:
        signals.append(f"{len(pending_habit_items)} habit(s) pending today")
    if not signals:
        signals.append("No active commitments need attention right now.")

    return {
        "date": today.isoformat(),
        "load_score": load_score,
        "load_label": load_label,
        "counts": {
            "calendar_due_soon": len(upcoming_calendar),
            "active_goals": len(active_goal_items),
            "open_loops": len(open_loop_items),
            "pending_habits": len(pending_habit_items),
        },
        "calendar_due_soon": upcoming_calendar[:5],
        "active_goals": active_goal_items[:5],
        "open_loops": open_loop_items[:5],
        "pending_habits": pending_habit_items[:5],
        "next_actions": next_actions[:5],
        "signals": signals,
    }


def format_active_commitments_summary(summary: Mapping[str, Any]) -> str:
    """Render the active commitments snapshot as a concise string."""
    lines = [
        f"Active commitments for {summary.get('date', '')}:",
        f"- Load: {summary.get('load_label', 'unknown')} ({summary.get('load_score', 0)}/100)",
    ]

    counts = summary.get("counts", {})
    lines.append(
        "- Counts: "
        f"{counts.get('calendar_due_soon', 0)} calendar due soon, "
        f"{counts.get('active_goals', 0)} active goals, "
        f"{counts.get('open_loops', 0)} open loops, "
        f"{counts.get('pending_habits', 0)} habits pending"
    )

    calendar_items = summary.get("calendar_due_soon", [])
    if calendar_items:
        lines.append("- Calendar:")
        for item in calendar_items[:3]:
            line = f"  - {item.get('title', '')} on {item.get('date', '')}"
            if item.get("time"):
                line += f" at {item['time']}"
            if item.get("days_until") is not None:
                line += f" ({item['days_until']}d)"
            lines.append(line)

    goals = summary.get("active_goals", [])
    if goals:
        lines.append("- Goals:")
        for goal in goals[:3]:
            line = f"  - {goal.get('title', '')}"
            if goal.get("next_step"):
                line += f" | Next: {goal['next_step']}"
            if goal.get("target_date"):
                line += f" | Target: {goal['target_date']}"
            lines.append(line)

    open_loops = summary.get("open_loops", [])
    if open_loops:
        lines.append("- Open loops:")
        for loop in open_loops[:3]:
            line = f"  - {loop.get('topic', '')}"
            if loop.get("next_step"):
                line += f" | Next: {loop['next_step']}"
            if loop.get("due_date"):
                line += f" | Due: {loop['due_date']}"
            lines.append(line)

    pending_habits = summary.get("pending_habits", [])
    if pending_habits:
        lines.append("- Habits pending today:")
        for habit in pending_habits[:3]:
            line = f"  - {habit.get('name', '')}"
            if habit.get("category"):
                line += f" ({habit['category']})"
            if habit.get("streak"):
                line += f" | Streak {habit['streak']}"
            lines.append(line)

    next_actions = summary.get("next_actions", [])
    if next_actions:
        lines.append("- Next actions:")
        for action in next_actions[:5]:
            lines.append(f"  - {action}")

    signals = summary.get("signals", [])
    if signals:
        lines.append("- Signals: " + "; ".join(str(signal) for signal in signals))

    return "\n".join(lines)
