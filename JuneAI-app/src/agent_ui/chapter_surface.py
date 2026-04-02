"""Actionable chapter metadata and today-memory summaries for the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from agent.memory import Memory

CHAPTERS = [
    ("calendar", "Calendar"),
    ("gym", "Gym Schedule"),
    ("food", "Food Schedule"),
    ("trips", "Trips"),
    ("plans", "Plans"),
    ("habits", "Habits"),
    ("body", "Body Metrics"),
    ("workouts", "Workout Sessions"),
    ("nutrition", "Nutrition"),
    ("water", "Water"),
    ("dating", "Dating/Love"),
    ("family", "Family"),
    ("birthdays", "Birthdays"),
]

_CHAPTER_SUBTITLES = {
    "calendar": "Appointments, reminders, and events.",
    "gym": "Training routines and weekly splits.",
    "food": "Food structure and nutrition plans.",
    "trips": "Travel plans and movement events.",
    "plans": "Goals, open loops, and next steps.",
    "habits": "Daily routines and consistency tracking.",
    "body": "Weight, sleep, energy, and body metrics.",
    "workouts": "Completed sessions and training detail.",
    "nutrition": "Meals, calories, and protein logs.",
    "water": "Daily hydration tracking.",
    "dating": "Relationship memory for love and dating.",
    "family": "Family context and relationship notes.",
    "birthdays": "Birthday reminders and personal dates.",
}


@dataclass(frozen=True)
class ChapterStatus:
    """Compact metadata for a chapter card."""

    key: str
    title: str
    subtitle: str
    item_count: int
    freshness: str
    attention: str
    preview: str
    last_updated: str
    last_updated_iso: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TodayMemoryCard:
    """Compact metadata for a today-summary card."""

    key: str
    title: str
    value: str
    preview: str
    freshness: str
    attention: str
    last_updated: str
    last_updated_iso: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def chapter_subtitle(chapter_key: str) -> str:
    return _CHAPTER_SUBTITLES.get(chapter_key, "")


def _cal_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(field, "")).lower()
        for field in ("title", "details", "source", "status")
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        parsed_date = date.fromisoformat(text[:10])
    except ValueError:
        return None
    return datetime.combine(parsed_date, datetime.min.time())


def _format_last_updated(timestamp: datetime | None) -> tuple[str, str]:
    if timestamp is None:
        return "never", ""

    now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
    delta_days = (now.date() - timestamp.date()).days
    iso_value = timestamp.isoformat(timespec="minutes")
    if delta_days == 0:
        return f"today {timestamp.strftime('%H:%M')}", iso_value
    if delta_days == 1:
        return f"yesterday {timestamp.strftime('%H:%M')}", iso_value
    if delta_days > 1 and delta_days <= 7:
        return f"{delta_days}d ago", iso_value
    if delta_days < 0:
        return f"in {abs(delta_days)}d", iso_value
    return timestamp.strftime("%Y-%m-%d"), iso_value


def _record_timestamp(record: dict[str, Any]) -> datetime | None:
    for field in ("updated_at", "timestamp", "created_at"):
        parsed = _parse_datetime(record.get(field))
        if parsed is not None:
            return parsed

    completions = record.get("completions")
    if isinstance(completions, list) and completions:
        parsed_candidates: list[datetime] = [
            candidate
            for candidate in (_parse_datetime(value) for value in completions)
            if candidate is not None
        ]
        if parsed_candidates:
            return max(parsed_candidates)

    if record.get("date"):
        return _parse_datetime(record.get("date"))
    return None


def _freshness_label(timestamp: datetime | None, item_count: int) -> str:
    if item_count == 0:
        return "empty"
    if timestamp is None:
        return "unknown"

    delta_days = (datetime.now(timestamp.tzinfo).date() - timestamp.date()).days if timestamp.tzinfo else (datetime.now().date() - timestamp.date()).days
    if delta_days <= 0:
        return "today"
    if delta_days <= 3:
        return "recent"
    if delta_days <= 14:
        return "stale"
    return "cold"


def _attention_label(chapter_key: str, freshness: str, item_count: int, records: list[dict[str, Any]]) -> str:
    if item_count == 0:
        return "needs_attention"

    if chapter_key == "water":
        count = int(records[0].get("count", 0)) if records else 0
        return "steady" if count >= 8 else "watch"

    if chapter_key == "habits":
        done_today = sum(1 for record in records if record.get("done_today"))
        return "steady" if done_today == item_count else "watch"

    if chapter_key == "plans":
        open_loops = sum(1 for record in records if record.get("kind") == "open_loop")
        return "watch" if open_loops else "steady"

    if chapter_key == "body":
        latest = records[-1] if records else {}
        if latest.get("sleep_hours", 0) and float(latest.get("sleep_hours", 0)) < 6:
            return "watch"
        if latest.get("energy", 0) and int(latest.get("energy", 0)) <= 2:
            return "watch"
        if latest.get("stress", 0) and int(latest.get("stress", 0)) >= 4:
            return "watch"
        return "steady"

    if freshness in {"today", "recent"}:
        return "steady"
    if freshness == "unknown":
        return "watch"
    return "review"


def _chapter_records(memory: Memory, chapter_key: str) -> list[dict[str, Any]]:
    if chapter_key == "calendar":
        return memory.get_calendar_items(limit=12)
    if chapter_key == "gym":
        return memory.get_gym_plans(limit=12)
    if chapter_key == "food":
        return memory.get_food_programs(limit=12)
    if chapter_key == "trips":
        return [
            item
            for item in memory.get_calendar_items(limit=30)
            if any(kw in _cal_text(item) for kw in ("trip", "travel", "flight"))
        ]
    if chapter_key == "plans":
        return [
            {"kind": "goal", **item}
            for item in memory.get_goals(status="", limit=20)
        ] + [
            {"kind": "open_loop", **item}
            for item in memory.get_open_loops(status="", limit=20)
        ]
    if chapter_key == "habits":
        return memory.get_habits()
    if chapter_key == "body":
        return memory.get_body_metrics(days=14)
    if chapter_key == "workouts":
        return memory.get_workout_sessions(limit=12)
    if chapter_key == "nutrition":
        return memory.get_nutrition_recent(limit=12)
    if chapter_key == "water":
        count = memory.get_water_today()
        return [{"date": date.today().isoformat(), "count": count}] if count else []
    if chapter_key == "dating":
        return [
            item
            for item in memory.get_relationship_profiles()
            if any(
                t in item.get("relationship", "").lower()
                for t in ("dating", "love", "partner", "girlfriend", "boyfriend", "romantic", "spouse")
            )
        ]
    if chapter_key == "family":
        return [
            item
            for item in memory.get_relationship_profiles()
            if any(
                t in item.get("relationship", "").lower()
                for t in ("family", "mother", "father", "mom", "dad", "brother", "sister", "parent", "child", "cousin", "uncle", "aunt")
            )
        ]
    if chapter_key == "birthdays":
        return [
            item
            for item in memory.get_calendar_items(limit=30)
            if "birthday" in _cal_text(item)
        ]
    return []


def _chapter_preview(chapter_key: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return "No entries yet."

    if chapter_key == "calendar":
        item = records[0]
        parts = [item.get("title", "Calendar event"), item.get("date", "")]
        if item.get("time"):
            parts.append(str(item["time"]))
        if item.get("details"):
            parts.append(str(item["details"]))
        return " · ".join(part for part in parts if part)

    if chapter_key == "gym":
        item = records[0]
        parts = [item.get("name", "Gym plan"), item.get("schedule", "")]
        if item.get("goal"):
            parts.append(f"Goal: {item['goal']}")
        return " · ".join(part for part in parts if part)

    if chapter_key == "food":
        item = records[0]
        parts = [item.get("name", "Food program"), item.get("daily_structure", "")]
        if item.get("goal"):
            parts.append(f"Goal: {item['goal']}")
        return " · ".join(part for part in parts if part)

    if chapter_key == "trips":
        item = records[0]
        parts = [item.get("title", "Trip"), item.get("date", "")]
        if item.get("details"):
            parts.append(str(item["details"]))
        return " · ".join(part for part in parts if part)

    if chapter_key == "plans":
        goal_count = sum(1 for item in records if item.get("kind") == "goal")
        loop_count = sum(1 for item in records if item.get("kind") == "open_loop")
        parts = [f"{goal_count} goal{'s' if goal_count != 1 else ''}"]
        parts.append(f"{loop_count} open loop{'s' if loop_count != 1 else ''}")
        if records:
            candidate = next(
                (
                    item
                    for item in records
                    if item.get("next_step") or item.get("due_date")
                ),
                records[0],
            )
            if candidate.get("next_step"):
                parts.append(f"Next: {candidate['next_step']}")
            elif candidate.get("due_date"):
                parts.append(f"Due: {candidate['due_date']}")
        return " · ".join(parts)

    if chapter_key == "habits":
        done_today = sum(1 for item in records if item.get("done_today"))
        preview = f"{done_today}/{len(records)} done today"
        done_habit = next((item for item in records if item.get("done_today")), None)
        if done_habit and done_habit.get("streak"):
            preview += f" · {done_habit['name']} streak {done_habit['streak']}d"
        elif records:
            preview += f" · Focus: {records[0].get('name', 'habit')}"
        return preview

    if chapter_key == "body":
        item = records[-1]
        parts = []
        if item.get("sleep_hours"):
            parts.append(f"sleep {item['sleep_hours']}h")
        if item.get("energy"):
            parts.append(f"energy {item['energy']}/5")
        if item.get("stress"):
            parts.append(f"stress {item['stress']}/5")
        if item.get("weight_kg"):
            parts.append(f"weight {item['weight_kg']}kg")
        if item.get("notes"):
            parts.append(str(item["notes"]))
        return " · ".join(parts) or "Body metrics"

    if chapter_key == "workouts":
        item = records[-1]
        parts = [item.get("plan_name", "Workout")]
        if item.get("duration_min"):
            parts.append(f"{item['duration_min']} min")
        if item.get("energy_rating"):
            parts.append(f"energy {item['energy_rating']}/5")
        if item.get("notes"):
            parts.append(str(item["notes"]))
        return " · ".join(parts)

    if chapter_key == "nutrition":
        item = records[-1]
        parts = [item.get("meal", "meal").title(), item.get("description", "")]
        if item.get("calories_est"):
            parts.append(f"~{item['calories_est']} kcal")
        if item.get("protein_est"):
            parts.append(f"~{item['protein_est']}g protein")
        return " · ".join(part for part in parts if part)

    if chapter_key == "water":
        count = int(records[0].get("count", 0))
        return f"{count} glass{'es' if count != 1 else ''} today"

    if chapter_key in {"dating", "family"}:
        item = records[0]
        parts = [item.get("person", "Relationship"), item.get("relationship", "")]
        if item.get("summary"):
            parts.append(str(item["summary"]))
        return " · ".join(part for part in parts if part)

    if chapter_key == "birthdays":
        item = records[0]
        parts = [item.get("title", "Birthday"), item.get("date", "")]
        if item.get("details"):
            parts.append(str(item["details"]))
        return " · ".join(part for part in parts if part)

    item = records[0]
    return " · ".join(str(value) for value in item.values() if value)


def chapter_status(memory: Memory, chapter_key: str) -> ChapterStatus:
    records = _chapter_records(memory, chapter_key)
    last_updated = max(
        (
            timestamp
            for timestamp in (_record_timestamp(record) for record in records)
            if timestamp is not None
        ),
        default=None,
    )
    if chapter_key == "water" and records:
        last_updated = _parse_datetime(date.today().isoformat())
    last_updated_label, last_updated_iso = _format_last_updated(last_updated)
    item_count = len(records)
    freshness = _freshness_label(last_updated, item_count)
    attention = _attention_label(chapter_key, freshness, item_count, records)
    title = dict(CHAPTERS).get(chapter_key, chapter_key.replace("_", " ").title())
    return ChapterStatus(
        key=chapter_key,
        title=title,
        subtitle=chapter_subtitle(chapter_key),
        item_count=item_count,
        freshness=freshness,
        attention=attention,
        preview=_chapter_preview(chapter_key, records),
        last_updated=last_updated_label,
        last_updated_iso=last_updated_iso,
    )


def chapter_statuses(memory: Memory) -> list[ChapterStatus]:
    return [chapter_status(memory, key) for key, _label in CHAPTERS]


def chapter_status_cards(memory: Memory) -> list[dict[str, Any]]:
    return [status.as_dict() for status in chapter_statuses(memory)]


def _today_body_card(summary: dict[str, Any]) -> TodayMemoryCard:
    body = summary.get("body_metrics") or {}
    if body:
        value_parts = []
        if body.get("sleep_hours"):
            value_parts.append(f"sleep {body['sleep_hours']}h")
        if body.get("energy"):
            value_parts.append(f"energy {body['energy']}/5")
        if body.get("stress"):
            value_parts.append(f"stress {body['stress']}/5")
        if not value_parts and body.get("notes"):
            value_parts.append("Body log captured")
        preview = body.get("notes") or "Body metrics logged today."
        last_updated, last_updated_iso = _format_last_updated(_record_timestamp(body))
        attention = "watch" if (
            (body.get("sleep_hours") and float(body["sleep_hours"]) < 6)
            or (body.get("energy") and int(body["energy"]) <= 2)
            or (body.get("stress") and int(body["stress"]) >= 4)
        ) else "steady"
        return TodayMemoryCard(
            key="body",
            title="Body",
            value=" · ".join(value_parts) or "Body log captured",
            preview=preview,
            freshness="today",
            attention=attention,
            last_updated=last_updated,
            last_updated_iso=last_updated_iso,
        )

    return TodayMemoryCard(
        key="body",
        title="Body",
        value="No body log yet",
        preview="Capture sleep, recovery, and energy to help June infer patterns.",
        freshness="empty",
        attention="needs_attention",
        last_updated="never",
        last_updated_iso="",
    )


def _today_habits_card(summary: dict[str, Any]) -> TodayMemoryCard:
    total = int(summary.get("habits_total", 0))
    done = int(summary.get("habits_done", 0))
    done_names = summary.get("habits_done_names", []) or []
    pending_names = summary.get("habits_pending_names", []) or []
    value = f"{done}/{total} done" if total else "No habits yet"
    if pending_names:
        preview = "Pending: " + ", ".join(str(name) for name in pending_names[:3])
    elif done_names:
        preview = "Complete: " + ", ".join(str(name) for name in done_names[:3])
    else:
        preview = "Create a habit to start tracking consistency."
    attention = "steady" if total and done == total else ("watch" if total else "needs_attention")
    return TodayMemoryCard(
        key="habits",
        title="Habits",
        value=value,
        preview=preview,
        freshness="today" if total else "empty",
        attention=attention,
        last_updated=summary.get("date", date.today().isoformat()),
        last_updated_iso=summary.get("date", date.today().isoformat()),
    )


def _today_workout_card(summary: dict[str, Any]) -> TodayMemoryCard:
    workout = summary.get("workout") or {}
    if workout:
        value = workout.get("plan_name", "Workout")
        preview_parts = [workout.get("exercises", "")] if workout.get("exercises") else []
        if workout.get("duration_min"):
            preview_parts.append(f"{workout['duration_min']} min")
        if workout.get("energy_rating"):
            preview_parts.append(f"energy {workout['energy_rating']}/5")
        if workout.get("notes"):
            preview_parts.append(str(workout["notes"]))
        last_updated, last_updated_iso = _format_last_updated(_record_timestamp(workout))
        return TodayMemoryCard(
            key="workout",
            title="Workout",
            value=value,
            preview=" · ".join(preview_parts) or "Workout logged today.",
            freshness="today",
            attention="steady",
            last_updated=last_updated,
            last_updated_iso=last_updated_iso,
        )

    return TodayMemoryCard(
        key="workout",
        title="Workout",
        value="No workout logged",
        preview="June can prompt for a session once you start moving.",
        freshness="empty",
        attention="watch",
        last_updated="never",
        last_updated_iso="",
    )


def _today_nutrition_card(summary: dict[str, Any]) -> TodayMemoryCard:
    meals_logged = int(summary.get("meals_logged", 0))
    calories = int(summary.get("calories_est", 0))
    protein = int(summary.get("protein_est", 0))
    if meals_logged:
        preview = f"~{calories} kcal · ~{protein}g protein"
        attention = "steady"
    else:
        preview = "No meals logged today."
        attention = "watch"
    return TodayMemoryCard(
        key="nutrition",
        title="Nutrition",
        value=f"{meals_logged} meal{'s' if meals_logged != 1 else ''}" if meals_logged else "No meals yet",
        preview=preview,
        freshness="today" if meals_logged else "empty",
        attention=attention,
        last_updated=summary.get("date", date.today().isoformat()),
        last_updated_iso=summary.get("date", date.today().isoformat()),
    )


def _today_water_card(summary: dict[str, Any]) -> TodayMemoryCard:
    count = int(summary.get("water_glasses", 0))
    preview = f"{max(0, 8 - count)} to goal" if count < 8 else "Hydration goal met."
    attention = "steady" if count >= 8 else "watch"
    return TodayMemoryCard(
        key="water",
        title="Water",
        value=f"{count} glass{'es' if count != 1 else ''}",
        preview=preview,
        freshness="today" if count else "empty",
        attention=attention if count else "watch",
        last_updated=summary.get("date", date.today().isoformat()),
        last_updated_iso=summary.get("date", date.today().isoformat()),
    )


def _today_reminders_card(memory: Memory) -> TodayMemoryCard:
    notifications = memory.get_upcoming_notifications(limit=3)
    if notifications:
        next_item = notifications[0]
        preview_parts = [next_item.get("title", "Reminder")]
        if next_item.get("when"):
            preview_parts.append(str(next_item["when"]))
        if next_item.get("details"):
            preview_parts.append(str(next_item["details"]))
        last_updated = next_item.get("when") or date.today().isoformat()
        last_updated_dt = _parse_datetime(last_updated)
        formatted, iso_value = _format_last_updated(last_updated_dt)
        return TodayMemoryCard(
            key="reminders",
            title="Reminders",
            value=f"{len(notifications)} upcoming",
            preview=" · ".join(part for part in preview_parts if part),
            freshness="today",
            attention="watch" if next_item.get("days_until", 0) <= 3 else "steady",
            last_updated=formatted,
            last_updated_iso=iso_value,
        )

    return TodayMemoryCard(
        key="reminders",
        title="Reminders",
        value="No reminders due soon",
        preview="June can still surface upcoming events once they are added.",
        freshness="empty",
        attention="steady",
        last_updated="never",
        last_updated_iso="",
    )


def today_memory_cards(memory: Memory) -> list[dict[str, Any]]:
    summary = memory.get_today_summary()
    cards = [
        _today_body_card(summary),
        _today_habits_card(summary),
        _today_workout_card(summary),
        _today_nutrition_card(summary),
        _today_water_card(summary),
        _today_reminders_card(memory),
    ]
    return [card.as_dict() for card in cards]


def today_memory_summary(memory: Memory) -> dict[str, Any]:
    summary = memory.get_today_summary()
    cards = today_memory_cards(memory)
    headline_parts = [
        card["value"]
        for card in cards
        if card["key"] in {"body", "habits", "water"} and card["value"] not in {"No body log yet", "No habits yet", "No meals yet", "No workout logged", "No reminders due soon"}
    ]
    headline = " · ".join(headline_parts) if headline_parts else "No tracked activity yet."
    return {
        **summary,
        "headline": headline,
        "cards": cards,
        "by_key": {card["key"]: card for card in cards},
        "upcoming_notifications": memory.get_upcoming_notifications(limit=3),
    }
