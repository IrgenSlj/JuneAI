"""Chapter metadata and chapter rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def _cal_text(item: dict) -> str:
    return " ".join(
        str(item.get(field, "")).lower()
        for field in ("title", "details", "source", "status")
    )


def chapter_items(memory: "Memory", chapter_key: str) -> list[tuple[str, str]]:
    if chapter_key == "calendar":
        return [
            (
                item["title"],
                f"{item['date']}{' ' + item['time'] if item.get('time') else ''}"
                f"{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=12)
        ]
    if chapter_key == "gym":
        return [
            (
                item["name"],
                f"{item['schedule']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}",
            )
            for item in memory.get_gym_plans(limit=12)
        ]
    if chapter_key == "food":
        return [
            (
                item["name"],
                f"{item['daily_structure']}{' | Goal: ' + item['goal'] if item.get('goal') else ''}",
            )
            for item in memory.get_food_programs(limit=12)
        ]
    if chapter_key == "trips":
        return [
            (
                item["title"],
                f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=30)
            if any(kw in _cal_text(item) for kw in ("trip", "travel", "flight"))
        ]
    if chapter_key == "plans":
        return [
            (
                item["title"],
                f"Goal | {item['status']}{' | Next: ' + item['next_step'] if item.get('next_step') else ''}",
            )
            for item in memory.get_goals(status="", limit=20)
        ] + [
            (
                item["topic"],
                f"Open loop | {item['status']}{' | Due: ' + item['due_date'] if item.get('due_date') else ''}",
            )
            for item in memory.get_open_loops(status="", limit=20)
        ]
    if chapter_key == "habits":
        return [
            (
                item["name"],
                f"{item.get('category', '')} | {item.get('target_days', '')}"
                f"{' | streak ' + str(item.get('streak', 0)) + 'd' if item.get('streak') else ''}",
            )
            for item in memory.get_habits()
        ]
    if chapter_key == "body":
        return [
            (
                item["date"],
                " | ".join(
                    part
                    for part in [
                        f"weight {item['weight_kg']}kg" if item.get("weight_kg") else "",
                        f"sleep {item['sleep_hours']}h" if item.get("sleep_hours") else "",
                        f"energy {item['energy']}/5" if item.get("energy") else "",
                    ]
                    if part
                )
                or "Body metrics",
            )
            for item in memory.get_body_metrics(days=14)
        ]
    if chapter_key == "workouts":
        return [
            (
                item["plan_name"],
                f"{item['date']}{' | ' + str(item['duration_min']) + ' min' if item.get('duration_min') else ''}"
                f"{' | ' + item['notes'] if item.get('notes') else ''}",
            )
            for item in memory.get_workout_sessions(limit=12)
        ]
    if chapter_key == "nutrition":
        return [
            (
                item["meal"].title(),
                f"{item['date']} | {item['description']}"
                f"{' | ~' + str(item['calories_est']) + ' kcal' if item.get('calories_est') else ''}"
                f"{' | ~' + str(item['protein_est']) + 'g protein' if item.get('protein_est') else ''}",
            )
            for item in memory.get_nutrition_recent(limit=12)
        ]
    if chapter_key == "water":
        count = memory.get_water_today()
        return [("Today", f"{count} glass{'es' if count != 1 else ''} logged")] if count else []
    if chapter_key == "dating":
        return [
            (item["person"], f"{item['relationship']} | {item['summary']}")
            for item in memory.get_relationship_profiles()
            if any(
                t in item.get("relationship", "").lower()
                for t in ("dating", "love", "partner", "girlfriend", "boyfriend", "romantic", "spouse")
            )
        ]
    if chapter_key == "family":
        return [
            (item["person"], f"{item['relationship']} | {item['summary']}")
            for item in memory.get_relationship_profiles()
            if any(
                t in item.get("relationship", "").lower()
                for t in ("family", "mother", "father", "mom", "dad", "brother", "sister", "parent", "child", "cousin", "uncle", "aunt")
            )
        ]
    if chapter_key == "birthdays":
        return [
            (
                item["title"],
                f"{item['date']}{' | ' + item['details'] if item.get('details') else ''}",
            )
            for item in memory.get_calendar_items(limit=30)
            if "birthday" in _cal_text(item)
        ]
    return []


def chapter_subtitle(chapter_key: str) -> str:
    return {
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
    }.get(chapter_key, "")
