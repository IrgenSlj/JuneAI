"""June health skill — body metrics, workouts, water, and habits via MCP.

``log_body_metrics`` routes through ``MemoryManager.write`` so the daily
body check also lands as a paraphrased recall hit ("Body check on
YYYY-MM-DD: weight Xkg, slept Yh…"). Water, workouts, and habit
completions still go through ``Memory`` directly because their
recall-paraphrase shape is unclear; revisit when usage shows demand.
"""

from __future__ import annotations

from june_brain.memory import Memory, MemoryManager
from june_brain.skills.server import MCPStdioServer

server = MCPStdioServer(name="june-health", version="0.1.0")


@server.tool(
    name="log_body_metrics",
    description="Log today's body metrics: weight, sleep, energy, stress, soreness, resting HR, steps.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "weight_kg": {"type": "number", "default": 0.0},
            "sleep_hours": {"type": "number", "default": 0.0},
            "sleep_quality": {"type": "integer", "default": 0},
            "energy": {"type": "integer", "default": 0},
            "stress": {"type": "integer", "default": 0},
            "soreness": {"type": "integer", "default": 0},
            "resting_hr": {"type": "integer", "default": 0},
            "steps": {"type": "integer", "default": 0},
            "notes": {"type": "string", "default": ""},
        },
        "required": ["user_id"],
    },
)
def log_body_metrics(
    user_id: str,
    weight_kg: float = 0.0,
    sleep_hours: float = 0.0,
    sleep_quality: int = 0,
    energy: int = 0,
    stress: int = 0,
    soreness: int = 0,
    resting_hr: int = 0,
    steps: int = 0,
    notes: str = "",
) -> str:
    result = MemoryManager(user_id).write(
        {
            "kind": "body_metric",
            "fields": {
                "weight_kg": weight_kg,
                "sleep_hours": sleep_hours,
                "sleep_quality": sleep_quality,
                "energy": energy,
                "stress": stress,
                "soreness": soreness,
                "resting_hr": resting_hr,
                "steps": steps,
                "notes": notes,
            },
        },
        source="skill:health:log_body_metrics",
    )
    if not result.get("written"):
        return "Couldn't log those body metrics."
    return "Logged today's body metrics."


@server.tool(
    name="log_water",
    description="Log glasses of water consumed today. Returns the running total.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "glasses": {"type": "integer", "default": 1},
        },
        "required": ["user_id"],
    },
)
def log_water(user_id: str, glasses: int = 1) -> str:
    total = Memory(user_id).log_water(glasses)
    return f"Water logged — {total} glass(es) today."


@server.tool(
    name="log_workout_session",
    description="Log a completed workout session with exercises and duration in minutes.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "plan_name": {"type": "string"},
            "exercises": {"type": "string", "default": ""},
            "duration_min": {"type": "integer", "default": 0},
            "notes": {"type": "string", "default": ""},
            "energy_rating": {"type": "integer", "default": 0},
        },
        "required": ["user_id", "plan_name"],
    },
)
def log_workout_session(
    user_id: str,
    plan_name: str,
    exercises: str = "",
    duration_min: int = 0,
    notes: str = "",
    energy_rating: int = 0,
) -> str:
    item = Memory(user_id).log_workout_session(
        plan_name=plan_name,
        exercises=exercises,
        duration_min=duration_min,
        notes=notes,
        energy_rating=energy_rating,
    )
    return f"Logged {item['plan_name']} on {item['date']} ({item['duration_min']} min)."


@server.tool(
    name="log_habit_completion",
    description="Mark a habit as completed for today. Auto-creates the habit if missing.",
    input_schema={
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "habit_name": {"type": "string"},
        },
        "required": ["user_id", "habit_name"],
    },
)
def log_habit_completion(user_id: str, habit_name: str) -> str:
    item = Memory(user_id).log_habit_completion(habit_name=habit_name)
    streak = item.get("streak", 0)
    if streak > 1:
        return f"'{item['name']}' done. {streak}-day streak."
    return f"'{item['name']}' checked off."


@server.tool(
    name="get_today_summary",
    description="Full summary of today: habits, workout, body metrics, water, nutrition.",
    input_schema={
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
)
def get_today_summary(user_id: str) -> str:
    s = Memory(user_id).get_today_summary()
    parts = [f"Today ({s['date']}):"]
    if s["habits_total"] > 0:
        parts.append(f"Habits: {s['habits_done']}/{s['habits_total']} done")
    parts.append(f"Water: {s['water_glasses']} glasses")
    if s["workout"]:
        parts.append(f"Workout: {s['workout']['plan_name']} ({s['workout']['duration_min']} min)")
    if s["meals_logged"] > 0:
        parts.append(
            f"Nutrition: {s['meals_logged']} meals, ~{s['calories_est']} kcal, "
            f"~{s['protein_est']}g protein"
        )
    return "\n".join(parts)


def main() -> None:
    server.run()
