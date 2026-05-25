"""Daily orchestration — default schedules, prompts, and task carry-forward.

Part of the Personal Assistant Framework (see ADR 0013). Creates default
schedules for morning briefing, evening review, and weekly review on
first use.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MORNING_BRIEFING_PROMPT = (
    "Give me a morning briefing covering:\n"
    "1. Today's calendar items\n"
    "2. Incomplete tasks from yesterday\n"
    "3. Chores due today or overdue\n"
    "4. Habit streak status\n"
    "5. One proactive suggestion for today\n"
    "Keep it concise and actionable."
)

EVENING_REVIEW_PROMPT = (
    "Help me do my evening review:\n"
    "1. What did I accomplish today?\n"
    "2. What carries forward to tomorrow?\n"
    "3. Log my current mood\n"
    "4. Plan tomorrow's top 3 priorities\n"
    "Be brief and encouraging."
)

WEEKLY_REVIEW_PROMPT = (
    "Generate my weekly review covering:\n"
    "1. Goals progress this week\n"
    "2. Workout summary (sessions, metrics)\n"
    "3. Habit consistency (streaks, misses)\n"
    "4. Upcoming priorities for next week\n"
    "Keep it to one paragraph per section."
)


# ---------------------------------------------------------------------------
# Default schedule creation
# ---------------------------------------------------------------------------

def create_default_schedules(user_id: str) -> list[dict[str, Any]]:
    """Create default daily orchestration schedules for a user.

    Called once during first setup. Safe to call multiple times (idempotent
    by checking if schedules already exist).

    Returns the list of created schedules.
    """
    from june_brain.config import MEMORY_DIR
    from june_brain.memory.sqlite import _get_connection
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL, Schedule
    from june_brain.scheduler.store import ScheduleStore

    db_path = str(MEMORY_DIR / "june.db")
    conn = _get_connection(db_path)
    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    store = ScheduleStore(conn)

    # Check if defaults already exist
    existing = store.list(user_id)
    existing_names = {s.name for s in existing}

    defaults = [
        Schedule(
            user_id=user_id,
            name="Morning briefing",
            description=MORNING_BRIEFING_PROMPT,
            cron_expression="0 8 * * *",
            scheduled_at=datetime.now(UTC).isoformat(),
            action_type="agent_invoke",
            action_config={"prompt": MORNING_BRIEFING_PROMPT},
        ),
        Schedule(
            user_id=user_id,
            name="Evening review",
            description=EVENING_REVIEW_PROMPT,
            cron_expression="0 21 * * *",
            scheduled_at=datetime.now(UTC).isoformat(),
            action_type="agent_invoke",
            action_config={"prompt": EVENING_REVIEW_PROMPT},
        ),
        Schedule(
            user_id=user_id,
            name="Weekly review",
            description=WEEKLY_REVIEW_PROMPT,
            cron_expression="0 10 * * 0",
            scheduled_at=datetime.now(UTC).isoformat(),
            action_type="agent_invoke",
            action_config={"prompt": WEEKLY_REVIEW_PROMPT},
        ),
    ]

    created: list[dict[str, Any]] = []
    for sched in defaults:
        if sched.name in existing_names:
            continue
        store.create(sched)
        created.append({
            "id": sched.id,
            "name": sched.name,
            "cron": sched.cron_expression,
        })
        logger.info("Default schedule created: %s (%s)", sched.name, sched.cron_expression)

    return created


# ---------------------------------------------------------------------------
# Task carry-forward
# ---------------------------------------------------------------------------

def get_carried_tasks(user_id: str) -> list[dict[str, Any]]:
    """Return incomplete tasks from the previous day for the morning briefing.

    Tasks with status PLANNING, RUNNING, PAUSED, or AWAITING_USER that were
    created or updated before today are considered 'carried forward'.
    """

    from june_brain.tasks.store import TasksStore

    store = TasksStore(user_id)
    active = store.active()
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    carried: list[dict[str, Any]] = []
    for task in active:
        created = datetime.fromisoformat(task.created_at)
        if created < today_start:
            carried.append({
                "id": task.id,
                "goal": task.goal,
                "status": task.status.value,
                "created_at": task.created_at,
            })
    return carried
