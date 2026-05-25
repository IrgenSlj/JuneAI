"""CRUD REST API for the scheduler (``schedules`` table)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from june_brain.config import MEMORY_DIR
from june_brain.memory.sqlite import _get_connection
from june_brain.scheduler.models import Schedule
from june_brain.scheduler.store import ScheduleStore

router = APIRouter(prefix="/schedules", tags=["scheduler"])


def _store() -> ScheduleStore:
    db_dir = MEMORY_DIR
    db_path = str(db_dir / "june.db")
    conn = _get_connection(db_path)
    # Ensure the schedules table exists
    from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL

    conn.executescript(_SCHEDULES_TABLE_SQL)
    conn.commit()
    return ScheduleStore(conn)


def _schedule_to_dict(schedule: Schedule) -> dict[str, Any]:
    return {
        "id": schedule.id,
        "user_id": schedule.user_id,
        "name": schedule.name,
        "description": schedule.description,
        "cron_expression": schedule.cron_expression,
        "interval_seconds": schedule.interval_seconds,
        "scheduled_at": schedule.scheduled_at,
        "last_run_at": schedule.last_run_at,
        "action_type": schedule.action_type,
        "action_config": schedule.action_config,
        "max_runs": schedule.max_runs,
        "run_count": schedule.run_count,
        "enabled": schedule.enabled,
        "created_at": schedule.created_at,
        "updated_at": schedule.updated_at,
    }


@router.get("/{user_id}")
def list_schedules(user_id: str) -> list[dict[str, Any]]:
    """List all schedules for a user."""
    return [_schedule_to_dict(s) for s in _store().list(user_id)]


@router.get("/{user_id}/{schedule_id}")
def get_schedule(user_id: str, schedule_id: str) -> dict[str, Any]:
    """Get a single schedule by ID."""
    sched = _store().get(schedule_id)
    if sched is None or sched.user_id != user_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return _schedule_to_dict(sched)


@router.post("/{user_id}", status_code=201)
def create_schedule(user_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Create a new schedule."""
    sched = Schedule(
        user_id=user_id,
        name=body.get("name", ""),
        description=body.get("description", ""),
        cron_expression=body.get("cron_expression", ""),
        interval_seconds=body.get("interval_seconds", 0),
        scheduled_at=body.get("scheduled_at", datetime.now(timezone.utc).isoformat()),
        action_type=body.get("action_type", "agent_invoke"),
        action_config=body.get("action_config", {}),
        max_runs=body.get("max_runs", 0),
        enabled=body.get("enabled", True),
    )
    created = _store().create(sched)
    return _schedule_to_dict(created)


@router.patch("/{user_id}/{schedule_id}")
def update_schedule(user_id: str, schedule_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Update a schedule (partial)."""
    store = _store()
    sched = store.get(schedule_id)
    if sched is None or sched.user_id != user_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    for field in ("name", "description", "cron_expression", "interval_seconds",
                  "scheduled_at", "action_type", "max_runs", "enabled"):
        if field in body:
            setattr(sched, field, body[field])
    if "action_config" in body:
        sched.action_config = body["action_config"]
    store.update(sched)
    return _schedule_to_dict(sched)


@router.delete("/{user_id}/{schedule_id}")
def delete_schedule(user_id: str, schedule_id: str) -> dict[str, bool]:
    """Delete a schedule."""
    store = _store()
    sched = store.get(schedule_id)
    if sched is None or sched.user_id != user_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    store.delete(schedule_id)
    return {"deleted": True}
