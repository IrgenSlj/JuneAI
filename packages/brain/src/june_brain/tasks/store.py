"""SQLite store for tasks.

Tasks live alongside memory in the same ``june.db`` so a future view that
joins task results to memory entries (for example, "which memories did this
task touch?") stays local-only. The connection pool is shared with the
memory module to keep there to one connection per (thread, db_path).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..memory.sqlite import _current_memory_dir, _get_connection
from .models import Task, TaskStatus, TaskStep, TaskStepStatus

_UNSET = object()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    plan TEXT NOT NULL,
    owner_skill TEXT,
    schedule TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_user_updated ON tasks(user_id, updated_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TasksStore:
    """CRUD over the ``tasks`` table for one user."""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        db_dir = Path(_current_memory_dir())
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_dir / "june.db")
        conn = _get_connection(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @property
    def _conn(self):  # type: ignore[no-untyped-def]
        return _get_connection(self._db_path)

    # ------------------------------------------------------------------
    # Create / read
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        goal: str,
        owner_skill: str | None = None,
        schedule: str | None = None,
        plan: list[TaskStep] | None = None,
        status: TaskStatus = TaskStatus.PLANNING,
    ) -> Task:
        task = Task(
            user_id=self.user_id,
            goal=goal.strip(),
            status=status,
            plan=list(plan or []),
            owner_skill=owner_skill,
            schedule=schedule,
        )
        self._insert(task)
        return task

    def get(self, task_id: str) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id=? AND user_id=?",
            (task_id, self.user_id),
        ).fetchone()
        return _row_to_task(row) if row else None

    def list(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[Task]:
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                (self.user_id, int(limit)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE user_id=? AND status=? "
                "ORDER BY updated_at DESC LIMIT ?",
                (self.user_id, status.value, int(limit)),
            ).fetchall()
        return [_row_to_task(row) for row in rows]

    def active(self) -> list[Task]:
        """Tasks the user would call 'in progress' right now."""
        active = {
            TaskStatus.PLANNING.value,
            TaskStatus.RUNNING.value,
            TaskStatus.PAUSED.value,
            TaskStatus.AWAITING_USER.value,
        }
        placeholders = ",".join("?" for _ in active)
        rows = self._conn.execute(
            f"SELECT * FROM tasks WHERE user_id=? AND status IN ({placeholders}) "
            "ORDER BY updated_at DESC",
            (self.user_id, *active),
        ).fetchall()
        return [_row_to_task(row) for row in rows]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def set_plan(self, task_id: str, plan: list[TaskStep]) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.plan = list(plan)
        task.updated_at = _now()
        self._update(task)
        return task

    def append_step(self, task_id: str, step: TaskStep) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        step.index = len(task.plan)
        task.plan.append(step)
        task.updated_at = _now()
        self._update(task)
        return task

    def update_step(
        self,
        task_id: str,
        step_id: str,
        *,
        status: TaskStepStatus | None = None,
        tool_result: Any = _UNSET,
        error: str | None = None,
        model_provenance: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        for step in task.plan:
            if step.id != step_id:
                continue
            if status is not None:
                step.status = status
                if status == TaskStepStatus.RUNNING and step.started_at is None:
                    step.started_at = _now()
            if tool_result is not _UNSET:
                step.tool_result = tool_result
            if error is not None:
                step.error = error
            if model_provenance is not None:
                step.model_provenance = model_provenance
            if finished:
                step.finished_at = _now()
            task.updated_at = _now()
            self._update(task)
            return task
        return None

    def set_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.status = status
        now = _now()
        task.updated_at = now
        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = now
        if status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            task.finished_at = now
        if error is not None:
            task.error = error
        self._update(task)
        return task

    def delete(self, task_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM tasks WHERE id=? AND user_id=?",
            (task_id, self.user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _insert(self, task: Task) -> None:
        self._conn.execute(
            """INSERT INTO tasks
                (id, user_id, goal, status, plan, owner_skill, schedule, error,
                 created_at, updated_at, started_at, finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task.id,
                task.user_id,
                task.goal,
                task.status.value,
                json.dumps([step.to_dict() for step in task.plan]),
                task.owner_skill,
                task.schedule,
                task.error,
                task.created_at,
                task.updated_at,
                task.started_at,
                task.finished_at,
            ),
        )
        self._conn.commit()

    def _update(self, task: Task) -> None:
        self._conn.execute(
            """UPDATE tasks SET
                goal=?, status=?, plan=?, owner_skill=?, schedule=?, error=?,
                updated_at=?, started_at=?, finished_at=?
                WHERE id=? AND user_id=?""",
            (
                task.goal,
                task.status.value,
                json.dumps([step.to_dict() for step in task.plan]),
                task.owner_skill,
                task.schedule,
                task.error,
                task.updated_at,
                task.started_at,
                task.finished_at,
                task.id,
                task.user_id,
            ),
        )
        self._conn.commit()


def _row_to_task(row) -> Task:  # type: ignore[no-untyped-def]
    plan_raw = row["plan"] or "[]"
    try:
        plan_data = json.loads(plan_raw)
    except (TypeError, ValueError):
        plan_data = []
    return Task(
        id=row["id"],
        user_id=row["user_id"],
        goal=row["goal"],
        status=TaskStatus(row["status"]),
        plan=[TaskStep.from_dict(step) for step in plan_data],
        owner_skill=row["owner_skill"],
        schedule=row["schedule"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
