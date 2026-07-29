"""SQLite store for tasks.

Tasks live alongside memory in the same ``june.db`` so a future view that
joins task results to memory entries (for example, "which memories did this
task touch?") stays local-only. The connection pool is shared with the
memory module to keep there to one connection per (thread, db_path).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ..memory.sqlite import _get_connection, db_path
from .migration import ensure_tasks_migration
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
    due_at TEXT,
    error TEXT,
    blocked_reason TEXT,
    next_action TEXT,
    final_deliverable TEXT,
    approved_tools TEXT DEFAULT '[]',
    blocked_kind TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    is_recurring INTEGER DEFAULT 0,
    recurrence_rule TEXT DEFAULT '',
    parent_task_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    checkpoint_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_user_updated ON tasks(user_id, updated_at);
"""

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TasksStore:
    """CRUD over the ``tasks`` table for one user."""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self._db_path = db_path()
        conn = _get_connection(self._db_path)
        conn.executescript(_SCHEMA_SQL)
        ensure_tasks_migration(conn)
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
        due_at: str | None = None,
        plan: list[TaskStep] | None = None,
        status: TaskStatus = TaskStatus.PLANNING,
        blocked_reason: str | None = None,
        next_action: str | None = None,
        final_deliverable: str | None = None,
        is_recurring: bool = False,
        recurrence_rule: str = "",
        parent_task_id: str | None = None,
    ) -> Task:
        task = Task(
            user_id=self.user_id,
            goal=goal.strip(),
            status=status,
            plan=list(plan or []),
            owner_skill=owner_skill,
            schedule=schedule,
            due_at=due_at,
            blocked_reason=blocked_reason,
            next_action=next_action,
            final_deliverable=final_deliverable,
            is_recurring=is_recurring,
            recurrence_rule=recurrence_rule,
            parent_task_id=parent_task_id,
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
        final_deliverable: str | None = None,
        next_action: str | None = None,
    ) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        previous = task.status
        task.status = status
        now = _now()
        task.updated_at = now
        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = now
        if status == TaskStatus.RUNNING:
            task.blocked_reason = None
            task.next_action = None
            task.blocked_kind = None
            task.final_deliverable = None
            if previous in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                task.finished_at = None
        if status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            task.finished_at = now
        if status == TaskStatus.COMPLETED:
            task.blocked_reason = None
            task.next_action = None
            task.blocked_kind = None
        if error is not None:
            task.error = error
        if final_deliverable is not None:
            task.final_deliverable = final_deliverable
        if next_action is not None:
            task.next_action = next_action
        self._update(task)
        # Spawn a child task if this is a recurring task that just completed
        if status == TaskStatus.COMPLETED and task.is_recurring:
            self._spawn_recurring_child(task)
        return task

    def set_blocked(
        self,
        task_id: str,
        *,
        reason: str,
        next_action: str,
        final_deliverable: str | None = None,
        kind: str | None = None,
    ) -> Task | None:
        """Mark a task as waiting on the user with explicit promise metadata.

        ``kind`` records how it is blocked ("approval" or "local_only") so the
        UI can pick the right unblock control without parsing the reason text.
        """
        task = self.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.AWAITING_USER
        task.blocked_reason = reason.strip() or "Waiting on the user."
        task.next_action = next_action.strip() or "Review this promise and retry."
        task.blocked_kind = kind
        if final_deliverable is not None:
            task.final_deliverable = final_deliverable
        task.finished_at = None
        task.updated_at = _now()
        self._update(task)
        return task

    def set_due(self, task_id: str, due_at: str | None) -> Task | None:
        """Set or clear a promise's deadline. Pass None/empty to clear it."""
        task = self.get(task_id)
        if task is None:
            return None
        task.due_at = (due_at or "").strip() or None
        task.updated_at = _now()
        self._update(task)
        return task

    def approve_tool(self, task_id: str, tool_name: str) -> Task | None:
        """Add a tool to this promise's allow-list so a retry can run it.

        Idempotent — approving the same tool twice is a no-op beyond touching
        ``updated_at``. The guard still enforces taint rules at dispatch time,
        so an approval never waives the exfiltration pattern.
        """
        task = self.get(task_id)
        if task is None:
            return None
        name = tool_name.strip()
        if name and name not in task.approved_tools:
            task.approved_tools = [*task.approved_tools, name]
        task.updated_at = _now()
        self._update(task)
        return task

    def increment_attempts(self, task_id: str) -> int:
        """Atomically increment the failure/block attempt counter. Returns the new value.

        Call this only when a run actually fails or hits a retryable block —
        NOT on a restart-reconciliation or a successful run.
        """
        self._conn.execute(
            "UPDATE tasks SET attempts = attempts + 1, updated_at = ? WHERE id = ? AND user_id = ?",
            (_now(), task_id, self.user_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT attempts FROM tasks WHERE id = ? AND user_id = ?",
            (task_id, self.user_id),
        ).fetchone()
        return int(row["attempts"]) if row else 0

    def delete(self, task_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM tasks WHERE id=? AND user_id=?",
            (task_id, self.user_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Recurring tasks
    # ------------------------------------------------------------------

    @staticmethod
    def _next_recurrence(rule: str) -> str | None:
        """Compute the next ISO datetime from a recurrence rule.

        Supports: ``daily``, ``N days``, ``weekly``, ``N weeks``,
        ``mon,wed,fri`` (comma-separated weekday abbreviations).
        Returns ``None`` if the rule isn't recognised.
        """
        from datetime import timedelta

        now = datetime.now(UTC)
        rule = rule.strip().lower()

        if rule == "daily":
            return (now + timedelta(days=1)).isoformat()

        if rule == "weekly":
            return (now + timedelta(weeks=1)).isoformat()

        if rule in ("monthly", "every month"):
            # Approximate: add 30 days
            return (now + timedelta(days=30)).isoformat()

        if rule.startswith("every "):
            rest = rule.removeprefix("every ").strip()
            parts = rest.split()
            if len(parts) >= 2 and parts[1].startswith(("day", "week", "month")):
                try:
                    n = int(parts[0])
                    if "day" in parts[1]:
                        return (now + timedelta(days=n)).isoformat()
                    if "week" in parts[1]:
                        return (now + timedelta(weeks=n)).isoformat()
                    if "month" in parts[1]:
                        return (now + timedelta(days=n * 30)).isoformat()
                except (ValueError, IndexError):
                    pass

        # Weekday abbreviations: mon,wed,fri
        weekday_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        days = [weekday_map.get(d.strip()) for d in rule.split(",")]
        if all(d is not None for d in days):
            current_weekday = now.weekday()
            for offset in range(1, 8):
                candidate = (current_weekday + offset) % 7
                if candidate in days:
                    return (now + timedelta(days=offset)).isoformat()

        return None

    def _spawn_recurring_child(self, completed_task: Task) -> Task | None:
        """When a recurring task completes, create the next instance."""
        if not completed_task.is_recurring or not completed_task.recurrence_rule:
            return None
        next_due = self._next_recurrence(completed_task.recurrence_rule)
        if next_due is None:
            return None
        child = Task(
            user_id=completed_task.user_id,
            goal=completed_task.goal,
            status=TaskStatus.PLANNING,
            owner_skill=completed_task.owner_skill,
            schedule=completed_task.schedule,
            is_recurring=True,
            recurrence_rule=completed_task.recurrence_rule,
            parent_task_id=completed_task.id,
        )
        self._insert(child)
        logger.info(
            "Recurring task %s spawned child %s (next due: %s)",
            completed_task.id, child.id, next_due,
        )
        return child

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _insert(self, task: Task) -> None:
        self._conn.execute(
            """INSERT INTO tasks
                (id, user_id, goal, status, plan, owner_skill, schedule, due_at, error,
                 blocked_reason, next_action, final_deliverable, approved_tools,
                 blocked_kind, checkpoint_summary,
                 created_at, updated_at, started_at, finished_at,
                 is_recurring, recurrence_rule, parent_task_id, attempts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task.id,
                task.user_id,
                task.goal,
                task.status.value,
                json.dumps([step.to_dict() for step in task.plan]),
                task.owner_skill,
                task.schedule,
                task.due_at,
                task.error,
                task.blocked_reason,
                task.next_action,
                task.final_deliverable,
                json.dumps(list(task.approved_tools)),
                task.blocked_kind,
                task.checkpoint_summary,
                task.created_at,
                task.updated_at,
                task.started_at,
                task.finished_at,
                int(task.is_recurring),
                task.recurrence_rule,
                task.parent_task_id,
                task.attempts,
            ),
        )
        self._conn.commit()

    def _update(self, task: Task) -> None:
        self._conn.execute(
            """UPDATE tasks SET
                goal=?, status=?, plan=?, owner_skill=?, schedule=?, due_at=?, error=?,
                blocked_reason=?, next_action=?, final_deliverable=?, approved_tools=?,
                blocked_kind=?, checkpoint_summary=?,
                updated_at=?, started_at=?, finished_at=?,
                is_recurring=?, recurrence_rule=?, parent_task_id=?, attempts=?
                WHERE id=? AND user_id=?""",
            (
                task.goal,
                task.status.value,
                json.dumps([step.to_dict() for step in task.plan]),
                task.owner_skill,
                task.schedule,
                task.due_at,
                task.error,
                task.blocked_reason,
                task.next_action,
                task.final_deliverable,
                json.dumps(list(task.approved_tools)),
                task.blocked_kind,
                task.checkpoint_summary,
                task.updated_at,
                task.started_at,
                task.finished_at,
                int(task.is_recurring),
                task.recurrence_rule,
                task.parent_task_id,
                task.attempts,
                task.id,
                task.user_id,
            ),
        )
        self._conn.commit()


def reconcile_running_after_restart(*, now: str | None = None) -> int:
    """Transition all RUNNING tasks to AWAITING_USER after an app restart.

    Execution is an in-memory asyncio coroutine; a crash or restart loses it
    silently. This function is called at startup to mark those orphans so the
    user can see them and resume. It operates across all users in the shared db.

    ``blocked_kind`` is left NULL — a restart interrupt is neither an approval
    block nor a local-only block, and no new kind value is introduced.

    Returns the count of rows reconciled. Never raises.
    """
    try:
        ts = now if now is not None else _now()
        conn = _get_connection(db_path())
        cur = conn.execute(
            """UPDATE tasks
               SET status        = ?,
                   blocked_reason = ?,
                   next_action   = ?,
                   blocked_kind  = NULL,
                   updated_at    = ?
               WHERE status = ?""",
            (
                TaskStatus.AWAITING_USER.value,
                "Interrupted by an app restart",
                "Resume this promise to pick it back up.",
                ts,
                TaskStatus.RUNNING.value,
            ),
        )
        conn.commit()
        return cur.rowcount
    except sqlite3.OperationalError as exc:
        # A fresh install has no `tasks` table until the first write creates it.
        # That is the expected first-run state, not a failure — logging a
        # traceback for it makes a clean install look broken and buries the
        # tracebacks that do matter.
        if "no such table" in str(exc).lower():
            logger.debug("no tasks table yet (fresh install); nothing to reconcile")
            return 0
        logger.exception("reconcile_running_after_restart failed")
        return 0
    except Exception:  # noqa: BLE001
        logger.exception("reconcile_running_after_restart failed")
        return 0


def _row_to_task(row) -> Task:  # type: ignore[no-untyped-def]
    plan_raw = row["plan"] or "[]"
    try:
        plan_data = json.loads(plan_raw)
    except (TypeError, ValueError):
        plan_data = []
    d = dict(row)  # convert sqlite3.Row for safe access
    try:
        approved_tools = json.loads(d.get("approved_tools") or "[]")
    except (TypeError, ValueError):
        approved_tools = []
    return Task(
        id=d["id"],
        user_id=d["user_id"],
        goal=d["goal"],
        status=TaskStatus(d["status"]),
        plan=[TaskStep.from_dict(step) for step in plan_data],
        owner_skill=d.get("owner_skill"),
        schedule=d.get("schedule"),
        due_at=d.get("due_at"),
        error=d.get("error"),
        blocked_reason=d.get("blocked_reason"),
        next_action=d.get("next_action"),
        final_deliverable=d.get("final_deliverable"),
        blocked_kind=d.get("blocked_kind"),
        approved_tools=list(approved_tools) if isinstance(approved_tools, list) else [],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        started_at=d.get("started_at"),
        finished_at=d.get("finished_at"),
        is_recurring=bool(d.get("is_recurring", False)),
        recurrence_rule=str(d.get("recurrence_rule", "")),
        parent_task_id=d.get("parent_task_id"),
        attempts=int(d.get("attempts") or 0),
        checkpoint_summary=d.get("checkpoint_summary"),
    )
