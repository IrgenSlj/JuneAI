"""June's tasks primitive — long-running, observable units of work.

Per ADR 0010, a task is a first-class persistable object, separate from a
chat turn. The chat agent (or the user directly) creates tasks; the tasks
runtime executes them across multiple steps; the UI shows a live trace of
what is happening. This package owns the data model and the SQLite store.
The runtime that actually executes tasks lives elsewhere and is wired up in
a later slice.
"""

from .models import Task, TaskStatus, TaskStep, TaskStepStatus
from .runtime import TaskRuntime, execute_task_in_background
from .store import TasksStore

__all__ = [
    "Task",
    "TaskStatus",
    "TaskStep",
    "TaskStepStatus",
    "TasksStore",
    "TaskRuntime",
    "execute_task_in_background",
]
