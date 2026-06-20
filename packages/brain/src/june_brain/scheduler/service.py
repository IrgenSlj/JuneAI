"""Background scheduler service — polls for due schedules and dispatches them."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from .models import Schedule
from .store import ScheduleStore
from .worker import execute_schedule

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 15  # seconds


def _parse_cron(cron: str) -> list[int] | None:
    """Very basic cron parser — returns minute values this matches, or None.

    Supports only: ``*``, ``*/N``, ``N``, ``N,M,O`` in minute field.
    This is intentionally minimal; we'll use a proper library (croniter)
    when needed.
    """
    if not cron or cron == "* * * * *":
        return None  # every minute
    parts = cron.strip().split()
    if len(parts) != 5:
        return None
    raw_minutes = parts[0]
    if raw_minutes == "*":
        return None
    minutes: list[int] = []
    for token in raw_minutes.split(","):
        token = token.strip()
        if "/" in token:
            base, step_str = token.split("/", 1)
            base = base.strip()
            step = int(step_str)
            start = 0 if base == "*" else int(base)
            minutes.extend(range(start, 60, step))
        else:
            minutes.append(int(token))
    return sorted(set(minutes))


def compute_next_run(schedule: Schedule) -> str | None:
    """Compute the next ``scheduled_at`` ISO string for a schedule.

    Returns ``None`` if the schedule has exhausted its runs.
    """
    if schedule.max_runs > 0 and schedule.run_count >= schedule.max_runs:
        return None

    now = datetime.now(UTC)

    if schedule.cron_expression:
        minutes = _parse_cron(schedule.cron_expression)
        if minutes is not None:
            # Find next minute match
            candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            for _ in range(60):
                if candidate.minute in minutes:
                    return candidate.isoformat()
                candidate += timedelta(minutes=1)
            return candidate.isoformat()
        # Fall through to interval or default
        return now.isoformat()

    if schedule.interval_seconds > 0:
        return (now + timedelta(seconds=schedule.interval_seconds)).isoformat()

    # One-shot with no cron/interval — done after first run
    return None


class SchedulerService:
    """Background thread that polls for due schedules."""

    def __init__(
        self,
        store: ScheduleStore,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._store = store
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Scheduler started (poll interval=%ss)", self._poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Scheduler stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                due = self._store.list_due()
                for schedule in due:
                    logger.debug("Executing schedule %s (%s)", schedule.id, schedule.name)
                    execute_schedule(self._store, schedule)
            except Exception:  # noqa: BLE001
                logger.exception("Scheduler poll loop error")
            self._stop_event.wait(self._poll_interval)
