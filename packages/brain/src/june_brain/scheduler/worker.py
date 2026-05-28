"""Execute a scheduled action: invoke agent, send notification, etc."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .models import Schedule

if TYPE_CHECKING:
    from .store import ScheduleStore

logger = logging.getLogger(__name__)


def execute_schedule(store: ScheduleStore, schedule: Schedule) -> None:
    """Execute a schedule's action and record the run."""
    try:
        if schedule.action_type == "notification":
            _run_notification(schedule)
        elif schedule.action_type == "agent_invoke":
            _run_agent_invoke(schedule)
        elif schedule.action_type == "webhook":
            _run_webhook(schedule)
        else:
            logger.warning("Unknown action_type %r for schedule %s", schedule.action_type, schedule.id)
    except Exception:  # noqa: BLE001
        logger.exception("Schedule %s (%s) execution failed", schedule.id, schedule.name)

    store.mark_run(schedule)


def _run_notification(schedule: Schedule) -> None:
    """Dispatch a notification via the notification bus."""
    from ..notification import Notification, bus

    config = schedule.action_config
    bus.dispatch(
        Notification(
            title=config.get("title", schedule.name),
            body=config.get("body", schedule.description or schedule.name),
            priority=config.get("priority", "medium"),
            source=f"scheduler:{schedule.id}",
        )
    )


def _run_agent_invoke(schedule: Schedule) -> None:
    """Invoke the agent with a system prompt built from the schedule config.

    Used for user-requested, deterministic scheduled jobs. June never creates
    these on a timer for proactive engagement (see ADR 0016); a schedule exists
    only because the user asked for it.
    """
    from ..graph import run_agent_sync

    config = schedule.action_config
    prompt = config.get("prompt", "")
    if not prompt:
        prompt = schedule.description or f"Execute scheduled task: {schedule.name}"
        if schedule.description:
            prompt = schedule.description

    try:
        result = run_agent_sync(prompt, user_id=schedule.user_id)
        logger.info(
            "Scheduled agent invoke %s (%s) completed: %s…",
            schedule.id, schedule.name, str(result)[:120],
        )
    except Exception as exc:
        logger.error(
            "Scheduled agent invoke %s (%s) failed: %s",
            schedule.id, schedule.name, exc,
        )
        raise


def _run_webhook(schedule: Schedule) -> None:
    """POST to an external URL."""
    import urllib.request

    config = schedule.action_config
    url = config.get("url", "")
    payload = config.get("payload", {})
    if not url:
        logger.warning("Webhook schedule %s has no URL", schedule.id)
        return
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30)
    logger.info("Webhook %s sent to %s", schedule.id, url)
