# Daily Tasks Orchestrator Plan

## Overview

An intelligent daily planner that runs a morning briefing, manages the day's tasks, tracks progress, and conducts an evening review. This is the most visible "personal assistant" feature — it gives June a daily rhythm and makes the assistant feel proactive rather than purely reactive.

## Architecture

The orchestrator builds on:
1. **Scheduler Service** (Phase 1 Component 1) — time-based triggers
2. **Notification Bus** (Phase 1 Component 2) — deliver briefings
3. **Existing Tasks System** — the task primitive already supports creation/status/SSE trace
4. **Enhancements to Tasks** — recurrence, due-dates, priorities
5. **Pattern Detection** — existing `patterns.py` feeds into briefing content

## System Prompt Templates

### Morning Briefing Prompt

```
You are June, the user's personal assistant. It's {time} on {date}.
Deliver a warm, concise morning briefing covering:

1. GREETING — Greet the user by name. Mention the day of week and weather
   (if available from context).

2. TODAY'S CALENDAR — List all calendar items for today. If none, say so.

3. INCOMPLETE TASKS — Any tasks carried over from yesterday that are still open.
   If none, acknowledge.

4. HABIT STREAKS — Current streaks, especially notable ones (7+ days).

5. TODAY'S CHORES — Chores due today, especially overdue ones.

6. SHOPPING — Any price alerts triggered overnight.

7. PROACTIVE SUGGESTION — One observation from pattern detection or memory
   (e.g., "You haven't trained chest in 5 days", "Your energy has been low,
   want to plan a rest day?").

8. QUESTION — End with one open question to engage the user.

Keep the tone warm, personal, and concise. Use the user's name. 
This is a briefing, not a monologue — end with a question.
```

### Evening Review Prompt

```
You are June. It's {time} on {date}. Help the user wrap up their day.

1. TODAY'S ACCOMPLISHMENTS — What tasks did they complete? What workouts?
   What habits did they maintain? Be specific and appreciative.

2. WHAT CARRIES OVER — What incomplete items carry to tomorrow?
   Suggest a plan for the most important one.

3. MOOD CHECK — Ask how their day was. Offer to log their mood.

4. JOURNAL — Offer to save a brief journal entry about their day.

5. TOMORROW'S PREVIEW — Briefly mention tomorrow's calendar items and
   any scheduled tasks.

6. GOOD NIGHT — End on a warm note.
```

### Weekly Review Prompt

Reuses the existing `generate_weekly_summary` tool but delivers it proactively.

## Recurring Tasks Enhancement

### Schema Migration

```sql
ALTER TABLE tasks ADD COLUMN is_recurring INTEGER DEFAULT 0;
ALTER TABLE tasks ADD COLUMN recurrence_rule TEXT DEFAULT '';
ALTER TABLE tasks ADD COLUMN parent_task_id INTEGER REFERENCES tasks(id);

-- recurrence_rule format examples:
-- "daily" — every day
-- "weekly" — every week on the same day of week
-- "interval:3" — every 3 days
-- "mon,wed,fri" — specific days of week
-- "day:15" — every 15th of the month
```

### Recurrence Engine

```python
def compute_next_due(recurrence_rule: str, completed_at: date) -> date | None:
    """Calculate the next due date based on recurrence rule and completion date."""
    if recurrence_rule == "daily":
        return completed_at + timedelta(days=1)
    elif recurrence_rule == "weekly":
        return completed_at + timedelta(weeks=1)
    elif recurrence_rule.startswith("interval:"):
        days = int(recurrence_rule.split(":")[1])
        return completed_at + timedelta(days=days)
    elif recurrence_rule == "mon,wed,fri":
        # Next day that's Monday, Wednesday, or Friday
        ...
    elif recurrence_rule.startswith("day:"):
        # Next occurrence of the Nth calendar day
        ...
    return None
```

### Agent Tools

- `set_task_recurrence(task_id, recurrence_rule)` — make a task recurring
- `complete_recurring_task(task_id)` — complete, then spawn next instance
- `skip_recurring_instance(task_id)` — skip this instance, reschedule next

## Scheduler Integration

### Default Schedule Entries (created on first setup)

| Name | Cron | Action |
|------|------|--------|
| Morning Briefing | `0 8 * * *` | Agent invoke with briefing prompt |
| Evening Review | `0 21 * * *` | Agent invoke with review prompt |
| Weekly Review | `0 10 * * 0` | Agent invoke with weekly review prompt |
| Overdue Chore Check | `0 9 * * *` | Notification: list overdue chores |
| Task Carry-Forward | `0 6 * * *` | Move incomplete tasks to today |

### User Controls

- **Settings page**: Enable/disable each routine, customize times
- **Snooze**: "Not now, remind me in 30 minutes" inline action
- **Skip**: "Don't ask me today"
- **Manual trigger**: "Good morning, June" triggers the briefing now

## Implementation Order

### Session D1 — Recurring Tasks
- [ ] SQLite migration: `ALTER TABLE tasks ADD COLUMNS...`
- [ ] Recurrence engine in `tasks/models.py`
- [ ] `TasksStore` methods for recurring task lifecycle
- [ ] Agent tools: `set_task_recurrence`, `complete_recurring_task`, `skip_recurring_instance`
- [ ] Update `TaskRuntime` to handle recurrence on completion

### Session D2 — Schedule Defaults + Briefing Prompts
- [ ] Create default schedule entries in scheduler service
- [ ] Add briefing/review prompt templates (to `skills/prompts.py`)
- [ ] Wire schedule → agent invoke in scheduler service
- [ ] Test: schedule triggers, agent responds, notification delivered

### Session D3 — Task Carry-Forward
- [ ] Scheduler action: carry incomplete tasks (re-schedule for today)
- [ ] Mark tasks as "carried from yesterday" in metadata
- [ ] Agent tool: `get_carried_tasks()` for briefing context

### Session D4 — Evening Review
- [ ] Schedule evening review trigger
- [ ] Review collects: completed tasks, habits, workouts, mood (from memory)
- [ ] Saves journal entry automatically
- [ ] Offers to draft tomorrow's plan

### Session D5 — UI for Daily Orchestrator
- [ ] `/today` page: briefing display, task list, chore list, habit progress
- [ ] Inline complete/skip buttons
- [ ] Snooze and dismiss controls
- [ ] Settings: customize briefing time, enable/disable components

### Session D6 — Weekly Review
- [ ] Schedule weekly review trigger
- [ ] Reuse `generate_weekly_summary` tool
- [ ] Deliver via notification bus + Telegram
- [ ] Save as journal entry
