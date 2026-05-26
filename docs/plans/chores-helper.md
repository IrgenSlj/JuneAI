# Chores Helper Plan

> **Status:** Backlog feature plan. Chores should build on the v0.1.1 personal
> operating layer so reminders, completions, skips, and recurring schedules are
> captured in the same event ledger and approval model as every other action.

## Overview

A structured chore management system that tracks recurring household tasks, reminds the user when chores are due, and builds streaks for consistent completion.

## Architecture

Chores extend the existing habits/tracking system with:
1. **Dedicated schema** — `chores` table with interval-based scheduling
2. **Agent tools** — create, complete, skip, and manage chores
3. **Scheduler integration** — proactive reminders when chores are due
4. **Streak tracking** — borrowed from the habits system

## Domain Schema

```sql
CREATE TABLE chores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    interval_days INTEGER NOT NULL DEFAULT 7,
    last_done TEXT,           -- ISO date of last completion
    next_due TEXT NOT NULL,   -- ISO date of next expected completion
    notes TEXT DEFAULT '',
    estimated_minutes INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE chore_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    chore_id INTEGER NOT NULL REFERENCES chores(id),
    completed_at TEXT NOT NULL,
    note TEXT DEFAULT '',
    skipped INTEGER DEFAULT 0
);

-- View: current streak per chore
-- (calculated from chore_completions, not stored)
```

## Agent Tools

### Management
- `create_chore(name, category, interval_days, notes, estimated_minutes)` — add a new chore
- `list_chores(category, active_only)` — view all chores
- `update_chore(name, **fields)` — update chore details
- `delete_chore(name)` — remove a chore

### Execution
- `complete_chore(name, note)` — mark chore done for today, auto-schedule next due
- `skip_chore(name, note)` — skip a chore without completing it
- `get_chores_due_today()` — view chores that are due
- `get_chores_overdue()` — view chores past their due date

### Proactive
- `get_chore_summary()` — overview of due/overdue/upcoming chores
- `get_chore_streaks()` — current streaks for each chore

## Streak Logic

```python
def calculate_streak(chore_id: str, completions: list[dict]) -> int:
    """Calculate current streak based on consecutive intervals."""
    if not completions:
        return 0
    # Sort by completed_at descending
    sorted_completions = sorted(completions, key=lambda c: c['completed_at'], reverse=True)
    
    streak = 0
    interval = chore['interval_days']
    expected_date = date.today()
    
    for completion in sorted_completions:
        completion_date = date.fromisoformat(completion['completed_at'][:10])
        # If completion is within the interval of the expected date
        if (expected_date - completion_date).days <= interval:
            streak += 1
            expected_date = completion_date - timedelta(days=1)
        else:
            break
    
    return streak
```

## Integration with Daily Briefing

The chore summary feeds into the morning briefing:
- "You have 2 chores due today: vacuum living room, water plants"
- "3 chores are overdue: clean gutters (4 days), wash car (2 days)"
- "Chore streak: 7 days of making the bed 🔥"

## Implementation Order

### Session C1 — Schema + DAO
- [ ] Add `chores` and `chore_completions` tables to schema DDL
- [ ] Create alembic migration
- [ ] Implement `ChoresDAO` class with streak calculation
- [ ] Wire into `Memory` facade

### Session C2 — Agent Tools
- [ ] Implement all chore agent tools
- [ ] Add to `JUNE_TOOLS` list
- [ ] Test each tool with mock data

### Session C3 — UI
- [ ] Chores page in SvelteKit (`/chores`)
- [ ] Chore list grouped by: due today, overdue, upcoming
- [ ] Quick-complete button for each chore
- [ ] Add/edit chore form
- [ ] Streak display per chore

### Session C4 — Proactive + Integration
- [ ] Chore reminders via notification bus (due today, overdue)
- [ ] Chore summary in morning briefing
- [ ] Weekly chore report in weekly review
- [ ] "Good morning! X chores waiting for you" — proactive suggestion
