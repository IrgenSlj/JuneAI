# ADR 0013 — Personal Assistant Framework

## Status

Accepted

## Context

June's stated mission is to be a "personal agent that remembers you." However, the current architecture is fundamentally a "chat-with-memory" system: the user initiates every interaction, and June responds. To become a true personal assistant, June must:

1. **Act proactively** — surface observations, reminders, and suggestions without being asked
2. **Communicate through multiple channels** — not just the web UI but also Telegram, desktop notifications, future mobile push
3. **Run scheduled tasks** — daily briefings, recurring chores, deal monitoring, habit nudges
4. **Maintain persistent background processes** — long-polling Telegram bot, price watchers, calendar monitors
5. **Orchestrate daily routines** — morning briefing, mid-day check-in, evening review

These capabilities cut across the existing architecture and require deliberate design.

## Decision

We introduce five architectural extensions, each building on the previous:

### 1. Scheduler Service

A lightweight background thread/service within the API process that maintains a schedule of recurring and one-shot events.

- Schedule stored in `june.db` (`schedules` table)
- Events trigger agent invocations with predefined prompts
- Configurable intervals: cron expressions, simple intervals (every N hours/days), specific times of day
- Thread-safe: single writer, multiple readers
- Exposed via REST API (`GET/POST/DELETE /schedules`)
- Agent tools: `create_schedule`, `list_schedules`, `delete_schedule`

### 2. Notification Bus

An abstract notification channel that routes messages to the appropriate surface(s):

- **Desktop notification** — via Tauri notification plugin (already shipped)
- **Telegram** — via the Telegram skill (new)
- **Web PWA** — via Service Worker push (future)
- **Mobile push** — via APNS/FCM (future)

Each notification has a priority level (low/medium/high/urgent) and a channel routing policy.

- `Notification` dataclass: `{title, body, priority, channel_hint, source_tool}`
- `NotificationBus` singleton: `dispatch(notification) -> list[ChannelResult]`
- Channels register themselves: `NotificationBus.register("telegram", handler_fn)`
- Agent tool: `send_notification`

### 3. Background Daemon Support in MCP Supervisor (Skill Enhancement)

Currently, MCP skills are request-response only: the agent calls a tool, the skill responds. The Telegram skill needs to *push* incoming messages to the agent, which requires:

- **Event subscription model**: skills can subscribe to events from the supervisor
- **Inbound message queue**: the Telegram skill receives messages and enqueues them
- **Agent polling**: the scheduler periodically checks the queue and invokes the agent if there are new messages
- **Long-lived subprocess**: the Telegram skill runs continuously (unlike current skills which are spawned on demand)

Implementation: enhance `SkillSupervisor` to support "daemon" skills that maintain a persistent subprocess with bidirectional streaming (not just request-response). Add a message queue table in `june.db` for inbound events from daemon skills.

### 4. Domain Memory Expansion

Extend the SQLite memory schema with new domains:

- **Shopping**: `products` (name, category, preferred_price, notes, url, date_added), `purchase_history` (product_id, date, price, store), `price_alerts` (product_id, target_price, active)
- **Chores**: `chores` (name, category, interval_days, last_done, next_due, notes, active), `chore_completions` (chore_id, date, note)
- **Recurring tasks**: extend existing `tasks` table or add `recurring_tasks` (name, interval, next_run, last_run, template_goal)
- **Daily routines**: `routines` (time_of_day, actions_json, enabled)

### 5. Daily Orchestration Engine

A specialized scheduler that runs the user's daily routine:

- **Morning briefing** (configurable time): weather (future), calendar today, incomplete tasks from yesterday, habit streak status, a proactive suggestion
- **Mid-day check-in** (optional): how is the day going, any adjustments needed
- **Evening review** (configurable time): what was accomplished, what to carry forward, log mood/journal entry, plan tomorrow
- **Weekly review** (Sunday): extended summary with goal progress, trends, next week's priorities

The orchestration engine uses the scheduler + notification bus + agent invocation pipeline.

## Architectural Impact

```
┌─────────────────────────────────────────────────┐
│                 API Process                      │
│                                                   │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ Scheduler │  │ Notification│  │ Orchestrator │  │
│  │ Service   │  │ Bus        │  │ Engine       │  │
│  └─────┬─────┘  └──────┬─────┘  └──────┬───────┘  │
│        │               │               │          │
│  ┌─────┴───────────────┴───────────────┴───────┐  │
│  │           LangGraph Agent                    │  │
│  └─────┬───────────────┬───────────────┬───────┘  │
│        │               │               │          │
│  ┌─────┴─────┐  ┌──────┴──────┐  ┌────┴──────┐   │
│  │ Memory    │  │ MCP Skills  │  │ Tasks     │   │
│  │ Stores    │  │ (incl.     │  │ Runtime   │   │
│  │           │  │  Telegram) │  │           │   │
│  └───────────┘  └─────────────┘  └───────────┘   │
└─────────────────────────────────────────────────┘
```

## Consequences

**Positive:**
- Clean separation of concerns: scheduling, notification, and orchestration are independent services
- The Telegram skill fits naturally into the MCP model (daemon variant)
- Domain memory expansion follows the existing three-store pattern
- Everything builds on existing infrastructure (SQLite, LangGraph, skills)

**Negative:**
- Adds complexity to the API process (scheduler thread, notification bus)
- Requires significant refactoring of the skill supervisor for daemon support
- Daily orchestration requires careful prompt design to avoid being annoying
- The scheduler introduces a new class of bugs (timing, race conditions, missed triggers)

## References

- ADR 0004 — Memory Architecture (three-store pattern)
- ADR 0005 — Skills as MCP Servers
- ADR 0010 — Agentic Core
- Existing `detect_patterns()` in `patterns.py`
- Existing `tasks/` module
