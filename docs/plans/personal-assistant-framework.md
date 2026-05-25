# Personal Assistant Framework

The meta-feature that all personal-assistant capabilities build on. Architecture defined in ADR 0013.

## Component 1: Scheduler Service

### Schema (`june.db`)

```sql
CREATE TABLE schedules (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    cron_expression TEXT DEFAULT '',  -- empty = one-shot
    interval_seconds INTEGER DEFAULT 0,  -- 0 = use cron or one-shot
    scheduled_at TEXT NOT NULL,  -- next run time (ISO 8601)
    last_run_at TEXT,
    action_type TEXT NOT NULL DEFAULT 'agent_invoke',  -- agent_invoke | notification | webhook
    action_config TEXT NOT NULL DEFAULT '{}',  -- JSON: prompt, skill_hint, etc.
    max_runs INTEGER DEFAULT 0,  -- 0 = unlimited
    run_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### Service

```
packages/brain/src/june_brain/scheduler/
├── __init__.py
├── models.py          # Schedule dataclass, ScheduleConfig
├── store.py           # SQLite CRUD for schedules
├── service.py         # Background thread: poll loop, dispatch, next-run calc
└── worker.py          # Execute a scheduled action (invoke agent, send notification)
```

### Agent Tools

- `create_schedule(name, cron, action, description)` — create recurring schedule
- `list_schedules()` — view all schedules
- `delete_schedule(id)` — remove a schedule
- `pause_schedule(id)` / `resume_schedule(id)`

### Implementation Order

1. `models.py` + `store.py` — dataclasses + SQLite CRUD
2. `service.py` — background thread with configurable poll interval (default: 15s)
3. `worker.py` — invoke agent with scheduled prompt, collect result
4. REST endpoints: `GET/POST/PATCH/DELETE /schedules`
5. Agent tools

---

## Component 2: Notification Bus

### Schema (in-memory routing table, no DB needed)

```python
@dataclass
class Notification:
    title: str
    body: str
    priority: str  # "low" | "medium" | "high" | "urgent"
    channel_hint: str | None  # "telegram" | "desktop" | None (route all)
    source: str  # tool name or schedule that triggered it

class NotificationBus:
    _channels: dict[str, Callable] = {}
    
    def register(self, name: str, handler: Callable[[Notification], bool]) -> None
    def dispatch(self, notification: Notification) -> list[tuple[str, bool]]
```

### Channels

- **Desktop notifications** — already available via Tauri notification plugin; expose as channel handler
- **Telegram** — registered by the Telegram skill on startup
- **Log channel** — writes to activity_log (always registered, for audit trail)

### Agent Tools

- `send_notification(title, body, priority, channel)` — send a notification
- `get_notification_history(limit)` — view recent notifications

### Implementation Order

1. `Notification` dataclass + `NotificationBus` class in `packages/brain/src/june_brain/notification.py`
2. Log channel (always-on)
3. Desktop notification channel (via API → Tauri)
4. Wire into scheduler (schedules can trigger notifications)

---

## Component 3: Daemon MCP Skills

### Problem

The current `SkillSupervisor` spawns subprocesses for request-response MCP tools. Telegram needs a persistent subprocess that can *push* events to the supervisor.

### Solution

Extend `SkillManifestEntry` with a `daemon` flag:

```toml
[skills.telegram]
enabled = true
daemon = true
command = "uv"
args = ["run", "june-skill-telegram"]
response_timeout_seconds = 30
env = { TELEGRAM_BOT_TOKEN = "..." }
```

Daemon skills differ from standard skills:
1. **Bidirectional streaming** — supervisor maintains a `select` loop that reads both requests (from the agent) and notifications (from the skill)
2. **Inbound event queue** — daemon skills push events into a DB queue table; the scheduler polls it
3. **Keepalive** — supervisor restarts daemon skills on crash (with backoff)
4. **No agent-tool binding** — daemon skills don't expose tools to the agent (they communicate via the event queue)

### Enhancement to supervisor.py

```python
class SkillProcess:
    # New field for daemon mode
    is_daemon: bool = False
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    
    # New: read loop for daemon push events
    def read_events(self) -> None:
        while self.is_alive and self.is_daemon:
            line = self.stdout.readline()
            event = json.loads(line)
            if event.get("method") == "notification/message":
                self.event_queue.put(event["params"])
```

### Inbound Event Queue Schema

```sql
CREATE TABLE skill_inbound_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,  -- JSON
    received_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
    agent_invoked INTEGER DEFAULT 0
);
```

### Implementation Order

1. Add `daemon` field to `SkillManifestEntry` / `SkillManifest`
2. Extend `SkillSupervisor._spawn_locked` to detect daemon flag and start read loop
3. Add `skill_inbound_events` table to schema
4. Add event queue processing to scheduler service
5. Test with a mock daemon skill

---

## Component 4: Domain Memory Expansion

### Shopping Schema

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    preferred_price REAL,
    preferred_store TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    url TEXT DEFAULT '',
    date_added TEXT NOT NULL,
    active INTEGER DEFAULT 1
);

CREATE TABLE purchase_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    price REAL,
    store TEXT DEFAULT '',
    date TEXT NOT NULL,
    notes TEXT DEFAULT ''
);

CREATE TABLE price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    target_price REAL NOT NULL,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);
```

### Chores Schema

```sql
CREATE TABLE chores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',  -- cleaning, maintenance, errand, admin
    interval_days INTEGER NOT NULL DEFAULT 7,
    last_done TEXT,
    next_due TEXT,
    notes TEXT DEFAULT '',
    estimated_minutes INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE chore_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    chore_id INTEGER NOT NULL REFERENCES chores(id),
    completed_at TEXT NOT NULL,
    note TEXT DEFAULT '',
    skipped INTEGER DEFAULT 0
);
```

### Recurring Tasks (extend existing tasks system)

Add to the existing `tasks` table:
```sql
ALTER TABLE tasks ADD COLUMN is_recurring INTEGER DEFAULT 0;
ALTER TABLE tasks ADD COLUMN recurrence_rule TEXT DEFAULT '';  -- "daily", "weekly", "every 3 days", "mon,wed,fri"
ALTER TABLE tasks ADD COLUMN parent_task_id INTEGER REFERENCES tasks(id);
```

### Implementation Order

1. Shopping: `products` + `purchase_history` + `price_alerts` tables
2. Shopping: DAO + Memory methods
3. Shopping: Agent tools
4. Chores: `chores` + `chore_completions` tables
5. Chores: DAO + Memory methods
6. Chores: Agent tools
7. Recurring tasks migration on existing tasks table

---

## Component 5: Daily Orchestration Engine

### Schedule Entries

On first-run (or first-time setup), the orchestrator creates default schedules:

1. **Morning briefing** — `0 8 * * *` — "Give me a morning briefing including today's calendar, incomplete tasks, habit streak status, and one proactive suggestion."
2. **Evening review** — `0 21 * * *` — "Help me do my evening review: what did I accomplish today, what carries forward, log my mood, and plan tomorrow."
3. **Weekly review** — `0 10 * * 0` — "Generate my weekly review: goals progress, workout summary, habit consistency, upcoming week priorities."

Each schedule invokes the agent with a crafted system prompt that steers the response toward the desired format.

### Agent Integration

The existing `build_system_prompt()` in `skills/prompts.py` gains awareness of the user's daily routine state:
- Whether morning briefing has been delivered today
- Whether evening review is pending
- Any proactive observations from `detect_patterns()`

### Implementation Order

1. Create default schedules on first setup
2. Craft system prompt templates for briefing/review
3. Add "routines" page in settings (enable/disable, customize timing)
4. Test with scheduled agent invocations
5. Wire Telegram notifications for briefing delivery
