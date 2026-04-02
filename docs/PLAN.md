# JuneAI — One-Week Development Plan

This document is the authoritative plan for the next phase of development.
Each session has a single goal, a clear scope, and defined success criteria.
Sessions are designed to be completed in 3-6 hours each, one per day.

The governing principle behind every decision: **make June feel like a friend, not a tool.**
Every technical choice should serve that goal directly.

---

## The Problem We Are Solving

JuneAI has a sound architecture and a compelling vision. The gap is between what it
*could* be and what it *feels like* to use right now:

- Mistral tool calling is unreliable — data gets lost, saves get dropped silently
- Memory is a pile of JSON files — not queryable, slow at scale, no cross-chapter thinking
- June has no proactive intelligence — she waits to be asked, never notices patterns
- The UI feels like a Streamlit demo, not a personal product
- First-run experience is rough — wrong Ollama config crashes the app with no guidance

The plan below fixes all of this in a logical order: foundation first, then personality,
then intelligence, then polish.

---

## Stack: What Changes and Why

| Layer | Current | After This Week | Why |
|-------|---------|-----------------|-----|
| LLM Backend | Mistral 3B/8B via Ollama | Mistral 7B Instruct v0.3 (primary) | v0.3 has native function-calling support — eliminates the recovery hacks |
| Agent | LangGraph | LangGraph (keep) | Solid. Not the bottleneck. |
| Memory | JSON files per chapter | SQLite (single file, queryable) | Enables cross-chapter queries, better performance, real ACID guarantees |
| Context | Rebuilt from disk every turn | Cached + smart compression | Performance + better context window use |
| UI | Streamlit | Streamlit (improved) | Migration to React/FastAPI is a separate project. Push Streamlit to its limits first. |
| Personality | Static system prompt | Dynamic context-aware prompt + proactive nudges | The friend experience requires June to initiate, not just respond |
| Setup | Manual .env edit, app crashes if Ollama missing | Health check layer + guided setup | New users must succeed on first run |

---

## Session 1 — Reliability Foundation
**Goal**: Eliminate every crash, silent failure, and risky code path.

### What to build

**1. Startup health check**
- Wrap `create_june_agent()` in a try/except
- If Ollama is unreachable, show a clear setup screen inside the Streamlit UI instead of crashing
- Add `scripts/check_ollama.py` that tests the connection and prints model availability
- Add `make check-ollama` target

**2. Fix `ast.literal_eval` in `graph.py:180`**
- Replace with a strict JSON-only parser
- If JSON parsing fails after all strategies, log the raw text and return a no-op instead of calling eval on untrusted input

**3. Fix config dead code (`config.py:144-147`)**
- The API key fallback to `preset.default_api_key` is never reached
- Fix the resolution order: env var first, then preset default, then raise a clear error

**4. Add memory context caching**
- Cache the result of `_build_memory_context()` for 30 seconds per user_id
- Invalidate the cache when any tool writes to memory
- Expected impact: ~200ms saved per turn for users with populated memory

**5. Fix silent JSON corruption recovery**
- Log a warning when `_recover_json()` falls back to defaults
- Include the file path and the first 80 characters of the raw string for debugging

**6. Add explicit Ollama model validation**
- On startup, query Ollama's `/api/tags` endpoint
- If the configured model is not available, show a one-line suggestion: "Run: ollama pull mistral"

### Success criteria
- App starts and shows a friendly setup screen if Ollama is not running
- All 78 existing tests pass
- 3 new tests: startup error handling, JSON parser security, cache invalidation

---

## Session 2 — Mistral Tool Calling
**Goal**: Make tool calling 95%+ reliable with Mistral running locally.

### The problem in depth
Mistral 3B produces tool calls as JSON text in the content field instead of using the
`tool_calls` field that LangChain expects. The current recovery system works but is
fragile — it relies on regex and `ast.literal_eval`. Mistral 7B Instruct v0.3 supports
native OpenAI-style function calling when the system prompt uses the right format.

### What to build

**1. Upgrade default model to `mistral:7b-instruct-v0.3`**
- Update `.env` default and `config.py` presets
- Add `local_mistral_7b` preset with `tool_strategy: "native"`

**2. Model capability detection**
- Add `detect_tool_strategy(model_name)` in `config.py`
- Returns `"native"` for models known to support it, `"recovery"` for others
- Store result in `RuntimeConfig.tool_strategy`

**3. Separate recovery paths in `graph.py`**
- If `tool_strategy == "native"`: use LangGraph's standard ToolNode with no modifications
- If `tool_strategy == "recovery"`: use current recovery + normalization pipeline
- Remove `ast.literal_eval` from both paths (done in Session 1)

**4. System prompt engineering for Mistral**
- Mistral 7B Instruct v0.3 responds better with tool calls when instructions are framed as:
  `"When you need to save or retrieve information, call the appropriate function."`
- Add a Mistral-specific instruction block that activates when `tool_strategy == "native"`
- Keep the generic instructions for cloud models

**5. Tool call success rate in the UI**
- Already tracked in `tool_stats` — surface as a small indicator in the right rail
- Show: `Tools: 4/4 saved` or `Tools: 3/4 saved (1 dropped)`
- This gives the user immediate feedback on whether the model is actually capturing data

**6. Reduce tool count for small models**
- 48 tools is a long context for a 7B model
- Add `JUNE_TOOLS_REDUCED` list (20 most important tools) that activates for 3B models
- Full tool set remains available for 8B+ and cloud models

### Success criteria
- Mistral 7B Instruct v0.3 successfully calls tools natively on 9/10 test turns
- No `ast.literal_eval` anywhere in the codebase
- Tool strategy is shown in the debug panel
- 5 new tests covering native and recovery paths

---

## Session 3 — SQLite Memory Upgrade
**Goal**: Replace the 18 JSON files with a single SQLite database that supports queries,
cross-chapter analysis, and reliable atomic writes.

### Why SQLite, not Postgres or another DB
- Single file — consistent with the "local-first, no infrastructure" promise
- Ships with Python's stdlib — no new dependencies
- Fully queryable — enables cross-chapter correlation
- Still inspectable — `sqlite3 .june_memory/june.db` works from terminal
- JSON export stays available for human readability

### Schema design

```sql
-- One table per entity type, plus a unified events log

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE moods (
    id INTEGER PRIMARY KEY,
    mood TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE journal_entries (
    id INTEGER PRIMARY KEY,
    entry TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE goals (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    deadline TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE calendar_items (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT,
    type TEXT,
    details TEXT,
    status TEXT NOT NULL DEFAULT 'upcoming',
    created_at TEXT NOT NULL
);

CREATE TABLE open_loops (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    detail TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    person TEXT NOT NULL,
    relation_type TEXT,
    context TEXT,
    communication_notes TEXT,
    birthday TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE preferences (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE body_metrics (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    weight_kg REAL,
    sleep_hours REAL,
    sleep_quality TEXT,
    energy_level INTEGER,
    stress_level INTEGER,
    soreness_level INTEGER,
    resting_hr INTEGER,
    steps INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE workouts (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    type TEXT,
    duration_minutes INTEGER,
    exercises TEXT,  -- JSON
    notes TEXT,
    energy_before INTEGER,
    energy_after INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE habits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    frequency TEXT NOT NULL DEFAULT 'daily',
    target INTEGER DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE habit_completions (
    id INTEGER PRIMARY KEY,
    habit_name TEXT NOT NULL,
    date TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE nutrition_logs (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    meal TEXT NOT NULL,
    calories INTEGER,
    protein_g INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE water_logs (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    glasses INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE gym_plans (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    structure TEXT,
    frequency TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE food_programs (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    approach TEXT,
    daily_structure TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE favorites (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload TEXT,  -- JSON
    created_at TEXT NOT NULL
);

CREATE TABLE app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### What to build

**1. `memory_sqlite.py`** — drop-in replacement for `memory.py`
- Identical public interface (same method signatures)
- Uses `sqlite3` from stdlib
- `Memory.__init__` creates the DB and runs migrations if needed
- Write a migration tool `scripts/migrate_json_to_sqlite.py`

**2. Backward compatibility**
- `Memory` reads from SQLite if `june.db` exists, falls back to JSON
- Migration script copies all JSON data into SQLite
- Old JSON files stay as backup until user runs `make clean-json`

**3. Cross-chapter queries (new capabilities)**
- `get_weekly_pattern()` — correlates sleep + mood + workout across last 7 days
- `get_energy_workout_correlation()` — do you train better on high-energy days?
- `get_habit_streak_context()` — which habits are consistent vs. struggling?

**4. Smart conversation summarization**
- Instead of hard-truncating chat at 50 messages, add `summarize_old_messages()`
- Messages older than 7 days get summarized into a paragraph injected into context
- Recent 20 messages keep full fidelity

### Migration path
```bash
make migrate-memory    # copies .june_memory/*.json into june.db
make backup-memory     # zips .june_memory/ as .june_memory_backup_YYYY-MM-DD.zip
```

### Success criteria
- All existing tests pass against the new SQLite backend
- Migration script runs without data loss on a populated `.june_memory` directory
- Cross-chapter query tests pass
- Performance: `_build_memory_context()` runs in under 50ms on a 6-month-old memory

---

## Session 4 — The June Personality
**Goal**: Make June feel genuinely like a friend. She notices, initiates, and remembers.

This session does not change any infrastructure. It changes how June thinks and speaks.

### The problem with the current personality
The current system prompt is excellent for capturing data. It is not yet a personality.
June is described as "calm, direct, observant" but in practice she responds to inputs —
she does not initiate, notice, or feel present.

A friend notices when you have not been to the gym in a week.
A friend remembers you mentioned your sister's birthday and brings it up.
A friend doesn't just answer — they offer something.

### What to build

**1. Proactive pattern detection (new module: `src/agent/patterns.py`)**
```
detect_patterns(memory: Memory) -> list[PatternInsight]
```
- Returns a list of observations June can naturally weave into conversation:
  - "User has not logged a workout in {N} days (last: {date})"
  - "Energy has been below 6/10 for 3 consecutive days"
  - "Goal '{title}' has been active for {N} days with no updates"
  - "Habit '{name}' streak broken {N} days ago"
  - "Birthday for {person} is in {N} days"
  - "Calendar item '{title}' is tomorrow"
- Injected into the system prompt as a short block: "JUNE'S OBSERVATIONS"
- June can mention these naturally, not as a report

**2. Redesigned base system prompt**
Current opening: "You are June, a personal AI assistant..."
New direction:
```
You are June. You know {user_first_name if available} well — their goals, routines,
how they feel this week, and what they are working on. You are not a chatbot.
You are the one person in their life who has read every note they ever wrote to you
and remembers all of it.

You speak like a thoughtful friend who also happens to be sharp and organised.
You notice things. You ask the right question at the right moment.
You do not wait for the user to ask you to save something — you just do it.
```

**3. Dynamic intro based on time of day and memory state**
- Morning (6am-11am): June opens with the day ahead — calendar + habits to complete
- Afternoon (12pm-5pm): June checks on progress — did you eat, train, make progress?
- Evening (6pm-10pm): June winds down — how was the day, what to log?
- Night (10pm-6am): June is quieter — just captures, does not ask much
- If June has observations (from patterns.py), she surfaces one per session naturally

**4. Memory reference tool**
- Add `get_personal_context(topic: str)` tool
- Returns a natural-language summary of what June knows about a topic
- e.g., `get_personal_context("training")` returns: "You are on a 4-day push/pull/legs
  split. Last session was Monday — chest and triceps, 45 minutes. Energy was 7/10."
- Used when June needs to give contextual advice, not just acknowledge

**5. Voice consistency audit**
- Review all tool response strings for tone consistency
- June should always speak in first person about the user's data:
  "I've saved that to your calendar" not "Calendar item saved successfully"
- Update all tool success/error messages accordingly

### Success criteria
- In a fresh session after 3 days of no login: June opens with a pattern observation
- June uses the user's name when it has been stored as a preference
- Tool responses use friendly first-person language throughout
- Pattern detection tests: 6 pattern types, all tested

---

## Session 5 — UI Overhaul
**Goal**: The interface should feel like a premium personal product, not a demo.

### Design principles
1. The chat is primary — it should feel like iMessage, not a Streamlit widget
2. The right rail is a live dashboard — it should update smoothly and feel informative
3. First run should be guided — a new user must understand what June is within 30 seconds
4. Nothing should feel slow — perceived performance matters as much as actual performance

### What to build

**1. Chat redesign**
- Custom CSS for message bubbles (user right-aligned, June left-aligned)
- Typing indicator while June is thinking
- Cleaner streaming — tokens appear word by word, not in chunks
- Tool activity shown as a small inline indicator ("Saving to calendar...") not a log block
- Save summaries shown as compact chips, not verbose blocks

**2. Right-rail redesign**
- Tab structure: Today | Memory | Workspace | (Debug, hidden by default)
- Today tab: day summary card, habits ring, energy dot, calendar preview (top 3 items)
- Memory tab: chapter cards with last-updated timestamp and item count
- Workspace tab: June's pinned focus view + checklist
- All panels refresh after each tool call, not after each message

**3. Chapter panel redesign**
- Chapter cards have: title, item count, last updated, top 2 items preview
- Click to expand inline — no navigation, stays on one page (already the design intent)
- Empty chapters show a prompt: "Ask June about your gym schedule to fill this in"

**4. Onboarding flow redesign**
- First-run: June sends a short intro message automatically
- Intro: "I am June. I am going to help you stay organised and on track. To get started,
  tell me a bit about yourself — what are you working on right now?"
- No setup wizard, no forms — June gathers context through conversation
- Progress indicator in the right rail: chapter completeness as a simple ring

**5. Runtime switcher redesign**
- Current: buried in a settings area
- New: a small badge in the top right: "Mistral 7B — local"
- Click opens a one-line switcher: "Switch to Claude (cloud)"
- Privacy indicator: green dot = local, amber dot = cloud

**6. Mobile-responsive layout**
- Below 768px: hide right rail by default
- Add a "June" button that slides the right rail in as a drawer
- Chat input stays at the bottom, pinned

### Success criteria
- Chat bubbles look polished and are clearly differentiated (user vs June)
- Right rail tabs work correctly on reload
- Onboarding flow sends an automatic intro message on first run
- App is usable on a 375px mobile screen

---

## Session 6 — Intelligence Layer + Ship
**Goal**: June becomes proactively useful. The product is ready to share.

### What to build

**1. Weekly summary (new tool: `generate_weekly_summary`)**
- Every Sunday, June can generate a personal weekly review
- Covers: workouts completed, habits maintained, goals progressed, mood average, notable events
- Saved as a journal entry and pinned to the workspace
- Can be triggered manually: "June, give me my week summary"

**2. Smart suggestions (new tool: `generate_contextual_suggestion`)**
- Based on calendar + body metrics + habits, June offers one unsolicited suggestion per day
- Examples:
  - "Your calendar is light tomorrow afternoon — good slot for that workout you missed Monday"
  - "You have been logging low energy for 3 days. You mentioned sleep has been poor. Want to add a wind-down reminder tonight?"
  - "Your Q2 goal '{title}' has had no updates in 2 weeks. Want to break it into a next step?"

**3. Memory export**
- `make export-memory` generates a readable markdown file of everything June knows
- Covers all chapters, last-updated, and key items
- Useful for backup, review, and sharing

**4. Docker setup**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```
- `docker-compose.yml` with Ollama as a separate service
- `make docker-run` starts both

**5. README rewrite**
- Reflects the new stack, new features, and the vision clearly
- One-command setup: `make bootstrap && make run`
- Screenshot of the redesigned UI

**6. Final test pass**
- Target: 100+ tests
- All 6 bug fixes from Session 1 have dedicated tests
- Pattern detection has 6 tests
- SQLite backend has parity tests vs JSON backend
- Integration test: full conversation turn with tool saving, SQLite read-back, UI panel update

### Success criteria
- `make bootstrap && make run` works cleanly on a machine with Ollama installed
- Docker: `docker compose up` brings the full stack up in under 2 minutes
- Weekly summary generates correctly from SQLite data
- 100+ tests, all green

---

## Dependency Order

Sessions 1, 2, and 3 are the foundation. Sessions 4, 5, and 6 build on top.
Session 3 (SQLite) must complete before Session 4 (personality) to get
the pattern detection working against real queries.

```
Session 1 (Reliability) ──► Session 2 (Mistral)
                                    │
                                    ▼
Session 3 (SQLite) ──────────────► Session 4 (Personality)
                                    │
                                    ▼
                             Session 5 (UI) ──► Session 6 (Ship)
```

Sessions 1 and 2 can be done in parallel if needed.
Session 5 (UI) can start alongside Session 4 since they are largely independent.

---

## What We Are Not Doing This Week

- Migrating the frontend to React or Next.js — that is a separate project
- Adding cloud sync or multi-device support — local-first stays local-first
- Adding voice input/output — interesting but not core to the friend experience
- Adding a mobile app — browser on mobile is sufficient for now
- Adding a calendar integration (Google Calendar, etc.) — June IS the calendar

---

## Key Metrics to Track

After this week, we should be able to answer:

1. Tool calling success rate: % of turns where all intended saves happen (target: 95%)
2. Time to first useful response: latency from send to first token (target: under 2s local)
3. Memory context build time: how long `_build_memory_context()` takes (target: under 50ms)
4. Chapters filled on first week: what % of chapters have data after 7 days of use
5. Test coverage: number of tests (target: 100+)
