# June

June is the open personal agent that remembers you. It runs Gemma 4 locally for
chat and recall, reaches Gemini for agentic work when you allow it, and is
designed around one shared brain for browser, desktop, and future mobile
surfaces. The web PWA is usable today, and the Tauri desktop shell has produced
the v0.1.0 Apple Silicon macOS DMG.

This document describes what June is. For why it exists, read
[vision.md](../vision.md). For how it is built, read
[architecture/overview.md](../architecture/overview.md). For where it is going
next, read [roadmap.md](roadmap.md), [ADR 0014](../decisions/0014-personal-operating-layer.md),
and the [v0.1.1 scheduled development plan](../plans/v0.1.1-scheduled-development.md).

## What June Is

A single, persistent agent with three surfaces, one memory, and the ability to do work for you across the apps and services you already use.

- **Chat** — a fluent conversation that remembers. Every turn recalls relevant memories before responding and extracts new facts afterwards. The assistant feels like it knows you because it does.
- **Tasks** — long-running, observable units of work that June carries out on your behalf: drafting and sending an email, finding files across folders, watching a page for a change, planning a trip across calendar and browser. Tasks are persistable, schedulable, and survive the conversation that spawned them.
- **Memory** — an inspectable, editable, exportable record of everything June has learned. Three stores work together: structured facts in SQLite, semantic recall in ChromaDB, entities and relationships in a graph. Users see what June knows, correct mistakes, and delete anything at any time.
- **Skills** — capabilities the agent can call. First-party skills (calendar, gmail, files, health, browser, research) ship in the box. Any third-party MCP server is installable from the in-app registry. Each skill is a standalone MCP server, independently enabled, versioned, and swappable. Third parties can ship skills as pip packages.

## The Product in One Turn

A user opens June. The agent greets them by name, references something real from the last conversation, and asks a specific question. The user answers in natural language. June files the answer into memory, calls a skill if needed (log the workout, draft and send a message, check the calendar, fetch the flight status), streams a response token by token, and updates its understanding of the user. The whole turn happens in one screen, on one device, with the user seeing per-call which model ran where, which skills were touched, and what was written to memory.

The user closes the laptop and opens their phone. Same agent. Same memory. Same tasks running in the background.

## The Product Surface

### Primary screen: Daily Home

One calm first screen. Quick capture is the center. Around it June shows today,
open loops, promises, recent important memories, the next best action, and a
quiet emotional check-in. Chat remains available, but the product center moves
from "ask and answer" to "capture and operate."

### Chat

One column. Message list above, composer below, model and privacy status in the header. Streaming responses token by token. Tool calls render inline with their arguments and results. Per-message provenance shows which model handled which segment of the turn and which skills were called. The composer supports cancellation mid-stream. No sidebars, no tabs, no modals that break the conversation.

### Tasks

A list of active tasks (with a live step trace) and recently completed tasks. Each task shows its goal, the plan June produced, the steps taken, the model used per step, and the artifacts touched (files, services, memories). The user can pause, resume, edit, or cancel any task. Tasks can be spawned from a chat turn or directly from this screen.

### Memory browser

Three sections: structured facts, semantic memories, entities and relationships. Each memory is a card with its source, timestamp, and a delete button. The user can scan their own history at a glance, search it, and remove anything that is wrong or out of date. Manual add and edit are deferred; delete-and-re-learn is the workflow until the UI demands more.

### Skills registry

One card per installed skill. Each card shows the skill name, a description, a running/stopped/crashed status badge, the list of tools it exposes, the model policy (`local-only`, `cloud-allowed`, `cloud-required`), required OAuth scopes if any, and an enable/disable toggle. A separate "Browse skills" view lists installable third-party MCP servers from the registry; one-click install adds them under the same supervisor.

### System header

Active privacy tier (`local-only`, `private-by-default`, `cloud-first`), Ollama reachability, Gemini key state, and the model currently in use for the active turn. Visible on every screen so the user always knows where their data is going.

## The Product Boundary

- **No account.** June is installed, not subscribed to. No signup, no login, no cloud sync by default.
- **No telemetry without consent.** The brain never reports back unless the user opts in.
- **No third model.** Gemma 4 for local, Gemini for cloud. Any new provider must replace one of these, not add to them.
- **No shell-specific business logic.** Desktop and mobile shells are capability wrappers. The same UI runs in all three.
- **No silent cloud calls.** Every cloud-routed model call and every external service call is visible in the UI before and after it happens.

## Model Routing

Three tiers, one dial. See [ADR 0009](../decisions/0009-private-by-default-and-model-routing.md) for the decision record.

- **Local (Gemma via Ollama)** is the default for chat tone, memory recall, classification, short summarisation, journaling, and any turn the user keeps private.
- **Cloud-on-consent (Gemini)** handles agentic planning, long context, vision, computer use, and any skill whose policy requires it.
- **Per-skill policy** — every skill manifest declares `local-only`, `cloud-allowed`, or `cloud-required`. The router resolves the effective tier per tool call, not per turn. A single turn can mix local recall, local planning, and one cloud-required tool call.

The user holds a privacy dial in settings: `local-only` (never call cloud; agentic skills that need cloud are disabled with a visible explanation), `private-by-default` (the default — chat and recall are local, agentic skills may call cloud with confirmation on first call of each kind per session), `cloud-first` (prefer cloud for capability, fall back to local when offline).

## Memory Model

Three stores, one facade. The `MemoryManager` is the only way into memory:

- **SQLite** stores deterministic rows: user profile fields, preferences, habits, daily chapters, workouts, tasks.
- **ChromaDB** stores semantic chunks: conversational context, journal entries, arbitrary facts that need recall-by-meaning.
- **Graph** stores entities (people, places, projects, things) and the relationships between them.

Before every turn, the manager recalls from all three stores based on the incoming message. After every turn, it extracts new facts and writes them back. Skills read and write memory through MCP resources proxied by the brain. No skill opens its own connection.

## Skills Model

Each skill is a standalone Python package with a `python -m june_skill_<name>` entrypoint and an MCP manifest declaring its tools, required OAuth scopes (if any), and model policy. A supervisor in the brain starts each enabled skill as a subprocess, negotiates capabilities over MCP stdio, bridges each skill's tools into LangChain `StructuredTool` instances, and restarts on crash. A manifest under the user's data directory records which skills are enabled.

A skill subprocess that re-imports the brain is a fork bomb. The supervisor sets `JUNE_IS_SKILL_SUBPROCESS=1` and `JUNE_SKILLS_DISABLED=1` in the child environment; the brain's graph module skips agent construction under those flags. See [ADR 0005](../decisions/0005-skills-as-mcp.md).

Third-party MCP servers ship via the same supervisor. The user installs them from the in-app registry; they run with the same subprocess lifecycle, the same memory access proxy, and the same model-policy resolution. Unsigned third-party skills carry a visible badge and a one-time "this runs with your user privileges" warning before first use.

## Tasks Model

A task is a first-class, persistable, observable unit of work, separate from a chat turn. See [ADR 0010](../decisions/0010-agentic-core-tasks-oauth-computer-use.md).

A task carries a goal (free-form natural language), a plan (LLM-produced, editable, JSON), a status (`planning`, `running`, `paused`, `awaiting_user`, `completed`, `failed`), a step trace with model provenance per step, an optional owner skill, and an optional schedule. Tasks live in a new SQLite table; the API surfaces them at `POST /tasks`, `GET /tasks`, `GET /tasks/{id}/events` (SSE), `PATCH /tasks/{id}`, and `DELETE /tasks/{id}`. The chat composer can spawn a task with a slash command or via an inline confirmation when the agent suggests one.

## Personal Operating Layer

The v0.1.1 layer standardizes how future features behave:

1. Capture natural input.
2. Classify it as task, event, memory, decision, promise, feeling, idea,
   question, or note.
3. Create action intents for writes and external actions.
4. Ask approval when risk requires it.
5. Commit to memory, tasks, schedules, notifications, or skills.
6. Record the event in the durable ledger.
7. Bring it back through Daily Home, reviews, reminders, and search.

This is the shared path for calendar, promises, Telegram, agenda suggestions,
emotional support, and future service skills.

## Privacy Model

- Conversations, memories, and embeddings live on the user's machine.
- Gemini calls send only the turn's context to Google; nothing persists there on June's behalf.
- OAuth tokens live in the OS credential store: Keychain on macOS, Credential Manager on Windows, libsecret on Linux. The brain never sees them; the skill subprocess holds and refreshes them.
- No analytics, crash reporting, or usage metrics leave the device without an explicit opt-in.
- Export is one command; delete is one button.

## Status

June currently ships as a web application and an experimental macOS desktop
DMG. The brain, API, memory stores, model routing, tasks, scheduler,
notification bus, daily orchestration, Telegram foundation, and skills system
are implemented. Light mode is the default with a dark mode toggle.

The active priority is v0.1.1: **Quick Capture + Daily Home + Durable Intent
Ledger**. The durable event ledger and the local-first Quick Capture backend
and capture box have shipped; action approval/commit and the full Daily Home
layout are next. See [roadmap.md](roadmap.md) for the trigger-gated surfaces
beyond this release.
