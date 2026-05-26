# ADR 0014 — Personal Operating Layer

## Status

Accepted

## Context

June has crossed the first release line with a working local-first brain,
memory, tasks, scheduler, MCP skills, desktop shell, and a public macOS DMG. The
next risk is not lack of ambition. The risk is building many disconnected
features: chat, tasks, schedules, emotions, files, Telegram, calendar, and
daily reviews that do not share one durable operating model.

The product direction is now sharper:

> June is a local-first personal operating layer. You talk naturally; June
> remembers, plans, schedules, comforts, reminds, and acts in the background
> with visible consent boundaries.

External research points to a consistent architecture:

- ChatGPT separates saved memory from chat history and exposes user control.
- Claude Projects show the value of scoped context.
- Reclaim and Motion show that tasks become useful when they are placed into
  time.
- Granola and Limitless show that passive capture plus searchable recall is
  compelling.
- LangGraph supports persistent checkpoints and interrupts for durable,
  user-approved agent work.
- MCP is the right skill protocol, but security depends on explicit
  permissions, scopes, and user consent.

June already chose the right primitives: SQLite as local truth, LangGraph for
agent flow, MCP for skills, FastAPI for the local boundary, SvelteKit for the
shared UI, and Tauri for desktop capabilities. This ADR defines how those
primitives combine into the next system.

## Decision

June will build a **personal operating layer** around five primitives.

### 1. Capture Items

Every user utterance, Telegram message, voice note, pasted transcript, or quick
note enters the system as a `CaptureItem`.

A capture item is classified into one or more kinds:

- `task`
- `event`
- `memory`
- `decision`
- `promise`
- `feeling`
- `idea`
- `question`
- `note`

The classifier produces structured candidates, not hidden side effects. June can
save safe local facts directly, but actions with external impact become action
intents.

### 2. Event Ledger

June gets a durable append-only ledger in SQLite. The ledger records the
important things June observed or did:

- capture received
- capture classified
- action intent created
- approval requested
- action approved or rejected
- memory written, edited, or deleted
- task created or completed
- schedule created or fired
- tool call started or completed
- notification sent

The existing activity log stays as a short rolling operational log for the
system page. The event ledger becomes the durable product record used by memory,
daily review, debugging, export, and future sync.

### 3. Action Intents And Approvals

June distinguishes between understanding and acting.

An `ActionIntent` is a proposed write or external action, such as:

- save a memory
- create a task
- add an agenda block
- create or update a calendar event
- send a notification
- send a Telegram/email message
- delete user data
- call a cloud-required tool

Each intent has a risk level:

- `low`: local-only, reversible, no external side effect
- `medium`: local schedule or reminder with future interruption
- `high`: external service write, deletion, or sensitive personal data
- `external`: message sending, paid action, browser/computer use, or cloud-only
  operation

Low-risk actions may auto-commit if the user preference allows it. Medium and
higher risk actions require an explicit approval state. Sending messages and
deleting data always require approval.

### 4. Scoped, Governable Memory

Memory remains local-first and inspectable, but each memory gains more
governance metadata:

- source capture or event
- scope: global, room/project, relationship, health, money, work, app, etc.
- sensitivity
- confidence
- expiry or review date
- last reinforced date
- vector ref(s) derived from the memory

SQLite remains the source of truth. Chroma remains a derived semantic index for
now. Vector rows must be traceable back to the structured source that created
them.

### 5. Scheduled Background Work

June continues with the existing SQLite scheduler and notification bus. We do
not add Temporal, Hatchet, Prefect, or a separate worker system yet.

The first durable background jobs are:

- morning briefing
- evening review
- promise follow-up
- agenda placement suggestions
- stale open-loop review
- Telegram quick-capture processing

If background work grows beyond what the SQLite scheduler can safely support,
June can later adopt a dedicated durable workflow engine. That is a scale
response, not the first implementation.

## UI Consequence

The UI should become simpler, not busier.

The first screen is a daily home:

- quick capture input
- today
- open loops
- promises
- recent important memories
- next best action
- quiet emotional check-in

Detailed controls live behind focused surfaces: memory, tasks, skills, system,
and settings. The user should feel one simple product while the backend keeps a
complete ledger.

## Security And Privacy Rules

- Local-only mode forbids cloud calls even when an agent thinks the task would
  benefit from a cloud model.
- External writes require a visible approval unless explicitly whitelisted by
  the user for a narrow scope.
- MCP skills declare permissions and model policy before first use.
- Third-party or unsigned skills carry a visible warning and can be disabled
  per skill and per tool.
- Emotional support flows are supportive and practical, not therapeutic claims.
  June may ground, reflect, and suggest small next actions. It must encourage
  real-world support in severe or unsafe situations.

## Alternatives Considered

**Rewrite around a workflow engine now.** Rejected. Temporal and Hatchet solve
real problems, but June does not yet have the operational complexity to justify
a second server or worker runtime. SQLite durability is enough for the next
release.

**Use a cloud memory service.** Rejected. It violates the local-first promise and
turns the core moat into somebody else's backend.

**Build separate feature-specific data models first.** Rejected. Shopping,
chores, calendar, promises, and emotions all need the same capture, intent,
approval, event, and memory provenance layer. Building them separately repeats
logic and weakens trust.

**Make the UI more dashboard-like.** Rejected. June should feel calm and direct.
The system should be rich; the surface should stay quiet.

## Consequences

Positive:

- Every future feature has the same path: capture, classify, propose, approve,
  commit, record, review.
- Debugging and user trust improve because June can show what happened.
- Memory can become more accurate because every stored fact has a source and
  lifecycle.
- Background work becomes a first-class product behavior instead of scattered
  timers.

Negative:

- Adds schema and lifecycle complexity before the UI payoff is visible.
- Requires careful migration discipline because the event ledger becomes
  foundational.
- Approval UX must be excellent or June will feel slow and bureaucratic.

## Implementation Order

1. Add shared dataclasses/enums for capture items, action intents, risk, and
   approvals.
2. Add event-ledger SQLite schema and store.
3. Add quick-capture endpoint and classifier.
4. Add action-intent preview and approval APIs.
5. Add Daily Home UI around the new pipeline.
6. Feed tasks, schedules, notifications, and memory writes through the ledger.
7. Add Telegram quick capture as the first background input.

## References

- ADR 0004 — Memory Architecture
- ADR 0005 — Skills as MCP Servers
- ADR 0009 — Private by Default and Model Routing
- ADR 0010 — Agentic Core
- ADR 0013 — Personal Assistant Framework
- `docs/product/personal-operating-layer-research.md`
- `docs/plans/v0.1.1-scheduled-development.md`
