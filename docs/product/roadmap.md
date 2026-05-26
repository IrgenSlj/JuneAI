# Product Roadmap

This is the detailed product roadmap. The short public summary lives at
[`../../ROADMAP.md`](../../ROADMAP.md).

## Direction

June is becoming a local-first personal operating layer:

- capture natural input
- classify it into useful categories
- propose safe actions
- ask approval when risk requires it
- commit to memory, tasks, schedules, notifications, or skills
- record the event
- bring it back in daily use

The UI should feel simple. The system should be technically rigorous.

## Current Shipped Surface

### Web PWA

The PWA remains the primary shared UI surface. It exposes chat, memory, tasks,
skills, system activity, settings, and setup. It is installable through browser
PWA support.

### Desktop Shell

The Tauri desktop shell now builds and has produced the v0.1.0 Apple Silicon
DMG. It adds native process and OS capabilities that the browser cannot provide:

- Ollama supervision
- native notifications
- system tray
- global hotkey
- autostart
- future background work and local file/app capabilities

The current public DMG is ad-hoc signed and not notarized. Signed distribution
is deferred until external users justify the Apple Developer Program cost and
release-process work.

## Active Release: v0.1.1

Theme: **Quick Capture + Daily Home + Durable Intent Ledger**

Primary references:

- [ADR 0014 — Personal Operating Layer](../decisions/0014-personal-operating-layer.md)
- [v0.1.1 Scheduled Development Plan](../plans/v0.1.1-scheduled-development.md)
- [Personal Operating Layer Research](personal-operating-layer-research.md)

### P0 — Repo Truth And Planning

Align docs and codebase vocabulary around the new direction.

Done when:

- README, roadmap, docs index, desktop setup, and release docs are current.
- Python version references are accurate.
- Old plans are clearly historical or backlog.

### P1 — Shared Operating-Layer Models

Add shared primitives for the rest of the release:

- capture items
- capture kinds
- action intents
- action risk
- approval status
- event kinds

Done when:

- Local low-risk actions and high-risk external actions are represented with
  clear approval behavior.
- Unit tests cover the invariants.

### P2 — Event Ledger

Create the durable event record that underpins memory, tasks, reviews, and
debugging.

Done when:

- SQLite stores events, captures, intents, approvals, and memory sources.
- Export/import includes those records.
- Tests prove migration idempotency and event ordering.

### P3 — Quick Capture Backend

Add the first real product loop: a universal input that turns messy thoughts
into structured candidates.

Done when:

- Natural planning creates task/event candidates.
- Promises are detected.
- Feelings produce supportive, practical responses.
- Privacy dial behavior is enforced.

### P4 — Action Preview And Approval

Turn action proposals into safe execution.

Done when:

- Calendar writes ask before committing.
- Notifications ask if they interrupt later.
- Message sending and deletion always ask.
- Rejected intents are recorded and do not run.

### P5 — Daily Home

Make the first screen useful without turning it into a heavy dashboard.

Done when the first screen has:

- quick capture
- today
- open loops
- promises
- recent important memories
- next best action
- emotional check-in

### P6 — Promise And Agenda Engine

Make June remember commitments and suggest time placement.

Done when:

- June can answer "what did I promise?"
- Dated tasks receive agenda suggestions.
- Evening review can carry unfinished work forward.

### P7 — Telegram Quick Capture

Use Telegram as a cheap mobile surface before building native mobile.

Done when:

- Telegram messages enter the capture pipeline.
- Approved reminders and daily briefings can be delivered through Telegram.
- Sensitive actions still require approval in June.

### P8 — Release Hardening

Ship v0.1.1 cleanly.

Done when:

- Golden workflow tests pass.
- `./tools/check.sh` passes.
- `pnpm desktop:build` produces a DMG.
- v0.1.1 release notes are honest about alpha limitations.

## Later Feature Surfaces

These are trigger-gated.

### Signed Desktop Distribution

Trigger: real external testers are blocked by macOS warnings. Until then, the
unsigned/ad-hoc signed DMG is enough for alpha testing.

### OAuth Gmail And Calendar

Trigger: action preview and approval are solid. External service writes should
not ship before June has a durable consent and audit model.

### Browser And Computer Use

Trigger: June needs to complete important tasks that cannot be done through
APIs, local files, or MCP skills. This remains an escape hatch.

### Voice

Trigger: quick capture works and typing becomes the bottleneck. Likely first
implementation is desktop local speech-to-text.

### Mobile Shell

Trigger: Telegram and the PWA are not enough for mobile capture, share
extensions, or push notifications.

### Skill Marketplace

Trigger: at least three external contributors ship useful MCP skills.

### Sync

Trigger: export/import becomes a clear user pain and the privacy tradeoff is
worth designing.

## Not On The Roadmap

- Account-required modes.
- Cloud memory as the default.
- Team workspaces.
- A third model provider.
- Paid hosting dependency.
- Always-on audio capture.
