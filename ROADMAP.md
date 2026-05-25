# June AI — Roadmap

> **Status:** Refocusing on the personal assistant vision. Supersedes `docs/product/roadmap.md` and `docs/product/agentic-pivot-plan.md`.

## Mission

June is the open personal assistant that *remembers you* and *acts on your behalf*. It runs locally, is private by default, and spans every surface you use. Not a smarter chatbot — an assistant that knows you and does work for you.

## Principles

1. **Memory is the product** — every interaction feeds a personal knowledge graph that is yours, editable, portable, and local-first.
2. **Private by default, intelligence on tap** — local Gemma for chat/recall, cloud models for agentic work, with a three-position privacy dial.
3. **Personal assistant, not chatbot** — June acts: communicates via Telegram, manages shopping/chores/tasks, runs daily routines, reaches external services through MCP skills.

## Phases

### Phase 0 — Codebase Stabilisation (current)

Fix the architectural debt that blocks safe feature work:

- Bump Python to 3.13, modernise syntax
- Add API key auth
- Fix ActivityLog singleton race
- Replace broad `except Exception` with targeted catches
- Add schema migration system (Alembic)
- Split `sqlite.py` into per-domain DAOs
- Replace tool alias if/elif chain with data-driven table
- Add data portability (bulk export/import)
- Reactive agent rebuild on skill toggle (no manual reload)

### Phase 1 — Personal Assistant Framework

The architectural layer that all personal-assistant features build on:

- **Scheduler service** — cron-like background trigger for daily tasks, routines, proactive nudges
- **Push notification bus** — abstract notification channel (desktop, Telegram, future mobile)
- **Background daemon support in MCP supervisor** — skills that push events, not just respond to tool calls
- **Domain memory expansion** — shopping products, chores, recurring tasks schemas
- **Daily orchestration engine** — morning briefing, evening review, task dispatch

### Phase 2 — Personal Assistant Features

Built on the Phase 1 framework, shipped incrementally:

- **Telegram communication** — bidirectional chat with June via Telegram, notifications, proactive messages
- **Shopping assistant** — track products you want, price preferences, purchase history, get notified on deals
- **Chores helper** — recurring chore schedules, completion tracking, reminders
- **Daily tasks orchestrator** — morning routine, daily goals, end-of-day review, weekly planning

### Phase 3 — Polish & Release

- Signed desktop installers (macOS .dmg, Windows .msi)
- Three-question first-run flow
- Public landing page
- Closed beta (50 users)
- Bug bash and hardening

### Phase 4 — Expansion (post-v0.1.0, trigger-gated)

- Mobile shell (Capacitor/iOS/Android)
- Multi-device memory sync
- Voice input/output
- Skill marketplace
- Multi-user profiles

## Current Surface: Web PWA + Desktop Shell (Tauri)

The web PWA is the primary shipped surface. The desktop shell adds Ollama supervision, native notifications, system tray, and background tasks. Both share the same SvelteKit build.

## Feature Surface Ordering

Each feature is built directly into the personal assistant framework rather than as a standalone module:

1. **Scheduler + notification bus** — foundation (Phase 1)
2. **Daily task orchestrator** — morning briefing, daily goals, evening review (Phase 2)
3. **Shopping assistant** — preference tracking, price watching, deal alerts (Phase 2)
4. **Chores helper** — recurring chore engine with completion streaks (Phase 2)
5. **Telegram integration** — bidirectional chat, notifications, quick-capture (Phase 2)
6. **Chores + shopping merge into daily briefing** — unified morning/evening experience (Phase 2)

## Things Explicitly Not On This Roadmap

- Cloud sync (memories stay local; export/import is the cross-device story)
- Team or collaboration features
- Third model provider (Gemma + Gemini, period)
- Account-required modes
- Native mobile apps (Capacitor planned, trigger-gated)
