# Multi-Session Implementation Plan

This document breaks the entire v0.1.0 work into discrete implementation sessions. Each session is designed to be completed in one coding session, produces a working artifact, and can be committed independently.

Sessions within a phase are ordered by dependency. Sessions across phases can be parallelized.

---

## Phase 0 — Codebase Stabilisation (must complete before Phase 1)

| # | Session | Description | Dependencies | Est. Time |
|---|---------|-------------|--------------|-----------|
| P1.1 | Python 3.13 upgrade | Bump target, auto-migrate annotations, fix CI | None | 1 session |
| P1.2 | Fix ActivityLog race | Thread-safe singleton initialization | P1.1 | 1 session |
| P1.3 | Replace broad exception handlers | Audit all `# noqa: BLE001`, add specific catches + logging | P1.1 | 2 sessions |
| P1.4 | Add schema migration system | Alembic setup, initial migration, auto-run on DB init | P1.1 | 1 session |
| P1.5 | Split sqlite.py into DAOs | Create per-domain DAO classes, refactor Memory facade | P1.4 | 2 sessions |
| P1.6 | Tool alias data-driven table | Extract if/elif chain to TOOL_ALIASES dict | P1.1 | 1 session |
| P1.7 | Add API key auth | Key generation, middleware, exempt setup routes | P1.1 | 1 session |
| P1.8 | Add data portability | Export/import endpoints + CLI tools | P1.3 | 1 session |
| P1.9 | Reactive agent rebuild | Auto-reload agent after skill toggle | P1.1 | 1 session |

**Total Phase 0: ~11 sessions**

---

## Phase 1 — Personal Assistant Framework

| # | Session | Description | Dependencies | Est. Time |
|---|---------|-------------|--------------|-----------|
| P1A.1 | Scheduler models + store | `schedules` table, Schedule dataclass, CRUD | P0 done | 1 session |
| P1A.2 | Scheduler service | Background poll thread, dispatch, next-run calc | P1A.1 | 1 session |
| P1A.3 | Scheduler REST API + agent tools | CRUD endpoints, `create_schedule`/`list_schedules` tools | P1A.2 | 1 session |
| P1A.4 | Notification bus | Notification dataclass, bus with log channel | P0 done | 1 session |
| P1A.5 | Desktop notification channel | Wire Tauri notification plugin into bus | P1A.4 | 1 session |
| P1A.6 | Daemon MCP skill support | Extend manifest, supervisor, spawning, event read loop | P0 done | 2 sessions |
| P1A.7 | Inbound event queue | `skill_inbound_events` table, scheduler polling | P1A.6 | 1 session |

**Total Phase 1: ~8 sessions**

---

## Phase 2 — Personal Assistant Features

| # | Session | Description | Dependencies | Est. Time |
|---|---------|-------------|--------------|-----------|
| P2.1 | Chores schema + DAO | `chores` + `chore_completions` tables, DAO, streak calc | P0 done | 1 session |
| P2.2 | Chores agent tools + wire into Memory | All chore tools, add to JUNE_TOOLS | P2.1 | 1 session |
| P2.3 | Chores UI | `/chores` page, list, add, complete, streaks | P2.2 | 1 session |
| P2.4 | Shopping schema + DAO | `products` + `purchase_history` + `price_alerts`, DAO | P0 done | 1 session |
| P2.5 | Shopping agent tools + wire into Memory | All shopping tools, add to JUNE_TOOLS | P2.4 | 1 session |
| P2.6 | Shopping UI | `/shopping` page, product list, add, alerts | P2.5 | 1 session |
| P2.7 | Recurring tasks migration + engine | ALTER TABLE, recurrence rule parser, store methods | P0 done | 1 session |
| P2.8 | Recurring tasks agent tools + TaskRuntime update | `set_task_recurrence`, `complete_recurring_task`, runtime hooks | P2.7 | 1 session |
| P2.9 | Daily briefing prompt + schedule | Prompt templates, create default schedule on setup | P1A.2, P1A.4 | 1 session |
| P2.10 | Evening review prompt + schedule | Review prompt template, create default schedule | P1A.2, P1A.4 | 1 session |
| P2.11 | Task carry-forward | Scheduler action to move incomplete tasks, `get_carried_tasks()` | P2.8 | 1 session |
| P2.12 | `/today` orchestration UI | Dashboard with tasks, chores, habits, calendar | P2.3, P2.6, P2.11 | 2 sessions |
| P2.13 | Weekly review prompt + schedule | Reuse `generate_weekly_summary`, deliver via bus | P1A.2, P1A.4 | 1 session |
| P2.14 | Telegram skill — daemon MCP server | `skills/telegram/` package, MCP stdio server, long-polling | P1A.6, P1A.7 | 2 sessions |
| P2.15 | Telegram — pairing + commands | `/start` pairing flow, chat commands, linking table | P2.14 | 1 session |
| P2.16 | Telegram — notification channel | Wire Telegram into notification bus for outbound messages | P2.15 | 1 session |
| P2.17 | Chore + shopping proactive nudges | Overdue reminders, price drop notifications via bus | P2.3, P2.6, P1A.4 | 1 session |

**Total Phase 2: ~18 sessions**

---

## Phase 3 — Polish & Release

| # | Session | Description | Dependencies | Est. Time |
|---|---------|-------------|--------------|-----------|
| P3.1 | Desktop first compile | Install rustup, fix Tauri build, verify tray/hotkey/autostart | P0 done | 1 session |
| P3.2 | macOS signing + notarization | Set up Apple Developer account, codesign, notarize .dmg | P3.1 | 1 session |
| P3.3 | Landing page | Single-page site at june.ai with install button, privacy explainer | — | 1 session |
| P3.4 | First-run flow | Three-question setup: name, services, privacy default | P0 done | 1 session |
| P3.5 | Dogfooding + bug bash | Use June daily for 1 week, log issues, fix top 10 | All above | 1 week |
| P3.6 | Release | Tag, build, publish, announce | P3.5 | — |

**Total Phase 3: ~4 sessions + 1 week dogfooding**

---

## Total Estimate

- Phase 0: ~11 sessions
- Phase 1: ~8 sessions
- Phase 2: ~18 sessions
- Phase 3: ~4 sessions
- **Total: ~41 sessions**

At roughly 1 session/day, this is ~8 weeks of work to v0.1.0.

---

## Parallelization Opportunities

- Phase 0 sessions P1.x must be sequential (each depends on previous)
- Phase 1A.x sessions are sequential within their track
- Phase 2 chores (P2.1-2.3) and shopping (P2.4-2.6) can run in parallel
- Phase 2 recurring tasks (P2.7-2.8) is independent of chores/shopping
- Phase 2 Telegram (P2.14-2.16) needs daemon MCP support (P1A.6) first
- Phase 3 desktop (P3.1) is independent but has external dependency (rustup)

## Quick Wins (do first for visible progress)

1. P1.7 (API key auth) — fast, meaningful security improvement
2. P2.1-2.2 (chores schema + tools) — quick, highly visible feature
3. P2.4-2.5 (shopping schema + tools) — quick, leverages existing preferences code
4. P1A.1-1A.2 (scheduler) — foundation for everything proactive
