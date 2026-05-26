# June AI — Roadmap

## Current Direction

June is becoming a **local-first personal operating layer**:

> You talk naturally. June captures what matters, remembers it, proposes safe
> actions, schedules what needs time, and returns at the right moment.

The interface should stay calm and simple. The technical system underneath must
be durable, inspectable, and privacy-preserving.

## Principles

1. **Memory is the product.** Memories are local, editable, source-linked,
   portable, and scoped to the right life area.
2. **Agency needs consent.** June can act in the background, but external
   writes, sensitive actions, cloud calls, and deletions have visible approval
   boundaries.
3. **One quiet surface, serious backend.** The user gets Daily Home and quick
   capture. The backend keeps the event ledger, tasks, schedules, skills,
   approvals, and memory provenance.
4. **Private by default.** Local Gemma/Ollama remains the default path. Gemini
   is an optional capability with visible provenance and privacy-dial control.
5. **No paid complexity before users.** Use the existing local stack until real
   usage proves the need for paid infrastructure, app-store signing, or a
   dedicated workflow engine.

## Shipped

- Web PWA with chat, memory, settings, skills, tasks, and system activity.
- Tauri desktop shell with Ollama supervision, tray, hotkey, autostart, and
  native notification capability.
- Three-tier model routing with per-message provenance.
- Tasks runtime with live SSE trace and cancel.
- SQLite + Chroma + graph memory architecture.
- MCP skill supervisor and bundled skills.
- Scheduler, notification bus, daily orchestration, and Telegram foundation.
- v0.1.0 GitHub release with Apple Silicon macOS DMG.

## Active Track: v0.1.1

Theme: **Quick Capture + Daily Home + Durable Intent Ledger**

Detailed execution plan:

- [ADR 0014 — Personal Operating Layer](docs/decisions/0014-personal-operating-layer.md)
- [v0.1.1 Scheduled Development Plan](docs/plans/v0.1.1-scheduled-development.md)
- [Personal Operating Layer Research](docs/product/personal-operating-layer-research.md)

### v0.1.1 Work Packages

1. **Repo truth and planning**
   - Align README, docs, setup, roadmap, and release docs with the current
     product state.

2. **Shared operating-layer models**
   - Add typed models for capture items, action intents, approval status, risk,
     and event kinds.

3. **Event ledger**
   - Add durable SQLite tables for events, captures, action intents, approvals,
     and memory sources.

4. **Quick capture backend**
   - Add an endpoint and classifier that turns messy input into tasks, events,
     memories, decisions, promises, feelings, ideas, questions, and notes.

5. **Action preview and approval**
   - Gate calendar writes, notifications, messages, deletion, and cloud-required
     actions behind visible approval rules.

6. **Daily Home**
   - Make the first screen a simple personal command center: quick capture,
     today, open loops, promises, important memories, next action, and emotional
     check-in.

7. **Promise and agenda engine**
   - Track commitments and suggest when dated work should happen.

8. **Telegram quick capture**
   - Use Telegram as the cheap mobile input and notification surface before
     building a native mobile app.

9. **Release hardening**
   - Golden workflow tests, docs, DMG build, and v0.1.1 release notes.

## Next Tracks

These are trigger-gated. They are not started until v0.1.1 is useful in daily
dogfooding.

- **Signed and notarized desktop distribution.** Start when external users are
  blocked by macOS warnings enough to justify the Apple Developer Program cost.
- **Voice input/output.** Start when quick capture is useful and voice becomes
  the main friction. Likely path: local desktop speech-to-text first.
- **OAuth-backed Gmail/Calendar.** Start after the approval system is solid.
- **Browser/computer use.** Keep as an escape hatch, not the front door.
- **Mobile shell.** Start when Telegram and PWA are insufficient for capture,
  share extensions, or push.
- **Skill marketplace.** Start when external contributors have shipped useful
  skills.
- **Sync.** Start only when export/import is not enough for real users.

## Explicit Non-Goals

- No cloud account requirement.
- No cloud memory service.
- No team/collaboration layer.
- No third model provider.
- No paid hosting dependency.
- No always-on audio.
- No mobile app until usage proves the need.
