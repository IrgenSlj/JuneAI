# June AI - Development Plan

Prepared: 2026-06-28

This plan consolidates the external product, architecture, agentic AI, and design
review into an implementation sequence. It does not replace the worldview in
`overview.md`, the tier roadmap in `roadmap.md`, or the architectural decisions in
`docs/decisions/`. It is the working checklist for turning the current spine into
a reliable product.

## Operating Rules

- Keep the existing spine: FastAPI, SvelteKit, Tauri, the hand-written loop,
  local SQLite memory, salience recall, provenance, and supervised MCP skills.
- Ship small slices. Every slice should leave the app runnable and the gate green.
- Commit and push after each logical slice.
- Make runtime truth visible in the UI. A degraded system state must not look
  healthy.
- Treat trust and privacy as product features, not backend details.
- Prefer measured fixes over broad rewrites.

## Current Diagnosis

The app is structurally sound. The highest leverage work is not a greenfield
rewrite; it is making the runtime honest, resilient, and understandable for a
non-technical user.

## Product Direction - Trusted Continuity Engine

June should become less of a chat app and more of a trusted continuity engine.
Chat remains the most natural input surface, but the product center is the
standing record of what June is holding for the user:

- **Promises** — durable commitments, blocked states, waits, approvals, artifacts,
  and final deliverables.
- **Memory** — inspectable, editable, explainable, reversible records of what June
  believes and why.
- **Trust** — a calm operational surface for local/cloud boundaries, degraded
  modes, traces, approvals, and data retention.
- **Skills** — permissioned capabilities whose scopes and side effects are visible
  before use.
- **Time** — explicit deadlines and event-driven wakeups, never heartbeat-style
  background wandering.

The eventual home screen should not be an empty chat. It should answer, in plain
language, what June is holding:

> I am holding 4 open threads for you. One is waiting on your reply, one is
> blocked by local-only mode, and one has a deadline tomorrow. Nothing has left
> this machine today.

Design implication: June should feel like a personal control room, not a SaaS
dashboard and not a novelty assistant. Surfaces should be quiet, dense, legible,
and explicit about state: local, cloud, degraded, blocked, waiting, done.

Agentic implication: avoid fake autonomy. June should act only when the user
speaks, a subscribed event changes, or an explicit deadline arrives. If a task
needs a consequential action, June defers with an approval request rather than
pretending to continue.

Strengths:

- The local-first product direction is coherent.
- The hand-written loop and provider layer are a defensible center of gravity.
- Memory, provenance, tasks, skills, and setup already exist as product surfaces.
- The backend test suite is broad and the project has a single `tools/check.sh`
  gate.

Critical gaps:

- Raw model reasoning can leak through the streaming path.
- Streaming chat does not fully match non-streaming native tool behavior.
- Embedding readiness can silently degrade semantic recall.
- Settings and System can show contradictory runtime truth.
- Mobile layouts overflow in core screens.
- Promises have first-class blocked state and deliverables, but still need deeper
  artifacts, approvals, retry policy, and memory integration.
- The browser UI lacks regression coverage for the paths users actually touch.

## Phase 0 - Runtime Truth And Stability

Goal: make the shipped spine honest and safe to dogfood.

1. Stop raw reasoning leakage.
   - Do not stream raw native reasoning or chain-of-thought to the UI by default.
   - Replace it with concise observable activity events.
   - Keep detailed traces useful without storing unnecessary raw thought.
   - Acceptance: a streamed trivial turn emits no raw `<think>` or native
     reasoning frames by default.

2. Align streaming and non-streaming tool behavior.
   - Pass provider-native tool specs through the streaming path.
   - Normalize streamed tool-call handling with the non-streamed path where the
     provider supports it.
   - Acceptance: tests prove streamed turns receive the configured tool specs.

3. Make embedding readiness explicit.
   - Detect a missing local embedding model during setup/system checks.
   - Surface degraded semantic recall in UI copy and system status.
   - Acceptance: missing embeddings show as degraded, not healthy.

4. Fix visible runtime contradictions.
   - Distinguish active runtime, saved default, and environment override.
   - Update stale storage and setup-route copy.
   - Acceptance: Settings and System agree on active state and explain overrides.

5. Fix the chat composer shortcut.
   - Ensure keyboard behavior matches the visible shortcut.
   - Acceptance: multiline text is not accidentally submitted by a shortcut
     mismatch.

6. Repair mobile overflow in the core shell.
   - Prioritize Chat, Memory, Settings, System, and Skills.
   - Acceptance: common mobile widths do not clip header chips, cards, stats, or
     composer controls.

## Phase 1 - Trust Surface

Goal: turn privacy and safety state into usable product controls.

1. Add first-class approval cards.
   - Show action, scope, destination, risk, and expiry.
   - Persist approval decisions per turn or task.
   - Add allow-listing only where the risk is low and clear.

2. Add trace controls.
   - Clear, export, and retention controls should live in System/Trust.
   - Redact or summarize sensitive trace fields where possible.

3. Replace optimistic capability labels.
   - Use states such as unknown, measured, degraded, blocked, and failed.
   - Do not show green health before measurement.

4. Improve memory governance.
   - Add server-side search, filters, pagination, edit, merge, forget, and undo.
   - Show why a memory exists and when it was last used.

## Phase 2 - Agentic Core

Goal: make June continue work as promises, not one-off prompts.

1. Promote tasks into promises.
   - Add objective, current state, plan, step ledger, artifacts, blocked reason,
     approvals, retry policy, and final deliverable.
   - Make tasks resumable after process restart.

2. Add deferred proactivity without a heartbeat.
   - Support wakeups from user events, external changes, and explicit deadlines.
   - Avoid timer-based scanning that violates the product boundary.

3. Harden skill permissions.
   - Add manifest scopes for network, filesystem, memory, cloud, secrets, and
     external side effects.
   - Store secrets outside plaintext manifests.
   - Review permissions during install and before risky actions.

4. Build an agentic eval harness.
   - Track recall precision, first-token latency, tool success, approval rate,
     task completion, and prompt-injection resistance.

## Phase 3 - Product And Design

Goal: keep the calm local-first identity while improving density and clarity.

1. Rework information architecture around user concepts.
   - Candidate top level: Chat/Home, Promises, Memory, Skills, Trust.
   - Treat implementation details as part of Trust for non-developers.

2. Clarify state language.
   - Use consistent iconography and semantic color for local, cloud, degraded,
     blocked, running, and complete states.
   - Avoid one-note palette usage on operational screens.

3. Improve first-run value.
   - Explain local/private state quickly.
   - Guide model and embedding setup.
   - Offer import/bootstrap paths so June has useful memory on day one.

4. Add design regression checks.
   - Capture desktop and mobile screenshots for core screens.
   - Fail obvious overflow and blank-screen regressions.

## Phase 4 - Engineering Plumbing

Goal: keep the codebase easy to change as the product surface grows.

1. Split large modules by responsibility.
   - Prioritize the SQLite memory repository, tools registry, Skills page, and
     Trust page.
   - Do not split stable small facades just to make files smaller.

2. Add frontend tests.
   - Start with Playwright smoke tests for setup, chat, memory, tasks, settings,
     skills, and system/trust.

3. Harden local API access for desktop distribution.
   - Use a desktop-generated local token or equivalent protected channel by
     default.
   - Keep explicit API-key auth for exposed deployments.

4. Add local observability.
   - Record first-token latency, total latency, provider, model, cloud boundary,
     tool count, recall count, and degraded modes.

## Phase 5 - Distribution

Goal: a non-technical Mac user can install and use June without a terminal.

1. Finish the desktop sidecar.
   - The `.app` should start the API and web surface without a separate shell.

2. Manage local model dependencies.
   - Detect, pull, and validate chat and embedding models.

3. Package and sign the desktop app.
   - Produce a notarized DMG when external distribution is justified.

4. Add update and recovery flows.
   - Handle corrupt data dirs, missing models, crashed sidecars, and failed
     migrations with understandable recovery steps.

## Active Tranche

The Phase 0 runtime-truth tranche has shipped into `main`. The active tranche is
now the smallest useful version of the Trusted Continuity Engine:

1. Make the home surface answer "what is June holding for me?"
   - Show open promises, waiting states, blocked local-only work, degraded recall,
     and whether the current runtime can act locally or needs cloud.
   - Use existing APIs before introducing new backend shape.

2. Make Promises honest before making them powerful.
   - Keep blocked tool work in `awaiting_user`.
   - Show the waiting step, blocked tool, and retry path in the Promises surface.
   - Next: persist `blocked_reason`, `next_action`, artifacts, final deliverable,
     and approval references as first-class fields instead of only trace steps.

3. Turn Trust into the product's glass box.
   - Continue moving developer-facing System detail behind calm user language.
   - Add explicit approval records and retention choices.
   - Make every degraded mode observable before the user notices failure.

4. Grow Memory governance after Promises can explain themselves.
   - Add edit, merge, forget, undo, filters, and "why this exists" views.
   - Connect completed promise artifacts to memory writes only with visible user
     control.

5. Add Time last, with a strict no-heartbeat boundary.
   - Use explicit deadlines and subscribed external events.
   - Never add ambient polling that violates the privacy/efficiency premise.

Each step should still ship as a small runnable slice, validated locally, pushed
to `main`, and recorded below.

## Progress Log

### 2026-06-28 - Approval Gate As Visible State

Pushed on `main`. Made the guard's approval gate (ADR 0021, S6.2) a first-class,
observable product state instead of prose the model relays. Previously only the
Local-only egress block emitted a structured event; a consequential action the
guard withheld (network egress, code execution, tainted reads) reached the user
as a fake `tool_result` carrying "[ACTION BLOCKED]" text.

- Loop now records structured block details and emits a `tool_blocked` event
  with `needs_approval` and `action_class` for each withheld call, instead of a
  misleading `tool_result`. The model still gets the observation to relay.
- `/chat` SSE forwards `needs_approval` and `action_class` and stops hardcoding
  `network=True`, so the UI can tell a dial-change block from an approval gate.
- `ChatRequest` accepts a per-conversation `approved_tools` allow-list, fed into
  the session; the chat surface renders a distinct approval card and an
  "Approve & retry" path that adds the tool and re-sends.
- Promises distinguish the two blocks: the runtime branches on `needs_approval`,
  records a distinct `blocked_reason`/`next_action`, and persists a structured
  `blocked_kind` ("approval"|"local_only", additive migration) so the UI never
  parses reason text.
- Promises persist an `approved_tools` allow-list; `POST /tasks/{u}/{id}/approve`
  approves one tool and re-runs, and the runtime carries the list into its
  session. Taint-flagged network actions always ask regardless.
- Promises page shows an "Approve & retry" button for approval-blocked promises.

Validation:

- Focused tests: loop egress/guard, chat stream, tasks runtime/store/routes.
- Full gate: `./tools/check.sh` passed with `673` backend tests, frontend
  checks, OpenAPI drift check, Ruff, and the narrowed mypy real-bug gate.

### 2026-06-28 - Phase 0 Runtime Truth

Pushed on branch `codex/development-plan-implementation`:

- Added this consolidated development plan and linked it from the product roadmap.
- Suppressed raw model reasoning in provider and streamed loop paths by default.
- Passed native tool specs into streamed provider calls.
- Added focused tests for streamed tool specs and suppressed reasoning traces.
- Exposed semantic recall readiness in `GET /system`, including the configured
  embedding model and degraded keyword-fallback state.
- Regenerated OpenAPI and TypeScript API types.
- Clarified active runtime vs saved model overrides in Settings.
- Updated System copy from ChromaDB/old setup routes to sqlite-vec/current routes.
- Surfaced semantic recall readiness in System.
- Fixed the composer shortcut so Cmd/Ctrl+Enter sends and plain Enter inserts a
  newline.
- Added mobile overflow guards to Settings and System.
- Added System trace export/clear controls and hid unmeasured capability defaults
  behind `unknown` UI labels.

Validation:

- Focused backend/API tests: `62 passed`.
- Frontend package checks: `@june/ui` and `@june/web` passed.
- Full gate: `./tools/check.sh` passed with `656` backend tests, frontend checks,
  OpenAPI drift check, Ruff, and the narrowed mypy real-bug gate.

### 2026-06-28 - Trusted Continuity Tranche

Pushed on `main`:

- Documented the Trusted Continuity Engine direction in this plan and
  `overview.md`.
- Renamed the user-facing System nav and page language to Trust.
- Changed blocked task execution so tool-blocked work remains `awaiting_user`
  instead of completing optimistically.
- Reframed the Tasks navigation/page as Promises.
- Added a visible waiting card and Retry action for promises blocked by
  local-only tool policy.
- Added a home continuity summary for open promises, waiting promises, privacy
  mode, runtime mode, and semantic recall health.
- Promoted promise blocked reason, next action, and final deliverable to
  persisted backend fields exposed through OpenAPI.
- Updated the Promises UI to use first-class blocked metadata and show completed
  deliverables without opening the trace.

Validation:

- Task runtime focused tests: `10 passed`.
- Promises web check: `svelte-check found 0 errors and 0 warnings`.
- Promise metadata focused tests: `48 passed`.
- Frontend package checks: `@june/ui` and `@june/web` passed.
- Full gate: `./tools/check.sh` passed with `662` backend tests, frontend
  checks, OpenAPI drift check, Ruff, and the narrowed mypy real-bug gate.

### 2026-06-28 - Phase 1 Trust And Memory UX

Pushed on branch `codex/development-plan-implementation`:

- Added trace export and clear controls to System.
- Rendered unmeasured capability probe defaults as `unknown` instead of green
  `good` badges.
- Hardened the Memory page against mobile overflow.
- Made capped Memory lists honest with copy such as `30 of 40 facts`.
- Added server-side `q` support for `GET /memory/{user_id}` across visible fact
  fields and metadata.
- Wired the Memory search box to the server-side query with a short debounce,
  while preserving the local filter as a final pass.
- Added setup-screen visibility for degraded semantic recall when the embedding
  model is missing.

Validation:

- Focused memory route tests: `18 passed`.
- Frontend package checks: `@june/ui` and `@june/web` passed.
- Full gate: `./tools/check.sh` passed with `659` backend tests, frontend checks,
  OpenAPI drift check, Ruff, and the narrowed mypy real-bug gate.
