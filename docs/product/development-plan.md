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
- Tasks work mechanically but are not yet durable promises.
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
   - Candidate top level: Chat, Tasks, Memory, Skills, Trust.
   - Treat System details as part of Trust for non-developers.

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
     System page.
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

The first implementation tranche is Phase 0:

- P0.1 reasoning leakage
- P0.2 streaming tool specs
- P0.3 embedding readiness
- P0.4 runtime truth in Settings/System
- P0.5 composer shortcut
- P0.6 mobile overflow

After this tranche, run `./tools/check.sh`, inspect desktop and mobile screens,
commit, and push.
