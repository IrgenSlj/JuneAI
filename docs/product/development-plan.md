# June AI - Development Plan

> **SUPERSEDED (2026-07-06).** This pre-v0.2 plan is retained for history only.
> The active plan is [`JUNE_V02_BRIEF.md`](../../JUNE_V02_BRIEF.md); current
> project state is [`docs/CURRENT.md`](../CURRENT.md). Do not follow this document
> for new work. (Not physically archived because append-only ADRs link it.)

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
   - NOTE (2026-06-30): this default was intentionally REVERSED for the Glass Box
     by user direction (Track 6 / GB-5). Raw chain-of-thought is now visible by
     default. No new egress: reasoning is ingress from the model, rendered
     locally; local-only mode still blocks outbound calls.

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

## Autonomous Execution Plan (2026-06-28)

Ordered queue of gated, individually-pushed slices completing the Trusted
Continuity Engine. Each: small slice -> `./tools/check.sh` green -> commit ->
push. Delegated chunks go to implementer/reviewer subagents; the gate and push
stay on the main thread.

1. [DONE] Promise deadlines (`due_at`) — set/show a due date; home and Promises
   surface "due soon" / "overdue" computed at read time. No timer.
2. [DONE] Reversible forget for structured rows — every memory type is now
   reversible (facts, entities, goals, open_loops, calendar, journal,
   body_metrics).
3. [DONE] Memory "why this exists" — humanized origin label ("learned from a
   conversation", "added by you", "written by the X skill").
4. [DONE] Trust retention controls — empty the forgotten trash from the UI;
   approval records visible/filterable; activity clear already existed.
5. [TODO] Frontend Playwright smoke tests — setup, chat, memory, promises,
   trust. Deferred: infra-heavy and not yet wired into the gate.
6. [DONE] Skill permission scopes — each skill shows derived capability scopes
   before use; network/execute highlighted.

Remaining beyond this queue: event-driven wakeups (subscribed external changes)
as the larger half of Time, and the Playwright harness. Both are documented for
a future tranche.

Inventive but bounded; respects every invariant (no heartbeat, privacy visible,
honesty fixed, graceful degradation shipped in the same change).

## Execution Queue (2026-06-29) - Continuity Engine completion

Ordered, leverage-first. Each entry is a small gated slice: implementer subagent
edits, reviewer subagent flags, the main thread runs `./tools/check.sh` and pushes
(`fetch + rebase origin/main` first, since the checkout is shared). Tracks 1-2 are
the priority; 3-4 deepen the core; 5 is design-gated big-bet work.

### Track 1 - Close the Silence loop (make the shipped feature real)

The policy + store + three UI surfaces exist, but nothing calls `decide()`. Give it
real producers, a `now` sink, and a `batch` drain. No timer (ADR 0016): producers
run at read time.

1. Deadline producer (`silence/producers.py`, pure): promises with `due_at` ->
   `SurfacingCandidate` (salience from urgency) -> `SurfacingContext` (injected
   `now`, dismissals from the store) -> `decide()` -> `record()`. Done when a
   promise due in 2h yields `now` and one due in 5 days yields `batch`, both
   visible in `/system/surfacing`, with tests.
2. `batch` drain at the session-open boundary: call `drain_digest()` at the "user
   shows up" endpoint; surface drained items on Home with working Open / Dismiss
   (Dismiss -> `record_feedback`). Done when held items appear next open, are
   marked surfaced, and don't re-appear.
3. `now` sink: wire a `now` verdict to a real delivery channel (existing
   notification/SSE path; OS notification via the desktop shell if present). Honest
   fallback shipped in-slice: an in-app "needs you now" banner on Home.
4. Contradiction and promise-nudge producers (blocked promise needing input;
   memory contradiction if the belief engine exists, else defer).

### Track 2 - Playwright smoke tests (lock in the just-shipped UI)

The one open `[TODO]` above; overdue now that Home/Receipts/Silence shipped with no
browser coverage. Greenfield (no e2e harness exists today).

1. Harness + setup->chat smoke against a throwaway data dir.
2. Core-screen smokes: Home, Promises, Memory, Trust>Receipts (verify affordance),
   Trust>Silence.
3. Gate wiring behind a flag so the default gate stays fast (`JUNE_E2E=1`).

### Track 3 - Durable / resumable Promises

Map (confirmed): task state is fully persisted in SQLite `tasks`; execution is an
in-memory asyncio coroutine in FastAPI BackgroundTasks, lost on restart -> a killed
promise is a permanent `RUNNING` zombie that `_lifespan` never rescans and the 409
guard refuses to re-kick. No retry/attempt field exists.

1. Startup reconciliation in `_lifespan`: zombie `RUNNING` tasks move to a
   recoverable state ("interrupted by restart") with the prior trace preserved, so
   they can be resumed rather than stuck.
2. Attempt counter + bounded retry (additive columns); surfaced in the Promises UI;
   no infinite loops.
3. Honest resume-on-wake: a resumed promise is re-dispatched with its prior trace
   visible and a "resumed after restart" marker. (Idempotent step-level resume is a
   documented future enhancement; the loop re-runs from scratch today.)

### Track 4 - Skill permission hardening

Scopes are already derived from guard action classes and shown before use. Harden
enforcement and secret handling (`skills/manifest.py`, `secret_store.py`).

1. Declared manifest scopes checked against derived (actual) scopes; flag drift.
2. Skill secrets resolve through `secret_store` (OS keychain), never plaintext.
3. Scope review at install and before a first risky (network/execute) action;
   recorded to the ledger.

### Track 5 - Big bets (design-gated; pause for explicit go)

Each gets a short ADR before code (crypto / external accounts).

1. Privacy Mode 2 - client-side-encrypted data-dir backup (vetted crypto; keychain
   day-to-day, passphrase to move machines).
2. Privacy Mode 3 - Gmail/Calendar/Drive as per-service MCP skills (reads first,
   writes per-action, revocable, always visible).
3. Desktop Python sidecar - the `.app` runs the brain without a separate shell.

### Track 6 - The Glass Box: activity window as a complete view of every process (ELEVATED, user 2026-06-29)

The user is dissatisfied with the current activity popup and wants it reimagined
as a breathtaking, see-everything view of all agentic + LLM processes. Diagnosis:
today there are two disconnected surfaces - a live chat "activity terminal" wiped
every turn, and a separate 5s-polling trace viewer on Trust - and the richest
captured data (tokens, cloud-call lifecycle, per-memory salience) is dropped
before render, while sub-LLM calls emit nothing.

Locked decisions (asked + answered): (a) a dedicated full-screen Glass Box route
under Trust + a slim live strip on Chat; (b) "go big" - capture the hidden
thinking layers (difficulty classifier, compactor sub-LLM calls) and the full
turn tree; (c) raw reasoning visible by default (reverses the Phase-0 suppression
default - update that acceptance test and document the reversal; no new egress:
local reasoning stays local, cloud reasoning is already surfaced as egress).

Ordered so it is visible early, then gets richer:

1. GB-1: the dedicated Glass Box route under Trust + a `glass` store, rendering
   the already-persisted turn traces (`/system/traces*`) as an expandable
   turn-tree timeline (route -> recall -> prompt -> llm -> tool -> provenance).
   No backend change. Trust page links to it.
2. GB-2: unify live + history - pipe the chat SSE into the persistent glass store
   (stop wiping every turn) + the slim live strip on Chat linking in.
3. GB-3: plumb already-captured data into live events - tokens in/out per call,
   cloud-call start/end lifecycle, per-recall salience scores.
4. GB-4: capture the thinking layers - difficulty classifier + compactor sub-LLM
   calls become nested events (model, prompt, tokens, latency) under the turn.
5. GB-5: reasoning always visible - stop suppressing raw CoT; emit + persist it;
   update the Phase-0 "no reasoning by default" test; document the reversal.
6. GB-6: history parity - replayed turns show the same enriched tree as live.

## Progress Log

### 2026-07-02 - Deploy path PROVEN + prepped + hardened; loopback auth; recovery

Continuation of the autonomous drive. The deploy path is no longer just built —
it is proven with a real artifact and prepped to the founder's signature.

**The app builds and runs (capstone).** A full unsigned `tauri build` produces
`June.app` (103 MB) + `June_0.1.0_aarch64.dmg` (43 MB). Verified headlessly from
inside the `.app`: the bundled brain serves `/system` 200 (with the git SHA
baked in) and `june-api --run-skill calendar` answers an MCP handshake as
`june-calendar` with zero stray processes. Fixed one latent build bug
(`beforeBuildCommand` path resolved outside the repo). Report:
`docs/product/tauri-build-report.md`. This closes gap 1 end-to-end; only the
founder's Developer ID signing remains. NOTE: this means Track 5 item 3
(desktop sidecar) and Phase 5 items 1-3 are DONE — the plan above is stale there.

**Signing/notarization infrastructure (gap 4).** `entitlements.plist` for the
frozen sidecar under hardened runtime, a GitHub Actions release workflow that
signs+notarizes only when the Apple secrets are present (degrades to unsigned
otherwise; injects signing via a `--config` override so the unsigned build stays
intact), and `apps/desktop/RELEASING.md`. The founder adds a cert + secrets and
pushes a `vX.Y.Z` tag.

**Loopback API auth (end to end).** Opt-in middleware enforces `X-June-Token`
only when `JUNE_API_TOKEN` is set; the Tauri shell generates a per-launch uuid,
sets it on the sidecar env, and hands it to the fetch-based web client via a
`get_api_token` command. Dev/browser stay unauthenticated (no token ->
pass-through). No other local process or page can read June's data through the
API. Verified: cargo check, check.sh, live token flow (401/200/200).

**Recovery flows (gap 7).** (a) A corrupt `june.db` is detected on first open
(PRAGMA quick_check), moved aside to `june.db.corrupt-<ts>`, and replaced with a
fresh DB — transient locks are never mistaken for corruption. (b) A Tauri
watchdog restarts a crashed sidecar once it has been healthy at least once
(never fights the cold start), with capped backoff and clean shutdown.
Remaining recovery item: failed-migration handling (assess whether a runtime
migration path exists).

**Also:** Track 4 completed (scope-drift detection, keychain secrets, ledger
scope-review); License Slice 2 (entitlement surfaced in `/system`); the
Silence contradiction producer (calendar double-booking) completing Track 1.3;
the draggable chat/activity splitter; and dependency-honesty fixes.

**Still genuinely founder-gated:** the wedge decision
(`strategic-review-2026-07-01.md`), Apple Developer enrollment, pricing +
payment processor + OSS license, and Track 5 items 1-2 (encrypted backup,
Google skills — crypto + external accounts, an ADR each).

**Cold-start profiled (bounded spike).** Frozen sidecar startup is dominated by
OS cost (macOS verifying the unsigned ~85 MB onedir on first launch, ~7-8 s),
not our Python — not cheaply reducible with lazy imports; the highest-ROI lever
is code-signing the sidecar (a packaging change for the founder's cert). One
safe win landed: deferred the `openai` import off the boot path (~0.18-0.47 s).
Full profile: `docs/product/cold-start-notes.md`.

**Adversarial-review hardening pass.** Ran a 4-agent read-only review of the
deploy-critical code that runs unattended on users' Macs (Rust sidecar
lifecycle, offline license verifier, loopback auth, corrupt-DB recovery) and
fixed every real finding, gate-green:
- License verifier: a naive (tz-less) `expires_at` raised TypeError and locked
  out a paying holder — fixed with a never-raising UTC-normalizing parse;
  documented the load-bearing canonical-JSON convention; added regression tests.
- Auth: exemptions were prefix-matched (`startswith`), which would silently
  exempt a future `/healthz_admin`-style route — made exact-match.
- Corrupt-DB recovery: serialized under a lock with a collision-safe,
  microsecond-stamped backup name so concurrent startup workers can't overwrite
  each other's backup; documented the load-bearing except ordering. Narrowed
  migration 5's blanket `except` so a genuine ALTER failure retries instead of
  being marked applied.
- Sidecar: closed an orphan-on-quit race (re-check `shutting_down` under the
  child lock after spawn) and made the loopback token **persistent per install**
  (0600 in the app-config dir) so an adopted, already-running sidecar shares the
  secret — the earlier per-launch uuid would 401 the webview against a survivor.
- Frontend: root `load()` now awaits the token before its first request
  (closing a desktop 401 race); `get_api_token` failures log loudly instead of
  degrading silently.
Verified false positive: `stop_api`'s tokio `kill().await` already reaps (unlike
std), so no zombie.

### 2026-07-01 - Overnight autonomous session: deploy path + monetization core

Worked as autonomous technical co-founder toward the north star (deploy + first
revenue). 13 gated commits on `main`. Every commit passed `./tools/check.sh`
(or `cargo check` for the Rust); no founder-gated line was crossed.

**Strategy (two new docs).**
- `docs/product/ship-to-revenue.md` - the deploy + monetization plan (Obsidian
  model, offline Ed25519 licensing rule, prioritized deployment-gap table).
- `docs/product/strategic-review-2026-07-01.md` - **NEEDS A FOUNDER DECISION.**
  A sharpen pass + market research found the ship-to-revenue lead option
  ($59 generic private assistant via Show HN) is the *weakest* credible wedge;
  a narrow compliance-driven vertical ICP (e.g. therapists' private session
  notes) is the most credible. This changes GTM, not the engine, so the build
  below serves any wedge. Decision framed there as options A/B/C; my read is B or C.

**The deploy blocker is broken (gap 1).** Previously the packaged app couldn't
start the brain at all. Now, end to end:
- Spike proved PyInstaller freezes `june-api` into a relocatable ~40MB bundle
  with `sqlite-vec` working (`docs/product/sidecar-spike-findings.md`, spec at
  `tools/packaging/june-api.spec`).
- Rust supervision wired (`apps/desktop/src-tauri/src/sidecar.rs` + `lib.rs`):
  spawn on setup, 60s health-wait on `/system`, kill on exit; `cargo check` clean.
- Build pipeline (`tools/packaging/build-sidecar.sh` + `tauri.conf.json`
  resources) stages the frozen brain to `Resources/june-api/` and bakes the
  build SHA. Verified: staged binary serves `/system` 200.
- Remaining: Apple Developer cert + notarization (FOUNDER), a `tauri build` +
  GUI launch test (human), and frozen-skills launch (Ship-3, task #43).

**First-run honesty.** FR-1: a hard turn no longer hard-errors when the deep
model (`gemma4:e4b`) isn't pulled - it degrades to the fast tier with visible
provenance. FR-2: onboarding pulls the *right* tag (derived from live status,
was a hardcoded mismatch) + the embedder, no terminal.

**Clean-install hardening.** Declared `openai` (a real but undeclared brain dep
that broke clean installs / the freeze), then `pydantic`/`httpx` (brain) and
`starlette` (api) after an audit - the dep graph no longer leans on transitive
closure. Audit's one remaining functional gap: skills don't launch in the frozen
bundle (Ship-3, #43).

**Monetization primitive.** `docs/product/license-design.md` + Slice 1: the
offline Ed25519 license verification core (`june_brain/licensing/`). Signature
verified before any field is read; every failure (missing/tampered/expired/
unknown key) degrades to FREE; no phone-home. `PUBLIC_KEYS` ships empty, so
everything is FREE until the founder adds the production key. Slices 2-3 (API
surface, gate wiring) deferred until distribution exists.

**Product.** Track 3 (durable promises) complete: 3.2b surfaces retry attempts
and a distinct "needs a new approach" exhausted state in the Promises UI.
Track 1.3: added the deterministic promise-nudge Silence producer (surfaces
awaiting-user promises through the existing decide() policy; no nagging).
Contradiction producer split out as a follow-on (#45).

**Needs the founder (batched, none blocking the above):** the wedge decision
(strategic-review doc); Apple Developer enrollment ($99/yr) for signing; a
payment processor + price points + OSS license choice; whether to operate cloud
sync/relay at all. See ship-to-revenue.md "Human-gated decisions".

**Queued next:** Ship-3 (frozen skills), the contradiction producer (#45),
loopback API auth token (gap 5), Track 4 (skill permission hardening).

### 2026-06-29 - Silence/Trust UI (slices 9-10), from Claude Design

Pushed on `main`, in five gated UI slices (A-E). Implements the handoff's
slices 9-10 against the approved Claude Design project ("JuneAI", file
`June.html`), imported via the design MCP. The React/Babel prototype was
re-implemented faithfully in the SvelteKit app (the app's CSS-variable tokens
mirror the design's palette 1:1).

- Client: getLedger / verifyLedger / getSurfacing / sendSurfacingFeedback /
  getHoldings added to `@june/ui` with type re-exports.
- Shared trust primitives in `@june/ui`: ReasonChip, EgressLine,
  VerifyAffordance (controlled unknown/verified/verifying/tampered).
- `/system/receipts` — the Trust Ledger made human: VerifyAffordance, kind
  filter, receipt cards driven by the redacted payload ('left the device' vs
  'stayed local', legible chain footer), broken-entry highlight, cursor paging.
- `/system/silence` — the decision inspector: surfaced/batched/suppressed with
  truthful reasons + feature chips, suppressed-but-visible, "was this the right
  call?" feedback wired to the API, and a held digest.
- `/` is now the control-room Home (consuming /home/{user}/holdings): summary,
  what-you're-holding promises, held digest, right rail (next deadline,
  blocked-by-local-only, egress card, chain-verified line), and a calm
  "you're clear" state. Chat moved to `/chat`; nav gains Home + Chat; the Trust
  page gains Receipts + Silence entry cards.

IA decisions (confirmed with the user): Home at `/` with chat at `/chat`;
Receipts/Silence as sub-routes under Trust. Out of scope: the design's
SurfacingScreen (a static intrusiveness-tiers explainer, not data-wired).
Validation: `./tools/check.sh` green at each slice (763 backend tests; 251 web
files, 0 svelte-check errors).

### 2026-06-29 - Silence Model + Trust Ledger (backend, slices 1-8)

Pushed on `main`. Implements the two flagship differentiators from the senior
review per `docs/product/CLAUDE_HANDOFF_silence_and_trust.md`, in eight gated
slices (each `./tools/check.sh` green, one commit, pushed). 761 backend tests.

- ADRs 0022 (Trust Ledger) and 0023 (Silence Model) written and indexed; the
  handoff saved into the repo as the anchoring spec.
- Trust Ledger (`packages/brain/src/june_brain/trust/`): append-only,
  hash-chained (blake2b/32, 64-zero genesis), device-global provenance in the
  single june.db. Seq assigned under a process lock so racing appends cannot
  fork the chain. Payloads redacted in the writer (no secret can enter).
  `verify_chain` reports the first broken seq. Optional Ed25519 signing via
  PyNaCl (lazy; added as the crypto-exception dep); unsigned is fully valid.
- Ledger wired centrally and unbypassably: one egress entry per cloud call
  (provenance `start`), an action entry per gated action that executes at the
  guard dispatch chokepoint, and approval entries (June asks / user grants).
  Export carries the full ledger. New autouse brain-test isolation keeps every
  test off the user's real data dir now that more seams write to june.db.
- Ledger API: `GET /system/ledger`, `POST /system/ledger/verify`, and a
  `ledger_summary` on `/system` (count, last entry, egress-today, last
  verification — cached so the status badge never pays an O(n) re-hash).
- Silence Model (`packages/brain/src/june_brain/silence/`): a pure, local,
  rules-first `decide()` over June-INITIATED candidates only -> now | batch |
  suppress + a truthful reason + features. No model, no egress, no clock
  (`now` injected). A structural test proves the reply path never imports it
  (June always answers when spoken to).
- Silence storage: `surfacing_decisions` + batched digest drained only by an
  explicit event-boundary function (no timer; tokenize-based invariant test).
  Each decision mirrored into the ledger. `GET /system/surfacing` +
  `POST /system/surfacing/{id}/feedback` (verdict maps honestly to the
  dismissal signal).
- `GET /home/{user_id}/holdings`: read-only aggregation of open promises,
  the held digest, and ledger egress/chain status. No new state.
- Out of scope this round (future): the belief/contradiction engine,
  presence-triggered consolidation, local preference learning, Silence v2.
  Slices 9-10 (the `/system` Trust + control-room home UI) are deferred
  pending `docs/product/CLAUDE_DESIGN_BRIEF_silence_and_trust.md`.

### 2026-06-28 - Reversible Forget Complete + Review Hardening

Pushed on `main`.

- Reversible forget now covers EVERY memory type. Structured rows (goals,
  open_loops, calendar, journal, body_metrics) archive to `forgotten_structured`
  before deletion and restore by replaying `mgr.write` (recreating the row and
  its vector paraphrase via tested code). `MemoryManager.list_forgotten/restore/
  purge_forgotten` span all three stores by ref prefix. Verified live (goal
  forget -> trash -> restore -> back in snapshot).
- Review hardening (from a reviewer subagent pass over the session's diff):
  - Restore is atomic (single transaction) for facts and entities; a crash can
    no longer leave a memory in both the live store and the trash.
  - Restore preserves the original `created_at`/`updated_at`, so a restored
    memory no longer jumps to the top of recency-ordered views.
  - Forget made atomic too, for lifecycle symmetry.
  - body_metric restore preserved (it would have landed on today and could
    overwrite a real metric); threaded an optional date through.

Validation: full gate green, 699 backend tests; live smoke tests of approval,
deadline, and structured forget/restore against the running API.

### 2026-06-28 - Autonomous Tranche (deadlines, trash lifecycle, skill scopes)

Pushed on `main`, continuing the continuity engine autonomously.

- Reversible forget extended to graph entities; trash surface generalized to
  ref-based (`ForgottenMemory{ref,kind,text,...}`, restore by ref). Facts +
  entities are now both reversible. Structured rows remain a documented follow-up.
- Promise deadlines: `due_at` persisted (additive migration), set on create and
  via PATCH (empty clears, null leaves unchanged), surfaced on the Promises page
  (urgency badge + inline editor) and the home continuity summary (overdue /
  due-soon counts, metric, status dot). Due state derived at read time — no
  timer (ADR 0016).
- Trash lifecycle completed: `DELETE /memory/{user}/forgotten` purges the trash;
  Memory page gained an "Empty trash" control. Reversible until explicitly
  emptied.
- Skill capability scopes: each skill shows plain-language scopes ("sends data
  off device", "runs code", ...) derived from its tools' guard action classes
  (ADR 0021) — honest, not a manifest claim; network/execute highlighted.

Validation: full gate green after each slice (688 backend tests at the last
slice); OpenAPI client regenerated where schemas changed. Live API smoke-tested
end-to-end (deadline create/clear, approve+activity record).

### 2026-06-28 - Reversible Forget (Memory Governance)

Pushed on `main`. Closed a stated non-negotiable the code violated: forgetting
was a hard delete, but the vision and CLAUDE.md require it to be conservative
and reversible.

- Semantic facts (and raw fact-id forgets) are archived to a `forgotten_facts`
  trash table before they leave the live store, so recall never sees forgotten
  data yet the fact can be restored with its original id. Capture-before-delete
  keeps every read path unchanged — no risk of a forgotten fact leaking back
  into recall.
- `MemoryManager.list_forgotten` / `restore` over new vector-store trash methods.
- `GET /memory/{user}/forgotten` and `POST /memory/{user}/forgotten/{id}/restore`.
- Memory page shows a "Recently forgotten" section with one-click Restore.
- Scope: semantic facts. Structured rows (goals/calendar/journal) and graph
  nodes still hard-delete; extending reversibility to them is the next slice.

Validation:

- Focused tests: vector store, memory manager (forget/restore cycle), memory
  routes (trash list + restore).
- Full gate: `./tools/check.sh` green; `679` backend tests, frontend checks,
  OpenAPI drift, Ruff, mypy.

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
- Both approval moments persist to the activity log as `kind=approval`: a chat
  block records the withheld action (`decision=requested`), the promise approve
  endpoint records the grant (`decision=approved`). Best-effort; never breaks the
  stream or approve path. Filterable via `/system/activity?kind=approval`.
- Trust renders approval records with an accent border and bold kind badge so
  trust-critical "June asked / you decided" moments stand out from routine rows.

Validation:

- Focused tests: loop egress/guard, chat stream, tasks runtime/store/routes,
  approval activity records.
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
