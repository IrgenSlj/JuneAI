# Open Source Readiness Plan

This plan turns June from a promising local alpha into a public alpha that
people can clone, run, trust, and contribute to. It was written after the
2026-05-12 repository review and is the active hardening track until the
release gates below are met.

The immediate goal is not to add another surface. The goal is to make the
current product promise true:

- The provider shown in the UI is the provider used by chat.
- Chat has enough conversation continuity for normal follow-ups.
- Memory edits and deletes remove stale recall copies from every store.
- A fresh clone has one obvious path to a working app.
- Local safety boundaries are explicit and enforced in code.
- CI catches the regressions that would embarrass a public alpha.

## Release Bar

June is ready to present as a useful open-source alpha when all of these are
true:

1. A fresh clone can run the web app with either Gemma/Ollama or Gemini without
   relying on hidden local state.
2. `/setup`, `/settings`, `/system`, and `/chat` agree on the active provider,
   model, key state, and privacy label.
3. A user can ask a follow-up question and June receives the relevant recent
   conversation context.
4. Deleting or editing a memory removes or updates every copy that can feed
   recall.
5. The local API has a basic same-machine authorization boundary, not just CORS
   and a localhost bind.
6. Python tests, frontend checks, backend lint/type gates, OpenAPI codegen, and
   desktop compilation run in CI.
7. The README is honest about what is shipped, what is experimental, and what
   a user should expect on first run.

## Non-Goals

These stay out of the readiness sprint:

- Cloud sync, accounts, teams, subscriptions, or hosted service work.
- Mobile app implementation.
- Skill marketplace.
- New model providers.
- Polished signed desktop distribution before the desktop code compiles in CI.

## Phase 0 - Product Correctness Blockers

Ship these before a public announcement. They protect the core promise.

### 0.1 Provider and Agent Lifecycle

Problem: stored config is applied after route imports, while the brain builds a
global agent at import time. `/setup/apply` verifies the requested provider, but
the cached chat agent can keep using the previous provider.

Implementation:

- Apply stored config before importing anything that can import
  `june_brain.graph`, or move graph import behind a lazy app startup path.
- Replace the import-time `june_agent` with a small agent registry that builds
  lazily from the current runtime config.
- Track a runtime fingerprint made from provider, model, base URL, and secret
  presence. Rebuild the agent when the fingerprint changes.
- After `/setup/apply` succeeds, reload the chat agent before returning success.
- After `/settings/forget-key`, clear the environment and reload or invalidate
  the agent so a stale Gemini client cannot continue serving chat.
- Make `/system` and `/chat` read from the same resolved runtime snapshot.

Acceptance tests:

- Starting with stored Gemini config builds a Gemini chat agent on the first
  request.
- Switching from Gemma to Gemini updates the next `/chat` request without
  restarting the API process.
- Forgetting the Gemini key makes Gemini chat unavailable until a key is added
  again.
- `/setup/status`, `/settings`, `/system`, and `/chat` report/use the same
  provider after every switch.

Suggested files:

- `packages/api/src/june_api/app.py`
- `packages/api/src/june_api/routes/setup.py`
- `packages/api/src/june_api/routes/settings.py`
- `packages/api/src/june_api/routes/chat.py`
- `packages/brain/src/june_brain/graph.py`

### 0.2 Conversation Continuity

Problem: the web client sends only the latest user message, and the API builds
agent state from only that one message. The SQLite chat history API exists but
is not used by production chat.

Implementation:

- Save the user message at the start of a successful `/chat` turn.
- Save the assistant message after the stream completes. If the stream is
  canceled, save a partial assistant message only if it is useful to future
  context; otherwise mark the turn as canceled and skip it.
- Seed the agent state with recent persisted messages plus the current user
  message. Keep the existing graph trimming logic so the context window stays
  bounded.
- Add stable per-message IDs if needed to avoid duplicate saves on retries or
  reconnects.
- Keep semantic extraction separate from transcript persistence. Transcript
  continuity should work even if memory extraction fails.

Acceptance tests:

- Two sequential `/chat` requests for the same user pass the first turn into
  the second agent invocation.
- `GET /memory/{user_id}` increments `recent_messages` after a completed chat.
- Regenerating a response does not duplicate old user turns in the persisted
  transcript.
- Canceling a stream leaves the system in a coherent state.

Suggested files:

- `packages/api/src/june_api/routes/chat.py`
- `packages/api/src/june_api/schemas/chat.py`
- `packages/brain/src/june_brain/memory/sqlite.py`
- `apps/web/src/lib/stores/chat.svelte.ts`

### 0.3 Memory Delete/Edit Correctness Across Stores

Problem: structured writes create semantic vector paraphrases, but structured
deletes and edits only touch SQLite. Stale vector facts can keep surfacing after
the user has forgotten or edited the structured memory.

Implementation:

- Give every structured paraphrase a deterministic vector identifier, or store a
  reliable `metadata.ref` lookup and add a delete-by-ref path.
- On structured delete, remove both the SQLite row and any semantic paraphrase
  that points at the same ref.
- On structured update, update the SQLite row and upsert the semantic paraphrase
  for the new ref. Remove the old paraphrase when the primary key changes.
- Add a small migration/repair command that can remove orphaned structured
  paraphrases from existing local data.
- Make `MemoryManager.forget()` return enough detail for API logs and tests:
  which stores were touched and whether stale vector rows were removed.

Acceptance tests:

- Writing then deleting `goal:Ship June` leaves no vector fact containing
  `Ship June`.
- Renaming a goal removes the old vector paraphrase and creates the new one.
- Calendar edits that change title/date/time return the new ref and remove the
  old paraphrase.
- Recall does not surface deleted structured facts.

Suggested files:

- `packages/brain/src/june_brain/memory/manager.py`
- `packages/brain/src/june_brain/memory/vector.py`
- `packages/api/src/june_api/routes/memory.py`
- `packages/brain/tests/unit_tests/test_memory_manager.py`
- `packages/api/tests/test_memory_routes.py`

### 0.4 Product-Correctness Regression Suite

Add focused tests that fail before the fixes above and pass after them.

Required tests:

- Provider switching reloads the chat agent.
- Key forgetting invalidates stale cloud clients.
- Chat follow-up context includes prior turns.
- Structured delete/edit removes stale semantic paraphrases.
- `/system` and `/settings` agree after setup changes.

## Phase 1 - Fresh Clone and CI Reliability

This phase makes the contributor path boring and repeatable.

### 1.1 Python Version and Dependencies

Problem: the package claims Python 3.10 support, but the TOML fallback imports
`tomli` without declaring it.

Choose one:

- Support Python 3.10: add `tomli` for Python `<3.11` and test Python 3.10 in
  CI.
- Require Python 3.11+: update every README, setup doc, CI job, and
  `requires-python`.

Acceptance:

- `./tools/dev.sh` works on the documented Python version from a clean venv.
- CI runs on every supported Python version.

### 1.2 Bootstrap Scripts

Split the current all-in-one developer flow into clearer commands:

- `tools/bootstrap.sh`: create venv, install Python workspace packages, run
  `pnpm install` if needed.
- `tools/check.sh`: run backend tests, frontend checks, codegen diff, and the
  enabled lint/type gates.
- `tools/dev.sh`: keep the current convenience path, but call bootstrap/check
  pieces instead of duplicating logic.

Acceptance:

- A contributor without Ollama can run all automated tests with one documented
  command.
- A contributor with Gemini can complete setup without installing Ollama.
- Failure messages point to the exact missing prerequisite.

### 1.3 Make Quality Gates Real

Problem: `ruff` and `mypy` are configured but not enforced and currently fail.

Implementation:

- Decide the intended gate level:
  - Full gate: fix all current ruff/mypy failures and add CI steps.
  - Transitional gate: scope ruff/mypy to changed or critical modules, document
    excluded debt, and create issues for the rest.
- Add `ruff check`, `mypy`, and generated OpenAPI diff to CI after the policy is
  chosen.
- Fix the PWA Workbox warning by removing or correcting the unmatched
  prerendered glob.

Acceptance:

- `pnpm check`, `pnpm build`, `pytest`, `ruff`, `mypy`, and `tools/codegen.sh`
  have a documented pass/fail policy.
- CI fails when generated API types drift from FastAPI schemas.

### 1.4 Smoke Test the Happy Path

Add a fake-model integration test that exercises:

1. Setup/status.
2. Chat stream.
3. Transcript persistence.
4. Memory extraction or manual write.
5. Memory browser API returns the new data.

This catches the actual user workflow instead of only isolated helpers.

## Phase 2 - Local Safety and Privacy Hardening

June is local-first, but a local API can still be abused by same-machine pages
or overly powerful skills. This phase makes the boundary explicit.

### 2.1 Local API Session Secret

Implementation:

- Generate a random local API token on first backend start.
- Store it in the user data dir with mode `0600`.
- Require the token on state-changing routes and `/chat`.
- Have the web/dev shell read the token through the documented local path or a
  dev-only env var.
- Keep `GET /healthz` unauthenticated.

Acceptance:

- A browser page on another origin cannot call `/chat` or mutate memory without
  the token.
- The PWA/dev app still works without manual token copying.
- Docs explain that this is local app authorization, not multi-user auth.

### 2.2 Tighten Demo and Settings Routes

Implementation:

- Disable `/demo/seed` unless `JUNE_DEMO_ROUTES=1`.
- Require the local token for setup/settings/memory/skills routes.
- Add explicit warning text when binding `JUNE_API_HOST` to anything other than
  loopback.

Acceptance:

- Demo data cannot be seeded accidentally in a normal run.
- Same-network exposure requires an explicit user choice and documentation.

### 2.3 Guard Network-Fetching Skills

Implementation:

- Block `file://`, non-HTTP(S), loopback, link-local, private IP ranges, and
  cloud metadata IPs.
- Resolve DNS before fetching and re-check the final redirected host.
- Limit response size before reading into memory.
- Add allowlist overrides only through explicit env vars.

Acceptance:

- Research/files skills cannot fetch localhost services or private network
  resources by default.
- Redirects to blocked hosts are rejected.
- Errors are readable by the model and the user.

### 2.4 Desktop Security Defaults

Implementation:

- Replace `csp: null` with a concrete Tauri CSP that allows the app shell,
  local API, and required assets only.
- Review Tauri capabilities and remove permissions not used by the current UI.
- Add a desktop smoke test in CI before claiming a desktop phase is shipped.

Acceptance:

- Desktop build passes with CSP enabled.
- CI compiles Rust on at least macOS and one Linux target.

## Phase 3 - Downloadable Alpha

This phase turns "clone and run" into "download and try."

### 3.1 Desktop Build CI

Implementation:

- Add GitHub Actions jobs for `pnpm desktop:build` on macOS, Windows, and Linux.
- Upload unsigned artifacts for internal testing first.
- Add a short manual QA checklist for each artifact.
- Only after artifacts work, add signing/notarization and update metadata.

Acceptance:

- Every release candidate has downloadable desktop artifacts.
- The app opens, reaches the local API, can start/pull Ollama where supported,
  and can complete first-run setup.

### 3.2 Honest Public README

Implementation:

- Add screenshots or a short GIF of chat, memory, setup, and skills.
- Add a "Known limitations" section with desktop, security, and model-size
  caveats.
- Add a "Use Gemini only" path for users who do not want to install Ollama.
- Add a "Reset local data" section with platform paths.

Acceptance:

- A non-contributor can tell whether June is worth trying in under two minutes.
- The README does not overstate desktop readiness.

### 3.3 Release Notes and Versioning

Implementation:

- Pick version semantics for alpha releases.
- Add a `CHANGELOG.md`.
- Tag the first public alpha only after Phase 0, Phase 1, and the relevant
  Phase 2 safety items are complete.

Acceptance:

- Users can see what changed and whether they need to migrate/reset data.
- Contributors can target issues to a concrete milestone.

## Phase 4 - Contributor Growth

Do this after the public alpha is stable enough that new contributors are not
only fixing setup.

- Add a skill authoring guide with a minimal MCP skill template.
- Add issue labels for `good first issue`, `help wanted`, `privacy`, `memory`,
  `desktop`, and `docs`.
- Add a maintainer checklist for reviewing privacy boundary changes.
- Add sample data or a guided demo profile that does not require real personal
  memory.
- Add architecture diagrams only where they help contributors avoid mistakes.

## Suggested Milestones

### Milestone A - Public Alpha Gate

Must include:

- Phase 0 complete.
- Python version/dependency policy complete.
- CI policy complete and passing.
- Local API token implemented for chat and mutation routes.
- README updated with limitations.

### Milestone B - Desktop Alpha Gate

Must include:

- Desktop compiles in CI.
- One-click Ollama path verified on macOS.
- Tauri CSP enabled.
- Unsigned artifacts available from GitHub Actions.
- Desktop docs updated from "experimental source" to "testable artifact."

### Milestone C - Contributor Gate

Must include:

- Skill authoring docs.
- Good-first-issue backlog.
- Stable bootstrap/check scripts.
- Public issue templates reflect current priorities.

## Tracking Notes

When an item in this plan ships, update this file and the roadmap in the same
pull request. If an implementation changes a privacy boundary, update
`SECURITY.md` and the relevant architecture decision or write a new ADR.
