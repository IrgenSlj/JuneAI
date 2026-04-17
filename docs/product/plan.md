# June v2 — Canonical 8-Week Plan

This document is the single source of truth for June's development. It replaces all prior planning documents (`docs/PLAN.md`, `docs/NEXT_SESSION.md`, `docs/product/roadmap.md`, `docs/product/next-sessions.md`, `JuneAI-app/docs/*`).

Each section below corresponds to one week (≈ one focused work session). Every week has: a goal, exact deliverables, exit criteria, and references.

## Before You Start

Read in this order:

1. [`docs/vision.md`](../vision.md) — the three non-negotiables
2. [`docs/architecture/overview.md`](../architecture/overview.md) — the layered model
3. [`docs/decisions/README.md`](../decisions/README.md) — the six ADRs that justify this plan

Then pick up the current week below.

## Current State

- v1 Streamlit app is functional on `main`. It is being retired, not improved.
- v1 brain modules (`JuneAI-app/src/agent/*`) are preserved and will be migrated.
- v1 user data in `JuneAI-app/.june_memory/` must be migrated to the new platform-appropriate location during Week 1.

## Week 1 — Foundation

**Goal:** Establish the monorepo skeleton. Move the brain. Cut model presets. Delete Streamlit.

**Deliverables:**

- Repo reshaped to the monorepo layout described in [ADR 0001](../decisions/0001-monorepo-structure.md).
- `packages/brain/` exists with the v1 `src/agent/*` code moved and tests updated. No behavior changes in this step — just relocation.
- `memory.py` (1,563 lines) split into `packages/brain/src/june_brain/memory/{sqlite,vector,graph,manager}.py`. `vector.py` and `graph.py` start as stubs that will fill in during Week 4.
- Model presets cut to Gemma 4 and Gemini only, per [ADR 0002](../decisions/0002-gemma-gemini-only.md). Delete Llama, Mistral, and Claude presets including the shared `LOCAL_LARGE_MODEL_NAME` override, the Anthropic provider branch, and the `claude_high` preset.
- `JuneAI-app/app.py` and `JuneAI-app/src/agent_ui/` deleted.
- v1 preserved on a `legacy/streamlit` branch for historical reference.
- User data migration script: move `JuneAI-app/.june_memory/june.db` to `~/Library/Application Support/June/june.db`.
- `pnpm-workspace.yaml` and root `pyproject.toml` (with uv workspaces) configured.
- `tools/dev.sh` one-command developer startup (spins up Ollama check + brain tests).

**Exit criteria:**

- All non-UI v1 tests pass against the new `packages/brain/` layout.
- `from june_brain import JuneAgent` works from a fresh Python venv.
- The repo has no Streamlit imports anywhere on `main`.

**Risks:**

- Test failures during the move. Mitigated by moving files first, updating imports, running tests, then splitting `memory.py` as a second commit.
- Losing user data. Mitigated by migration script that copies (not moves) first, verifies, then marks v1 location as archived.

## Week 2 — API Layer

**Goal:** Build the FastAPI boundary. Ship the first working `/chat` endpoint with streaming.

**Deliverables:**

- `packages/api/` scaffolded with FastAPI and uvicorn.
- Routes: `POST /chat` (SSE streaming), `GET /memory/{user_id}`, `GET /skills`, `GET /system`.
- Pydantic schemas under `packages/api/src/june_api/schemas/` covering all request and response types.
- `tools/codegen.sh` runs `openapi-typescript-codegen` (or equivalent) to emit `packages/ui/src/api/types.ts`.
- Integration test: start the API, POST a message to `/chat`, assert a streaming response from Gemma 4 via Ollama.
- First written decision record that isn't in the initial batch: `docs/decisions/0007-sse-over-websockets.md` with the rationale for SSE.

**Exit criteria:**

- `curl -N http://localhost:8000/chat -d '{"user_id":"me","message":"hi"}'` streams tokens from Gemma 4.
- `GET /system` reports Ollama status and active model.
- Generated TypeScript types compile.

**Risks:**

- SSE behavior across proxies. Mitigated by explicit `Content-Type: text/event-stream` and flush handling.
- LangGraph streaming shape changing under the hood. Mitigated by pinning LangGraph version.

## Week 3 — Web UI Skeleton

**Goal:** Ship the first working browser app. Round-trip a message from a real frontend.

**Deliverables:**

- `apps/web/` scaffolded with SvelteKit + TypeScript + Tailwind (or tokens from `packages/design/`).
- `packages/ui/` with the first three components: `ChatBubble`, `MessageList`, `Composer`.
- `packages/ui/src/api/client.ts` using the generated types.
- Chat route working: type a message, see tokens stream back.
- Deployed to Vercel or Netlify as a public preview URL.
- Design tokens in `packages/design/src/tokens.ts`: color palette, typography, spacing.

**Exit criteria:**

- A user can open the preview URL, paste a Gemini API key, send a message, and see a streamed reply.
- If Ollama is running on localhost, Gemma 4 is the default.
- Lighthouse PWA score above 90.

**Risks:**

- CORS between Vercel-hosted frontend and a local API. Mitigated by deploying the API alongside (Fly.io, Render, or Modal) or by running both locally for development.
- SvelteKit's SSR interacting with SSE. Mitigated by rendering chat routes client-only.

## Week 4 — Memory Excellence

**Goal:** Implement the three-store memory described in [ADR 0004](../decisions/0004-memory-architecture.md). Ship the recall-extract loop.

**Deliverables:**

- `packages/brain/src/june_brain/memory/vector.py` wired to ChromaDB (embedded mode).
- `packages/brain/src/june_brain/memory/graph.py` with nodes/edges tables and basic traversal.
- `MemoryManager.recall(user_id, message, k=5)` returns ranked facts from all three stores.
- `MemoryManager.extract(user_id, exchange)` pulls new facts and writes to all three stores.
- Pre-turn hook in the LangGraph agent injects recall results into the system prompt.
- Post-turn hook runs extraction on a background task.
- `/memory` route in the SvelteKit app: timeline view, search, fact editor.
- Local sentence-transformer embedder (start with `all-MiniLM-L6-v2`) bundled via `sentence-transformers`.

**Exit criteria:**

- Ask June "what did I tell you about X" two turns after mentioning X — correct recall.
- The memory browser shows conversations, facts, and entities.
- The user can edit or delete a fact and it disappears from recall on the next turn.

**Risks:**

- Embedding model download on first run is large. Mitigated by a one-time setup prompt and cache.
- Extraction quality depends on prompt engineering. Mitigated by keeping the extractor prompt in a dedicated file (`packages/brain/.../memory/extractor_prompt.txt`) that can be iterated on.

## Week 5 — Skills as MCP

**Goal:** Decompose `tools.py` into five MCP skill servers. Wire them through the brain.

**Deliverables:**

- `skills/calendar/`, `skills/health/`, `skills/research/`, `skills/files/`, `skills/daily/`, each with `pyproject.toml` and an MCP server entry point.
- `packages/brain/src/june_brain/skills/loader.py` — MCP client that spawns skill processes, multiplexes tool calls, handles reconnection on crash.
- Skill manifest at `~/Library/Application Support/June/skills.toml` — lists installed skills and enable/disable state.
- `/skills` route in the SvelteKit app: list skills, toggle on/off.
- First external skill: `skills/research/` with a web search tool powered by Brave Search's free API (or SearXNG if self-hosted).
- `skills/files/` reads local PDFs and webpages (via `trafilatura` or similar).

**Exit criteria:**

- June can answer "what's the weather in Lisbon today" using the research skill.
- Disabling a skill immediately removes its tools from the agent.
- Skill crashes do not take down the agent (supervisor restarts the skill).

**Risks:**

- MCP spec changes. Mitigated by pinning protocol version.
- Long skill startup times. Mitigated by eager start on app launch, not on first tool call.

## Week 6 — Mac Desktop App

**Goal:** Ship a downloadable `.dmg` that gives June a real home on macOS.

**Deliverables:**

- `apps/desktop/` scaffolded with Tauri 2.x.
- Tauri commands: system tray icon, global hotkey (⌘⇧J to summon), native macOS notifications, Ollama lifecycle management (start on app launch, stop on quit).
- Autostart on login.
- Window chrome: frameless, acrylic background, traffic lights only.
- First-run setup: detect Ollama, offer to install if missing, pull Gemma 4 with progress.
- CI produces a signed (ad-hoc for now) `.dmg` on every release tag.

**Exit criteria:**

- Fresh Mac: download `.dmg`, double-click, drag to Applications, open — June is running.
- ⌘⇧J from any app brings June to the front.
- Quitting the app cleanly shuts Ollama down.

**Risks:**

- Notarization and App Store requirements. Deferred to Week 8; Week 6 produces an ad-hoc signed build.
- Tauri webview inconsistencies. Mitigated by testing on macOS 13+ only.

## Week 7 — iPhone App

**Goal:** Ship a TestFlight build. Real iOS shell, not a PWA.

**Deliverables:**

- `apps/mobile/` scaffolded with Capacitor 6.
- iOS-specific plugins: push notifications (local scheduler for proactive June messages), share extension (share any webpage to June), voice input via the Web Speech API with a Whisper fallback via a Capacitor plugin.
- Mobile-specific layout tweaks in `packages/ui/`: larger touch targets, bottom composer, swipe navigation.
- App icon, launch screen, and screenshots prepared.
- First TestFlight build submitted.

**Exit criteria:**

- A user can install June from TestFlight, paste a Gemini API key, chat, and receive a proactive notification.
- Sharing a URL from Safari to June opens the app with that URL in the composer.

**Risks:**

- Apple developer account and review process. Mitigated by starting this ahead of Week 7 if possible.
- Push notifications require backend infrastructure (APNs). Simplified in Week 7 to local notifications only; server-initiated push deferred.

## Week 8 — Polish and Launch

**Goal:** Public release. README, docs site, App Store submissions, announcement.

**Deliverables:**

- Performance pass: streaming latency, memory recall speed, first-paint on each shell.
- Onboarding flow: "Connect your local Gemma 4 or paste a Gemini key" — three-step wizard that works on every shell.
- Docs site under `docs/site/` using VitePress, published to `june.ai` (or similar) via GitHub Pages.
- README rewritten for a public audience.
- Screenshots, screen recording, demo video.
- Mac App Store submission (notarized build).
- iOS App Store submission (from TestFlight).
- Public GitHub announcement; optional: HN / Product Hunt / r/LocalLLaMA posts.

**Exit criteria:**

- A stranger can land on `june.ai`, understand what June is in 10 seconds, download it, and have it running in 2 minutes.
- Both App Store submissions in review.

**Risks:**

- App Store review delays. Mitigated by submitting early in the week and having the PWA + direct `.dmg` download as the primary distribution until approval.

## What Success Looks Like at Week 8

- June runs locally on Gemma 4 or cloud on Gemini.
- Memory works. Users feel the difference from ChatGPT within three conversations.
- The same codebase powers browser, Mac, and iPhone.
- Skills can be toggled and extended.
- The project is open source under a permissive license with clear contribution guidelines.

## How We Work

- One week at a time. No skipping ahead.
- Each week's exit criteria are binary. A week is not complete until they are met.
- Every architectural decision gets an ADR. If it is worth debating, it is worth recording.
- The vision document (`docs/vision.md`) is the tiebreaker. When in doubt, read it again.

## Environment Reference

Canonical environment variables for v2 are documented in [`docs/setup/environment.md`](../setup/environment.md). The list is short on purpose — see [ADR 0002](../decisions/0002-gemma-gemini-only.md).
