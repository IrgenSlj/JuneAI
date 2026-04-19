# Architecture Overview

This document describes how June is built. For the rationale behind each choice, see the Architecture Decision Records under `docs/decisions/`.

## Layered View

June is organized in four horizontal layers, each with a single responsibility:

```
┌───────────────────────────────────────────────────────────────┐
│  SHELLS       Tauri (macOS)   Capacitor (iOS)   PWA (Web)     │
├───────────────────────────────────────────────────────────────┤
│  UI           SvelteKit app + shared TypeScript components    │
├───────────────────────────────────────────────────────────────┤
│  API          FastAPI · REST + SSE streaming                  │
├───────────────────────────────────────────────────────────────┤
│  BRAIN        LangGraph agent · memory · skills loader        │
├───────────────────────────────────────────────────────────────┤
│  PROVIDERS    Ollama/Gemma 4   Gemini API                     │
└───────────────────────────────────────────────────────────────┘
                              ↑
                              │
                    ┌─────────┴─────────┐
                    │  SKILLS (MCP)     │
                    │  calendar, health,│
                    │  research, files, │
                    │  daily            │
                    └───────────────────┘
```

A layer only calls into the layer directly below it. Shells consume the UI. The UI consumes the API. The API consumes the Brain. The Brain consumes the Providers and the Skills. No layer reaches across another.

## The Brain

`packages/brain/` — Python, installable as `june-brain`.

The brain is the intelligence. Anything that is model-facing or memory-facing lives here. The brain is designed to be usable without the API — a Python developer can `pip install june-brain` and embed June in their own system.

Internal modules:

- **`agent/`** — LangGraph state machine, routing, streaming orchestration.
- **`memory/`** — three-store memory (see ADR 0004): `sqlite.py`, `vector.py`, `graph.py`, with `manager.py` as the unified facade.
- **`runtime/`** — model provider implementations: `gemma.py`, `gemini.py`, plus `router.py` for local-first-with-cloud-escape-valve routing.
- **`skills/`** — MCP client that discovers and connects to skill servers.
- **`patterns/`** — chapter and pattern detection, injected into system prompts.
- **`telemetry/`** — structured event logging, stays local.

The brain exposes one primary class: `JuneAgent`. Construction takes a `MemoryManager`, a `SkillRegistry`, and a `ModelProvider`. Calling `agent.stream(user_id, message)` yields streaming events (tokens, tool calls, memory saves).

## The API

`packages/api/` — Python, FastAPI.

The API is the network boundary. It is deliberately thin: routes wrap `JuneAgent` methods, handle HTTP and SSE concerns, and nothing else. Business logic belongs in the Brain.

Routes:

- **`POST /chat`** — stream an agent turn over SSE. Request: user ID, message. Response: token stream, tool call events, memory events.
- **`GET /memory/{user_id}`** — snapshot across all three stores: structured rows, semantic facts, entities.
- **`DELETE /memory/{user_id}/fact/{ref}`** — fact removal. The `ref` carries a source prefix (`semantic:`, `node:`, `edge:`) so the handler routes to the correct store through `MemoryManager.forget`.
- **`GET /skills`** — discovered skill servers with their tool lists. *(stub until Week 5)*
- **`POST /skills/{name}/enable`** and **`/disable`** — runtime toggling. *(Week 5)*
- **`GET /system`** — model provider status, Ollama health, memory paths.

Manual fact editing (`POST /memory/{user_id}/fact`) is deferred. The memory browser currently supports delete-and-re-learn; a `POST`/`PATCH` surface lands when the UI needs it.

All request and response schemas are defined as Pydantic models in `packages/api/src/june_api/schemas/`. A codegen step (`tools/codegen.sh`) produces TypeScript types under `packages/ui/src/api/types.ts`. The UI never defines its own API types.

## The UI

`apps/web/` + `packages/ui/` — TypeScript, SvelteKit.

The UI is the shared frontend. Every shell serves this same build.

`packages/ui/` holds reusable components and stores:

- **`components/`** — `ChatBubble`, `MessageList`, `Composer`, `MemoryCard`, `SkillToggle`, etc.
- **`stores/`** — Svelte stores for the active conversation, memory index, user settings.
- **`api/`** — typed client generated from the API's Pydantic schemas.

`apps/web/` holds routes and app-specific layout:

- `/` — main chat surface.
- `/memory` — memory browser and editor.
- `/skills` — skill registry.
- `/settings` — model provider, API keys, preferences.

A small capability layer (`packages/ui/src/platform.ts`) exposes platform features (`showNotification`, `registerHotkey`, `readFile`) with implementations that route to Tauri commands, Capacitor plugins, or web APIs depending on the runtime.

## The Shells

Three thin shells wrap the UI:

- **`apps/desktop/`** — Tauri. Rust commands for system tray, global hotkey (⌘⇧J), native notifications, Ollama process supervision, filesystem access, autostart. Ships a macOS `.dmg` (and Windows/Linux for free).
- **`apps/mobile/`** — Capacitor. Swift plugins for iOS push notifications, share extensions, voice input via AVFoundation. Ships an iOS `.ipa` for TestFlight and the App Store.
- **Web** — the SvelteKit PWA served directly. Installable via the browser's native install flow.

Shells do not contain business logic. If a shell needs to do something, there is a Tauri command or a Capacitor plugin, and the UI calls it through the capability layer.

## The Skills

`skills/` — one folder per skill, each a standalone MCP server.

Each skill is a pip-installable Python package that exposes a Model Context Protocol server. The brain's skills loader discovers skills from a manifest file, spawns each server as a child process, and multiplexes tool calls over the MCP stdio transport.

Skills read and write to memory through MCP resources, which are proxied by the brain to its `MemoryManager`. Skills do not talk to SQLite or ChromaDB directly.

## Data Flow: One Turn

```
user types a message in apps/web
        │
        ▼
SvelteKit composer → POST /chat (SSE) ──── @packages/api
        │
        ▼
api.main calls JuneAgent.stream() ──────── @packages/brain
        │
        ├──► MemoryManager.recall(message)
        │        │
        │        ├─► sqlite (structured facts)
        │        ├─► chromadb (semantic recall)
        │        └─► graph (entity relations)
        │        │
        │        ▼
        │    top-K memories
        │
        ▼
LangGraph builds prompt, calls ModelProvider
        │
        ▼
Gemma 4 (Ollama) or Gemini API — streams tokens
        │
        ▼
Tokens stream back to SSE → SvelteKit renders
        │
        ▼
On tool call: brain's skill loader → MCP → skill process
        │                                       │
        │                                       ▼
        │                             skill reads/writes memory
        │                                       │
        │                                       ▼
        ◄── tool result ───────────────────────────
        │
        ▼
Model continues until done
        │
        ▼
Post-turn: MemoryManager.extract(conversation)
        │
        ├─► new structured facts → sqlite
        ├─► new embeddings → chromadb
        └─► new entities/edges → graph
```

## Where User Data Lives

- **Database and vector index:** `~/Library/Application Support/June/` on macOS, `~/.local/share/June/` on Linux, `%APPDATA%/June/` on Windows. On iOS, the app's sandboxed container.
- **Logs and telemetry:** `~/Library/Logs/June/` on macOS.
- **Config:** `~/Library/Application Support/June/config.toml`.

The repository never contains user data.
