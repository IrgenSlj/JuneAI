# June 1.0

June is the open personal AI that remembers you. It runs privately on your laptop via Gemma 4, reaches the cloud via Gemini when you ask it to, and works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

This document describes what June is. For why it exists, read [vision.md](../vision.md). For how it is built, read [architecture/overview.md](../architecture/overview.md). For where it is going next, read [roadmap.md](roadmap.md).

## What June Is

A single, persistent assistant with three surfaces and one memory.

- **Chat** — a fluent conversation that remembers. Every turn recalls relevant memories before responding and extracts new facts afterwards. The assistant feels like it knows you because it does.
- **Memory** — an inspectable, editable, exportable record of everything June has learned. Three stores work together: structured facts in SQLite, semantic recall in ChromaDB, entities and relationships in a graph. Users see what June knows, correct mistakes, and delete anything at any time.
- **Skills** — capabilities the assistant can call. Calendar, health, research, files, daily. Each skill is a standalone MCP server, independently enabled, versioned, and swappable. Third parties can ship skills as pip packages.

## The Product in One Turn

A user opens June. The assistant greets them by name, references something real from the last conversation, and asks a specific question. The user answers in natural language. June files the answer into memory, calls a skill if needed (log the workout, draft the message, check the calendar), streams a response token by token, and updates its understanding of the user. The whole turn happens in one screen, on one device, with no login, no cloud round-trip unless the user opted into Gemini, and no data leaving the machine.

The user closes the laptop and opens their phone. Same assistant. Same memory. Same conversation if they want.

## The Product Surface

### Primary screen: Chat

One column. Message list above, composer below, model and provider status in the header. Streaming responses token by token. Tool calls render inline with their arguments and results. The composer supports cancellation mid-stream. No sidebars, no tabs, no modals that break the conversation.

### Memory browser

Three sections: structured facts, semantic memories, entities and relationships. Each memory is a card with its source, timestamp, and a delete button. The user can scan their own history at a glance, search it, and remove anything that is wrong or out of date. Manual add and edit are deferred; delete-and-re-learn is the workflow until the UI demands more.

### Skills registry

One card per installed skill. Each card shows the skill name, a description, a running/stopped/crashed status badge, the list of tools it exposes, and an enable/disable toggle. Toggling hot-reloads the agent so the next turn sees the new tool list.

### System header

Model provider, active model, Ollama reachability, and a one-word privacy label (`local-only` or `cloud-opt-in`). Visible on every screen so the user always knows where their turn is running.

## The Product Boundary

- **No account.** June is installed, not subscribed to. There is no signup, no login, no cloud sync by default.
- **No telemetry without consent.** The brain never reports back unless the user opts in.
- **No third model.** Gemma 4 for local, Gemini for cloud. Any new provider must replace one of these, not add to them.
- **No shell-specific business logic.** Desktop and mobile shells are capability wrappers. The same UI runs in all three.
- **No cloud-only features.** If a feature cannot work without an internet connection, it does not ship.

## Model Routing

Gemma 4 via Ollama is the default. It handles the daily conversational load at zero marginal cost. The user can paste a Gemini API key and switch the active provider with one setting. Both can be configured at once; only the active provider is called.

There is no fallback chain. The active provider is the active provider. If Ollama is offline, June surfaces that in the header and the user either starts Ollama or switches to Gemini.

## Memory Model

Three stores, one facade. The `MemoryManager` is the only way into memory:

- **SQLite** stores deterministic rows: user profile fields, preferences, habits, daily chapters, workouts.
- **ChromaDB** stores semantic chunks: conversational context, journal entries, arbitrary facts that need recall-by-meaning.
- **Graph** stores entities (people, places, projects, things) and the relationships between them.

Before every turn, the manager recalls from all three stores based on the incoming message. After every turn, it extracts new facts and writes them back. Skills read and write memory through MCP resources proxied by the brain. No skill opens its own connection.

## Skills Model

Each skill is a standalone Python package with a `python -m june_skill_<name>` entrypoint. A supervisor in the brain starts each enabled skill as a subprocess, negotiates capabilities over MCP stdio, bridges each skill's tools into LangChain `StructuredTool` instances, and restarts on crash. A manifest under the user's data directory records which skills are enabled.

A skill subprocess that re-imports the brain is a fork bomb. The supervisor sets `JUNE_IS_SKILL_SUBPROCESS=1` and `JUNE_SKILLS_DISABLED=1` in the child environment; the brain's graph module skips agent construction under those flags. This is documented in [ADR 0005](../decisions/0005-skills-as-mcp.md).

## Privacy Model

- Conversations, memories, and embeddings live on the user's machine.
- Gemini calls send only the turn's context to Google; nothing persists there on June's behalf.
- No analytics, crash reporting, or usage metrics leave the device without an explicit opt-in.
- Export is one command; delete is one button.

## Status

June 1.0 ships as a web application. The brain, the API, the memory stores, and the skills system are complete. The shared UI renders the chat, memory, and skills surfaces. Light mode is the default with dark mode toggle available.

The desktop shell is in progress. The Ollama capability gap fired the trigger on 2026-04-27; the concrete plan lives in [desktop-shell-plan.md](desktop-shell-plan.md) with touch and tablet hardening detailed in [responsive-plan.md](responsive-plan.md).

See [roadmap.md](roadmap.md) for what ships next.
