# Vision

June is the open personal AI that remembers you. It runs privately on your laptop via Gemma 4, scales to the cloud via Gemini, and works identically in your browser, on your Mac, and on your iPhone. Everything is open source. Everything is free.

## Why June Exists

Every AI assistant today has the same failure mode: it forgets you between sessions. You re-explain your goals, your context, your history, your constraints, over and over. The assistant feels capable but never feels personal.

June is built on a different premise: the assistant that remembers you is more valuable than the assistant with the biggest model. Memory is the product. The model is infrastructure.

## The Three Non-Negotiables

Every feature, decision, and dependency is measured against these three principles. If a request cannot be justified by at least one of them, the answer is no.

### 1. Memory is the product

Every conversation feeds a personal knowledge graph that is yours, editable, portable, and local-first. June recalls relevant memories before every response and extracts new facts after every response. The memory layer is inspectable, exportable, and never leaves your machine without your consent.

### 2. Local-first, cloud as escape valve

Gemma 4 on Ollama handles the daily conversational load. Gemini Flash is the optional turbo for long-context work, vision, and reasoning-heavy asks. Users can run June fully offline. Users who prefer cloud can paste a Gemini API key and never install a local model. No third model. No fallback chain.

### 3. One codebase, every surface

Browser, desktop, and mobile share the same frontend code. The same brain, the same memory, the same API. New features land in one place and appear everywhere. This is how a small team ships a multi-platform product.

## What June Is Not

- **Not a chatbot.** June is a personal assistant with persistent identity and context.
- **Not a wrapper around a single model.** The model layer is swappable; the memory and skills layer is the moat.
- **Not a commercial SaaS.** June is open source under a permissive license. No account. No cloud dependency. No telemetry that leaves the device without consent.
- **Not a research project.** June ships. Each week produces something a user can actually use.

## North Star User Experience

A user opens June on their Mac at 8am. June greets them by name, reminds them that yesterday they mentioned an upcoming gym session at 8, and asks how it went. They answer. June logs the workout, updates their streak, notices they missed water logging yesterday, and gently surfaces it. They ask June to draft a message to a collaborator. June knows who that collaborator is, how they communicate, and what the current project context is. The draft is good on the first try.

The same user opens June on their iPhone at lunch. Same memory. Same context. Same conversation, if they want.

Later they open the browser app on a shared machine, log in to nothing, paste their Gemini key, and continue where they left off.

Nothing in that experience requires an internet connection except optional Gemini calls. Nothing is synced to a vendor cloud. Everything is June's — and therefore theirs.

## How This Document Is Used

This vision governs architecture decisions and product scope. When in doubt, open this file. When the answer is still unclear, write an Architecture Decision Record under `docs/decisions/`.
