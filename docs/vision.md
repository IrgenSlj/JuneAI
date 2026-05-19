# Vision

June is the open personal agent that remembers you. It is private by default — chat and recall stay on your machine; agentic capability reaches Gemini when you ask June to do real work, with the call visible before and after it happens. One brain across browser, desktop, and mobile surfaces. The web PWA is the current shipped surface; desktop is in active development; mobile is planned.

## Why June Exists

Every AI assistant today has the same failure mode: it forgets you between sessions. You re-explain your goals, your context, your history, your constraints, over and over. The assistant feels capable but never feels personal.

And even when an assistant remembers, it cannot act. It can describe what to do; it cannot do it. The user is still the courier between the chatbot's suggestion and the apps and services where the work actually lives.

June is built on a different premise: an assistant that *remembers* you and can *do work for you* across the apps you already use is the product. The model is infrastructure. Memory is the moat. Agency is the unlock.

## The Three Non-Negotiables

Every feature, decision, and dependency is measured against these three principles. If a request cannot be justified by at least one of them, the answer is no.

### 1. Memory is the product

Every conversation feeds a personal knowledge graph that is yours, editable, portable, and local-first. June recalls relevant memories before every response and extracts new facts after every response. The memory layer is inspectable, exportable, and never leaves your machine without your consent.

### 2. Private by default, intelligence on tap

June runs Gemma 4 locally for chat, recall, and any turn the user keeps private. June reaches Gemini for agentic capability — multi-step planning, long context, vision, computer use — only when the user's policy allows it, and every cloud call is visible in the UI before and after it happens. The user holds the dial: `local-only`, `private-by-default`, or `cloud-first`. Memory never leaves the machine. Cloud calls send only the turn's context and are not used for training. See [ADR 0009](decisions/0009-private-by-default-and-model-routing.md).

### 3. One codebase, every surface

Browser, desktop, and mobile share the same frontend code. The same brain, the same memory, the same API. New features land in one place and appear everywhere. This is how a small team ships a multi-platform product.

## What June Is Not

- **Not a chatbot.** June is a personal agent. It does work for you across files, apps, and services — drafting and sending emails in your Gmail, scheduling in your Calendar, watching a page, finding a file, summarising a thread, running multi-step tasks in the background.
- **Not a wrapper around a single model.** The model layer is swappable; the memory and skills layer is the moat.
- **Not an account-required service.** June installs onto your machine. No signup. No login. No cloud dependency by default. Telemetry never leaves the device without consent.
- **Not a research project.** June ships. Each week produces something a user can actually use.

## North Star User Experience

A user opens June on their Mac at 8am. June greets them by name, reminds them that yesterday they mentioned an upcoming gym session at 8, and asks how it went. They answer. June logs the workout, updates their streak, and notices they missed water logging yesterday. They ask June to draft and send a reply to a collaborator. June pulls the last three messages with that person from Gmail, drafts a reply in their voice, asks for confirmation, and sends it. The reply lands in the inbox; the action lands in memory.

Later they ask June to "watch this flight and add the gate to my calendar when it's announced." June creates a task, runs in the background, and reopens that conversation when the gate is found.

The same user opens June on their iPhone at lunch. Same memory. Same context. Same conversation, if they want.

Later they open the browser app on a shared machine, log in to nothing, paste their Gemini key, and continue where they left off.

Nothing in that experience requires an internet connection except the cloud calls the user authorised. Nothing is synced to a vendor cloud. Everything is June's — and therefore theirs.

## How This Document Is Used

This vision governs architecture decisions and product scope. When in doubt, open this file. When the answer is still unclear, write an Architecture Decision Record under `docs/decisions/`. The current strategic direction is anchored by [ADR 0009](decisions/0009-private-by-default-and-model-routing.md), [ADR 0010](decisions/0010-agentic-core-tasks-oauth-computer-use.md), and the twelve-week [agentic pivot plan](product/agentic-pivot-plan.md).
