# ADR 0008: In-App Ollama Supervision (Use, Do Not Bundle)

**Status:** Accepted
**Date:** 2026-04-27

## Context

The desktop shell (see [ADR 0006](0006-desktop-and-mobile-shells.md)) needs to hide the fact that June depends on Ollama for the local model runtime. Today the PWA can detect that Ollama is reachable but cannot install it, start it, or pull a model — the browser sandbox forbids it. The result is a `/help/ollama` page that asks the user to open a terminal and run shell commands. That cliff disqualifies every non-technical user.

Two questions then need answering for the desktop shell:

1. Should the shell **manage** an installed Ollama on the user's behalf (start it, pull models, restart on crash)?
2. Should the shell **bundle** Ollama itself, so the user does not need to install Ollama separately?

## Decision

**Yes to (1). No to (2).**

The desktop shell manages Ollama as a child process: it detects whether Ollama is installed, installs it for the user with one click if it is missing, starts it on launch if it is not running, pulls Gemma 4 if it is not pulled, and reports progress back to the UI in real time. The Rust side does the actual subprocess work; the UI calls into it through the platform capability layer.

The shell **does not** ship with an Ollama binary inside its installer. We use the Ollama the user has installed (or that we just helped them install via the official installer). Our installer stays small.

## Why Manage But Not Bundle

**Bundling makes the installer huge.** Ollama is roughly 500 MB on disk. Bundling it inside our Tauri installer would multiply the download size by something like 30× and erase Tauri's main user-visible advantage over Electron. We chose Tauri for size; we do not undo that choice in the first feature.

**Bundling forks our Ollama from theirs.** If we bundle our own copy, we own the upgrade path for the model runtime. Users who already have Ollama installed end up with two of them, two copies of the model weights, and two daemons fighting over port 11434. Using the user's installed Ollama keeps a single source of truth on the machine.

**Bundling complicates licensing and signing.** Shipping a third-party binary inside our installer means we are responsible for its provenance, its license terms, and re-signing it on macOS. Pointing the user at the official installer pushes those concerns where they belong.

**Managing without bundling closes the actual gap.** The friction the user feels is "I have to open a terminal." That goes away the moment we run the official installer for them and supervise the process. We do not need to bundle to fix that friction.

## Implementation Sketch

The Rust side exposes five commands, all in `apps/desktop/src-tauri/src/ollama.rs`:

- `is_ollama_installed()` — `which ollama` or platform equivalent.
- `install_ollama()` — downloads the official installer for the user's OS (`ollama.com/download/Ollama-darwin.zip` etc.), runs it, emits progress events.
- `start_ollama()` — spawns `ollama serve` if it is not already running. Supervised: restart on crash, terminate on app exit.
- `is_model_pulled(tag)` — calls `http://localhost:11434/api/tags`.
- `pull_model(tag)` — runs `ollama pull <tag>` and emits progress events.

The UI calls these through `packages/ui/src/platform.ts`. On the web shell, the same calls return "unsupported" and the existing text-based `/help/ollama` page remains.

## Consequences

**Positive:**

- The user installs June and reaches a working chat in under three minutes from a clean machine, with no terminal involvement.
- The desktop shell installer stays small (target: under 20 MB).
- We do not own the Ollama upgrade path; users get whatever Ollama version the official installer provides.
- The web shell's text-based `/help/ollama` page continues to work for users who cannot or will not install the desktop shell.

**Negative:**

- We depend on Ollama's official installer continuing to be a reliable, signed, single-file installer. If their distribution model changes, we adapt.
- Users with non-standard Ollama setups (custom port, different binary location) need a manual override. We add this in `/settings` only when a real user asks for it.
- A future Ollama version could break our process supervision (e.g., daemon mode changes). Integration tests in Rust catch the obvious breakages; users catch the subtle ones.

## Alternatives Considered

**Bundle Ollama.** Rejected for the size, fork, and licensing reasons above.

**Replace Ollama with a Rust-native runtime (`mistral.rs`, `llama.cpp` via `llama-cpp-rs`).** Considered seriously. This is the long-term direction once a user feels the cost of "Ollama is a separate dependency." Today, Ollama is robust, well-tested, and supports Gemma 4 out of the box. Re-implementing the runtime layer is a multi-week project with no user-facing benefit on day one. Revisit when one of these triggers: Ollama drops support for a model we need, the bundled-runtime size argument flips (e.g., GGUF runtimes get small enough), or a user complains about the dependency.

**Do nothing — keep the text-based help page.** Rejected because the cliff is the single biggest reason a non-technical user gives up on June before getting to their first conversation.

**Web-only management via WebSerial / WebUSB / a local helper service.** Rejected because none of these grant the kind of process supervision Ollama needs, and a "local helper service" is a desktop shell by another name.

## Revisit When

- Ollama distribution changes in a way that breaks our auto-install path.
- A user reports that managing two Ollama installations (theirs and ours) creates confusion.
- The Rust-native runtime alternatives mature to the point where the size and dependency story flips.
