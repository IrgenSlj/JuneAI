# Roadmap

This document describes what is left to ship and when each additional surface becomes worth planning in detail. It is organized by trigger, not by week number. A surface is planned when its trigger fires. A surface is implemented when the plan says it is ready.

## The One Rule

> Implement what the current users need. Plan the next surface when the current one has users.

No parallel construction. No half-finished platforms. Each surface must reach real users before the next one gets detailed design.

## Active Track: Agentic Pivot (Sprints 1-4)

The full execution plan is at [agentic-pivot-plan.md](agentic-pivot-plan.md). The strategic decisions behind it are [ADR 0009](../decisions/0009-private-by-default-and-model-routing.md) and [ADR 0010](../decisions/0010-agentic-core-tasks-oauth-computer-use.md).

The twelve-week plan replaces, in priority terms, the open-source-readiness pass and the memory-skills Phase C items. Those plans remain valid as backlog and will be folded back in once the pivot's Sprint 1 has shipped.

- **Sprint 1 (weeks 1-3) — Agentic Core.** Three-tier model router, tasks primitive, real files/gmail/gcal skills, browser skill, MCP registry connector, desktop shell first compile. **Status as of 2026-05-19:** router, tasks (with runtime), files (expanded), MCP registry connector, and a Batch 1 of cross-department UX adds (TaskRuntime / MemoryStats / SkillPlayground / SystemActivity) are shipped. Gmail/Calendar OAuth, browser skill, chat-event provenance, and desktop first-compile remain.
- **Sprint 2 (weeks 4-6) — Dogfooding.** Owner uses June daily, journals failures, rewrites Sprint 3 backlog from observed pain.
- **Sprint 3 (weeks 7-9) — Installable for humans.** Signed installers, three-question first-run flow, public landing page, README rewrite for mainstream users.
- **Sprint 4 (weeks 10-12) — 50-user closed beta.** Discord, weekly office hours, per-week metric tracking, written go/no-go decision.

## Current Surface: Web PWA (shipped, evolving with the pivot)

The browser application is the first surface for June 1.0. Installable through the browser's native install flow. Works offline against a local Ollama. Works online against Gemini. No account, no cloud dependency beyond the optional model call. The first prototype checklist is fully shipped as of 2026-04-20.

The web PWA remains the primary surface during the pivot. The desktop shell does not retire it; the same SvelteKit build serves both.

The agentic capabilities being added in Sprint 1 will appear in the PWA where the browser's sandbox allows: file access via the File System Access API where supported, OAuth via same-origin popups, MCP-server installation in registry-browse-only mode. Browser-controlled automation and computer use are desktop-only by physical necessity.

### Done Criteria for the Prototype (achieved 2026-04-20)

A first-time user opens the URL, completes setup in under two minutes, has their first conversation with Gemma or Gemini, sees a memory land in the browser, and toggles at least one skill. The browser prompts them to install. They close the tab and the next day open the installed app from their dock or home screen and continue the conversation.

## Next Surface: Desktop Shell — In Progress

### Trigger Fired

The Ollama process-supervision capability gap fired the trigger on 2026-04-27. The PWA can detect Ollama reachability but cannot install it, start it, or pull a model on the user's behalf, leaving non-technical users at a terminal-instructions cliff. The desktop shell is also the only surface where the agentic core can run at full capability: filesystem access, browser automation, OAuth via loopback redirect, system tray, background tasks, native notifications.

The full plan is in [desktop-shell-plan.md](desktop-shell-plan.md). Phases 1-4 have shipped (scaffold, capability layer, Ollama supervision, native affordances). Phase 4.5 (First Compile — install rustup and verify the Rust code in Phases 3-4 actually builds) is Sprint 1.7 of the agentic pivot. Phases 5-7 (touch hardening, distribution, polish) merge into Sprint 3 of the pivot ("installable for humans").

### What This Unblocks

- The mobile-shell trigger (push, share extensions, voice) becomes the next one to watch once the desktop shell is in users' hands.
- Background task execution (Sprint 1.2 of the pivot) becomes meaningful: tasks survive laptop sleep, native notifications surface results, the tray icon shows running work.
- OAuth flows for service skills become reliable: a loopback redirect on `127.0.0.1` is much more robust than the popup-with-postMessage path the PWA must use.

## Later Surface: Mobile Shell

### Trigger to Plan

The desktop shell has shipped and is stable, and one of the following is true:

- Users ask for push notifications on their phone.
- Users want to share content from Safari or Mail into June.
- Users want voice input on the go.

### What It Is

A Capacitor shell that wraps the same SvelteKit build. Swift plugins expose iOS push, share extensions, and voice input via AVFoundation. Ships to TestFlight first, App Store second.

### Why Not Now

Mobile adds an App Store submission process, code signing, and a second capability layer to maintain. The PWA is installable on iOS via Safari's Add to Home Screen; that path serves users until the native features above become load-bearing.

### When Implemented

Estimate two to three weeks once started. Most of the UI already works; the budget is for capability plugins, icon variants, and App Store review cycles.

## Feature Surfaces

Features live on top of the surfaces above. Ordered by user-visible impact.

### Voice

Speech-to-text input and text-to-speech output. The PWA uses the Web Speech API where available. Desktop and mobile shells use native APIs. Plan when at least one user asks for it; implement when three have.

### Proactive Agent

June surfaces its own thoughts without being prompted: reminders, gentle nudges, pattern observations, results of background tasks. Requires the tasks primitive (shipping in Sprint 1.2) plus the scheduler and a notification channel on each shell. Plan once the desktop shell's native notifications are exercised by Sprint 1 tasks; implement once mobile push lands.

### Skill Marketplace

A browsable registry of community skills, installable in one click. The Sprint 1.6 MCP registry connector ships a minimal version of this against a static, curated index. A richer marketplace with descriptions, ratings, signing, and a review process is planned when three external contributors have shipped skills.

### Multi-User

One installation, multiple profiles. Requires memory partitioning by user, a profile picker, and per-profile settings. Plan when a user asks (families, couples). Not before.

## Things That Are Not On The Roadmap

- **Cloud sync.** Memories stay local. Export and manual import are the cross-device story until a user shows the pain is bigger than the privacy cost. Cross-device memory sync is considered for a later phase but is not on the twelve-week pivot.
- **Team or collaboration features.** June is a personal agent. An organisation layer is a different product.
- **A third model provider.** Gemma and Gemini. A new provider replaces one; it does not add to them.
- **Account-required modes.** June installs onto your machine. No signup, no login, no cloud dependency by default.
- **A native app on a platform not listed above.** Android, Linux-only, watchOS, etc. are considered if a contributor ships them, not planned by us.
