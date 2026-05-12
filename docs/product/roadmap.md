# Roadmap

This document describes what is left to ship for a useful first prototype, and when each additional surface becomes worth planning in detail. It is organized by trigger, not by week number. A surface is planned when its trigger fires. A surface is implemented when the plan says it is ready.

## The One Rule

> Implement what the current users need. Plan the next surface when the current one has users.

No parallel construction. No half-finished platforms. Each surface must reach real users before the next one gets detailed design.

## Active Hardening Track: Open Source Readiness

Before June is presented as broadly download-and-use software, the project is
in a focused open-source readiness pass. The detailed execution plan lives in
[open-source-readiness-plan.md](open-source-readiness-plan.md).

The release bar is:

1. Provider correctness: `/setup`, `/settings`, `/system`, and `/chat` agree on
   the active provider, model, key state, and privacy label.
2. Conversation continuity: recent turns are available to the chat agent, not
   only the latest user message.
3. Memory correctness: editing or deleting a memory updates every store that can
   feed recall, including semantic paraphrases.
4. Fresh-clone reliability: the documented Python version, bootstrap scripts,
   and model-provider paths work from an empty checkout.
5. Local safety: the API has a basic same-machine authorization boundary, demo
   routes are opt-in, and network-fetching skills reject private targets.
6. CI coverage: frontend checks, backend tests, lint/type policy, OpenAPI
   codegen, and desktop compilation are all enforced or explicitly scoped.
7. Honest release docs: the README separates working web alpha behavior from
   experimental desktop source.

This hardening track temporarily outranks new feature surfaces. Once the
Public Alpha Gate in the readiness plan is complete, feature work returns to the
trigger-gated roadmap below.

## Current Surface: Web PWA (Shipped)

The browser application is the first surface for June 1.0. Installable through the browser's native install flow. Works offline against a local Ollama. Works online against Gemini. No account. No cloud dependency beyond the optional model call. The prototype checklist below is fully shipped as of 2026-04-20.

The web PWA remains the primary surface, but the open-source readiness track
above is the current development priority before inviting broad public usage.

The desktop shell (next section) does not retire the PWA. The PWA remains a first-class surface and the same SvelteKit build serves both.

### Remaining Work for the First Working Prototype

Ordered by dependency, not priority.

1. **First-run setup flow.** A `/setup` route that detects Ollama reachability, lets the user pick a provider, paste a Gemini key if they chose cloud, and verifies end to end before landing them on the chat screen. Until this exists, a new user has to read the README to get past the first screen. _Shipped._
2. **API key entry UI.** A settings screen that reads and writes `GEMINI_API_KEY` through a new API surface. Keys are stored in the platform's native credential store when available and in `config.json` with mode 0600 otherwise. Never logged, never echoed back to the UI after save. _Shipped._
3. **Ollama detection and guidance.** When the provider is `gemma` and Ollama is not reachable, the header's warning should deep-link to a one-screen troubleshooting page with the exact commands to install, pull, and start Ollama for the user's OS. _Shipped._
4. **PWA installability.** `manifest.webmanifest`, a service worker that caches the shell, icons at the required sizes, and a theme color. `vite-plugin-pwa` generates these. Verify install prompts on Chrome, Edge, and mobile Safari. _Shipped._
5. **Offline fallback screen.** When the brain is unreachable, render a useful offline state instead of a fetch error. Chat history and memory browser are read-only offline because they fetch from the API; show that clearly rather than spinning. _Shipped._
6. **Branding.** A wordmark, an app icon set, and a coherent visual identity. See [design/claude-design-prompt.md](../design/claude-design-prompt.md) for the design brief. _Shipped — black "J" wordmark, light mode default, dark mode toggle._
7. **Chat polish.** Keyboard shortcuts (Cmd+Enter to send, Cmd+K to focus, Esc to cancel stream). Message selection and copy. Regenerate last response. Scroll-to-bottom pinning. _Shipped._
8. **Memory browser polish.** Search box that filters across all three stores. Grouping by source and date. Empty states that teach the user what to expect. _Shipped._
9. **Skills registry polish.** A tools-documentation view per skill. Per-tool enable/disable within a skill (skill-level toggle is live). Status tooltips that explain `starting`, `crashed`, `stopped`. _Shipped — skill-level toggle, status tooltips, and collapsible per-skill tool list are live; per-tool toggle deferred until a user asks._
10. **Accessibility pass.** Keyboard navigation, focus rings, semantic landmarks, color-contrast audit. Screen-reader announcement for streaming tokens is deferred until complaints arrive. _Shipped._

### Done Criteria for the Prototype

A first-time user opens the URL, completes setup in under two minutes, has their first conversation with Gemma or Gemini, sees a memory land in the browser, and toggles at least one skill. The browser prompts them to install. They close the tab and tomorrow open the installed app from their dock or home screen and continue the conversation.

## Current Depth Track: Memory and Skills

The web prototype is shipped, but the contracts between memory, skills, and the chat UI are weaker than the product's first non-negotiable demands ("memory is the product"). This track deepens those contracts in three phases — making memory editable, making recall legible, and making skill writes feed recall. It runs parallel to the desktop-shell track because it touches separate subsystems; both can advance independently. Full plan in [memory-skills-plan.md](memory-skills-plan.md).

The first slice — making goals, open loops, and calendar items deletable — is the smallest end-to-end pattern that proves the architecture move (`MemoryManager.forget` dispatches across stores). Subsequent slices repeat that pattern.

## Next Surface: Desktop Shell — In Progress

### Trigger Fired

The Ollama process-supervision capability gap fired the trigger on 2026-04-27. The PWA can detect Ollama reachability but cannot install it, start it, or pull a model on the user's behalf, leaving non-technical users at a terminal-instructions cliff. Closing that cliff requires shell access the browser does not grant. The native shell is the way it gets closed.

The full plan is in [desktop-shell-plan.md](desktop-shell-plan.md). The architectural decision behind the choice of Tauri lives in [ADR 0006](../decisions/0006-desktop-and-mobile-shells.md); the architectural decision behind in-app Ollama supervision lives in [ADR 0008](../decisions/0008-ollama-supervision.md). Touch and tablet hardening that ships alongside the shell is in [responsive-plan.md](responsive-plan.md).

### What It Is

A Tauri 2.x shell at `apps/desktop/` that wraps the same SvelteKit build. Rust commands expose native capabilities (Ollama supervision, system tray, global hotkey, native notifications, autostart, filesystem) to the UI through the capability layer at `packages/ui/src/platform/`. Distribution packages come after Rust CI, signing, and release automation.

### The Phases (full detail in desktop-shell-plan.md)

1. **Scaffold** — _Shipped (`e2639312`)._ Existing UI runs unchanged inside a Tauri window.
2. **Capability layer** — _Shipped (`2cd0408b`)._ Typed `packages/ui/src/platform/` interface with Tauri, Capacitor, and Web backends.
3. **Ollama supervision** — _Shipped (`49400967`)._ Install (opens OS installer), start, pull with streamed progress, model check; one-click `/help/ollama` flow on desktop.
4. **Native affordances** — _Shipped (`f5e24dfa`)._ Tray, global hotkey, notifications, autostart, window state. Hidden-inset title bar deferred to 4.5.
5. **Touch and responsive hardening** — _Next._ See [responsive-plan.md](responsive-plan.md).
6. **Distribution** — code signing, auto-update, GitHub Actions build pipeline.
7. **Migration and polish** — first-run welcome, opt-in crash reporting, data-path consolidation.

### Estimate

Roughly nine working days plus a one-week external test period. Phases 1–4 took two implementation sessions (TypeScript verified clean; the Rust in 3 and 4 awaits its first compile on a machine with rustup).

### What This Unblocks

- The mobile-shell trigger (push, share extensions, voice) becomes the next one to watch once the desktop shell is in users' hands.
- The proactive assistant feature plan can begin in parallel late in Phase 4 because tray and notifications are its prerequisites.

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

### Proactive Assistant

June surfaces its own thoughts without being prompted: reminders, gentle nudges, pattern observations. Requires the scheduler (a background loop that inspects memory and pushes messages into the next opened conversation) and a notification channel on each shell. Plan alongside the desktop shell once Phase 4 (native affordances, including notifications) lands; implement after mobile push lands.

### Skill Marketplace

A browsable registry of community skills, installable in one click. Requires a public index, signing for safety, and a review process. Plan when three external contributors have shipped skills; implement after.

### Multi-User

One installation, multiple profiles. Requires memory partitioning by user, a profile picker, and per-profile settings. Plan when a user asks (families, couples). Not before.

## Things That Are Not On The Roadmap

- **Cloud sync.** Memories stay local. Export and manual import are the cross-device story until a user shows the pain is bigger than the privacy cost.
- **Team or collaboration features.** June is a personal assistant. An organization layer is a different product.
- **A third model provider.** Gemma and Gemini. A new provider replaces one; it does not add to them.
- **In-app payments, subscriptions, accounts.** June is free and open.
- **A native app on a platform not listed above.** Android, Linux-only, watchOS, etc. are considered if a contributor ships them, not planned by us.
