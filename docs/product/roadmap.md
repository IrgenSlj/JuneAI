# Roadmap

This document describes what is left to ship for a useful first prototype, and when each additional surface becomes worth planning in detail. It is organized by trigger, not by week number. A surface is planned when its trigger fires. A surface is implemented when the plan says it is ready.

## The One Rule

> Implement what the current users need. Plan the next surface when the current one has users.

No parallel construction. No half-finished platforms. Each surface must reach real users before the next one gets detailed design.

## Current Surface: Web PWA

The browser application is the first and only surface for June 1.0. Installable through the browser's native install flow. Works offline against a local Ollama. Works online against Gemini. No account. No cloud dependency beyond the optional model call.

### Remaining Work for the First Working Prototype

Ordered by dependency, not priority.

1. **First-run setup flow.** A `/setup` route that detects Ollama reachability, lets the user pick a provider, paste a Gemini key if they chose cloud, and verifies end to end before landing them on the chat screen. Until this exists, a new user has to read the README to get past the first screen. _Shipped._
2. **API key entry UI.** A settings screen that reads and writes `GEMINI_API_KEY` through a new API surface. Keys are stored in the platform's native credential store when available and in `config.toml` otherwise. Never logged, never echoed back to the UI after save. _Shipped._
3. **Ollama detection and guidance.** When the provider is `gemma` and Ollama is not reachable, the header's warning should deep-link to a one-screen troubleshooting page with the exact commands to install, pull, and start Ollama for the user's OS. _Shipped._
4. **PWA installability.** `manifest.webmanifest`, a service worker that caches the shell, icons at the required sizes, and a theme color. `vite-plugin-pwa` generates these. Verify install prompts on Chrome, Edge, and mobile Safari. _Shipped._
5. **Offline fallback screen.** When the brain is unreachable, render a useful offline state instead of a fetch error. Chat history and memory browser are read-only offline because they fetch from the API; show that clearly rather than spinning. _Shipped._
6. **Branding.** A wordmark, an app icon set, and a coherent visual identity. See [design/claude-design-prompt.md](../design/claude-design-prompt.md) for the design brief. _In progress — placeholder amber "J" mark shipped alongside the PWA work; full identity pending design iteration._
7. **Chat polish.** Keyboard shortcuts (Cmd+Enter to send, Cmd+K to focus, Esc to cancel stream). Message selection and copy. Regenerate last response. Scroll-to-bottom pinning. _Shipped._
8. **Memory browser polish.** Search box that filters across all three stores. Grouping by source and date. Empty states that teach the user what to expect. _Shipped._
9. **Skills registry polish.** A tools-documentation view per skill. Per-tool enable/disable within a skill (skill-level toggle is live). Status tooltips that explain `starting`, `crashed`, `stopped`. _Shipped — skill-level toggle, status tooltips, and collapsible per-skill tool list are live; per-tool toggle deferred until a user asks._
10. **Accessibility pass.** Keyboard navigation, focus rings, semantic landmarks, color-contrast audit. Screen-reader announcement for streaming tokens is deferred until complaints arrive. _Shipped._

### Done Criteria for the Prototype

A first-time user opens the URL, completes setup in under two minutes, has their first conversation with Gemma or Gemini, sees a memory land in the browser, and toggles at least one skill. The browser prompts them to install. They close the tab and tomorrow open the installed app from their dock or home screen and continue the conversation.

## Next Surface: Desktop Shell

### Trigger to Plan

The web PWA has at least one hundred active users, or one of the following capability gaps is blocking real usage:

- Global hotkey. Users want to summon June from any app with a keyboard shortcut.
- System tray presence. Users want June quietly running with unread-memory badges.
- Native notifications. Reminders, proactive nudges, calendar alerts.
- Autostart on login.
- Filesystem reach. Reading documents the user points at, without upload steps.
- Ollama process supervision. Start and stop the local model server from inside June.

### What It Is

A Tauri 2.x shell that wraps the same SvelteKit build. Rust commands expose the capabilities above to the UI through the capability layer at `packages/ui/src/platform.ts`. Ships a macOS `.dmg`, a Windows installer, and a Linux AppImage from one build pipeline.

### Why Not Now

Tauri requires the Rust toolchain in every contributor's environment. The PWA already delivers install-to-dock via the browser. Until a capability gap is blocking a real user, the native shell adds friction without adding value.

### When Implemented

The plan is written when the trigger fires. Implementation follows the plan. Estimate one to two weeks once started; most of the UI already works.

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

June surfaces its own thoughts without being prompted: reminders, gentle nudges, pattern observations. Requires the scheduler (a background loop that inspects memory and pushes messages into the next opened conversation) and a notification channel on each shell. Plan alongside the desktop shell; implement after mobile push lands.

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
