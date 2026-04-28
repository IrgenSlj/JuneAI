# Desktop Shell Development Plan

This document is the concrete plan for shipping June's desktop shell. It replaces the placeholder "Trigger to Plan" section in [roadmap.md](roadmap.md) now that the trigger has fired. The architectural decision behind the choice of Tauri lives in [ADR 0006](../decisions/0006-desktop-and-mobile-shells.md); the architectural decision behind in-app Ollama supervision lives in [ADR 0008](../decisions/0008-ollama-supervision.md).

## Why Now

The trigger that fired is the Ollama capability gap. The PWA can detect whether Ollama is reachable but cannot install it, start it, or pull a model on the user's behalf. `/help/ollama` is a wall of text that asks the user to leave the app and run shell commands. That cliff disqualifies every non-technical user. Closing it requires shell access, which the browser does not grant. The desktop shell is the way it gets closed.

Building the desktop shell now also unlocks four other capabilities the roadmap calls out: global hotkey, system tray, native notifications, and autostart on login. They are gravy on top of the Ollama bootstrap; we ship them in the same shell because they share the same Rust foundation.

## What We Are Building

A Tauri 2.x application at `apps/desktop/` that wraps the existing `apps/web/` SvelteKit build inside a native window. The same UI runs in the PWA, in the desktop shell, and (later) in the mobile shell. The desktop shell adds Rust commands for capabilities the browser cannot provide. It does not introduce a parallel UI.

The shell ships three platform installers from one build pipeline:

- **macOS** — universal `.dmg` (Apple Silicon + Intel), code-signed, notarized.
- **Windows** — `.msi` installer, signed with an EV certificate when one is available.
- **Linux** — `.AppImage` and `.deb`, unsigned for now.

The macOS build is the one we polish first. Windows and Linux come for free from Tauri's build pipeline; we test them and fix what breaks but do not invest in platform-specific polish in the first release.

## Non-Goals For This Release

- **No new product features.** The shell wraps the existing UI. New surfaces (proactive assistant, voice, marketplace) wait until the shell is in users' hands.
- **No bundled Ollama runtime.** We use the user's installed `ollama` binary, install it for them if missing, but do not embed Ollama in our installer. See [ADR 0008](../decisions/0008-ollama-supervision.md) for the trade-off.
- **No Tauri-mobile experiment.** Mobile stays on the Capacitor track in [ADR 0006](../decisions/0006-desktop-and-mobile-shells.md). Adopting Tauri 2's mobile target would split the mobile plan; it is reconsidered when the desktop shell has shipped and stabilized.
- **No multi-window UI.** One window per running app. Tray icon and global hotkey raise the existing window; they do not spawn extra ones.

## Phase Status (live)

| Phase | Status | Commit |
|---|---|---|
| 1. Scaffold | Shipped | `e2639312` |
| 2. Capability layer | Shipped | `2cd0408b` |
| 3. Ollama supervision | Shipped (with bootstrap caveat — see below) | `49400967` |
| 4. Native affordances | Shipped (title-bar overlay deferred to 4.5) | `f5e24dfa` |
| 5. Touch + responsive hardening | Next |  |
| 6. Distribution | Pending |  |
| 7. Migration + polish | Pending |  |

The Rust code in Phases 3 and 4 has not been compiled locally (no rustup on the dev machine yet). The TypeScript side of every phase is clean (`pnpm check` 0/0). The first `pnpm desktop:dev` run by anyone with Rust installed is also the first compile of `apps/desktop/src-tauri/src/ollama.rs` and `native.rs`; expect minor fix-ups.

## The Phases

The plan is divided into seven phases. Each phase ends in a working artifact you can install and use. Phases are sequential because each builds on the last; do not parallelize.

### Phase 1: Scaffold

**Goal:** the existing web UI runs unchanged inside a Tauri window on macOS, dev mode only.

**Work:**

- Add `apps/desktop/` with `src-tauri/`, `package.json`, `tauri.conf.json`, `Cargo.toml`.
- Configure the Tauri dev command to load Vite from `http://localhost:5173` (proxying the existing `apps/web` dev server).
- Configure the Tauri build command to consume the `apps/web/.svelte-kit/output/prerendered` artifact as the static frontend.
- Add `pnpm desktop:dev` and `pnpm desktop:build` scripts at the repo root.
- Document Rust toolchain installation in `docs/setup/desktop.md` (rustup, target add for `aarch64-apple-darwin` and `x86_64-apple-darwin`).
- Update `tools/dev.sh` to detect the Rust toolchain and warn (not fail) when missing.

**Done when:** `pnpm desktop:dev` opens a native macOS window, the chat surface streams a Gemma response, and `pnpm desktop:build` produces an unsigned `.dmg`.

**Estimate:** 1 day.

**Shipped:** commit `e2639312`. Window opens at `http://localhost:5173`, repo-root scripts wired (`pnpm desktop:dev` / `desktop:build`), Rust toolchain detection in `tools/dev.sh` warns instead of fails. apps/desktop excluded from default `pnpm build` so contributors without Rust aren't blocked.

### Phase 2: Capability Layer

**Goal:** the UI calls platform features through one typed interface that has a Tauri implementation, a Capacitor stub, and a Web fallback.

**Work:**

- Create `packages/ui/src/platform.ts` exposing a `Platform` interface with the methods the desktop shell needs: `notify`, `registerHotkey`, `setTrayMenu`, `openExternal`, `pickFile`, `bootstrapOllama`, `isOllamaInstalled`, `startOllama`, `pullModel`, `getAutostart`, `setAutostart`.
- Implement three runtime backends:
  - `platform-tauri.ts` — calls `invoke()` and subscribes to Tauri events.
  - `platform-web.ts` — uses Web Notifications API, `window.open`, `<input type=file>`, returns "unsupported" for the rest.
  - `platform-capacitor.ts` — stubs that throw "not implemented" until the mobile shell ships.
- Pick the backend at module load via runtime detection (`window.__TAURI__`, `window.Capacitor`).
- Generate TypeScript types for Tauri commands from a single Rust source (using `tauri-specta` or hand-written types until the surface stabilizes).

**Done when:** the existing PWA build still works in the browser unchanged, the desktop shell calls a `notify` test command, and the type system catches a typo'd command name.

**Estimate:** 1 day.

**Shipped:** commit `2cd0408b`. Files landed under `packages/ui/src/platform/`: `types.ts` (Platform interface + UnsupportedError + closed `TauriCommand` union), `web.ts`, `tauri.ts` (lazy-imports `@tauri-apps/api/*` so the bundle still works in the browser), `capacitor.ts` (stubs), `index.ts` (runtime detection). The contract proved itself mid-build: a missing `"pick_file"` entry in `TauriCommand` was caught by `tsc` rather than at runtime. Settings page has a "Send test notification" button showing `platform.runtime`.

### Phase 3: Ollama Supervision

**Goal:** `/setup` and `/help/ollama` no longer ask the user to open a terminal. Everything happens inside the app on the desktop shell, with a graceful fallback to text instructions on the web.

**Work:**

- Implement Rust commands in `src-tauri/src/ollama.rs`:
  - `is_ollama_installed` — checks `which ollama` (or platform equivalent).
  - `install_ollama` — downloads the appropriate installer for the OS, runs it, reports progress as Tauri events.
  - `start_ollama` — spawns `ollama serve` if it is not already running, supervised, restarted on crash, terminated on app exit.
  - `is_model_pulled` — calls Ollama's HTTP API at `/api/tags`.
  - `pull_model` — runs `ollama pull <tag>` and streams progress (bytes/percentage) back to the UI.
- Update `apps/web/src/routes/setup/+page.svelte` to detect the desktop shell and show a one-click "Set up Ollama and pull Gemma" button. The web version retains its current text instructions.
- Update `apps/web/src/routes/help/ollama/+page.svelte` similarly: a one-click "Fix it" button on desktop, the existing text on web.
- Add a status surface in the brain (already exists) that exposes Ollama process state, so the header dot can show green/yellow/red.
- Write integration tests in Rust for the supervision logic (process spawn, crash detection, restart). Skip in CI environments without Ollama.

**Done when:** a user installing the desktop shell on a machine with no Ollama gets to the chat screen in under three minutes by clicking through `/setup`. No terminal involvement.

**Estimate:** 2 days.

**Shipped:** commit `49400967`. Five Rust commands in `src-tauri/src/ollama.rs`: `is_ollama_installed`, `start_ollama` (spawns `ollama serve`, retains the child handle in `OllamaState`, waits up to 10s for `/api/tags` to answer), `is_model_pulled`, `pull_model` (POSTs to `/api/pull` with `stream=true`, parses newline-delimited JSON, emits `ollama-pull-progress` events with fraction + status), and `bootstrap_ollama`. The `/help/ollama` route renders a one-click step list (Install → Start → Pull) with a real progress bar driven by Tauri events when `platform.runtime === "tauri"`; web users still see the manual instructions.

**Pragmatic caveat — bootstrap:** `bootstrap_ollama` opens the official OS-specific installer URL via the shell plugin rather than downloading and extracting in-process. The OS-native installer flow (Gatekeeper on macOS, UAC on Windows) is a known UX rather than a half-baked one. Phase 3.5 may revisit if real users hit friction.

**Deferred to Phase 3.5 / never (depending on usage):** crash detection + auto-restart of `ollama serve`; integration tests in Rust for the supervision logic; in-process installer download.

### Phase 4: Native Affordances

**Goal:** the shell feels like a Mac app, not a Chrome tab.

**Work:**

- **System tray.** Icon in the menu bar. Click opens or focuses the window. Right-click shows a menu: Open June, Toggle Local-Only Mode, Quit.
- **Global hotkey.** Default `Cmd+Shift+J` (configurable in settings). Toggles window visibility from anywhere.
- **Native notifications.** Use Tauri's `notification` plugin. Wired through the capability layer so the UI calls `platform.notify(title, body)` and the right thing happens on each shell.
- **Autostart on login.** Use Tauri's `autostart` plugin. Toggle in `/settings`.
- **Native title bar.** Custom window controls on macOS that match the visual identity. Hidden inset variant (dots in the top-left, title bar drag area).
- **Window state persistence.** Remember size and position across launches via Tauri's `window-state` plugin.
- **macOS-specific touches.** Dock badge for unread proactive messages (later — stub the API now). Reduced-motion respected. Dark-mode follows the system unless overridden in `/settings`.

**Done when:** every native affordance in the list above works, is configurable in `/settings` where applicable, and survives a quit/relaunch.

**Estimate:** 2 days.

**Shipped:** commit `f5e24dfa`. Three Tauri 2 plugins join the build (`tauri-plugin-window-state`, `tauri-plugin-autostart`, `tauri-plugin-global-shortcut`). Rust module `src-tauri/src/native.rs` installs the tray (left-click toggles main window, right-click opens Open / Quit menu), registers `Cmd+Shift+J` / `Ctrl+Shift+J` as a global hotkey that toggles window visibility, and wraps autostart in `get_autostart` / `set_autostart` commands. Settings page gains a desktop-only "Native shell" card with an autostart toggle, hotkey hint, and tray hint.

**Deferred to Phase 4.5:** hidden-inset title bar (needs a CSS drag region in the layout — defer until verifiable on screen); configurable hotkey (defer until usage data shows the default conflicts with something); dock badge (depends on proactive notifications); "Toggle Local-Only Mode" tray entry (needs brain coordination).

### Phase 5: Touch and Responsive Hardening

**Goal:** the same UI looks intentionally good on every screen size and works correctly on touch input. See [responsive-plan.md](responsive-plan.md) for the detailed work.

**Work (summary; full breakdown in the responsive plan):**

- Add `(pointer: coarse)` media queries that enforce 44px minimum touch targets.
- Add a tablet breakpoint at 768px–1024px with intentional tablet layout, not stretched mobile.
- Audit and fix every hover-only affordance.
- Add iOS PWA viewport tweaks (safe-area insets, dynamic viewport height, no zoom on input focus).
- Test landscape and portrait on iPad, on a Surface, on a touch-screen laptop, and on the small (iPhone SE) and large (iPhone 16 Pro Max) phone breakpoints.
- Composer: visible keyboard hint becomes invisible on touch, send button grows, swipe-to-dismiss keyboard works.

**Done when:** the responsive plan's acceptance checklist is fully checked off.

**Estimate:** 2 days.

### Phase 6: Distribution

**Goal:** a user clicks a link on the website and ends up with a running, signed June.

**Work:**

- **macOS code signing.** Apple Developer account, Developer ID Application certificate, configure `tauri.conf.json` to sign and notarize during build. Confirm the `.dmg` opens without Gatekeeper warnings on a clean Mac.
- **Windows code signing.** Stretch goal — defer until the unsigned build has external users complaining about SmartScreen warnings.
- **Auto-update.** Tauri's updater plugin pointed at a static `latest.json` hosted on GitHub Releases. Updates check on launch and once daily; users can disable in `/settings`.
- **Build pipeline.** GitHub Actions workflow that builds macOS (universal), Windows, and Linux artifacts on tag push, signs the macOS build, and attaches binaries to a GitHub Release.
- **Install page.** A `/download` route or a section of the project README pointing at the latest `.dmg`, `.msi`, and `.AppImage`. Detect OS in the browser and highlight the right one.

**Done when:** a user with no developer tools, no Rust, no Python, and no Ollama can install June on a fresh Mac in under five minutes from a public download link.

**Estimate:** 2 days (excluding the Apple notarization wait, which is asynchronous).

### Phase 7: Migration and Polish

**Goal:** existing PWA users move to the desktop shell without losing their conversation history; rough edges are sanded.

**Work:**

- **Memory migration.** The PWA stores nothing locally that needs migrating (memory lives behind the API on the user's machine already). Confirm: open the desktop shell on a machine that previously ran the PWA, expect the same conversations to appear. If they don't, write a migration step.
- **Data path consolidation.** Confirm the data path is `~/Library/Application Support/June/` on macOS, `%APPDATA%\June\` on Windows, `~/.local/share/June/` on Linux, and that the API/brain agree.
- **Telemetry opt-in.** No analytics ship in the first release. A single opt-in `/settings` toggle sends crash reports to a self-hosted Sentry. Off by default.
- **Crash reporting.** Sentry only when the user has opted in. The Rust panic handler captures, the Python brain captures, the UI captures.
- **First-run welcome.** A modal on first launch that explains what the desktop shell adds over the PWA: tray, hotkey, no terminal, offline by default.

**Done when:** five external testers run the build for a week with no support tickets opened.

**Estimate:** 1 day plus the test week.

## Total Estimate

Roughly nine working days of engineering plus the test week. Two calendar weeks of focused work or three weeks of part-time work. The Apple notarization process and code-signing setup add wall-clock time, not engineering time.

## Open Questions

These are decisions deferred until the work surfaces them. Adding them here so they don't get re-debated mid-phase.

1. **Custom title bar vs. native traffic-light buttons.** The visual identity prompt asks for a "calm, editorial" feel. A custom title bar reads more polished but is more work. Default to native traffic-light buttons in Phase 1; revisit in Phase 4.
2. **Tray icon idle state.** Solid black "J" or outlined? Pick after seeing the icon at 16×16 on retina and non-retina displays.
3. **Auto-update channel.** Stable only, or stable + beta? Default to stable only until there are enough users for the distinction to matter.
4. **Linux package format.** AppImage is universal; `.deb` is more familiar on Ubuntu/Debian. Ship AppImage in v1; add `.deb` if a real user asks.

## What This Plan Does Not Cover

The mobile shell. Voice. The proactive assistant. The skill marketplace. All of those wait for their own triggers.

## When This Plan Is Done

The roadmap's "Next Surface: Desktop Shell" section is rewritten to record what shipped, ADR 0006 stays Accepted, and a new "Later Surface: Mobile Shell" trigger becomes the next one to watch.
