# Desktop Shell Setup

This file documents how to run, build, and develop the June desktop shell at `apps/desktop/`. The shell is a Tauri 2.x wrapper around the SvelteKit build that already powers the PWA. The strategic plan lives in [`docs/product/desktop-shell-plan.md`](../product/desktop-shell-plan.md); this doc is the operational reference.

## Prerequisites

The desktop shell needs the same toolchain as the web app, plus Rust.

| Tool | Why | How |
|---|---|---|
| Node + pnpm | The web build the desktop shell wraps | Already required for the web app. |
| Python 3.13 + the brain venv | The API the desktop shell talks to | Run `./tools/bootstrap.sh` from the repo root. |
| Ollama with Gemma 4 | The local model | Already required for `MODEL_PROVIDER=gemma`. |
| **Rust toolchain (stable)** | Compiles the Tauri shell | See below. |
| **Platform build tools** | Native linker | See below. |

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version    # expect 1.77 or newer
```

`rustup` installs `cargo`, `rustc`, and the standard toolchain into `~/.cargo/`. Restart your shell or `source` the env file once before continuing.

### Install Platform Build Tools

- **macOS** — `xcode-select --install` (one-time). Tauri needs the system linker and Apple's WebKit headers.
- **Linux** — package depends on distro. On Debian/Ubuntu: `sudo apt install -y libwebkit2gtk-4.1-dev build-essential libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev`.
- **Windows** — Microsoft C++ Build Tools (Visual Studio Installer → "Desktop development with C++" workload) plus WebView2, which ships with Windows 11 by default.

The first build downloads and compiles roughly 200 Rust crates. Plan for 5–10 minutes on the first run; subsequent builds are incremental and fast.

## Running

From the repo root:

```bash
pnpm desktop:dev      # opens the UI in a native window with hot-reload
```

`tauri dev` starts `pnpm --filter @june/web dev` first (Vite at `localhost:5173`), then opens a native window pointed at it. Edits to Svelte or CSS hot-reload as in the browser. Edits to Rust code recompile and relaunch the window.

The desktop shell still talks to the FastAPI backend on `localhost:8000`, exactly like the PWA does. Run the brain in a second terminal:

```bash
packages/brain/.venv/bin/python -m june_api
```

## Building

```bash
pnpm desktop:build    # produces installers under apps/desktop/src-tauri/target/release/bundle/
```

Output paths by platform:

- **macOS:** `bundle/dmg/June_<version>_<arch>.dmg` and `bundle/macos/June.app`.
- **Windows:** `bundle/msi/June_<version>_<arch>_en-US.msi` and `bundle/nsis/June_<version>_<arch>-setup.exe`.
- **Linux:** `bundle/appimage/June_<version>_<arch>.AppImage` and `bundle/deb/june_<version>_<arch>.deb`.

The current macOS build is ad-hoc signed and not notarized. It is acceptable for
alpha testing but macOS may show a first-launch warning. Developer ID signing
and notarization are deferred until external testers justify the Apple Developer
Program cost.

## Generating Icons

The icons in `apps/desktop/src-tauri/icons/` are seeded from the PWA icons in `apps/web/static/`. Phase 1 only ships PNG variants (the minimum `tauri dev` needs). For platform-correct iconsets (`.icns` on macOS, `.ico` on Windows) generate them once:

```bash
pnpm --filter @june/desktop tauri icon ../../web/static/icon-512.png
```

Then update `apps/desktop/src-tauri/tauri.conf.json` to reference the generated `icon.icns` and `icon.ico` files. Commit the regenerated icons.

## Troubleshooting

**`error: linker 'cc' not found` on macOS** — Run `xcode-select --install` and retry.

**First build is very slow** — Expected. Tauri pulls a few hundred crates the first time. Subsequent builds use the local cargo cache.

**`pnpm desktop:dev` opens a blank window** — The Vite dev server probably did not start. Check `localhost:5173` in a regular browser. If the PWA loads but the Tauri window does not, restart the brain API and reopen.

**`localhost:8000` unreachable from the Tauri window** — The brain is not running. Tauri's webview honors the same CORS rules as a browser; the API has CORS configured for `localhost:5173` and `localhost:1420` (Tauri's default), so no extra setup is required.

**WebView2 missing on Windows 10** — Install the Evergreen runtime from Microsoft's WebView2 page. Windows 11 ships it by default.

## What's Wired Today

The shell is past the scaffold. As of the latest push:

### Capability layer (Phase 2)

The UI calls every native feature through `getPlatform()` from `@june/ui/platform`. On the desktop shell those calls become Tauri `invoke`s; in the browser they fall back to Web APIs or throw `UnsupportedError` for native-only methods. Settings has a "Send test notification" button you can use to verify your runtime: `web` falls back to the Web Notifications API, `tauri` goes through `tauri-plugin-notification`.

### Ollama supervision (Phase 3)

Visit `/help/ollama` inside the desktop shell. You'll see a one-click "Install Ollama → Start Ollama → Pull gemma4:e2b" panel that browser users don't get. The Rust side spawns `ollama serve` in-process (via `tokio::process`) and streams pull progress to the UI through Tauri events.

`bootstrap_ollama` opens the official OS-specific installer URL — Gatekeeper handles the macOS hand-off, UAC handles Windows, the install.sh page handles Linux. You return to the app and click "Start Ollama". This is by design (Phase 3 plan, "pragmatic caveat — bootstrap"); Phase 3.5 may swap to in-process download if real users ask.

### Native affordances (Phase 4)

- **Tray icon** — appears in the menu bar on launch. Left-click toggles the main window; right-click opens an Open / Quit menu.
- **Global hotkey** — `Cmd+Shift+J` on macOS, `Ctrl+Shift+J` on Windows and Linux. Fires from anywhere; toggles window visibility.
- **Autostart** — toggle in `/settings` under "Native shell". Off by default. Backed by `tauri-plugin-autostart` (`LaunchAgent` on macOS).
- **Window state** — size and position persist across launches via `tauri-plugin-window-state`.

If any of the above doesn't work on your machine, the most likely cause is a missing capability in `apps/desktop/src-tauri/capabilities/default.json` — which already lists `core:tray:default`, `global-shortcut:default`, `autostart:default`, `window-state:default`, `notification:default`, `shell:allow-open`. File a fix.

## Phase Status

| Phase | Status |
|---|---|
| 1. Scaffold | Shipped |
| 2. Capability layer | Shipped |
| 3. Ollama supervision | Shipped (bootstrap opens OS installer URL; in-process download deferred to 3.5) |
| 4. Native affordances | Shipped (hidden-inset title bar deferred to 4.5) |
| 5. Touch + responsive hardening | Backlog — see [responsive-plan.md](../product/responsive-plan.md) |
| 6. Distribution | Partial — v0.1.0 Apple Silicon DMG published; signing/notarization pending |
| 7. Migration + polish | Backlog |
