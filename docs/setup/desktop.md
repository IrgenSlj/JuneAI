# Desktop Shell Setup

This file documents how to run, build, and develop the June desktop shell at `apps/desktop/`. The shell is a Tauri 2.x wrapper around the SvelteKit build that already powers the PWA. The strategic plan lives in [`docs/product/desktop-shell-plan.md`](../product/desktop-shell-plan.md); this doc is the operational reference.

## Prerequisites

The desktop shell needs the same toolchain as the web app, plus Rust.

| Tool | Why | How |
|---|---|---|
| Node + pnpm | The web build the desktop shell wraps | Already required for the web app. |
| Python 3.10+ + the brain venv | The API the desktop shell talks to | Already required for the web app. |
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

The build is unsigned in Phase 1. Code signing for macOS lands in Phase 6 (see the desktop-shell plan).

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

## Phase Status

This setup doc covers what is actually wired today (Phase 1: scaffold). As later phases land, expect this doc to grow:

- Phase 3 will add a section on running Ollama supervision integration tests.
- Phase 4 will document the global hotkey, tray, and autostart toggles.
- Phase 6 will document code signing, notarization, and the GitHub Actions release workflow.
