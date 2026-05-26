# June Desktop Shell

Tauri 2.x wrapper around the SvelteKit build at `apps/web/`. Same UI as the PWA, plus Rust commands for native capabilities.

This is the native desktop surface — see [`docs/product/desktop-shell-plan.md`](../../docs/product/desktop-shell-plan.md) for status and [`docs/setup/desktop.md`](../../docs/setup/desktop.md) for first-time setup.

## Layout

```
apps/desktop/
├── package.json           # @june/desktop — pnpm wrapper around Tauri CLI
└── src-tauri/
    ├── Cargo.toml         # Rust crate definition
    ├── build.rs           # Tauri build script
    ├── tauri.conf.json    # window, bundle, build-pipeline config
    ├── capabilities/      # permissions granted to each window
    └── src/
        ├── main.rs        # binary entry point
        └── lib.rs         # Tauri builder + plugin registration
```

## Running

```bash
# from the repo root
pnpm desktop:dev      # opens the UI in a native window, hot-reload via Vite
pnpm desktop:build    # produces platform installers under src-tauri/target/release/bundle
```

Both commands assume the Rust toolchain is installed. See [`docs/setup/desktop.md`](../../docs/setup/desktop.md) if `cargo` is not on your PATH.

## Phase Status

- **Phase 1 (scaffold)** — _Shipped._ Window opens, UI loads, dev server proxied via Vite.
- **Phase 2 (capability layer)** — _Shipped._ Typed `Platform` interface in `packages/ui/src/platform/`; Tauri / Web / Capacitor backends; `notify` end-to-end.
- **Phase 3 (Ollama supervision)** — _Shipped._ Five Rust commands in `src-tauri/src/ollama.rs`; `/help/ollama` drives one-click install/start/pull on the desktop shell. See [ADR 0008](../../docs/decisions/0008-ollama-supervision.md). `bootstrap_ollama` opens the OS-native installer URL; in-process download is a Phase 3.5 candidate.
- **Phase 4 (native affordances)** — _Shipped._ Tray icon with Open/Quit menu, global hotkey (`Cmd+Shift+J` / `Ctrl+Shift+J`), autostart toggle in `/settings`, window-state persistence. Hidden-inset title bar deferred to Phase 4.5.
- **Phase 5 (touch + responsive hardening)** — _Backlog._ See [responsive-plan.md](../../docs/product/responsive-plan.md).
- **Phase 6 (distribution)** — _Partial._ v0.1.0 Apple Silicon DMG is published on GitHub Releases. Developer ID signing and notarization are pending.
- **Phase 7 (migration + polish)** — _Backlog._ See the desktop-shell plan.

## Distribution Caveat

The current macOS DMG is ad-hoc signed and not notarized. It is suitable for
alpha testing, but macOS may show a first-launch security warning. Developer ID
signing and notarization are deferred until external testers justify the cost
and release-process work.
