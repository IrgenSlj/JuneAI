# June Desktop Shell

Tauri 2.x wrapper around the SvelteKit build at `apps/web/`. Same UI as the PWA, plus Rust commands for native capabilities.

This is the active development surface — see [`docs/product/desktop-shell-plan.md`](../../docs/product/desktop-shell-plan.md) for the phased plan and [`docs/setup/desktop.md`](../../docs/setup/desktop.md) for first-time setup.

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
pnpm desktop:build    # produces unsigned platform installers under src-tauri/target/release/bundle
```

Both commands assume the Rust toolchain is installed. See [`docs/setup/desktop.md`](../../docs/setup/desktop.md) if `cargo` is not on your PATH.

## Phase Status

- **Phase 1 (scaffold)** — current. Window opens, UI loads, no Rust commands yet.
- **Phase 2 (capability layer)** — pending. Typed `platform.ts` with Tauri / Capacitor / Web backends.
- **Phase 3 (Ollama supervision)** — pending. See [ADR 0008](../../docs/decisions/0008-ollama-supervision.md).
- **Phases 4–7** — pending. See the desktop-shell plan.
