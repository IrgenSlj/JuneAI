# Desktop Icons

Phase 1 seeds these from the PWA icons in `apps/web/static/`. They are the minimum needed for `tauri dev` to run.

For `tauri build` you also need `icon.icns` (macOS) and `icon.ico` (Windows). Generate them once with the Tauri CLI:

```bash
pnpm --filter @june/desktop tauri icon ../../web/static/icon-512.png
```

That command rewrites this directory with the full set of platform variants. Update `tauri.conf.json` to reference the generated `icon.icns` and `icon.ico` after running it.

The source of truth for the icon design lives in `packages/design/`; do not edit pixels here directly.
