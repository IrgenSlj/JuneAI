# Tauri desktop build report — unsigned `June.app`

Capstone integration proof of the desktop deploy path: a full **unsigned**
`tauri build` of the June desktop app, followed by headless verification that the
produced `June.app` bundles and runs the frozen brain + skills from the real
`.app` location.

- Date: 2026-07-02
- Host: macOS (Darwin 25.0), Apple Silicon (`arm64` / `aarch64-apple-darwin`)
- Toolchain: tauri-cli 2.10.1 (`apps/desktop/node_modules/.bin/tauri`), cargo (release), PyInstaller (onedir), Python 3.14.3 (build venv), pnpm/vite (web build)
- Repo HEAD at build time: `3d7d2e39`
- Invocation: `cd apps/desktop && pnpm exec tauri build` (no target flag needed — host arch is the default)

## Does an unsigned `June.app` build? YES

Both bundles built successfully (`tauri build` exit code 0):

| Artifact | Path | Size |
| --- | --- | --- |
| `June.app` | `apps/desktop/src-tauri/target/release/bundle/macos/June.app` | 103 MB |
| `June_0.1.0_aarch64.dmg` | `apps/desktop/src-tauri/target/release/bundle/dmg/June_0.1.0_aarch64.dmg` | 43 MB |

- Bundle identifier `com.juneai.desktop`, version `0.1.0`.
- Signing: `Signature=adhoc`, `TeamIdentifier=not set` — i.e. **unsigned / ad-hoc only** (no Developer ID, no notarization), exactly as intended. Tauri applies an ad-hoc signature automatically because arm64 macOS requires at least an ad-hoc signature to execute; this is not a Developer ID signature and does not satisfy Gatekeeper for distribution.
- Build wall-clock this run: ~2m26s (19:17:21 → 19:19:47). This is far below the 15-25 min cold estimate **only because the caches were warm**: the cargo `target/` was already ~4.3 GB (incremental release compile: `Finished release in 1m14s`) and the PyInstaller build venv under `$TMPDIR/june-sidecar-build` was reused. A clean-machine first build will still take the full 15-25 min (cold cargo LTO + fresh venv + full freeze).

## The brain binary + skills are bundled under Resources

Inside `June.app/Contents/Resources/june-api/`:

- `june-api` — the frozen entry-point executable. `Mach-O 64-bit executable arm64`, executable bit set, ad-hoc signed.
- `_internal/` — the PyInstaller onedir payload (85 MB): frozen `june_brain` + `june_api`, third-party deps, and the embedded PYZ.
- `_internal/sqlite_vec/vec0.dylib` — the loadable SQLite vector extension (load-bearing for semantic recall, ADR 0019). Present.
- `_internal/_build_sha.txt` — `3d7d2e39` (the git short SHA stamped by `build-sidecar.sh`, read by the runtime hook so `/system` reports the real build).
- First-party skills (`june_skill_calendar/health/research/files/daily`) are **pure-Python modules compiled into the embedded PYZ** inside the `june-api` executable — they do not appear as loose directories under `_internal/`. Their presence and importability is proven functionally below (`--run-skill calendar`).

## Headless run proofs (from the real `.app` location)

Launched the bundled binary directly (no GUI). Loopback TCP is blocked under the
default sandbox, so this run was done with the sandbox disabled (same constraint
the sidecar spike hit); this is a local-only loopback bind with a throwaway
`JUNE_DATA_DIR`.

### Proof 1 — bundled `/system` returns 200

Command (abbreviated):

```
JUNE_DATA_DIR=<tmp> JUNE_SKIP_MODEL_CHECK=1 JUNE_SKILLS_DISABLED=1 \
  JUNE_API_HOST=127.0.0.1 JUNE_API_PORT=<free> \
  June.app/Contents/Resources/june-api/june-api
curl http://127.0.0.1:<port>/system
```

- **HTTP 200** after ~5 s (frozen cold start, well within the 60 s health budget in `sidecar.rs`).
- Body reported `provider: gemma`, `label: "Gemma 4 (local)"`, `version: 3d7d2e39` (matches the stamped build SHA), plus honest degradation fields (`semantic_recall_status: degraded`, embedding model not pulled). The FastAPI app boots and serves from the frozen bundle.

### Proof 2 — `--run-skill calendar` MCP handshake

Command (abbreviated):

```
printf '<initialize JSON-RPC>\n' | \
  JUNE_IS_SKILL_SUBPROCESS=1 JUNE_SKILLS_DISABLED=1 JUNE_DATA_DIR=<tmp> \
  June.app/Contents/Resources/june-api/june-api --run-skill calendar
```

Response on stdout:

```json
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "june-calendar", "version": "0.1.0"}}}
```

- Valid JSON-RPC `initialize` result — the skill's MCP stdio server runs from the real bundle, which proves `june_skill_calendar` (and its `june_brain` import chain) is bundled and importable.
- Process exited on its own in ~2 s on stdin EOF; **0 stray `june-api` processes remained** afterward — the fork-bomb guard (`JUNE_IS_SKILL_SUBPROCESS`/`JUNE_SKILLS_DISABLED`) held.

All started processes were killed; temp data dirs removed.

## Config fix made (minimal, one line)

**File:** `apps/desktop/src-tauri/tauri.conf.json` — `build.beforeBuildCommand`.

The first end-to-end `tauri build` surfaced a latent path bug (the sidecar
freeze had previously only been exercised by running `build-sidecar.sh`
directly, never through the Tauri hook).

- Before: `pnpm --filter @june/web build && bash ../../../tools/packaging/build-sidecar.sh`
- After: `pnpm --filter @june/web build && bash "$(git rev-parse --show-toplevel)/tools/packaging/build-sidecar.sh"`

Root cause: Tauri runs `beforeBuildCommand` from the **invocation directory**
(`apps/desktop`, where `pnpm exec tauri build` is run — confirmed empirically),
not from `src-tauri`. The `../../../` (three levels up) therefore resolved to
`/Users/admin/tools/...` (nonexistent) and the hook failed with exit 127
(`No such file or directory`). The fix resolves the script via the git repo root,
so it is cwd-independent regardless of where the build is invoked — mirroring how
the web build already uses `pnpm --filter` for cwd-independence. It does not
touch the sidecar/resources wiring. (git is already a requirement of
`build-sidecar.sh`, which shells out to it for the build SHA.)

This is the only source change; it is left **unstaged** for review.

## Remaining steps that need the founder (signing + notarization)

The build is unsigned. To ship a Gatekeeper-passing, notarized `.dmg`, the
founder (who holds the Apple Developer account) must:

1. **Developer ID Application certificate** — enroll in the Apple Developer
   Program ($99/yr) and create/install a "Developer ID Application" cert in the
   login keychain (and a "Developer ID Installer" cert if a `.pkg` is ever
   wanted). Note the 10-char Team ID.

2. **Point Tauri at the signing identity.** Add a `bundle.macOS` block to
   `apps/desktop/src-tauri/tauri.conf.json`, e.g.:

   ```json
   "bundle": {
     "macOS": {
       "signingIdentity": "Developer ID Application: <Name> (<TEAMID>)",
       "hardenedRuntime": true,
       "entitlements": "src-tauri/entitlements.plist"
     }
   }
   ```

   or supply it at build time via env: `APPLE_SIGNING_IDENTITY="Developer ID Application: <Name> (<TEAMID>)"`.
   Hardened Runtime is required for notarization; a PyInstaller sidecar that
   maps writable/executable pages typically needs the entitlement
   `com.apple.security.cs.allow-unsigned-executable-memory` (and/or
   `allow-jit`, `disable-library-validation`) in `entitlements.plist` — validate
   empirically once signing is in place.

3. **Notarization credentials** (Tauri auto-notarizes + staples when signing and
   these are present). Provide **either**:
   - App Store Connect API key: `APPLE_API_ISSUER`, `APPLE_API_KEY`, `APPLE_API_KEY_PATH` (path to the `.p8`), **or**
   - Apple ID: `APPLE_ID`, `APPLE_PASSWORD` (an app-specific password), `APPLE_TEAM_ID`.

4. **Build signed + notarized:**

   ```
   cd apps/desktop
   APPLE_SIGNING_IDENTITY="Developer ID Application: <Name> (<TEAMID>)" \
   APPLE_TEAM_ID=<TEAMID> APPLE_ID=<apple-id> APPLE_PASSWORD=<app-specific-pw> \
   pnpm exec tauri build
   ```

   Then verify: `codesign --verify --deep --strict --verbose=2 June.app`,
   `spctl -a -vvv -t install June.app` (expect "accepted, source=Notarized
   Developer ID"), and `xcrun stapler validate June.app`.

These four steps are the only gap between this unsigned build and a
distributable, notarized June desktop app.

## Build artifacts are not staged

`git status` shows only the one-line `apps/desktop/src-tauri/tauri.conf.json`
change (plus this report). The `.app`/`.dmg` live under `target/` and the staged
onedir under `binaries/` — both gitignored and confirmed untracked/unstaged. The
project `.venv` was not touched (the freeze uses its own scratch venv under
`$TMPDIR`).
