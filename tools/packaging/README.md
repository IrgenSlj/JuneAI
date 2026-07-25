# Packaging the june-api sidecar

`june-api` (FastAPI + uvicorn on top of `june-brain`) is frozen with PyInstaller
into a self-contained, relocatable bundle that the Tauri desktop shell spawns as
a sidecar. Without it, a packaged `June.app` is a web window with nothing serving
`http://127.0.0.1:<port>`.

The spike that validated this approach is written up in
`docs/product/sidecar-spike-findings.md` (read it before changing the spec).

## Prerequisites

- Python matching the target (the spike used 3.14.3 on macOS arm64).
- A FRESH build venv, OUTSIDE the repo. Do NOT use `packages/brain/.venv` — the
  test gate depends on it, and PyInstaller pulls in build-only deps.

## Build

```sh
# 1. Fresh build venv outside the repo
python3 -m venv /tmp/june-build-venv

# 2. Install the two workspace packages, the (currently undeclared) openai
#    runtime dep, and PyInstaller into it.
/tmp/june-build-venv/bin/pip install \
    ./packages/brain ./packages/api "openai==2.32.0" pyinstaller

# 3. Freeze. Keep build artifacts OUT of the repo via --distpath/--workpath.
/tmp/june-build-venv/bin/pyinstaller --noconfirm \
    --distpath /tmp/june-dist --workpath /tmp/june-build \
    tools/packaging/june-api.spec
```

Output: an onedir bundle at `/tmp/june-dist/june-api/` (~40 MB). The executable
is `/tmp/june-dist/june-api/june-api`; its dependencies live in `_internal/`
(including `_internal/sqlite_vec/vec0.dylib`, the loadable SQLite vector
extension).

> `openai` must eventually be added to `packages/brain/pyproject.toml`
> dependencies. It is imported at module load by the Gemini provider but is not
> declared, so a clean install (and this freeze) needs it added explicitly.

## Building the whole app locally (`tauri build`)

Use `CI=true`:

```sh
cd apps/desktop && CI=true pnpm exec tauri build
```

Without it, `bundle_dmg.sh` fails with exit 64 at "Running AppleScript to make
Finder stuff pretty" whenever the shell cannot send Apple Events (any non-GUI
session, including an agent or CI-like terminal). `CI=true` skips that purely
cosmetic Finder-window styling step and the DMG builds normally. GitHub Actions
sets `CI` itself, so `.github/workflows/release.yml` never hits this.

If a build did fail there, detach the leftover volume before retrying —
`hdiutil detach /Volumes/dmg.*` — and delete the stray
`target/release/bundle/macos/rw.*.dmg` intermediate.

## Smoke test

The frozen sidecar must be smoke-tested by running **the packaged binary**, not
the dev entry point. They do not behave identically: the frozen bundle is
ad-hoc signed, so it has a different code identity from the dev interpreter, and
anything identity-scoped (macOS Keychain ACLs above all) takes a different path.
That difference hid a total chat hang through an entire green test gate — see
[`../../docs/product/cold-install-log-2026-07-25.md`](../../docs/product/cold-install-log-2026-07-25.md).

```sh
JUNE_DATA_DIR=/tmp/june-testdata \
JUNE_SKIP_MODEL_CHECK=1 \
JUNE_SKILLS_DISABLED=1 \
JUNE_API_PORT=8137 \
JUNE_BUILD_SHA=$(git rev-parse --short HEAD) \
  /tmp/june-dist/june-api/june-api &

# First boot imports the whole dep tree from the frozen archive — allow ~30s.
curl -s --retry 60 --retry-delay 1 --retry-connrefused http://127.0.0.1:8137/system
```

A 200 with JSON means uvicorn + FastAPI + brain config all work frozen. Set
`JUNE_BUILD_SHA` at package time; without it `/system` reports
`"version":"unknown"` (there is no git checkout inside the bundle).

## Wiring into the desktop shell

The bundle is copied into `June.app/Contents/Resources` (via Tauri `resources`
or `externalBin`) and spawned from `apps/desktop/src-tauri/src/lib.rs` on setup,
mirroring the existing `start_ollama` supervision in `src/ollama.rs`
(spawn → health-wait on `/system` → terminate the child on app exit). See the
findings doc's "Next step" section for the concrete plan and port strategy.
