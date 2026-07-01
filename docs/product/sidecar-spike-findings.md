# Sidecar packaging spike — freezing `june-api` for the desktop shell

Status: complete. Timeboxed spike. This is a DECISION with evidence, not shipped code.

## The deploy blocker this answers

`apps/desktop/src-tauri/src/lib.rs` supervises Ollama and native affordances but
never starts the Python brain/API. A packaged `June.app` is therefore a web UI
window with nothing serving `http://127.0.0.1:<port>` — every API call fails. To
ship a double-clickable app, `june-api` must be bundled as a sidecar the Tauri
shell spawns. The prerequisite question: **can `june-api` and its native
extensions be frozen into a relocatable, self-contained binary that boots and
serves HTTP?**

## 1. Verdict

**YES.** PyInstaller (6.21.0, Python 3.14.3, macOS arm64) freezes `june-api`
with all of its native extensions into a relocatable onedir bundle that boots
uvicorn, serves the FastAPI app, and successfully loads and executes the
sqlite-vec C extension — the single biggest risk.

## 2. Evidence

### Build
- Fresh build venv outside the repo; installed `./packages/brain ./packages/api`
  `openai==2.32.0 pyinstaller`. Confirmed all import before freezing:
  `june_api, june_brain, sqlite_vec, nacl, uvicorn, pydantic_core, openai` +
  `june_api.app`.
- Froze via a spec (final version at `tools/packaging/june-api.spec`). Onedir.

### Working spec / flags
The spec's load-bearing pieces:
- `collect_submodules("june_brain"|"june_api")` — uvicorn imports `june_api.app`
  by string and routes import lazily.
- `collect_data_files("june_brain")` — pulls the package-data read at runtime via
  `__file__` / `importlib.resources`: `providers/providers.toml`,
  `memory/extractor_prompt.txt`, `skills/registry_index.json`. All three
  confirmed present under `_internal/june_brain/...` in the bundle.
- **`collect_data_files("sqlite_vec")`** — the critical line. `vec0.dylib` is a
  loadable SQLite extension loaded at runtime via `conn.load_extension`, not
  linked to the package, so PyInstaller's analysis misses it. Collecting it as
  data lands it at `_internal/sqlite_vec/vec0.dylib`, exactly where
  `sqlite_vec.loadable_path()` (= `dirname(sqlite_vec.__file__)/vec0`) looks.
- `collect_submodules("uvicorn")` + explicit hidden imports for the
  loops/protocols/lifespan/logging leaves and `httptools/websockets/uvloop/
  watchfiles`.
- `collect_submodules("openai")` — see the dependency-gap note below.

`pynacl`'s libsodium (`_internal/nacl/_sodium.abi3.so`) and pydantic-core's Rust
extension (`_internal/pydantic_core/_pydantic_core.cpython-314-darwin.so`) were
collected automatically by PyInstaller's normal analysis — no extra config
needed.

### It runs and serves HTTP
Launched the frozen binary with `JUNE_DATA_DIR=<temp>`, `JUNE_SKIP_MODEL_CHECK=1`,
`JUNE_SKILLS_DISABLED=1` on a test port:

```
GET /system  -> HTTP 200
{"provider":"gemma","label":"Gemma 4 (local)","model":"gemma4:e2b","mode":"local",
 "privacy_label":"local-only","privacy_dial":"private_by_default",
 "base_url":"http://localhost:11434/v1","ollama_reachable":true,"ollama_has_model":true,
 "embedding_model":"nomic-embed-text","embedding_available":false,
 "semantic_recall_status":"degraded","semantic_recall_detail":"...",
 "api_key_present":false,"version":"unknown",
 "ledger_summary":{"count":0,"last_entry_ts":null,"egress_today":0,
 "chain_verified":null,"chain_verified_at":null}}
GET /healthz -> HTTP 200 {"status":"ok"}
```

A 200 with JSON proves uvicorn + FastAPI + `june_brain` config resolution all
work frozen. `"version":"unknown"` is `build_info.build_version()` degrading
gracefully — there is no git checkout in the bundle. Setting `JUNE_BUILD_SHA`
fixed it: a second run reported `"version":"spike-test"`. **Packaging must set
`JUNE_BUILD_SHA`.**

Startup also ran the lifespan hooks against a fresh empty DB and degraded
cleanly: `startup vec backfill skipped (no such table: semantic_facts)` and the
task-reconcile no-op — both expected on an empty data dir, neither fatal.

### sqlite-vec exercised (the whole risk) — PASS
`/system` alone does not create/query a vec0 table, so I proved the dylib works
inside the frozen runtime directly. A `vecprobe` entry (scratch only, not
shipped) run by the frozen binary:

```
sqlite_vec.__file__ = .../dist/june-api/_internal/sqlite_vec/__init__.py
loadable_path       = .../dist/june-api/_internal/sqlite_vec/vec0
vec_version         = v0.1.9
knn_query           = [(1, 0.0)]        # created a vec0 table, inserted, KNN queried
VECPROBE_OK
```

This is airtight: the frozen process loaded `vec0.dylib`, created a `vec0`
virtual table, and ran a nearest-neighbour query. Independently, the running
server's startup opened a DB connection (which calls `load_extension` in
`memory/sqlite.py`) and logged **no** "sqlite-vec unavailable" warning — that
warning only fires on load failure.

### Relocatable — PASS (critical for shipping into `June.app/Contents/Resources`)
Copied the whole bundle to a different directory and re-ran both `vecprobe` and
the full server there: `vec_version=v0.1.9`, `knn_query=[(1, 0.0)]`, `VECPROBE_OK`,
and `/system` -> 200 from the new path. PyInstaller resolves everything relative
to the executable at runtime, so the bundle can be moved into an `.app` freely.

### Binary size
Onedir bundle: **~40 MB** total (executable ~10 MB, deps in `_internal/`). This
is the API/brain only — Ollama and models are separate and already supervised.

## 3. The one real gotcha found (not a PyInstaller problem)

**`openai` is an undeclared runtime dependency.** `june_brain/providers/gemini.py`
does `from openai import AsyncOpenAI` at module load, and `providers/__init__.py`
imports `GeminiProvider` at top level — so `import june_api.app` hard-fails
without `openai`. Yet `openai` is **not** in `packages/brain/pyproject.toml`
dependencies; it is only present in the project venv incidentally. A clean
install (CI, a new contributor, or this freeze) misses it.

Action: add `openai` (spike used `2.32.0`) to `packages/brain/pyproject.toml`.
This is a latent bug independent of packaging and should be fixed regardless.

## 4. Residual risks / untested

- **Cold start ~13–30 s on first boot.** Importing the full dep tree (fastapi,
  pydantic, openai, uvicorn, brain) from the frozen archive is slow on first
  run; warm runs are faster. The sidecar health-wait in `lib.rs` must tolerate
  this (poll `/system` with a generous timeout, e.g. up to 60 s — not the 10 s
  Ollama uses).
- **Code signing / notarization not tested.** The spike binary is ad-hoc signed
  by PyInstaller. For a notarizable app every Mach-O in `_internal/` (including
  `vec0.dylib`, `_sodium.abi3.so`, `_pydantic_core...so`) must be signed with a
  Developer ID and hardened-runtime, then the `.app` notarized. Standard Tauri
  signing covers the app shell; the sidecar payload needs signing too. This is
  the next unknown to retire before a public build.
- **Only macOS arm64 built.** Windows/Linux freezes are expected to work the
  same way (same spec, platform-native `vec0`), but were not built here.
- **`chat` / embedding path not exercised end-to-end** (needs Ollama +
  `nomic-embed-text` pulled). The vec0 SQL path — the actual risk — was proven
  directly; the embedding call is ordinary HTTP to Ollama and low-risk.

## 5. Recommendation

**Use PyInstaller.** For a solo founder shipping a notarizable macOS app it is
the right call:

- It works today, end to end, including the sqlite-vec dylib that was the whole
  reason to spike — proven, not assumed.
- One relocatable onedir bundle drops straight into `June.app/Contents/Resources`
  and is spawned like any sidecar. It mirrors the existing `start_ollama`
  supervision, so there is one mental model for both child processes.
- The spec is ~90 lines and captures every native-extension quirk explicitly, so
  it is auditable and reproducible.
- ~40 MB is negligible next to Ollama + models.

Versus the fallback **python-build-standalone** (Astral relocatable CPython +
the app as a venv/zipapp resource): it avoids freezing entirely and sidesteps
hidden-import hunting, but it ships a full interpreter + site-packages tree
(larger, more files to sign/notarize), and still needs the same `vec0.dylib`
placement and `JUNE_BUILD_SHA` handling. It is the right escape hatch **only if**
notarization of the PyInstaller `_internal/` Mach-O set proves painful — which we
have not hit yet. Keep it in reserve; do not adopt it preemptively.

Recommendation: proceed with PyInstaller; keep python-build-standalone as the
documented fallback if signing/notarization of the frozen bundle blocks.

## 6. Next step — wire the sidecar into `lib.rs`

Mirror the `start_ollama` pattern in `apps/desktop/src-tauri/src/ollama.rs`.
Concretely, add a `brain.rs` module + `BrainState { child: Mutex<Option<Child>> }`:

1. **Bundle the artifact.** Add the onedir `june-api` to the Tauri bundle as a
   `resources` entry (or a per-target `externalBin`) so it lands in
   `June.app/Contents/Resources/june-api/`. Resolve it at runtime via
   `app.path().resolve("june-api/june-api", BaseDirectory::Resource)`.
2. **Spawn on setup.** In the `.setup(|app| ...)` closure (alongside
   `install_tray` / `register_hotkey`), spawn the resolved binary with
   `tokio::process::Command`, passing env: `JUNE_API_HOST=127.0.0.1`,
   `JUNE_API_PORT=<port>`, `JUNE_DATA_DIR=<app data dir>`, and
   `JUNE_BUILD_SHA=<baked at build>`. Store the `Child` in `BrainState` (managed
   like `OllamaState`).
3. **Health-wait on `/system`.** Poll `GET http://127.0.0.1:<port>/system` with
   `reqwest` until 200, up to ~60 s (frozen cold start is slower than Ollama's;
   do not reuse the 10 s budget). Emit progress events so the UI can show
   "Starting June…" instead of a dead window.
4. **Shut down on exit.** Register an exit/`RunEvent::ExitRequested` handler (or
   Drop on `BrainState`) that kills the child, exactly as Ollama's child is
   terminated, so no orphaned `june-api` survives the app.

**Port strategy: fixed port, recommended.** Use a fixed loopback port so the web
UI needs no runtime injection. The web UI resolves the API base from
`PUBLIC_JUNE_API_URL || "http://localhost:8000"` (`apps/web/src/lib/api.ts`),
and `$env/dynamic/public` is baked at build time for the static Tauri bundle —
there is no server to inject a runtime-chosen port. So either keep the default
`8000`, or pick a dedicated June port (e.g. `8137`) and set
`PUBLIC_JUNE_API_URL=http://127.0.0.1:8137` at desktop build time; pass the same
port to the sidecar via `JUNE_API_PORT`. Add a preflight check: if the port is
already in use, surface a clear error (or fall back to a small candidate list
and, only then, inject the chosen port via a Tauri-emitted config the client
reads at boot). Dynamic-port robustness is a later refinement, not needed for v1.

---

### Reproduction (all outside the repo)

Build venv, install, freeze, and smoke-test commands are in
`tools/packaging/README.md`. The final spec is `tools/packaging/june-api.spec`.
