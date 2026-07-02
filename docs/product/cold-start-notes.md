# Sidecar cold-start: profile + findings (bounded spike)

_Frozen `june-api` sidecar: time from process launch to first `/healthz` 200._
_This is a UX polish item — the warming UI already covers the gap gracefully._
_Measured on macOS (Apple Silicon), PyInstaller **onedir** bundle, Python 3.14._

## (a) Baseline cold-start (old staged binary)

Launch the frozen `june-api` with a temp `JUNE_DATA_DIR`, `JUNE_SKIP_MODEL_CHECK=1`,
`JUNE_SKILLS_DISABLED=1`, a free port; poll `/healthz` until 200:

| run | time |
|-----|------|
| 1 (cold FS cache) | **25.4 s** |
| 2 (warm)          | 4.9 s |
| 3 (warm)          | 2.0 s |

The 13-30 s figure is the **truly-cold first launch**. Once the OS file cache is
warm the same binary answers in ~2 s. The variance is the tell: the dominant
first-run cost is not our Python, it's the OS.

## (b) Where the time goes (profile breakdown)

Instrumented the frozen binary (temporary env-gated trace in `__main__.main()`,
since removed) to split launch into phases:

| phase | cold (1st run) | warm |
|-------|----------------|------|
| launch -> `main()` (PyInstaller bootloader unpack + dylib load + **macOS security verification** + interpreter init) | **7.8 s** | 0.12 s |
| `import uvicorn` | 1.3 s | 0.04 s |
| `import june_api.app` (fastapi + pydantic + june_brain graph + all routes) | **7.0 s** | 1.7–2.2 s |
| `import openai` (now deferred; measured separately) | 0.47 s | 0.18 s |

Pure-Python import cost of `import june_api.app`, measured warm in the build venv
with `python -X importtime` (representative of the frozen import *work*, minus
the PYZ decompression multiplier):

- Total module self-time: **696 ms** before, **434 ms** after the openai deferral.
- Biggest roots: `openai` 263 ms (deferrable — only used mid-call), `fastapi`
  209 ms (needed), the rest is `pydantic` + the `june_brain` loop/context/router
  graph (all needed to build the app that answers `/healthz`).

**Takeaways:**
- **Cold** is dominated by the OS: PyInstaller unpacking the ~85 MB `_internal`
  onedir and macOS verifying each unsigned `.dylib` on first launch (~7.8 s
  bootloader) plus cold-cache reads of every module during app import (~7 s).
  Lazy imports barely touch this — the modules still ship and still get verified;
  they just get *imported* later.
- **Warm** is dominated by `import june_api.app` (~1.7–2.2 s): the framework +
  brain graph, decompressed from the PYZ archive. This is ~4× the loose-`.pyc`
  cost because frozen imports pay zlib-decompress + unmarshal per module.

## (c) Change made (the cheap, safe win)

Deferred `from openai import AsyncOpenAI` from module top to inside `_client()`
in both providers:

- `packages/brain/src/june_brain/providers/gemini.py`
- `packages/brain/src/june_brain/providers/gemma.py`

**Why it's safe:**
- `openai` was imported at module load in exactly these two files (nowhere else
  in the startup path). The provider layer is imported on every API boot via the
  loop/context/router chain, so this dragged the whole `openai` SDK tree
  (a large pydantic model surface) into the `/healthz` critical path.
- The name is only *evaluated* at the construction site; both files already use
  `from __future__ import annotations`, so the `-> AsyncOpenAI` /
  `AsyncOpenAI | None` annotations are strings (kept resolvable for mypy via a
  `TYPE_CHECKING` import). No public API, behavior, or import-graph change.
- Tests patch `GeminiProvider._client` / `GemmaProvider._client` (the methods),
  not the module-level `AsyncOpenAI` name, so no test relied on the eager import.
  The one test that builds a real client (`test_gemini_caches_client_across_calls`)
  still works — the deferred import runs at call time.
- `/healthz` never touches a provider, and model warmup is skipped under
  `JUNE_SKIP_MODEL_CHECK`, so `openai` is now imported only when inference
  actually runs.

## (d) After timing (new staged binary)

| run | time |
|-----|------|
| 1 (cold, post-rebuild) | 9.1 s |
| 2–5 (warm) | 1.32 / 1.25 / 1.19 / 1.19 s |

The rigorous, noise-free signal is the traced breakdown: `openai` (**~0.18 s warm /
~0.47 s cold**) is now off the `/healthz` path entirely. End-to-end warm floor
sits ~1.2 s (was ~2.0 s), though part of that spread is measurement noise
(scheduler, cache state, uvicorn socket setup). Defensible measured win:
**~0.18–0.47 s** removed from the critical path, zero risk.

## (e) Gate

`./tools/check.sh` -> **EXIT=0** (brain+api pytest, frontend check, OpenAPI drift,
ruff, mypy all green).

## (f) Honest assessment + recommendations

The **bulk of the cold-start cost is not cheaply reducible with lazy imports.**
It splits into two costs neither of which is a Python-import problem:

1. **Cold bootloader (~7–8 s, first launch only): macOS security verification of
   the unsigned onedir.** This is the single biggest lever and it's a *packaging*
   fix, not a code fix:
   - **Codesign + notarize** the frozen `june-api` and its bundled `.dylib`s in
     the Tauri build pipeline. Gatekeeper caches verification per signed file, so
     the first-launch scan collapses. Highest ROI.
   - At minimum, ad-hoc `codesign` the sidecar and strip `com.apple.quarantine`
     on install.
2. **Warm app import (~1.7 s): fastapi + pydantic + the june_brain graph.** All of
   it is genuinely needed to *construct the app that answers `/healthz`* — the
   whole point of `/healthz` is that answering it means "ready." Deferring routers
   to answer `/healthz` sooner would make it a **dishonest readiness signal**, so
   we deliberately did **not** do that.

Other levers, ranked:
- **Trim the bundle** (smaller `_internal` = less to unpack + verify). `openai`
  ships a large `types` tree; if only `chat.completions` is used, most of it is
  dead weight. But spec `excludes` are risky (a wrong exclude breaks the frozen
  app), so measure first — prefer lazy imports over excludes, as the guardrails say.
- **Do NOT switch onedir -> onefile.** onefile unpacks the whole archive to a
  temp dir on *every* launch — it makes cold start worse, not better.
- **Keep the openai deferral** (done): free ~0.18–0.47 s, no downside.

**Recommendation:** the warming UI already hides this gap, so treat cold start as
a low-priority polish item. If pursued, the one high-value next step is
**code-signing/notarizing the sidecar in the desktop bundle pipeline** — that
targets the 7–8 s cold bootloader that lazy imports can't. Everything else is
marginal.

## (g) Deviations

- Baseline was measured on the previously-staged binary; "after" on the rebuilt
  binary. Both use identical methodology; the clean apples-to-apples signal is the
  traced phase breakdown, not the noisy end-to-end wall clock.
- The build venv installs local packages non-editably and pip skips same-version
  reinstalls, so edited `brain`/`api` were `--force-reinstall --no-deps`'d into the
  existing scratch build venv before each freeze to guarantee edits landed. The
  project venv was never touched.
- Temporary env-gated timing instrumentation was added to `__main__.py` for the
  breakdown, then fully removed; the final staged binary and source contain none.
  Only the two provider files are left modified (unstaged).
