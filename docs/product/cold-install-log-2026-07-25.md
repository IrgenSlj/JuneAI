# Cold-install verification — v0.1.0 candidate, 2026-07-25

Slice 1.2 of the [v0.3 execution plan](v0.3-execution-plan.md). This is the
written record of walking the packaged app from a state that has never run the
repo, and the defects that walk surfaced.

Related: [`cold-start-notes.md`](cold-start-notes.md) covers sidecar *launch
latency*; this covers whether the packaged product actually works.

## Method

- Built the DMG from `main` (`ac42baee` + Phase 0 docs) with
  `CI=true pnpm exec tauri build`.
- Mounted the DMG and ran `June.app/Contents/Resources/june-api/june-api`
  directly, with `JUNE_DATA_DIR` pointed at an empty scratch directory and a
  non-default port, so no repo venv, config, or memory was in scope.
- Compared every result against the dev entry point
  (`packages/brain/.venv/bin/june-api`) on its own empty data directory, so any
  difference is attributable to packaging rather than to the code.

## What worked

| Check | Result |
|---|---|
| Frozen sidecar boots on an empty data dir | Yes |
| `/healthz` | 200 |
| `/setup/status` | Correct: `is_configured: false`, Ollama and `gemma4:e2b` detected |
| First-party skills inside the frozen bundle | All five spawn (`health`, `calendar`, `files`, `research`, `daily`) |
| Chat produces a real answer | Yes |
| Memory write + recall across turns | Yes — told June the dog is "Biscuit", asked in a later turn, got "Your dog is named Biscuit." |
| Bundle contents | `Contents/Resources/june-api/june-api` present, 43MB DMG / 103MB app, ad-hoc signed |

Memory working end-to-end on a cold install — including with embeddings
unavailable, via the keyword fallback — is the single most important result here.

## Defect 1 (release-blocking) — every chat turn hangs in the packaged app

**Symptom.** The frozen sidecar streamed tokens and produced a complete answer,
then never emitted `provenance` or `done`. The stream never closed. Two turns
were killed at 180s and 300s. The dev entry point, same machine, same prompt,
same model, completed in **18s** with both frames.

**Root cause.** `sample(1)` on the hung process showed the main thread parked in:

```
SecItemCopyMatching -> SecItemCopyMatching_osx -> SecKeychainItemCopyContent
  -> ItemImpl::getContent -> SSGroupImpl::decodeDataBlob -> CSSM_DecryptDataFinal
  -> ClientSession::decrypt -> mach_msg -> mach_msg2_trap
```

A macOS Keychain read blocked on an authorization decision that never arrives.
The packaged binary is ad-hoc signed, so its code identity differs from the dev
interpreter that created the keychain items; the ACL check escalates to a user
prompt, and a headless sidecar has no way to answer one. `trust/signing.py`
loads the ledger's Ed25519 device key from that store at the end of every turn,
which is exactly where the turn stopped. `config_store` reads the Gemini key the
same way, which is why a later request blocked before its first byte.

**Proof.** Same frozen binary, same prompt, with
`PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring`: **2 seconds**, with
`provenance` and `done` emitted.

**Why it was invisible until now.** Only the packaged build has a different code
identity. Every dev run, every test, and the whole CI gate use an interpreter the
keychain already trusts. No amount of `check.sh` would have caught this — only
running the artifact a user would actually download.

**Fix.** `ec7d92ec` — every credential-store call runs under a hard deadline
(`JUNE_KEYRING_TIMEOUT_S`, default 2s) in a daemon thread, and the module latches
into a process-wide degraded mode on the first overrun, returning the
file-fallback answer immediately thereafter. A thread blocked in the platform
keychain cannot be cancelled, so the latch is what keeps one parked thread per
process from becoming one per call. Degradation is logged once and readable via
`secret_store.keyring_unresponsive()`.

**Note for signed builds.** Notarized builds with a stable Developer ID will have
one consistent code identity, so the ACL prompt should appear once and be
answerable. The deadline still matters: it converts "June hangs" into "June keeps
working with file-backed secrets," which is the correct behaviour either way.

## Defect 2 (cosmetic, fixed) — fresh install logs a traceback

`reconcile_running_after_restart` ran before the `tasks` table existed and logged
a full traceback on a clean install. Never raised, but it makes a healthy first
run look broken and buries tracebacks that matter. Fixed in `ec7d92ec`: the
"no such table" case is now a debug line.

## Defect 3 (open) — the embedding model is an invisible second download

`/setup/status` correctly reported `embedding_available: false` and
`semantic_recall_status: degraded`, because `nomic-embed-text` is a *separate*
Ollama pull from `gemma4:e2b`. The setup screen's readiness gate only checks
`ollama_reachable && ollama_has_model`, so a new user can complete setup, see
"Ollama is ready," and run June with semantic recall silently degraded to a
keyword scan.

Memory is the product. Shipping a default path where the memory system is
degraded and the setup screen says "ready" is the wrong first impression. The
`/help/ollama` guide does pull the embedding model, so the capability exists —
the readiness *gate* is what needs to account for it.

Tracked as a Phase 1.3 follow-up.

## Defect 4 (open, low) — local DMG bundling needs `CI=true`

`pnpm exec tauri build` fails locally at `bundle_dmg.sh` with exit 64 when the
Finder AppleScript step cannot send Apple Events from a non-GUI shell. `CI=true`
skips the cosmetic step and the build succeeds. GitHub Actions sets `CI`
automatically, so the release workflow is unaffected. Worth documenting in the
packaging README so the next person does not lose time to it.
