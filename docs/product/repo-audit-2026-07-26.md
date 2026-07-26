# Repo audit — 2026-07-26

A file-by-file pass over the repository: what is tracked, what is dead, what is
duplicated, and what is wired inefficiently. Findings are separated by whether
they were fixed in this pass, need a decision, or are deliberately left alone.

Scope: 483 tracked files (292 `packages/`, 94 `docs/`, 70 `apps/`, 19 `tools/`,
19 `skills/`, 7 `assets/`).

---

## Health summary

| Check | Result |
|---|---|
| Tracked junk (caches, venvs, `.DS_Store`, build output) | **None.** `.gitignore` covers every case found |
| Untracked-but-unignored files | **None.** `git status` is clean |
| Unused UI components | **None.** Every `packages/ui` component has a consumer |
| Dead modules in `loop/` | **None.** All eight are imported; `engine.py` still has 9 referrers |
| Workspace wiring (`pnpm-workspace.yaml`) | Correct — `apps/*`, `packages/ui`, `packages/design` |
| Python dependency declarations | Clean, and each non-obvious one carries a comment explaining why it is direct |
| CI parity with the local gate | CI runs `./tools/check.sh` — the same gate, not a divergent copy |
| Generated artifacts tracked on purpose | `openapi.json` and `types.ts` are committed so the drift check can fail |

The codebase itself is in good shape. Everything below is either doc hygiene,
history weight, or a coverage gap — not structural rot.

---

## Fixed in this pass

1. **`tools/migrate_v1_data.py` deleted.** Zero references anywhere; it migrated
   data from the v1 Streamlit app, which was removed from git in `a248ab92`.
   Recoverable from history if ever needed.
2. **`tools/migrate_chroma_to_sqlitevec.py` marked HISTORICAL.** Chroma stopped
   being a dependency at ADR 0019. The file stays only because that ADR
   references it by path and ADRs are append-only — the same convention the doc
   archive uses.
3. **`JUNE_V02_BRIEF.md` and `v0.2-execution-plan.md` given SUPERSEDED banners.**
   Both were listed as superseded in `docs/archive/README.md` but carried no
   banner in the file itself, so anyone opening them directly — the likely path
   for an agent — would read a stale plan as current. This was the single most
   dangerous doc defect in the repo.
4. **`CHANGELOG.md` brought current.** It stopped at 2026-06-28 and recorded
   none of the release repair, retrieval measurement, or visual work.
5. **`cold-install-log-2026-07-25.md` added to the archive index** as a
   historical record rather than a live spec.

---

## Needs a decision

### D1 — `.git` is 240MB because of committed v1 artifacts

History contains a committed virtualenv and training data that no longer exist
in the working tree:

| Blob | Size |
|---|---|
| `misc/models/june_0_0.pt` | 68.0 MB |
| `junevenv/` site-packages (numpy, zstandard, cryptography, notebook, debugpy…) | ~90 MB combined |
| `data/mnist/mnist.pkl.gz` | 15.4 MB |

Every clone pays for these forever. A `git filter-repo` pass would cut the repo
to roughly a tenth of its size. It rewrites every commit hash, which normally
makes it a hard sell — but this repo has **0 forks and 2 watchers**, so the
blast radius is one machine. This is the cheapest it will ever be to do.

**Recommendation:** do it, before the project has contributors. Requires a force
push and re-cloning locally.

### D2 — Untracked local directories (not in git, deletion is unrecoverable)

| Path | Size | What it is |
|---|---|---|
| `JuneAI-app/` | 508 KB | v1 Streamlit source; the tracked copy was deleted in `a248ab92` and remains in history |
| `.june_memory/` | 4.2 MB | v1 data directory — contains `chroma/` and a `june.db`. **May hold real personal data** |
| `.tmp/` | 11 MB | scratch dirs (`c4`, `c4f2`, `c5`, …) |
| `apps/*/target/`, `.mypy_cache/`, `.venv/`, `node_modules/` | ~5.5 GB | rebuildable caches |

All are correctly gitignored, so they affect only local disk. `.june_memory/`
is the one to look at before deleting — it is the only item here that could
contain something you want.

### D3 — Positioning (from the sharpen pass)

Still open, and it gates how the remaining plan is ordered: does June compete
with OpenClaw on *auditability*, or plug into that ecosystem as memory
infrastructure? See "Remaining development" below.

---

## Deliberately left alone

- **Six plan documents.** `JUNE_V02_BRIEF`, `rebuild-plan`, `development-plan`,
  `v0.2-execution-plan`, plus the two v0.3 docs. This looks like sprawl, and is,
  but `docs/archive/README.md` sets a deliberate policy: superseded plans stay at
  their original paths with banners, because append-only ADRs link them and
  moving files would either break those links or force an ADR edit. The policy is
  sound; the defect was the missing banners, now fixed.
- **Two roadmaps** (`ROADMAP.md`, `docs/product/roadmap.md`). The root one is the
  public track view, the docs one sequences tiers. Both are current and
  cross-linked.
- **`loop/engine.py`.** Reads as a leftover from the pre-ADR-0018 era but is
  still imported by nine modules. Not dead.
- **`packages/brain/build/`, `skills/*/build/`.** Local `pip install` artifacts,
  correctly ignored. Worth knowing they exist because they duplicate the entire
  source tree and will pollute any `grep -r` that does not exclude them.

---

## Coverage gaps found

| Gap | Risk |
|---|---|
| No e2e spec for `/setup` | **Highest.** It is the first-run flow — the one path every new user takes, and it was changed in this session's Phase 1.3 |
| No e2e spec for `/settings`, `/skills`, `/help/ollama` | `/help/ollama` is what a new user is sent to when a model is missing |
| `ollama_manager.py` (357 lines) referenced by no test | 5 modules import it; it drives install/pull/start |
| `tool_aliases.py` (391 lines) referenced by no test | 1 importer |
| Nothing exercises the packaged artifact | The Keychain hang passed 986 green tests. Only running the DMG caught it |
