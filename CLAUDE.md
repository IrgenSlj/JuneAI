# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

June is being rebuilt (2026-11). It is the user's own interface to AI: a Mac
app (and the same UI in a browser at localhost) for chatting with any model the
user brings, with every conversation, note and plan kept as plain files in a
vault they own. Read `PLAN.md` (the only plan: decisions, runtime, API,
providers, file formats, milestones and their "done when" lists) and
`lab/README.md` (how work is done here) before starting. The previous June lives
on branch `legacy` and tag `v0-final`; its UI and Tauri shell are reused as
`PLAN.md` lists, its Python is not.

## Commands

- `pnpm install` — install everything (pnpm workspace, Node 26 as pinned in `.node-version`).
- `pnpm test:watch` — the inner loop: Vitest in watch mode across all packages. Keep it running while you work.
- `pnpm test` — run all tests once. Narrow with `pnpm test ids` (file name filter) or `pnpm test -t "sortable"` (test name). Do not put `--` before the filter: pnpm passes it through and Vitest then runs everything.
- `pnpm typecheck` — strict TypeScript across packages.
- `pnpm dev` — the daemon plus the UI with hot reload; prints the URL (with its token) to open.
- `pnpm dev:app` — builds the UI and daemon, then runs the Mac app (Tauri, debug) around them. Needs Rust. `JUNE_LOG_REQUESTS=1` makes the daemon log method, path and status.
- `pnpm test:e2e` — builds the UI and daemon, then runs the Playwright tests against the real daemon.
- `pnpm check` — the gate: typecheck, unit tests, e2e. Must pass before every commit. CI runs typecheck and unit tests on macOS, Windows and Linux, and e2e on macOS.

## Architecture in one paragraph

A pnpm monorepo in strict TypeScript. A local daemon (`apps/server`, `june
serve`, Node on 127.0.0.1 with a random port and a per-launch `X-June-Token`)
owns the vault and talks to model providers. The UI (`apps/web`, static
SvelteKit, components in `packages/ui`, tokens in `packages/design`) talks to it
over HTTP and SSE. The Mac app (`apps/desktop`, Tauri 2) is a thin window that
runs the daemon as a Node sidecar. Inside the daemon: `packages/engine` runs the
turn loop, `packages/providers` holds model adapters (network only through an
injected fetch, processes through an injected spawner), `packages/node` holds
the Node adapters, and `packages/core` owns vault formats, importers, receipts
and trace. Dependencies point one way:
`core <- node`, `core <- providers <- engine <- server`. `core` and `providers`
never import Node built-ins (`boundaries.test.ts` fails the gate if they do);
tests use the in-memory adapters in `packages/core/src/adapters/`. Packages and
apps are created when their milestone needs them.

Patterns that span packages:

- **Contract suites.** An adapter interface in `core` (for example `VaultFs`) has a shared suite in `packages/core/src/testing/`; both the in-memory and the Node adapter run it, so tests on memory cannot drift from the disk.
- **Cross-process safety.** A read-then-write that other processes might race (the receipts chain) runs inside a `VaultLock`: an in-process queue in tests, `nodeFileLock` (a lock file in `.june/locks/`) on disk. Behaviour across processes is tested with real child processes, bundled with esbuild (see `packages/node/src/lock.test.ts`).
- **Where things stand.** The Status table in `PLAN.md` names the current milestone and its next slices; start there.

## How we work: the lab

- **Ideas start as questions.** Anything non-trivial (a new feature, a library choice, a format decision, a performance claim) goes through the lab loop in `lab/README.md`: question, hypothesis, cheapest experiment, result, decision. One short file per experiment.
- **Innovate out loud.** When you see a better idea than the plan, say so with the evidence or propose the experiment that would produce it. Add smaller ideas to the backlog in `lab/README.md`. Do not build them unasked.
- **The founder makes product calls.** Ask with `AskUserQuestion` when a choice changes what the product is (audience, scope, a paid service, a dependency the rules below flag). Decide engineering details yourself and say what you chose.
- **Teach the why.** The founder is learning the stack. When you make a decision, give the reason in a sentence or two, in plain words.
- **Real data stays private.** Exports and real conversations go in `lab/private/` (git-ignored). Commit only small redacted fixtures.

## Rules

1. **One plan.** `PLAN.md` is the plan of record. Do not write new plans, ADRs or audits. If the plan is wrong, edit it (with the founder's agreement for product changes) and update its Status table when a milestone moves.
2. **One milestone at a time,** in the order of the milestone list in `PLAN.md`. Do not start code for a later milestone or anything on the plan's "deliberately out" list. Lab experiments may look ahead; production code may not.
3. **Files are the truth.** Anything in a vault's `.june/` must be rebuildable from the vault's files. Add a test for that whenever the index changes.
4. **The model never decides whether something was saved.** Writes are done by code paths with a result the UI reads. Model text is never the confirmation.
5. **Nothing leaves without a label.** Every network call to a model goes through `packages/providers`, is preceded by a label that is computed, recorded in the receipt and shown according to the user's transparency level, and has a receipt. If the receipt cannot be written, the call does not happen.
6. **Untrusted stays marked.** Imported text, web content and tool output carry `trust: untrusted` into any note derived from them, across turns and restarts.
7. **Tests come from "done when".** Each milestone's "done when" list becomes Vitest or Playwright tests before the feature is written. Unit tests alone do not count; the old repo had 1,214 of them and missed every cross-module bug.
8. **Measure, do not tune.** No retrieval or prompt tuning without a failing test that states the target.
9. **Small commits, plain messages.** Say what changed and why in two lines. One slice, `pnpm check` green, one commit, push. Never mention Claude or any AI tool in commit messages or co-author lines.
10. **Ask before adding a dependency** that has native code, a network call at install, or more than a few transitive packages. The choices fixed in `PLAN.md` are already approved.
11. **Reuse the UI and shell from `legacy` as listed in the plan.** Do not port any Python.
12. **Subscription routes ship only after their policy check is written in `lab/`** with the clause, the URL and the date.

## Conventions

- No emojis in code comments, README or docs.
- June is "June" or "it", never "she".
- Ids are ULIDs with a type prefix (`th_`, `nt_`, `pl_`), from `packages/core/src/ids.ts`.
- Keep docs short. The old repo had one line of documentation for every three lines of code.
