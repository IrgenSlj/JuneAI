# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

June is being rebuilt from scratch (2026-11). It is a private, EU-first AI
assistant whose first product is a local vault of every AI conversation the user
has, drawn as one Map. Read `PLAN.md` (the only plan: decisions, file formats,
architecture, milestones and their "done when" lists) and `lab/README.md` (how
work is done here) before starting. The previous Python June lives on branch
`legacy` and tag `v0-final`; it is reference for ideas only.

## Commands

- `pnpm install` — install everything (pnpm workspace, Node 22+).
- `pnpm test:watch` — the inner loop: Vitest in watch mode across all packages. Keep it running while you work.
- `pnpm test` — run all tests once. Narrow with `pnpm test -- ids` (file name filter) or `pnpm test -- -t "monotonic"` (test name).
- `pnpm typecheck` — strict TypeScript across packages.
- `pnpm check` — the gate: typecheck then tests. Must pass before every commit. CI runs it on macOS, Windows and Linux.

## Architecture in one paragraph

A pnpm monorepo in strict TypeScript. `packages/core` owns the vault: file
formats, importers, index, links, receipts, revisions, trace queries. It is
platform-agnostic: it reaches files, the index and git only through the
interfaces in `packages/core/src/adapters.ts`, never through Node built-ins,
Electron or the network. `boundaries.test.ts` fails the gate if a source file
imports one. Desktop (Electron, `apps/desktop`, M0) plugs in Node adapters;
tests use the in-memory adapters in `packages/core/src/adapters/`; a phone shell
later plugs in its own. Other packages (`providers`, `context`, `mcp`) and apps
are created when their milestone starts, not before.

## How we work: the lab

- **Ideas start as questions.** Anything non-trivial (a new feature, a library choice, a format decision, a performance claim) goes through the lab loop in `lab/README.md`: question, hypothesis, cheapest experiment, result, decision. One short file per experiment.
- **Innovate out loud.** When you see a better idea than the plan, say so with the evidence or propose the experiment that would produce it. Add smaller ideas to the backlog in `lab/README.md`. Do not build them unasked.
- **The founder makes product calls.** Ask with `AskUserQuestion` when a choice changes what the product is (audience, scope, a paid service, a dependency the rules below flag). Decide engineering details yourself and say what you chose.
- **Teach the why.** The founder is learning the stack. When you make a decision, give the reason in a sentence or two, in plain words.
- **Real data stays private.** Exports and real conversations go in `lab/private/` (git-ignored). Commit only small redacted fixtures.

## Rules

1. **One plan.** `PLAN.md` is the plan of record. Do not write new plans, ADRs or audits. If the plan is wrong, edit it (with the founder's agreement for product changes) and update its Status table when a milestone moves.
2. **One milestone at a time.** Do not start code for a later milestone or anything on the plan's "not in this plan" list. Lab experiments may look ahead; production code may not.
3. **Files are the truth.** Anything in a vault's `.june/` must be rebuildable from the vault's files. Add a test for that whenever the index changes.
4. **The model never decides whether something was saved.** Writes are done by code paths with a result the UI reads. Model text is never the confirmation.
5. **Nothing leaves without a label.** Every network call to a model goes through `packages/providers`, is preceded by a label the user saw, and is followed by a receipt. If the receipt cannot be written, the call does not happen.
6. **Untrusted stays marked.** Imported text, web content and tool output carry `trust: untrusted` into any note derived from them, across turns and restarts.
7. **Tests come from "done when".** Each milestone's "done when" list becomes Vitest or Playwright tests before the feature is written. Unit tests alone do not count; the old repo had 1,214 of them and missed every cross-module bug.
8. **Measure, do not tune.** No retrieval or prompt tuning without a failing test that states the target.
9. **Small commits, plain messages.** Say what changed and why in two lines. One slice, `pnpm check` green, one commit, push. Never mention Claude or any AI tool in commit messages or co-author lines.
10. **Ask before adding a dependency** that has native code, a network call at install, or more than a few transitive packages. The choices fixed in `PLAN.md` are already approved.
11. **Do not port old code.** `legacy` is reference for ideas only.

## Conventions

- No emojis in code comments, README or docs.
- June is "June" or "it", never "she".
- Ids are ULIDs with a type prefix (`th_`, `nt_`, `pl_`), from `packages/core/src/ids.ts`.
- Keep docs short. The old repo had one line of documentation for every three lines of code.
