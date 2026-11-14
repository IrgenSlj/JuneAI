# June — plan of record

As of 2026-11-14. This is the only plan. Edit it in place when the plan is
wrong; do not write a second one. Experiments that change it live in `lab/`.

June is a private, EU-first AI assistant that lives on your machine. Its first
product is a **local vault for every AI conversation you have**, with one visual
**Map** that traces what was said, what was remembered and what was planned.
The destination is a premium daily assistant for professional, intellectual and
family life, running on the user's model of choice, with June owning the
interface, the harness, the memory and the agentic flows. The vault comes first
because an assistant whose memory you cannot trust is worth nothing, and an
imported history gives value on day one.

Origin: the rebuild brief written with the founder on 2026-11-14 (claude.ai doc
"June rebuild — development plan for Claude Code"), plus the edits agreed the
same day: audience, destination, Linux, a phone milestone after the gate, and
the adapter rule. The Python June is on branch `legacy` / tag `v0-final`.

## Status

| Milestone | State |
| --- | --- |
| M0 Reset | In progress: old tree removed, TypeScript monorepo and `packages/core` with test loop and CI in place. Electron shell, vault picker and empty Map still to do. |
| M1 Vault and import | Not started. Blocked on lab 001 (inspect the founder's real Claude export). |
| M2 onwards | Not started. |

## Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Audience (first) | EU professionals and knowledge workers who use two or more assistants | They feel the scattered-history problem now and will pay for private and premium. Family use follows, with the phone |
| Destination | Premium daily assistant, BYO model, June owns harness, memory, UX and agentic flows | The vault is the memory layer that assistant needs |
| Product centre (first) | Vault first, chat second | History search is a commodity; the vault plus the Map is not |
| Storage | Plain Markdown and JSONL files in a folder the user picks; SQLite is a rebuildable index only | File over app. The vault stays readable without June |
| Stack | TypeScript everywhere: Electron desktop shell, Svelte 5 UI, platform-agnostic core | One language across app, importers, extension and MCP server |
| Inference | Bring your own key, plus optional local models through Ollama. No hosted tier. Using existing subscriptions is open: lab 002 | No server means no logs, no controller duties and no bill |
| Capture | Official exports first (M1). Browser extension last (M6) | Exports are reliable and allowed; page capture is fragile |
| Platforms | Develop on macOS; CI builds macOS, Windows and Linux. Phone companion (iPhone first) after the gate | Electron gives all three desktops from one codebase; the Map needs a large screen; daily use lives on the phone |
| Agent scope | Now: plans are files; actions are vault operations and approved MCP tool calls only. Richer agentic flows after the gate, earned by numbers | The old June failed by letting a small model decide too much |
| Licence | MIT, open source | Trust claims need readable code |
| Signing | Unsigned builds until before the M2 gate | Signing needs paid accounts; the founder sets them up |

Four ideas carried over from the old June as design, not code: the hash-chained
receipts log, taint tracking for untrusted content, consent-gated MCP access,
and reversible forgetting.

## The Map

The Map is the home screen and the one thing no competitor has: a single time
canvas where conversations, memory and plans are the same picture. Read it left
to right: a message in an old ChatGPT thread wrote a budget note, a later Claude
turn read that note, the first planned step depends on it. Selecting any one of
those lights the whole path and dims everything else.

- **Lanes** are projects or topics, not providers. Colour carries the assistant.
- **Threads** are lines, turns are dots. Continuing a thread on another model draws a branch.
- **Notes** sit in a band underneath. A note is one remembered fact or decision, stored as a file with its revisions.
- **Plans** are the same lines continued past Now as dashed track. Each step is a stop; done steps turn solid and slide into the past.
- **Receipts** are a strip along the bottom: one tick per send, memory read, memory write and tool call.

Three zoom levels: **Map** (everything over time; edges hidden until something
is selected), **Thread** (one conversation as a reader, with a gutter showing
notes in, notes out, route label and receipt per turn), **Note or Plan** (the
file, its revision history and every place it was used).

**The trace** is the signature interaction. Backward: where did this note come
from. Forward: which answers did it shape. Forward trace exists only for turns
sent through June, because only then does June know what context was attached.
Imported threads get backward trace only, and the UI says so.

## The vault

The vault is a folder of plain files and the only source of truth. Deleting
`.june/` must lose nothing; the app rebuilds it.

```
MyVault/
  threads/2026/11/2026-11-14-kitchen-quotes.md   one conversation per file
  notes/budget-kitchen.md                        one fact or decision per file
  plans/kitchen-renovation.md                    one plan per file
  lanes.md                                       lane names, order, colours
  packs/claude-desktop.md                        what one assistant may read
  attachments/                                   files referenced by threads
  receipts/2026-11.jsonl                         append-only, hash-chained
  .june/index.sqlite                             derived: search, links, layout
  .june/revisions/                               derived from git history
```

**Thread file.** YAML frontmatter, then one `## ` heading per turn with a stable anchor.

```markdown
---
id: th_01JD3...
title: Kitchen quotes
lane: kitchen-renovation
source: chatgpt-export        # or claude-export, june, capture
source_id: 6734ab...          # provider's own id, used for dedupe
model: <as exported>          # copied from the export; never guessed
started: 2026-09-02T18:04:11Z
parent: th_01JC9...#m7        # set when this thread is a branch
---

## user ^m1
What should a 12 m2 kitchen cost?

## assistant ^m2
...
```

**Note file.** One claim, its status and its sources. A note with no source link is invalid.

```markdown
---
id: nt_01JD4...
kind: fact                    # fact | decision | preference | question
status: accepted              # proposed | accepted | retired
trust: user                   # user | model | untrusted
sources: ["th_01JD3...#m5"]
---
Kitchen budget is 18k maximum, including appliances.
```

**Plan file.** A goal and a checklist. Each step carries a state and the notes it depends on.

```markdown
---
id: pl_01JD5...
lane: kitchen-renovation
status: active
---
# Kitchen renovation
- [ ] Get three quotes ^s1  {state: approved, needs: [nt_01JD4...]}
- [ ] Pick a contractor ^s2 {state: proposed}
```

**Rules that hold everywhere**

- Filenames are ASCII slugs derived from titles plus a date; the real title lives in frontmatter. Paths obey `vaultPath()` in core, so a vault opens on macOS, Windows and Linux (reserved names, case-only collisions, Unicode NFC).
- Ids are ULIDs with a type prefix (`th_`, `nt_`, `pl_`). Links are `[[id]]` or `[[id#anchor]]`, so files stay valid in Obsidian.
- Imports are idempotent: the same export imported twice changes nothing. Dedupe on `source` plus `source_id`.
- Timestamps are the provider's originals, never the import time.
- The vault is a git repository managed by June. Every accepted change to a note or plan is one commit; that is the revision history.
- A receipt line holds kind, time, actor, target ids, a payload hash and the previous line's hash. It never holds message content.
- Keys live in the OS keychain. Nothing secret is ever written into the vault.

## Architecture

A pnpm monorepo in TypeScript, strict mode.

| Package | Owns | Must not |
| --- | --- | --- |
| `packages/core` | Vault read and write, file formats, importers, index, links, receipts, revisions, trace queries | Import Electron or any Node built-in, touch the network, or call a model |
| `packages/providers` | Model adapters (OpenAI-compatible, Anthropic, Mistral, Ollama), streaming, route metadata for the label | Read the vault directly |
| `packages/context` | Decides which notes attach to a turn, records that set, runs redaction | Send anything; it returns a proposal the UI shows |
| `packages/mcp` | MCP server over the core: scoped reads by pack, `propose_note`, receipts | Write accepted notes; proposals only |
| `apps/desktop` | Electron main process, IPC, keychain, Svelte 5 UI, the Map renderer, Node adapters for core | Hold business logic; it calls core |
| `apps/extension` | Browser capture for provider pages (M6) | Ship before M6 |

**The adapter rule.** Core reaches files, the index and git only through the
interfaces in `packages/core/src/adapters.ts`. Desktop plugs in Node
implementations; tests use the in-memory ones; a phone shell later plugs in
its own. This keeps the phone and browser open without building either now,
and it is enforced by `packages/core/src/boundaries.test.ts`. Packages are
created when their milestone starts, not before.

**Fixed technical choices**

- Index: SQLite with FTS5 behind the index adapter (desktop: `better-sqlite3` or `node:sqlite`, chosen in M1). Vector search is deferred until a user asks for it.
- Map renderer: Canvas 2D with `d3-zoom` and a quadtree for hit-testing. SVG will not hold 5,000 threads at 60 fps.
- Revisions: `isomorphic-git`, so no system git is required.
- Electron hardening from the first commit: context isolation on, node integration off, sandboxed renderer, a strict CSP, a typed IPC allowlist.
- All model output and all imported text render as sanitised Markdown. No raw HTML, ever.
- Tests: Vitest for packages, Playwright for the desktop app. Importers are tested against fixture exports committed with personal data removed.

## Milestones

Built in order, with a hard stop for outside users after M2. Sizes are relative.

1. **M0 Reset (S).** Tag `v0-final`, branch `legacy`, clean `main`. Monorepo, CI (typecheck, Vitest on three OSes, one Playwright smoke test), unsigned desktop builds.
    - Done when: a fresh clone builds, the app opens, lets you pick a vault folder and shows an empty Map.
2. **M1 Vault and import (M).** File formats, the core read and write API, importers for Claude then ChatGPT exports, the index, full-text search and the Thread reader.
    - Done when: a 1,000-conversation export imports in under 60 seconds; a second import changes zero files; deleting `.june/` and reopening rebuilds an identical index; search returns in under 100 ms; the founder's full Claude history imports with every conversation, original timestamps, attachments and project grouping intact, and a count report (found, imported, skipped, with reasons) matches the export.
    - First task: lab 001. Inspect the real export before writing the parser; formats change and must not be written from memory. The export stays outside the repo; commit only small redacted fixtures.
3. **M2 The Map, read-only (L).** Lanes, threads, turns, zoom, pan, filter by assistant and date, lane assignment by drag, open a thread from the Map.
    - Done when: 5,000 threads pan and zoom at 60 fps on a five-year-old laptop; every dot opens the right turn; keyboard navigation works.
4. **Gate: ten outside users.** Ship M2 as a free offline viewer for AI history. Do not start M3 until ten people who are not the founder have imported their own history and three have come back in a second week.
5. **M3 Chat, labels and receipts (M).** Provider adapters, streaming chat that writes thread files, continue any imported thread on any model as a branch, the route label before send, the receipts log and a `verify` command.
    - Done when: every send shows model, provider, jurisdiction and attached context before it leaves; three processes appending receipts at once lose zero lines and the chain verifies.
6. **M4 Notes and the trace (L).** Note files, git revisions, the notes band, backward and forward trace, and distillation: the model proposes notes from a thread and the user accepts or rejects each.
    - Done when: an explicit "remember that" stores a note without a model deciding, 100 times out of 100; no reply claims a note was saved unless a file was written; proposals from any turn that included web or pasted external content arrive marked untrusted.
7. **M5 Plans (M).** Plan files, step states, the dashed track right of Now, done steps becoming history, execution limited to vault operations.
    - Done when: no step runs without an approval click; every executed step has receipts; a plan edited in any text editor is picked up.
8. **M6 Packs, MCP and capture (L).** Scoped packs, the MCP server with `propose_note`, redacted export, the browser extension for live capture.
    - Done when: Claude Desktop reads only what its pack allows, each read leaves a receipt, and proposed notes land as `proposed`, never `accepted`.
9. **M7 Phone companion (L, after the gate numbers hold).** iPhone first: capture, chat and read notes against the same vault, the core running through phone adapters. Scope set by what M2 to M6 teach.

Not in this plan, by sequence rather than rejection: sync, hosted inference, a
skills system, scheduled or background agents, voice, vector search, Android.
Each comes back only through a lab result and an edit to this file.

## Risks and kill criteria

The biggest risk is the one that sank the last version: building for months with
no outside user. The gate after M2 exists to prevent that.

| Risk | What it would look like | Response |
| --- | --- | --- |
| Nobody wants a history viewer | Fewer than ten outside imports within a month of shipping M2 | Stop. Do not build M3 to M6 on hope |
| The Map is pretty and useless | Users open it once and then only use search | Make search the home screen and keep the Map as a view |
| Providers change export formats | An importer fails on a fresh export | Fixture tests per provider, a clear error naming the format version, a fast patch release |
| Importing is already free | Obsidian plugins import ChatGPT, Claude, Le Chat, Perplexity and Grok exports to Markdown today ([Nexus](https://github.com/Superkikim/nexus-ai-chat-importer)) | Do not compete on importing. Compete on the Map, the trace, receipts and the vault being usable from every assistant (lab 002) |
| Providers ship this themselves | A major assistant adds cross-provider history with provenance. Already moving: Claude imports memory from ChatGPT, Gemini and Copilot since March 2026, one way, into Claude | Lean on what they will not do: files you own and receipts across competitors |
| Edge hairball at scale | A trace on a heavily used note lights hundreds of turns | Cap visible edges, cluster by thread, show a count |
| Electron security mistake | Model output reaches a privileged API | The hardening list is a release blocker, tested in CI |
| Scope creep back into agents | Requests for background tasks, skills, scheduling | Refuse until M6 ships and the gate numbers hold |

One honest limit to state in the product: a local copy gives ownership, not
erasure. What a provider retains on its side is outside June's control, and the
send label is where the user learns that before each message.

## Design brief

Claude Design produces the Map first, because every other screen is a zoom
level of it. Five frames, in order: the Map with real density (about 300
threads over 18 months in six lanes); the Map with a trace selected (one note
lit, its source turn and three later turns ringed, everything else at 20%);
the Thread view with its gutter; the send label (model, provider, jurisdiction,
retention, attached notes, one being removed); a plan (one step done, one
awaiting approval, one proposed).

Direction: a transit map crossed with a git graph, calm lines, strong colour only
for assistants and the trace. Dark and light themes, both designed. One
interface typeface, one monospace for ids and receipts. Motion carries meaning in
exactly two places: the trace lighting up, and a finished step sliding across
Now. No chat-app cliches. The empty state is a single lane and one action:
import your history. Every meaningful colour has a second cue for colour-blind
users.

## Open questions for the founder

- Is the first audience EU professionals and knowledge workers, with family later? (Agreed in direction 2026-11-14; confirm wording before the M2 landing page.)
- Should a read-only "vault in every assistant" milestone (MCP) move before the Map, so the gate measures a reason to return? Decided by lab 002.
- How does June make money? Lab 003.
- How is the gate measured without telemetry? June sends nothing home, so "three came back in week two" needs either a strictly opt-in, content-free usage ping (shown in the receipts like any other egress) or ten short user calls. Decide before shipping M2.
- Index library: `better-sqlite3` (native, must match Electron's version) or `node:sqlite` (built in, newer). Decided by a spike in M1.
