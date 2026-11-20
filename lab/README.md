# The lab

June is built as lab work: every non-trivial idea starts as a question, gets the
cheapest experiment that could answer it, and ends in a decision that is written
down. The founder and the agent are co-investigators. The agent proposes ideas
and challenges assumptions; the founder makes the product calls.

## The loop

1. **Question.** One sentence the founder would care about.
2. **Hypothesis.** What we expect, with a number where one exists.
3. **Cheapest experiment.** A script, a spike branch, a measurement, reading real data. Hours, not weeks.
4. **Result.** What actually happened, with the numbers. Surprises first.
5. **Decision.** Keep, change or drop, and the edit it makes to `PLAN.md` (or "none").

An experiment is one file, `lab/NNN-short-name.md`, using the five headings
above. Keep it under a page. Throwaway code lives in `lab/scripts/` or on a
`lab/NNN-*` branch, never in `packages/` until the decision says so.

Private material (exports, real conversations, keys) goes in `lab/private/`,
which git ignores. Only redacted fixtures are ever committed. `tools/privacy-check.mjs`
blocks exports, archives, vault files and anything in `lab/private/` at commit
(a hook `pnpm install` sets up) and in the gate. Findings from real data are
written as counts and shapes, never quoted text.

## Index

| # | Question | State |
| --- | --- | --- |
| [001](001-claude-export.md) | What does a real Claude export contain, and can it round-trip into the vault format losslessly? | Open: the founder has the export; it comes in at the start of M2 |
| [002](002-subscriptions.md) | Can June serve the vault read-only to other assistants over MCP? | Open; now part of M7 (subscriptions are reached through official routes in M4 instead) |
| [004](004-sqlite-fts5.md) | Does the pinned Node build's `node:sqlite` include FTS5? | Done: yes; use a trigram index, because word search misses compounds like "Küchenbudget" |
| [005](005-data-notes.md) | What does each remote route do with what June sends? | Done 2026-11-15: notes on every preset; Zen's free models no longer work outside OpenCode (founder question) |
| [006](006-free-route.md) | What should "free" mean now that Zen's free models are closed to June? | Proposed: local or a key, no separate free gateway; founder decides |
| [003](003-business-model.md) | How does June make money without breaking the privacy promise? | Open: founder picks the first hypothesis |

## Idea backlog

Ideas worth an experiment later. Add freely; promote one to a numbered file
when it is next.

- Map density: does Canvas 2D hold 5,000 threads at 60 fps on old hardware? (gates M6)
- EU jurisdiction label: can the send label state a provider's processing region from public data, honestly?
- Cost-aware routing across the user's own keys and local models: which turns does a small local model handle as well as a frontier one?
- GDPR access helper: exports are incomplete (Claude reportedly omits artifacts); June drafts an Article 15 access request for the missing data. Needs a legal read before shipping.
- Append-only thread writes: append each turn block instead of rewriting the file, so no writer ever overwrites another. Needs a torn-write answer first (a crash mid-append leaves half a turn); today's atomic rewrite plus a thread lock is the safe default.
