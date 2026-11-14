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
which git ignores. Only redacted fixtures are ever committed.

## Index

| # | Question | State |
| --- | --- | --- |
| [001](001-claude-export.md) | What does a real Claude export contain, and can it round-trip into the vault format losslessly? | Open: waiting for the founder's export |
| [002](002-subscriptions.md) | Can June serve the vault read-only to the user's own assistants over MCP, as the daily-use reason? | Open, research done; prototype next |
| [003](003-business-model.md) | How does June make money without breaking the privacy promise? | Open: founder picks the first hypothesis |

## Idea backlog

Ideas worth an experiment later. Add freely; promote one to a numbered file
when it is next.

- Map density: does Canvas 2D hold 5,000 threads at 60 fps on old hardware? (gates M2)
- EU jurisdiction label: can the send label state a provider's processing region from public data, honestly?
- Cost-aware routing across the user's own keys and local models: which turns does a small local model handle as well as a frontier one?
- GDPR access helper: exports are incomplete (Claude reportedly omits artifacts); June drafts an Article 15 access request for the missing data. Needs a legal read before shipping.
- Watch Downloads: a new export zip is detected and merged automatically; imports are already idempotent.
- Phone core: does `packages/core` run unchanged inside an iOS shell through phone adapters? (gates M7)
