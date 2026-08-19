# Tool-selection accuracy — August 2026 (D.5d)

D.5a cut `JUNE_TOOLS_GEMMA` from 24 tools to 5 on the argument that a small
model picking between near-synonyms is where wrong tool calls come from, and
that `tool_aliases.py` existed to paper over those wrong calls. D.5d is where
that argument gets checked instead of restated.

**Instrument.** `tools/tool_selection_harness.py`, with the scoring in
`june_brain/experiments/tool_selection.py` (pure, unit-tested — a benchmark
whose own arithmetic is unverified is worse than none, because it produces a
number people quote). 24 cases, three passes, n=72. `gemma4:e2b` via Ollama,
driven through the real loop with tool dispatch live and writes sent to a
throwaway data dir. MacBook, Apple Silicon.

**This is the first measurement in this repo taken with skills running.**
`check.sh` sets `JUNE_SKILLS_DISABLED=1`, so no test had ever seen the tool list
the model is actually offered. On a default install that list is **15 tools** —
the 5 native ones plus calendar (3), research (2) and files (5) — not 5.

## Two metrics, because one hides the answer

- **first-call accuracy** — the first tool of the turn is the expected one.
- **reached-tool accuracy** — the expected tool ran *at some point* in the turn.

They differ where a model looks before it writes. Answering "the passport
renewal is done" with `list_promises` and then `update_promise` is right in two
steps, and the user's promise does get updated; first-call alone records that as
a failure. Where the two diverge is the interesting part of this table.

## Results

Run 1 is the surface as D.5a left it. Run 2 is after the one change run 1
motivated (below).

| Metric | run 1 | run 2 | |
|---|---|---|---|
| first-call accuracy | 66.7% | 68.1% | +1.4 |
| **reached-tool accuracy** | **68.1%** | **75.0%** | **+6.9** |
| tool-turn accuracy | 52.9% | 54.9% | +2.0 |
| abstention accuracy | 100% | 100% | — |

| Failure mode | run 1 | run 2 |
|---|---|---|
| missed — should have called, did not | 21 | 17 |
| wrong tool | 3 | 6 |
| spurious — called on a turn needing a plain answer | **0** | **0** |
| alias table fired | **0** | **0** |

Per tool, first-call (run 2):

| Expected | Correct | of | |
|---|---|---|---|
| `list_promises` | 12 | 12 | 100% |
| `remember` | 9 | 15 | 60% |
| `forget` | 7 | 12 | 58% |
| `update_promise` | 0 | 12 | 0% |
| no tool (abstain) | 21 | 21 | 100% |

## What the numbers say

**The D.5a argument held, in the direction it was made.** Wrong-tool errors are
3 of 51 tool turns in run 1 (6%). A surface of five non-overlapping tools does
not produce near-synonym confusion. That was the prediction and it is met.

**The opposite risk did not materialize.** Zero spurious calls across 144 turns
over both runs, and 100% abstention including cases built to bait one — "i had a
long day", "what's the difference between a promise and a reminder?". The
rewritten prompt did not make a 2B model trigger-happy, which was the live worry
in shrinking the list: fewer tools can make each remaining one look *more*
applicable.

**The real failure is under-calling, and the accuracy column understates how bad
it is.** 21 of 24 failures in run 1 are turns where June answered in prose and
stored nothing. The user says "remember that my sister is called Mira", gets a
warm reply, and nothing is written — with nothing in the interface saying so.
That is a Glass Box problem, not a scoring problem, and trimming the tool list
did not cause it. It is the largest open item this benchmark found.

**The alias table never fired.** Zero rewrites in 144 turns. Its one surviving
entry points at `save_calendar_item`, which the calendar *skill* serves and no
case in this corpus targets. That answers D.5d's question directly:
`tool_aliases.py` no longer serves the native path at all.

## The one change, and how it actually worked

Run 1 put `update_promise` at 0/12 — the only total failure, stable across all
three passes rather than noise. The cause was the signature, not the model:
`promise_id` exists only inside a `list_promises` result, so the tool could not
be reached without chaining two calls. A promise can now be named the way the
user names it, matched over open promises on content words.

**The prediction was half right, and the half that was wrong is the more
interesting one.** The expectation was that removing the unobtainable handle
would let the model call `update_promise` directly. It did not:
first-call `update_promise` is still 0/12. What changed is that the *chain now
completes*. In run 1 the model called `list_promises` first on 2 of 12 update
cases and then stopped, having ids it could not confidently use; in run 2 it
does so on 6 of 12 and follows through. Turns that reached the right tool
without leading with it went from 1 to 5, and `missed` fell from 21 to 17 while
`wrong_tool` rose from 3 to 6 — the same turns, moved from *not acting* to
*acting in two steps*.

So the user-visible number is reached-tool accuracy, **68.1% → 75.0%**, and the
end-to-end behaviour is correct: the promise gets updated within the turn.

This also means the corpus's expectation of a single call for `update_promise`
encodes an assumption worth doubting. Looking before writing is defensible — for
a tool that mutates a standing intention it may be the better behaviour, and it
is what the four inversions would argue for. The first-call metric is kept
because it is what would show a regression into blind writes, not because 0/12
on it is necessarily a defect.

## Open, in priority order

1. **Under-calling on `remember` and `forget`** (60% / 58%). June accepts a
   memory instruction, replies warmly, and writes nothing, silently. Worth
   attacking from the Glass Box side as much as the prompt side: a turn where
   the user asked June to remember something and no write happened is something
   the interface should be able to say.
2. **Re-measure on `gemma4:e4b`** (`local-deep`). Every number here is the 2B
   model; the routing layer sends harder turns to 4B and this has not been
   measured there.
3. `tool_aliases.py` is now one entry serving one skill tool. Decide whether it
   survives as a module or folds into the calendar skill.

## Reproducing

```
packages/brain/.venv/bin/python -u tools/tool_selection_harness.py --repeat 3 --json results.json
```

Needs a running Ollama with `gemma4:e2b`. Roughly 45 minutes for 72 turns; the
JSON carries every call of every turn, so a two-step answer can be told from a
stall without re-running it.
