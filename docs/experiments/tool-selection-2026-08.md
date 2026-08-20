# Tool-selection accuracy — August 2026 (D.5d)

D.5a cut `JUNE_TOOLS_GEMMA` from 24 tools to 5 on the argument that a small
model picking between near-synonyms is where wrong tool calls come from, and
that `tool_aliases.py` existed to paper over those wrong calls. D.5d is where
that argument gets checked instead of restated.

**Instrument.** `tools/tool_selection_harness.py`, scoring in
`june_brain/experiments/tool_selection.py` (pure, unit-tested — a benchmark
whose own arithmetic is unverified is worse than none, because it produces a
number people quote). 24 cases, three passes, n=72 per run. `gemma4:e2b` via
Ollama, driven through the real loop with dispatch live and writes sent to a
throwaway data dir. MacBook, Apple Silicon.

**This is the first measurement in this repo taken with skills running.**
`check.sh` sets `JUNE_SKILLS_DISABLED=1`, so no test had ever seen the tool list
the model is actually offered. On a default install that list is **15 tools** —
5 native plus calendar (3), research (2) and files (5) — not 5.

## Two metrics, because one hides the answer

- **first-call accuracy** — the first tool of the turn is the expected one.
- **reached-tool accuracy** — the expected tool ran at some point in the turn.

They differ where a model looks before it writes. Answering "the passport
renewal is done" with `list_promises` then `update_promise` is right in two
steps, and the promise does get updated; first-call alone records that as a
failure.

## Results

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| first-call accuracy | 66.7% | 68.1% | 66.7% | **72.2%** |
| reached-tool accuracy | 68.1% | 75.0% | 72.2% | **77.8%** |
| tool-turn accuracy | 52.9% | 54.9% | 52.9% | **60.8%** |
| abstention accuracy | 100% | 100% | 100% | **100%** |
| spurious calls | 0 | 0 | 0 | **0** |
| alias table fired | 0 | 0 | 0 | **0** |

Per tool, first-call:

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| `list_promises` | 100% | 100% | 100% | 100% |
| `remember` | 60% | 60% | 53% | **60%** |
| `forget` | 50% | 58% | 50% | **67%** |
| `update_promise` | 0% | 0% | 8% | **17%** |
| abstain | 100% | 100% | 100% | 100% |

What changed between runs: **run 2** gave `update_promise` a name instead of an
id; **run 3** rewrote `build_system_prompt`'s triggers and seeded the memories
the `forget` cases refer to; **run 4** put the tool-use rule where the live path
actually reads it.

## What held

**The D.5a argument, in the direction it was made.** Wrong-tool errors are 8 of
51 tool turns and most of those are `update_promise` reaching for
`list_promises` first, which is defensible behaviour. Five non-overlapping tools
do not produce near-synonym confusion.

**The opposite risk never materialised.** Zero spurious calls across 288 turns,
and 100% abstention in every run, including cases written to bait one ("i had a
long day", "what's the difference between a promise and a reminder?"). Adding
explicit tool-use guidance in run 4 did not make a 2B model trigger-happy, which
was the live worry.

**`tool_aliases.py` never fired once** in 288 turns. Its one surviving entry
points at `save_calendar_item`, which the calendar *skill* serves. It no longer
serves the native path at all — which is the question this slice existed to
answer.

## The finding that mattered, and it was not a prompting problem

Run 3 measured as **no effect**. Rather than guess again, the failing turns were
probed directly, and the model turned out to be choosing correctly the whole
time: on a turn scored as a miss, the provider emits a correct native
`remember` call through both `generate()` and `stream()`. The seam is fine.

Running the same utterance through the real loop returned **no tool calls at
all**, and the reply read:

> I have remembered that you are vegetarian.

Nothing was stored. That is not under-calling, which is how the accuracy column
frames it. It is June making a true-sounding statement about the user's own
data that is false — the exact failure the Glass Box exists to prevent, reached
without any component behaving incorrectly.

**Cause:** `build_system_prompt` carries the "when to use a tool" rules and has
one production caller, the scheduler. The live chat path assembles its own
context, and `ContextAssembler._DEFAULT_SYSTEM_PROMPT` says "Be helpful, honest,
conversational, and concise" and nothing about tools. The model was handed
fifteen tools and no criterion. One rule, written twice, missing from the copy
that runs — the shape every finding in this stream has had.

`TOOL_USE_GUIDANCE` now has one definition, injected into the live path by
`make_tools_block` and substituted into the scheduler prompt, with tests pinning
both. Run 4 is that change: **+7.8 points on tool turns, +9.7 on reached-tool
against the baseline, with abstention and spurious unchanged.**

## What is still broken

**`remember` sits at 60% and has not moved.** 9/15, 9/15, 8/15, 9/15 across four
runs and two prompt interventions. That stability is the signal: it is a ceiling,
not noise, and further prompt tuning looks unpromising. Two of every five times a
user asks June to remember something, nothing is stored.

**And the reply still claims otherwise.** On the turns that make no call, the
model says "I have remembered that you are vegetarian" regardless. The guidance
tells it not to; a 2B model does not reliably obey, and no wording will make it.

That is the argument for the structural half rather than more prompting, and it
is why the turn frame now reports `memories_written` alongside
`memories_recalled`: a turn that stored nothing looks different from one that
did, whatever the reply says. The prompt reduces how often that matters; it
cannot be the guarantee. This is the guard layer's own thesis applied to
honesty rather than to safety — *defence is primarily structural; the gates hold
regardless of what the content says.*

## Open, in priority order

1. **The gap between what June says and what June did.** The frame reports it;
   the sentence the user actually reads still overstates. Worth a structural
   answer — a turn that claims a memory and wrote none is detectable in the
   places both facts are already known.
2. **Re-measure on `gemma4:e4b`** (`local-deep`). Every number here is the 2B
   model; the router sends harder turns to 4B and that path is unmeasured.
3. `tool_aliases.py` is one entry serving one skill tool. Decide whether it
   survives as a module or folds into the calendar skill.

## Reproducing

```
packages/brain/.venv/bin/python -u tools/tool_selection_harness.py --repeat 3 --json results.json
```

Needs a running Ollama with `gemma4:e2b`. Roughly 45 minutes for 72 turns; the
JSON carries every call of every turn, so a two-step answer can be told from a
stall without re-running it.
