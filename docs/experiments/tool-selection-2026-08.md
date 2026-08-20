# Tool-selection accuracy — August 2026 (D.5d)

D.5a cut `JUNE_TOOLS_GEMMA` from 24 tools to 5 on the argument that a small
model picking between near-synonyms is where wrong tool calls come from, and
that `tool_aliases.py` existed to paper over those wrong calls. D.5d is where
that argument gets checked instead of restated.

> **Correction (2026-08-20, after the fact).** Everything below measures
> **tool-call accuracy**, not memory reliability, and the two are not the same
> thing. The harness drives `loop.run_turn` directly; the real chat path is the
> `/chat` route, which *also* runs `MemoryManager.extract` on the exchange after
> the response is sent. A fact therefore has **two** chances to land and this
> instrument only ever watched one, so every `remember` and `forget` number here
> is a floor rather than the user-facing rate. `--extract` now runs the second
> path too. Read the per-tool numbers as "how often the model reached for the
> right tool", which is what they are.
>
> The correction is smaller than it sounds: spot-checking the failing case
> ("please keep in mind that I'm vegetarian") found extraction storing nothing
> either, and isolating the extractor showed it is itself roughly a coin flip on
> this model — 0 facts and 1 fact on the *same* user text with different
> assistant replies. Both mechanisms are `gemma4:e2b` judgment calls, so they
> are not independent in the way redundancy needs.

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

## The deeper local model is not the answer either

`--role local-deep` runs the same corpus against `gemma4:e4b`. n=48 (two
passes), against e2b's n=72:

| | `gemma4:e2b` | `gemma4:e4b` |
|---|---|---|
| first-call accuracy | 72.2% | 68.8% |
| reached-tool accuracy | 77.8% | 75.0% |
| tool-turn accuracy | 60.8% | 55.9% |
| abstention accuracy | 100% | 100% |
| `remember` | 60% | 50% |
| `forget` | 67% | 62% |

**Doubling the model does not help.** Every number is flat-to-worse — within
noise at these sample sizes, but nowhere near the improvement that would justify
routing memory instructions to `local-deep`. That idea is dead, and this is the
measurement that killed it rather than an opinion about it.

Which locates the remaining gap: it is not model capacity at this scale, and two
prompt interventions have taken it as far as prompting seems to go (the second
was worth +7.8 points, the first worth nothing). A materially better number
needs a different mechanism, not a bigger local model or more wording.

## Open, in priority order

1. **The gap between what June says and what June did.** The frame reports it;
   the sentence the user reads still overstates. Note the useful asymmetry found
   while auditing: `forget` and `update_promise` report their no-match cases
   truthfully, so **June tells the truth whenever a tool actually runs** — the
   false claim appears only on turns where no tool ran. That makes this a
   call-rate problem with an honesty backstop, not a lying-model problem, and it
   is why the structural half (`memories_written` in the turn frame) is the part
   that holds.
2. **Try the cloud tier** (`--role cloud-capable`). Unmeasured here because it
   needs a key and egress, both of which are the user's call. If cloud is near
   100% the ceiling is model quality, and "a memory write is worth a cloud call
   when the user permits it" becomes a real product option.
3. ~~`tool_aliases.py` — decide whether it survives.~~ **Settled: it stays, and
   a bug in it is fixed.** The corpus never fired it, but the corpus has no
   calendar cases, so that was absence of evidence. A targeted probe of six
   calendar utterances found the model emitting the canonical
   `save_calendar_item` on 7 of 7 calls with canonical parameters on all of
   them — the tools block names tools canonically now and the model copies what
   it is shown. The aliases are unused rather than harmful, and n=7 is thin
   grounds for deleting a fallback that costs nothing when it does not fire.

   The probe did find a real defect: `_normalize_save_calendar_item` rebuilt a
   fixed four-key dict and so **dropped every argument it did not know about**,
   measurably `status` and `source` on every real call. `source` is the
   provenance tag the memory browser uses to say where a saved item came from,
   so a pass whose job is repairing the model's arguments was degrading the
   record it exists to improve. It now merges.

## Reproducing

```
packages/brain/.venv/bin/python -u tools/tool_selection_harness.py --repeat 3 --json results.json
```

Needs a running Ollama with `gemma4:e2b`. Roughly 45 minutes for 72 turns; the
JSON carries every call of every turn, so a two-step answer can be told from a
stall without re-running it.
