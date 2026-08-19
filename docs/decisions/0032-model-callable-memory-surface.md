# ADR 0032 — June's model-callable memory surface is four deliberate tools

## Status

Accepted 2026-08-19. Settles the tranche 2 question left open by D.5a in
`docs/product/v0.4-development-plan.md`, which D.5b and D.5d both block on.
Implements inversions 2 and 3 (ADR 0015) as callable surface. Constrained by
ADR 0021 (guard layer) and ADR 0022 (Trust Ledger).

## Context

The 2026-08-18 audit found that the v1 life-coach product still owned most of
June's tool surface. D.5a tranche 1 removed the unambiguous part — health,
fitness, mood, chapters, conversation coaching, and five no-op UI tools. What
remained was a set the audit deliberately refused to delete inside a cleanup
slice: journal, relationships, goals, open loops, preferences, calendar and
favorites. These read as v1 residue by origin but overlap the current product by
function. `save_user_preference` is close to the heart of "remembers what
matters". `save_open_loop` is a standing intention, which is inversion 2 under an
older name.

Underneath that sat the finding that made it a design question rather than a
cleanup one: **June had no model-callable memory tools at all.** Recall is
automatic in `ContextAssembler`; Promises are managed through `/tasks` by the
user, not by the model. So the eleven tools the default local model received
after tranche 1 were v1 memory-writes plus runtime controls, and the product's
two headline capabilities — remembering, and carrying an intention forward — were
things June could not do on purpose when asked.

The plan named two options. (a) Keep tranche 2 and rename it to match the
product. (b) Delete tranche 2 and design a small deliberate set. (a) is cheaper
and preserves working storage. It also keeps a seven-table domain schema whose
shape was chosen for a fitness-and-journaling app, and asks the model to express
"remember that my sister's name is Mira" by choosing between
`save_relationship_profile`, `save_user_preference` and `save_journal_entry` —
three writes to three tables that recall then has to fuse back together. The
schema's granularity is not a feature here; it is a classification burden placed
on the smallest model in the stack, and wrong picks are why `tool_aliases.py`
exists.

## Decision

Take option (b). June's model-callable memory surface is **four tools**:

| Tool | Backed by | Inversion |
|---|---|---|
| `remember(text)` | `MemoryManager.write({"kind": "fact"}, source="tool:remember")` | 3 — the user decides what is kept |
| `forget(description)` | `MemoryManager.forget(ref)` (tombstone, restorable) | 3 — forgetting is first-class |
| `list_promises()` | `TasksStore.active()` | 2 — standing intentions, not TODOs |
| `update_promise(promise, status, next_action)` | `TasksStore.set_status` / `set_blocked` | 2 |

Three properties are load-bearing, not incidental:

**One write target, not seven.** `remember` writes a semantic fact through the
existing `write` seam, which indexes it for recall and feeds the same fusion
path as extracted facts. The model chooses *whether* something is durable, never
*which table* it belongs in. Structure is recall's job, not the model's.

**`forget` defers rather than guesses.** A description is resolved by recall.
Exactly one clear match is forgotten and named back to the user. Ambiguity
between candidates is *not* resolved by picking the top score — the tool returns
the candidates and forgets nothing, which is inversion 1 in the one place where
being wrong destroys user data. No match says so plainly.

**Forgetting stays reversible.** `MemoryManager.forget` tombstones into the
`forgotten_*` tables and `restore(ref)` brings a memory back, so the tool's
report names what was forgotten and says it can be restored. Conservative and
reversible was already the store's behaviour; this makes the tool honest about
it rather than reporting a deletion that did not happen.

**`update_promise` cannot claim work it did not do.** The model may move a
promise to `completed`, `cancelled` or `paused`, and may set a next action. It
may **not** set `running` — that status means the runtime is executing the
promise, and a tool that wrote it would make the Promises view assert work that
nobody started.

**Amended 2026-08-19 by measurement.** `update_promise` first took a
`promise_id` and scored 0/12 on the local model: that id exists only inside a
`list_promises` result, so the tool could not be reached without chaining two
calls, which a 2B model does not do reliably. A promise is now named the way the
user names it and June matches it over open promises — the same resolution
`forget` does, including the refusal to break a tie by ranking. See
[`experiments/tool-selection-2026-08.md`](../experiments/tool-selection-2026-08.md).

## Consequences

- `JUNE_TOOLS_GEMMA` becomes the product's tools instead of a fitness app's, and
  gets smaller: the default local model chooses among a handful of tools whose
  descriptions do not overlap.
- The seven v1 domain tools go, and their names join `RETIRED_TOOL_NAMES` so a
  skill cannot unshadow them (the failure mode D.9 found).
- **The tables stay.** Only the tools are removed in this step. `recall`'s
  keyword channel and `context_intelligence` still read those rows, so existing
  user data keeps surfacing through recall and nothing needs exporting. Removing
  the store layer is D.5b, and that slice owns the export question.
- June can now write memory on request, which it could not before. That is new
  capability, so it is guard-classified and ledgered like any other action.
- What this does *not* add: a model-callable search. Recall already runs every
  turn on the user's message, and a second retrieval path the model triggers by
  hand would be a second answer to a question `ContextAssembler` already
  answers. Revisit only with evidence of a query the turn-level recall misses.

## Alternatives rejected

**Keep tranche 2, renamed (option (a)).** Cheaper, and defensible for calendar
alone (the Time track in `ROADMAP.md` will want real calendar structure). But it
locks the product's memory model to a schema chosen for a different product, and
leaves the smallest model doing table classification on every write. Calendar is
better rebuilt by the Time track against its own requirements than inherited.

**Expose the structured tables through one `remember(kind, fields)`.** Keeps the
granularity while collapsing the tool count. Rejected because it moves the same
classification burden from tool choice into an argument, where it is harder to
advertise and impossible to validate at dispatch.
