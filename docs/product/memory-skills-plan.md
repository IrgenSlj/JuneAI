# Memory and Skills Depth Plan

> **Status:** Backlog and design reference. Its trust goals are still valid, but
> active sequencing now flows through ADR 0014 and the v0.1.1 event-ledger,
> capture, approval, and Daily Home plan. Memory edits, recall provenance, and
> skill writes should use the durable source/event model when implemented.

This document is the plan for deepening June's two load-bearing systems — memory and skills — across all shells. It is parallel to the desktop-shell track in [desktop-shell-plan.md](desktop-shell-plan.md): the desktop shell is how June reaches users, this plan is what June actually does for them once they arrive. Both can advance independently because they touch separate subsystems.

The trigger that justifies this plan is non-negotiable #1 of the project: **memory is the product**. The web prototype shipped a working memory subsystem, but the contracts between memory, skills, and the chat UI are weaker than they need to be for June to feel like a confidant rather than a chat box that occasionally remembers.

## The Frame

June already has the right components: three memory stores (SQLite for structured rows, ChromaDB for semantic facts, a knowledge graph for entities and relationships), five MCP skills, a recall step before every turn, and an extract step after. The plumbing works. The gaps are not architectural; they are **contracts between components that should already exist**.

Three concrete gaps, all observed in current code:

1. **Memory is read-only where it matters most.** `packages/api/src/june_api/routes/memory.py` carries a comment that "SQLite rows remain read-only for now" — goals, open loops, calendar items, journal entries, and body metrics cannot be edited or deleted. The product promises "remembers you, you stay in control"; half of memory currently violates the second clause.
2. **Recall is invisible.** `MemoryManager.recall()` runs every turn and injects facts into the system prompt at `packages/brain/src/june_brain/graph.py:611`, but the chat UI never shows which memories were drawn on. Users cannot tell if June used a real memory, hallucinated one, or used neither. This is the largest trust gap in the product.
3. **Skills and memory are siloed.** When a skill writes a structured row (a journal entry, a workout, a goal), that row never enters the vector store or the graph. So next conversation, recall does not surface skill writes unless extract happens to re-derive them from the chat itself. Skills write a parallel memory the recall ranker does not see.

## The Principle

**Simple but dense** means no new components and no new top-level concepts. Every existing component gets a sharper contract with its neighbors. Three phases. Each ships a visible UX upgrade and a small architecture upgrade together. The architecture moves are three contract changes, not three new classes:

- **`MemoryManager.write()`** becomes the single entry point for any code path that creates a memory — extract, skill writes, manual entries from the UI. It fans out to whichever stores apply to that payload.
- **`MemoryManager.recall()`** already merges across stores. It gets a return shape the API hands to the chat stream as a `recall_block` event so the UI can render which memories were used.
- **`MemoryManager.forget(ref)`** generalizes from "delete a fact" to "delete any memory by reference." Every memory the user sees in the browser becomes deletable.

Three contract changes. No new classes. No new stores. No new top-level concepts.

## Phase A — Memory Becomes Editable

**Goal:** the user can fix or forget anything June remembers, including structured rows.

### Why first

Without this, every other improvement amplifies an existing problem. June remembers more, more confidently, and the user still cannot fix it. Edit and delete are the foundation for trust; trust is the foundation for everything else.

### Slices

- **A.1 — Structured-row delete.** Add `delete_goal`, `delete_open_loop`, `delete_calendar_item` to `Memory` (SQLite). Extend `MemoryManager.forget()` to dispatch `goal:`, `open_loop:`, `calendar:` refs. Drop the "read-only" comment in `routes/memory.py`. Flip `deletable: true` for those sections in `apps/web/src/routes/memory/+page.svelte`. _Done when:_ I open `/memory`, click "Forget" on a goal, see it disappear, and confirm it does not come back in the next recall.
- **A.2 — Structured-row edit.** Add `update_goal`, `update_open_loop`, `update_calendar_item` SQLite methods. Add `PATCH /memory/{user_id}/fact/{ref}` API. Add inline editor in the memory browser (click a fact, edit fields, save). _Done when:_ I can rename a goal in `/memory`, save, and the new title appears the next time June recalls it.
- **A.3 — Surface journal and body metrics.** Add journal entries and body metrics to `MemorySnapshot`. Render them as new sections in the memory browser. Make them deletable. _Done when:_ a journal entry written by the daily skill appears in `/memory` and can be removed.
- **A.4 — Extract-failure visibility.** `chat.py:258` swallows extract failures silently. Persist failures to a small `extract_failures` table (turn id, error, timestamp). Surface a "Recent extracts that failed" section in the memory browser when there is at least one. _Done when:_ I can force an extract failure (e.g. break the extractor prompt) and see the failure listed in `/memory`.
- **A.5 — `/forget` slash command.** Typing `/forget` in the composer opens a picker showing the memories June recalled most recently for this user. Selecting one deletes it. _Done when:_ after a conversation that used a recalled fact, I can type `/forget`, see that fact in the picker, and remove it.

### Done criteria for Phase A

The memory browser displays every store the user might want to inspect. Every visible memory is editable or deletable. Extract failures are visible rather than silent. The user has a fast path from "June got this wrong" to "June will not say that again."

## Phase B — Recall Becomes Visible

**Goal:** the user can see which memories June drew on for each answer, and rate them.

### Why second

Once memory is editable, the next gap is legibility. The difference between "magical" and "creepy" is whether the user can see the work. Phase B is also the prerequisite for the Proactive Assistant feature (in `roadmap.md`); proactive suggestions are unjustifiable without recall provenance.

### Slices

- **B.1 — Stream the recall block.** Extend `ChatEvent` with type `recall`, emitted before the assistant's first token. Payload: the same hits the recall block was built from, with refs that resolve in `/memory`. _Done when:_ the chat SSE stream includes a `recall` event whose payload matches what was injected into the system prompt.
- **B.2 — "Memories used" disclosure.** Each assistant message gets a small disclosure showing facts/entities/relations used. Click a memory → opens it in `/memory`. Collapsed by default; reduced-motion respected. _Done when:_ I ask June something I told her in March, and the answer shows the March fact as a clickable disclosure.
- **B.3 — Inline citations.** When the assistant directly relies on a recalled fact, render that fact as a tooltip-pill in the message text. Heuristic match in the brain; the pill is metadata in the SSE payload, not LLM output. _Done when:_ a message that uses a recalled fact shows the fact inline as a tooltip on hover/tap.
- **B.4 — Memory feedback.** `POST /memory/feedback` for thumbs up/down on a recalled memory. Persist a `feedback` column on `semantic_facts` and a `relevance_score` on graph edges. Use feedback in recall ranking: thumbs-down halves the score for that user/topic; thumbs-up doubles it. _Done when:_ a thumbs-down on a fact noticeably reduces its rank in subsequent recalls.

### Done criteria for Phase B

For any assistant message, the user can see what June drew on, click through to the source memory, and influence future recalls with one click. Recall stops being a black box.

## Phase C — Skills Feed Memory, Memory Shapes Skills

**Goal:** every skill write becomes recallable. Tool-level control lands.

### Why third

Phases A and B make existing memory better. Phase C is what makes skills and memory feel like one product. Doing it before A would amplify the read-only problem; doing it before B would make the new memories invisible.

### Slices

- **C.1 — `MemoryManager.write()`.** A new public method that takes a payload `{kind, fields, source}` and fans out: keep the structured row (no regression), paraphrase the row into a one-sentence fact and upsert into the vector store, extract named entities and add nodes/edges to the graph. Existing `extract()` is reimplemented as a thin caller of `write()`.
- **C.2 — Skills route through `write()`.** Each skill in `skills/*/src/.../tools.py` updates one call site so that what was `mem.save_goal(...)` becomes `manager.write({...})`. The contract change is in the supervisor's `_bridge_tool` so individual skill modules do not import `MemoryManager` themselves. Five skills, ~20 minutes per skill.
- **C.3 — Per-tool toggle in the skills page.** The deferred item from the prototype roadmap, now justified because users can reason about which tool's writes feed memory. `apps/web/src/routes/skills/+page.svelte` gets a per-tool checkbox; the supervisor enforces the gate when the model emits a tool call.
- **C.4 — "What this skill remembered" view.** Per-skill modal listing the facts that skill has written (from the `source=skill:<name>:<tool>` tag in the vector store). _Done when:_ I log a workout via the health skill, then 3 days later ask "when did I last train legs?" and June answers from recall, not from a structured query.

### Done criteria for Phase C

Skills are first-class memory writers. Their writes show up in the same recall pipeline as extracted facts. The user can inspect what each skill has remembered and toggle individual tools.

## What This Plan Does Not Do

Per the One Rule of `roadmap.md`, *implement what current users need*. The following are explicitly out of scope until their own triggers fire:

- **No new skills.** Five is enough until users ask for a sixth. Deepen, do not widen.
- **No skill marketplace.** Trigger (three external contributors who have shipped skills) has not fired.
- **No proactive assistant.** Gated on mobile push and Phase 4 native notifications. Phase B above is its prerequisite — once recall is legible, proactive suggestions become justifiable.
- **No multi-user.** No request.
- **No new memory store.** Episodic, time-series, etc. The three we have are not yet maxed out.
- **No voice.** No requests.
- **No auto-summarization of old conversations.** Extract already does this turn-by-turn; revisit only if recall starts missing things.

## Estimate

Phases run in order. Within a phase, slices ship independently and each is shippable in one focused session.

- Phase A — five slices, roughly two sessions.
- Phase B — four slices, roughly two sessions.
- Phase C — four slices, roughly two sessions.

Six focused sessions, give or take. The exact pace is set by what real users surface as we ship each slice; later slices can be re-prioritized or dropped if their problem turns out to be smaller than expected.

## When This Plan Is Done

The roadmap's first non-negotiable — memory is the product — has stopped being aspirational and started being the user's experience. Every slice above is a contract tightened, not a new component. When all three phases land, the next planning cycle is set up to consider the proactive-assistant feature plan with confidence rather than speculation, because recall is already legible and editable.
