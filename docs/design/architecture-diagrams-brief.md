# June — Architecture Diagram Brief

**For:** Claude (claude.ai, Artifacts enabled) or any design tool that can emit SVG.
**Deliverable:** a coherent set of nine architectural diagrams for the GitHub front
page (`README.md`) and `docs/architecture/overview.md`.
**Created:** 2026-07-25. **Source of truth for facts:** `docs/CURRENT.md`.

Everything in section 2 is verified against the shipped code as of commit
`ac42baee`. Do not invent components, do not soften the claims, do not add
anything that is "planned" as if it existed. Section 7 lists the specific
statements that must not be misrepresented.

---

## 0. How to use this brief

Work in this order. Do not skip ahead — diagram 1 establishes the visual language
that the other eight reuse.

1. Read sections 1-3 (what June is, the verified system facts, the visual system).
2. Produce **Diagram 1 (System Map)** first, in both themes, and stop. It sets the
   grammar: box shapes, line weights, arrow semantics, type scale, spacing rhythm.
3. Once diagram 1 is locked, produce diagrams 2-9 reusing that grammar exactly.
4. Deliver each diagram as **two standalone `.svg` files** (light + dark) per the
   output contract in section 4.

After the first diagram, name the three things you would change about it and ask
which way to push before continuing.

---

## 1. The product in one paragraph (paste-ready, accurate)

June is a personal AI you can audit. She runs on the user's machine, remembers what
matters to them, forgets what doesn't, explains every action, and never phones home.
Chat and recall run locally on Gemma 4 via Ollama; cloud capability (Gemini) is
reached only on explicit agentic paths, is surfaced in the UI before and after, and
is written to a hash-chained local ledger — a capability cannot egress and skip the
record. Her center of gravity is the user, not the task: she **defers** to the user
instead of verifying against ground truth, **continues** standing intentions instead
of completing and exiting, **forgets** gracefully instead of accumulating, and
**stays quiet** instead of acting fast.

The diagrams exist to make one thing legible to a technical stranger in ten seconds:
**June's trust properties are structural, not promised.** Every diagram should leave
the viewer able to point at the place in the system where a guarantee is enforced.

---

## 2. Verified system facts

Use these names, these layers, these directions. Labels in `code font` are real
module or file names and should appear verbatim in the diagrams where indicated.

### 2.1 Layering

`Shell -> API -> Brain -> Providers`. Each layer calls only into the one below.
Everything persists under one versioned data directory (ADR 0019).

- **Shell** — `apps/desktop`: Tauri 2 (macOS Apple Silicon) supervising a
  PyInstaller-frozen Python sidecar at `Contents/Resources/june-api/`, with a
  watchdog, corrupt-DB recovery, and loopback-token auth. `apps/web`: a SvelteKit
  PWA — the same build every shell wraps.
- **API** — `packages/api`: thin FastAPI REST + SSE. Pydantic schemas are the single
  source of truth; the TypeScript client is generated from OpenAPI and drift is
  CI-gated. On startup it reconciles promises orphaned by a restart.
- **Brain** — `packages/brain`: one hand-written harness loop
  (`loop/handwritten.py`, ADR 0018 — no agent framework), model-specific providers,
  layered context with anchored compaction, self-authored character with a fixed
  honesty + safety floor. The loop *shape* is fixed and never self-modified; only
  choices (tier, tools) flow through it as data.
- **Providers** — three roles (ADR 0017): `local-fast` (`gemma4:e2b`), `local-deep`
  (`gemma4:e4b`), `cloud-capable` (`gemini-2.0-flash`).

### 2.2 Memory (`packages/brain/src/june_brain/memory/`)

One SQLite file, `june.db`, behind one facade, `MemoryManager`, over three stores:

| Store | What it holds |
|---|---|
| structured rows | conversations, `semantic_facts` (composite PK `user_id` + `fact_id`), bi-temporal validity columns |
| `sqlite-vec` (`vec0`) | the semantic index; embeddings from local Ollama `nomic-embed-text` |
| entity graph | `graph_nodes` / `graph_edges` |

Retrieval (`memory/recall.py::gather_hits`) fuses **four signals** — vector
similarity, **BM25** over an FTS5 mirror (`semantic_facts_fts`, kept current by
insert/update/delete triggers), **entity overlap** with the query, and **temporal
validity** — via **Reciprocal Rank Fusion (`rrf_k = 60`, tunable through
`RetrievalConfig`)**, then reranks by **salience** = recency x frequency x relevance.
Forgetting is first-class, conservative and reversible: content is tombstoned into
`forgotten_*` tables, never silently dropped. Schema is versioned
(`memory/migration.py`, latest **v7**), and FTS5 absence degrades gracefully to the
vector channel alone.

### 2.3 Trust Ledger (`trust/`, ADR 0022)

Append-only, **blake2b hash-chained** local event log (`trust_ledger`) with
tail-truncation-aware chain verification and optional **Ed25519** signing. Event
kinds today: `egress`, `action`, `approval`, `system`. Rendered in the UI as
**Receipts** under the **Trust** screen (`/system`), with a verify affordance.

### 2.4 The egress chokepoint (`providers/provenance.py`)

Every cloud-routed model call passes through **one** function,
`record_cloud_call`, which brackets the call with a `start` and an end event and
writes an `egress` entry to the ledger. As of ADR 0022's enforcement extension the
chokepoint is no longer passive: in local-only mode the `start` phase raises
`CloudEgressBlockedError` **before** the request leaves. The live loop routes by
difficulty (`router/difficulty.py`) and **never auto-escalates to cloud** — cloud is
reached only on explicit agentic/skill paths.

### 2.5 Guard layer (`guard/actions.py`, ADR 0021)

A single seam, `evaluate_call`, sits between the loop and every tool. It:
classifies each call into an action class (`read`, `read_network`, `write`,
`write_network`, `execute`); tracks **taint** (content that came back from an
untrusted result and is now flowing into a new action); gates `execute`,
`write_network`, and tainted network calls behind explicit user approval; frames
every tool result as untrusted content; and redacts secrets before anything reaches
the ledger. The defense is **structural** — there is no content-based
injection-phrase detector yet, and the diagrams must not imply one.

### 2.6 Silence Model (`silence/`, ADR 0023)

Governs June-*initiated* surfacing only — never the reply path. A pure, clockless,
model-free rules policy, `decide()`, maps a candidate to `now | batch | suppress`,
gated by salience and presence (presence is derived from recency-of-activity;
there is no OS idle or power signal). Every decision — **including the decision to
stay quiet** — is mirrored to the ledger.

### 2.7 Promises (`tasks/`)

Standing intentions, not terminating TODOs. Persisted in `tasks` with a per-step
trace, explicit `blocked_reason` / `next_action` / `final_deliverable`, retries
(cap 5), recurrence, and restart reconciliation. Exposed at `/tasks`, rendered in
the UI as **Promises**. Resuming re-runs the goal — there is no mid-plan checkpoint
resume yet.

### 2.8 Skills and Scheduler

Skills (`skills/`, ADR 0005) are standalone **MCP** servers over stdio — one
supervised subprocess each, independently toggled, guard-fronted. Shipped:
`calendar`, `daily`, `files`, `health`, `research`, `telegram`. The scheduler
(ADR 0016) runs **deterministic, user-requested jobs only** (cron / interval / at).
There is **no heartbeat and no timer-driven proactivity**; a separate event poller
drains real-world skill events, which is the sanctioned "the world changed" wake.

---

## 3. Visual language

The diagrams must look like they came out of the product, not out of a generic
diagramming tool. These are the shipped tokens (`packages/design/src/tokens.css`).

### 3.1 Palette

| Role | Dark (default) | Light |
|---|---|---|
| page background | `#13110E` | `#FAF9F7` |
| raised surface (node fill) | `#1A1814` | `#FFFFFF` |
| sunken surface (group fill) | `#22201B` | `#F3F1ED` |
| primary text | `#F1EEE7` | `#141410` |
| secondary text | `#D8D3C9` | `#2A2824` |
| muted text (annotations) | `#8E8778` | `#6B665D` |
| accent (June) | `#F2AC6E` | `#E8965A` |
| accent muted | `#E0945A` | `#D17F3F` |
| success / local-safe | `#8AA884` | `#4E6B4A` |
| danger / blocked | `#C88080` | `#8A3B3B` |
| warn / gated | `#C8A260` | `#8A6A2F` |
| border | `rgba(255,250,240,0.08)` | `rgba(20,16,10,0.08)` |
| border strong | `rgba(255,250,240,0.16)` | `rgba(20,16,10,0.14)` |
| terminal / ledger surface | `#0E0C09` | `#F0ECE3` |

Warm, low-chroma, quiet. The accent is precious — reserve it for **June's own
boundary** (the brain, and the one thing each diagram is actually about). Never
color more than roughly a fifth of any diagram with the accent.

### 3.2 Semantic encoding (must be consistent across all nine)

| Meaning | Encoding |
|---|---|
| stays on device | solid stroke, `border-strong`, raised fill |
| leaves the device | **dashed** stroke in `warn`, always labeled with what leaves |
| blocked in local-only mode | dashed stroke in `danger` with a small stop-glyph at the boundary crossing |
| requires user approval | `warn` stroke plus a small key glyph on the edge |
| append-only / tamper-evident | the ledger surface color, with a chain-link motif in the node's corner |
| data at rest | cylinder |
| single chokepoint / seam | a narrow accent-stroked bar the flow must pass through, never a box with bypass edges |

Nothing may cross the device boundary without a dashed line. If a diagram has no
dashed lines, say so explicitly in a caption ("nothing on this diagram leaves the
machine") rather than leaving it ambiguous.

### 3.3 Type and geometry

- Font: system stack only — `-apple-system, "Helvetica Neue", Helvetica, Arial,
  sans-serif`. **No webfonts, no `@import`** (see the output contract).
- Sizes: title 20px / 600, node label 13px / 500, sub-label 11px / 400 in muted,
  edge label 10px / 500 in muted, caption 11px italic.
- Radii: nodes `10px`, groups `14px`, pills `999px`. Stroke `1px` for structure,
  `1.5px` for the flow the diagram is about.
- Spacing on an 8px grid. Generous whitespace; these are read at README width
  (roughly 850 CSS px) and must survive it.
- Arrowheads: small, filled, 6px. One arrow direction per relationship; if a
  relationship is bidirectional, use two separate offset arrows, never a
  double-headed one.

### 3.4 Tone

Calm, engineered, warm. Think a well-set technical paper figure, not a SaaS
marketing graphic. No 3D, no gradients beyond a barely-there surface lift, no drop
shadows on more than one z-level, no emoji, no icon soup. Any glyph used (lock,
key, chain link, stop) must be drawn as a path in the same stroke weight as the
diagram, not imported from an icon font.

---

## 4. Output contract

- **Format:** hand-authored SVG. One file per theme per diagram:
  `docs/architecture/diagrams/<slug>-dark.svg` and `<slug>-light.svg`.
- **Sizing:** `viewBox` with a fixed aspect per diagram (specified below), no fixed
  `width`/`height` attributes, `preserveAspectRatio="xMidYMid meet"`. Design for a
  1600px-wide render and verify legibility at 850px.
- **Self-contained:** no external references of any kind — no webfonts, no images,
  no CSS `@import`, no scripts. GitHub renders these through an image proxy; an
  external reference will silently fail.
- **Theme pairing in the README** uses GitHub's supported `<picture>` form:

  ```html
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/architecture/diagrams/system-map-dark.svg">
    <img alt="June system map: Shell, API, Brain, Providers, and the single SQLite store" src="docs/architecture/diagrams/system-map-light.svg">
  </picture>
  ```

  Do not rely on `prefers-color-scheme` media queries *inside* the SVG — behavior
  is inconsistent when an SVG is loaded as an image.
- **Accessibility:** every SVG carries `role="img"`, a `<title>` and a `<desc>`
  that reads as a complete sentence describing the flow. Contrast of text against
  its own fill must be at least 4.5:1 in both themes. The diagram must remain
  readable in grayscale — never encode meaning in hue alone; pair every color cue
  with a stroke style or a glyph.
- **Text handling:** live `<text>` elements, not outlined paths (keeps the files
  small, selectable, and diffable). Keep every label under 34 characters; push
  detail into sub-labels or the caption.
- **Budget:** aim for under 60KB per file, hand-readable markup, grouped with `<g
  id="...">` per logical cluster so the files stay maintainable in git.

---

## 5. The nine diagrams

Each spec gives: where it lives, the one sentence it must land, the exact content,
and the failure mode to avoid.

### D1 — System Map *(README hero; 16:9)*

**Lands:** "Four layers, one local database, one visible door to the outside."

Vertical stack, top to bottom: **Shell** (Tauri desktop · PWA · mobile marked
*planned* in muted, not equal weight) → **API** (`FastAPI · REST + SSE`) → **Brain**
(one accent-bordered group containing: hand-written loop, layered context,
character, router, guard, silence) → **Providers** (a `local` cluster —
`gemma4:e2b`, `gemma4:e4b`, `nomic-embed-text`, all via Ollama, in success color;
and a single `cloud` node, `gemini-2.0-flash`, outside a clearly drawn **device
boundary**). To the side of the Brain: **Skills (MCP)** as a stack of supervised
subprocesses. Under the Brain: **`MemoryManager`** → cylinder `june.db`, with three
inner compartments labeled `structured rows`, `sqlite-vec (vec0)`, `entity graph`,
plus the `trust_ledger` shown as a chain-marked band inside the same file.

The device boundary is the strongest line on the page. Exactly one edge crosses it:
Brain → cloud, dashed, `warn`, labeled `surfaced + ledgered · opt-in only`, with the
stop-glyph annotation `blocked in local-only mode`.

**Avoid:** making the mobile shell look shipped; drawing more than one crossing edge;
letting `june.db` look like three separate files.

### D2 — The Life of a Turn *(docs; 4:3 or a tall sequence)*

**Lands:** "Every message follows the same fixed path, and you can see all of it."

A left-to-right sequence with numbered stages: **1** user message arrives at
`/chat` (SSE opens) → **2** recall: `gather_hits` runs the four-signal fusion → **3**
context assembly with anchored compaction → **4** difficulty routing picks a tier
(annotate: *never auto-escalates to cloud*) → **5** provider streams tokens → **6**
tool calls, each through `guard.evaluate_call` (branch: allowed / approval-gated /
refused) → **7** results framed as untrusted content, fed back into the loop → **8**
memory write + salience update → **9** provenance frame rendered under the turn →
**10** ledger append.

Show the loop-back from 7 to 5 explicitly, and mark stage 10 as append-only. Put a
small "what the user sees" rail along the bottom, aligned to stages 5, 6, and 9
(streaming text, the approval prompt, the provenance line).

**Avoid:** implying the model decides the shape of the pipeline; hiding the tool
loop-back; drawing the ledger as optional.

### D3 — Memory Architecture *(README section + docs; 16:10)*

**Lands:** "One file, three stores, one facade, four retrieval signals, and a
tombstone instead of a delete."

Left: a write path — conversation → extractor → `semantic_facts` (note the
bi-temporal validity columns) → embedding via local `nomic-embed-text` → `vec0`
index; entity extraction → `graph_nodes` / `graph_edges`; FTS5 triggers keeping
`semantic_facts_fts` in sync (draw the triggers as small automatic couplings, not
as a separate service).

Right: the read path — a query fanning into four channels (**vector**, **BM25**,
**entity overlap**, **temporal validity**) that converge on a single **RRF (k=60)**
fusion bar, then a **salience rerank** node labeled `recency x frequency x relevance`,
then the top-k hits into context.

Bottom rail: **forgetting** — a fact moving into `forgotten_*` with a reversible
arrow back, labeled `conservative · reversible · user is the source of truth`.

**Avoid:** drawing the three stores as three databases; showing similarity alone as
the ranking; making forgetting look like deletion.

### D4 — The Visible Cloud Boundary *(README; wide, 21:9)*

**Lands:** "There is exactly one door, it is always logged, and it can be locked."

A single horizontal band. Everything left of the boundary is the machine; the only
thing right of it is `gemini-2.0-flash`. Center on the boundary: the chokepoint bar,
labeled `providers/provenance.py :: record_cloud_call`, with three outputs — the
call itself (dashed, `warn`), the `egress` ledger entry (chain motif), and the
per-turn provenance frame rendered in the UI. Overlay a second state, drawn in
`danger` and clearly marked as the local-only mode: `CloudEgressBlockedError` raised
at the `start` phase, the dashed edge terminated at the boundary with the stop glyph.

Show that skills route through the same bar — a skill cannot egress and skip the
ledger. That is the whole point of the diagram.

**Avoid:** a second unlogged path; implying the ledger is written after the fact
only; suggesting local-only is a UI preference rather than an enforced raise.

### D5 — Guard Layer and Taint *(docs; 4:3)*

**Lands:** "Untrusted content can be read, but it cannot silently become an action."

The loop proposes a tool call → `evaluate_call` classifies it into
`read | read_network | write | write_network | execute` → three outcomes: **allowed**,
**approval-gated** (key glyph, user prompt shown), **refused**. Alongside, a taint
channel: a tool result enters framed as untrusted content, carries a taint marker,
and when that tainted content flows into a new `read_network` or `write_network`
call, the call is escalated to approval. A separate small box, in muted, states
plainly: *structural defense; no content-based injection detector yet.*

**Avoid:** claiming pattern-matching defense; hiding the redaction step before the
ledger write.

### D6 — The Silence Model *(README section; 16:10)*

**Lands:** "Restraint is a decision, and it is on the record too."

Candidates (a promise needing a nudge, a skill event, a recall worth surfacing) enter
a policy node marked **pure · clockless · model-free**, with its inputs shown as
injected rather than read: `now` (ISO-8601, passed in), salience, presence (derived
from recency of activity, annotated *no OS idle signal*), `active_thread_open`.
Three outputs: **now**, **batch** (held for the next natural opening), **suppress**.
All three — including `suppress` — feed the ledger. Add the boundary note:
*governs June-initiated surfacing only; never the reply path.*

**Avoid:** any clock, timer, or cron imagery; making `suppress` look like a dead end.

### D7 — Promises Lifecycle *(docs; 4:3)*

**Lands:** "A promise is a standing intention, not a task that dies when it exits."

State machine: `pending → running → (blocked | waiting-on-user | done | failed)`
with `retry` (cap 5), `recurrence`, and `restart reconciliation` (the API reconciles
`running` promises orphaned by a restart). Each blocked state carries the three
explicit fields — `blocked_reason`, `next_action`, `final_deliverable` — drawn as
required attributes, because the UI must never infer state from trace text. Mark the
current limitation honestly: *resume re-runs the goal; no mid-plan checkpoint yet.*

**Avoid:** drawing a terminating TODO lifecycle; omitting the honest limitation.

### D8 — Runtime Topology and Distribution *(docs + release page; 16:9)*

**Lands:** "It is one app, one loopback port, one data directory — and nothing else."

The macOS bundle: `June.app` → Tauri shell (Rust) supervising the PyInstaller
sidecar at `Contents/Resources/june-api/` (watchdog, corrupt-DB recovery, loopback
token auth), serving the SvelteKit build. Beside it, the separately installed Ollama
daemon on `127.0.0.1:11434`. Below, the versioned data directory with `memory/june.db`
and config. Annotate every arrow with `127.0.0.1` to make the loopback-only property
visible. Show the browser/PWA path reaching the same API as an alternative shell.

**Avoid:** implying models are bundled in the DMG; hiding the Ollama prerequisite.

### D9 — The Four Inversions *(README, near the top; 16:9 conceptual)*

**Lands:** "June is a coding agent's skeleton with all four operations inverted."

Two columns. Left, muted: a coding agent — *verifies against ground truth · completes
and exits · accumulates context · optimizes for speed*. Right, accent: June —
**defers** to the user · **continues** standing intentions · **forgets** gracefully ·
**stays quiet**. Between each pair, an inversion glyph. Under each right-hand item,
the module that implements it in small code font: approval gates (`guard/`),
`tasks/`, `memory/` forgetting + tombstones, `silence/`.

This is the only diagram allowed to be conceptual rather than structural — and it
earns that by naming the module under every claim.

**Avoid:** strawmanning coding agents; leaving any inversion without an
implementation pointer.

---

## 6. Front-page placement

The README should carry **four** of the nine (D9, D1, D3, D4 — in that order,
interleaved with prose). D6 is optional if the page still breathes. The rest live in
`docs/architecture/overview.md`. A README that opens with five diagrams reads as a
pitch deck; one that opens with the inversions, then the system map, reads as
engineering.

---

## 7. Accuracy guardrails (do not misstate)

1. Cloud is **never** auto-escalated to. Difficulty routing chooses among local
   tiers; cloud is reached on explicit agentic/skill paths only.
2. Local-only mode **raises and blocks** at the chokepoint. It is not advisory.
3. There is **no heartbeat**, no timer-driven proactivity, no daily orchestration
   (ADR 0016). The scheduler runs user-requested deterministic jobs only.
4. The guard's defense is **structural**. No injection-phrase detector exists yet.
5. `june.db` is **one file**. The three stores are inside it.
6. Forgetting **tombstones**; it does not delete.
7. The Silence Model touches **initiated surfacing only**, never replies.
8. Promise resume **re-runs the goal**; there is no mid-plan checkpoint.
9. The loop shape is **fixed and never self-modified**; only choices flow as data.
10. Mobile is **planned**, not shipped. Voice is **not started**. Do not draw either
    as present.

---

## 8. Acceptance checklist

- [ ] Nine diagrams, eighteen SVG files, consistent grammar across all of them.
- [ ] Every device-boundary crossing is dashed and labeled with what crosses.
- [ ] No diagram encodes meaning in hue alone; all readable in grayscale.
- [ ] Every SVG has `role="img"`, `<title>`, and a full-sentence `<desc>`.
- [ ] No external font, image, script, or stylesheet reference in any file.
- [ ] Every label is under 34 characters; legible at 850px render width.
- [ ] Each of the ten guardrails in section 7 survives a read of the diagram set.
- [ ] Every claim in D9 names the module that implements it.
- [ ] Both themes verified against the exact token values in section 3.1.
