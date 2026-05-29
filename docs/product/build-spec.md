# June AI — Build Specification & Implementation Plan

> **Status:** Final, ready for implementation. **Version:** 1.0 (28 May 2026).
> **Authority:** This is the single authoritative build document. It supersedes any earlier notes,
> brainstorms, or partial briefs. Where it conflicts with anything else, this wins.
> **How to use it:** Read Parts A–B once and keep them in mind throughout. Build Part C (Tier 1) in
> order; do not start Tier 2 until Tier 1 ships and has been used. Treat file paths and the existing
> code's exact shapes as *intent, not gospel* — adapt names to what the repo actually contains; the
> decisions, data shapes, invariants, and ordering are the load-bearing parts.

**Repo adoption note (28 May 2026):** this document was adopted as June's canonical direction on the
`main` branch. It supersedes the "personal operating layer / Quick Capture" framing carried by the
earlier docs (ADR 0013, ADR 0014, the agentic-pivot plan, and the v0.1.1 scheduled-development plan),
which are retained as historical context only. The product overview, vision, architecture overview,
and roadmap have been realigned to this spec. Where the spec names file paths or code shapes that do
not yet exist in the repo, they describe build *intent*; the decisions, data shapes, invariants, and
ordering are what bind.

**Build status (29 May 2026):** the Tier 1 spine (C.0-C.6 in Part C) is implemented, tested, and on
`main`; the file paths it names now exist. The remaining Tier 1 work is to make the hand-written loop
the live chat path (route the provider layer, layered context, character block, difficulty router, and
capability probe through it, replacing the LangGraph agent as the live path while keeping it as a
flagged fallback). Do not start Tier 2 until that lands and the spine has been used. Parts A, B, and
the C.0-C.6 decisions remain authoritative.

**Audience:** Claude Code. **Purpose:** everything needed to start implementing today.
**Repo:** `IrgenSlj/JuneAI` — local-first personal AI agent. Brain = Python/LangGraph in
`packages/brain`; API = FastAPI (REST + SSE) in `packages/api`; UI = SvelteKit in `apps/web` +
`packages/ui`; skills = MCP servers in `skills/`. Stores: SQLite + ChromaDB + a graph, behind one
`MemoryManager`. Gate: `./tools/check.sh` (pytest, svelte-check, OpenAPI drift). Codegen:
`./tools/codegen.sh` after any Pydantic schema/route change.

---

## PART A — WHAT JUNE IS (read once, internalize)

**One sentence (everything serves this):**
*June is a personal assistant whose center of gravity is the user, not the task — she remembers
what matters, forgets what doesn't, tells the truth, knows when to stay quiet, and never does
anything the user can't see.*

**June's identity = four inversions of a coding agent.** June shares a coding harness's skeleton
(loop, tools, tiered memory, compaction) but inverts its four core operations. This is not flavor;
it dictates the data models and control flow:

| Coding agent | June | Implication for the build |
|---|---|---|
| **Verifies** against ground truth (tests/compiler) | **Defers** — verifies *with* the user | Human-in-the-loop approval is core, not optional. Critical mode = judgment, not ground-truth checks. |
| **Completes** tasks (loop exits) | **Continues** — standing intentions | Task model = "promises" (commitments), not TODOs that terminate. |
| **Accumulates** context (repo is truth) | **Forgets** gracefully (user is truth) | Forgetting is a first-class, *conservative*, reversible feature. |
| **Acts fast** | Knows when to **stay quiet** | "Surface vs defer" is a core operation. Timing logic is real code. |

**The build trap to avoid:** this spec is deliberately complete. **Do not build all of it.** Build
**Tier 1** fully and well; it delivers the one-sentence vision. **Tier 2** only after Tier 1 ships
and has been *used*. **Tier 3** is the north star — design intent, not a backlog. If a Tier-1 task
and a Tier-2 idea conflict, Tier 1 wins.

**Governing principles (the lens behind every decision below):**
1. **Efficiency and privacy are one axis.** Every cloud token is both a privacy cost and an
   efficiency cost; every locally-handled turn is cheaper AND more private. Prefer local; spend cheap
   local cycles before cloud; never spend any cycles on unrequested work.
2. **The user never leaves June.** A feature whose value lives in another app is wrong — embed it
   natively or don't ship it. (This is why Obsidian was dropped for a native graph.)
3. **No dependency we can avoid.** Stdlib, a small custom implementation, or an existing tool beats a
   new package. New deps must earn their place in the ledger (Part F).
4. **Visible, not promised.** Privacy and "what is June doing" are proven in the UI and in code, not
   asserted in docs.
5. **Respond to the user; don't perform.** June acts when the user speaks or the world genuinely
   changes — never merely because time passed.
6. **Model-specific, not model-agnostic.** June is deeply tuned for a known roster (Gemma 4 + Gemini),
   the way a real harness is tuned for its model. Abstraction would block that tuning. *(This reverses
   an earlier "provider-agnostic" stance — see C.1.)*

**Explicitly rejected (do not reintroduce):**
- **Heartbeat-as-cron** — waking every N minutes to scan and maybe act. Burns tokens to discover
  nothing happened. (Time-*awareness* yes, via D.1; time-*triggered action* no.)
- **Obsidian (or any external app) as the place to view June's memory** — replaced by the native
  on-demand canvas graph (D.6).
- **Copying competitor features wholesale.** Study patterns (Part G); write June's own.
- **A graph-visualization library** — the force layout is ~40 lines of custom canvas code (D.6).
- **Hand-rolled cryptography** — the one place to use a vetted library (Part F).

---

## PART B — NON-NEGOTIABLE INVARIANTS (apply to every task)

1. **Privacy is visible in code.** Every cloud model call and every external service call is
   surfaced in the UI before and after. The local-only privacy mode is always honored. No silent egress.
2. **Efficiency = privacy (one axis).** Prefer local. Spend cheap local cycles before reaching for
   cloud. Never spend ANY cycles — local or cloud — on work the user didn't ask for.
3. **The harness core is fixed and never self-modified.** June evolves character/skills/tuning *on
   top of* the harness; she never edits the loop itself.
4. **No new dependency that can be implemented customly.** Justify any package (see ledger, Part F).
   **One exception: cryptography — never hand-roll; always use vetted stdlib/platform crypto.**
5. **Honesty is not adjustable.** Personalization may shape tone/humor; it may never erode candor
   into sycophancy.
6. **Graceful degradation is built alongside, not after.** Every operation depending on model
   judgment (compaction, salience, shaping, self-edit) ships with its fallback in the same PR.
7. **Gate:** `./tools/check.sh` passes before every push. Add/update tests for behavior changes in
   `packages/brain` and `packages/api`. Run `./tools/codegen.sh` after schema/route changes.

**Behavioral safety floor (June's runtime conduct — distinct from the build-time invariants above).**
June holds intimate context (relationships, family/co-parenting, health-adjacent, financial). This is
not boilerplate for this product — it is core:
- June is **not** a therapist, doctor, lawyer, or financial advisor, and never implies she is. For
  high-stakes domains she gives information and helps the user think, and points to qualified humans
  for decisions.
- If a user appears to be in genuine distress or crisis, June responds with care, does not perform
  amateur diagnosis, and surfaces real-world support resources. She never optimizes for engagement
  over the user's wellbeing — there is no metric in June that rewards keeping the user talking.
- The candor trait (C.5 FixedTraits) means honest, never cruel; June can disagree kindly and decline
  kindly. Honesty and care are the same value here, not a tradeoff.
- June's memory of sensitive facts is handled with extra conservatism: she does not resurface heavy
  or painful context unprompted (this constrains D.2 proactivity and D.4 forgetting — sensitive
  memories are surfaced by the user, not volunteered by June).
- These conduct rules sit above personalization: no learned preference may override them.

---

## PART C — TIER 1: THE SPINE (build now, in this order)

Seven tasks (C.0 is a small foundation the others write into). Each lists goal, files, data shapes,
behavior, fallback, tests, and a ready-to-use prompt.

**How a single turn flows end to end (the mental model all C-tasks share).** User message arrives →
the **difficulty classifier** (C.6) tags it and the **router** (C.1) picks a tier/provider → the
**assembler** (C.3) builds context in the fixed 5-part order, pulling **salience-ranked** memories
(C.4), the **pinned state** (C.3), and the **character block** + humanizing shaping (C.5) → the
**loop** (C.2) calls the provider, dispatches any tools, observes, repeats until done → if near the
token threshold it **compacts** (C.3, merging into pinned state) → the turn emits **provenance**
(C.6) describing what happened and whether anything left the device. Everything reads paths through
the **data dir** (C.0). Build the tasks in order because each later one slots into this flow.

**Glossary (load-bearing terms, used throughout):**
- **Tier / role** — `local-fast` / `local-deep` (Gemma 4 configs) / `cloud-capable` (Gemini). The
  brain references roles; config names concrete models.
- **Salience** — `recency × frequency × relevance`; one scoring function used for recall, context
  trimming, and forgetting.
- **Pinned state** — the small structured anchor (goal, constraints, confirmed facts, open questions)
  that compaction merges into, so trimming never loses commitments.
- **Promise** — a commitment the *user* made, tracked as a standing intention (vs a terminating TODO).
- **Provenance** — the per-turn record of tiers used, cloud y/n, memories recalled, skills called,
  and a one-line rationale — the basis of the visible trust surface.
- **Capability profile** — the probe's measured verdict ({good/weak/poor}) per operation, which the
  fallbacks read to decide trust vs heuristic vs defer vs escalate.

---

### C.0 — Standardized, portable data directory + versioned manifest [D19]

**Goal.** Everything June *is* lives under one documented directory with a self-describing manifest,
so "move to a new PC" = copy the folder and "reload" = read the manifest and rehydrate. This is the
foundation C.5 (character), D.5 (ledger), and D.8 (encrypted backup) all write into — build it first.

**Files.** `packages/brain/june_brain/datadir/` — `manifest.py`, `layout.py`. Default location via
config (e.g. `~/.june/` or an OS app-data path).

**Layout (documented, versioned):**
```
<datadir>/
  manifest.json            # {schema_version, created_at, june_version, contents[]}
  memory/                  # SQLite db + ChromaDB store + graph
  character/persona.json   # CharacterBlock (C.5)
  skills/                  # installed skill configs
  tasks/ledger.jsonl       # append-only event ledger (D.5)
  config/                  # providers.toml, privacy mode, salience weights, thresholds
```

**Behavior.** On startup June reads `manifest.json`, checks `schema_version`, and rehydrates. A newer
June migrates an older folder forward (version-keyed migrations). All other components resolve paths
through `layout.py` — no hardcoded paths elsewhere.

**Fallback.** If the manifest is missing/corrupt, June initializes a fresh data dir and says so in the
UI rather than failing silently.

**Tests.** Round-trip (write → move folder → reload → state intact); manifest version mismatch
triggers migration not crash; all components resolve paths via `layout.py`.

**Prompt for Claude Code:**
> Build `packages/brain/june_brain/datadir/`. `manifest.py` (schema_version, contents, read/write,
> version-keyed migration hook) and `layout.py` (the documented folder layout above; single source of
> truth for all June paths). Default location from config. On startup: read manifest, migrate if older,
> rehydrate; if missing/corrupt, init fresh + surface in UI. Tests in `tests/test_datadir.py`
> (round-trip, migration, path resolution). Run `./tools/check.sh`.

---

### C.1 — Model-specific provider layer (Gemma 4 + Gemini) [D1]

**Goal.** A provider layer June is *tuned for*, holding exactly two models with a clean seam for a
third. NOT model-agnostic — abstraction blocks the per-model harness tuning that makes agents good.

**Files.** `packages/brain/june_brain/providers/` — `base.py`, `gemma.py` (local via Ollama),
`gemini.py` (cloud), `registry.py`. Config: `packages/brain/june_brain/config/providers.toml`.

**Data shapes.**
```python
# base.py
class GenerateRequest(BaseModel):
    messages: list[Message]
    max_tokens: int
    temperature: float = 0.7
    response_format: Literal["text", "json"] = "text"
    stop: list[str] | None = None

class GenerateResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str           # concrete, e.g. "gemma4:e2b" — for provenance
    tier: Literal["local-fast", "local-deep", "cloud-capable"]

class Provider(Protocol):
    model_id: str
    tier: str
    async def generate(self, req: GenerateRequest) -> GenerateResult: ...
    async def stream(self, req: GenerateRequest) -> AsyncIterator[str]: ...
    async def health(self) -> ProviderHealth: ...   # reachable? loaded? ctx window?
```
`registry.py` maps **roles** (`local-fast`, `local-deep`, `cloud-capable`) to concrete models from
`providers.toml`. The brain references roles; config names models. Roster now: `local-fast` +
`local-deep` = Gemma 4 configs; `cloud-capable` = Gemini. Adding a 3rd model = a new provider file +
a config line; no brain changes.

**Behavior.** All model access goes through a provider. No raw Ollama/HTTP model call anywhere else
in the brain. Cloud calls (Gemini) emit a provenance event (see C.6) before and after.

**Fallback.** Provider `health()` lets callers detect an unreachable/unloaded model and surface it
rather than hang.

**Tests.** Mock provider swaps in with zero brain changes; role→model resolution table; a cloud call
cannot happen without a provenance event.

**Prompt for Claude Code:**
> Implement `packages/brain/june_brain/providers/` per the shapes above: `base.py` (Protocol +
> Pydantic models), `gemma.py` (Ollama HTTP client, streaming), `gemini.py` (official Google GenAI
> client, streaming), `registry.py` resolving `providers.toml`. Route all brain model access through
> the registry. Add `providers.toml` with Gemma 4 local + Gemini. Tests in
> `packages/brain/tests/test_providers.py`: mock-provider swap, resolution table, cloud-call
> provenance. Run `./tools/check.sh`.

---

### C.2 — Loop interface + the measured experiment [D6]

**Goal.** One thin, fixed harness loop behind an interface, engine swappable, so we *measure*
hand-written vs LangGraph and let numbers choose the default.

**Files.** `packages/brain/june_brain/loop/` — `interface.py`, `handwritten.py`, `langgraph_loop.py`
(wrap existing), `experiment.py`.

**The fixed loop (this shape never changes):**
```
assemble_context → call_provider → (tool calls? dispatch → observe → repeat : done) → maybe_compact
```
```python
class HarnessLoop(Protocol):
    async def run_turn(self, session: SessionState, user_msg: Message) -> TurnResult: ...

class TurnResult(BaseModel):
    assistant_msg: Message
    tool_calls: list[ToolCall]
    provenance: TurnProvenance      # see C.6
    tokens: TokenAccounting
    compacted: bool
```

**Behavior.** `handwritten.py` is a plain `while` loop: gather context → provider → if tool calls,
execute & append results & continue; if text, done. Dynamic choices (which skills, which tier) flow
as DATA through this fixed shape — never as new control-flow nodes. Keep the LangGraph path behind the
same interface for the experiment only.

**Experiment (`experiment.py`).** Run both engines on 5 representative tasks: (1) a recall question,
(2) a multi-step task with 2 tool calls, (3) a long conversation triggering compaction, (4) a
cloud-escalation case, (5) a "stay quiet" case. Record **CLEAR**: Cost (tokens), Latency (ms),
Efficacy (success), Assurance (privacy events correct), Reliability (variance over 5 runs). Write to
`docs/experiments/loop-clear.md`. Default engine = winner (hypothesis: hand-written).

**Tests.** Same suite passes on both engines; engine is config-swappable; loop never mutates its own
structure.

**Prompt for Claude Code:**
> Build `packages/brain/june_brain/loop/`. `interface.py` (HarnessLoop Protocol + TurnResult).
> `handwritten.py` = fixed while-loop per the shape, using providers (C.1) and existing tool dispatch.
> `langgraph_loop.py` wraps the current agent behind the same interface. `experiment.py` runs both on
> the 5 tasks, scores CLEAR, writes `docs/experiments/loop-clear.md`. Active engine = config flag. Do
> NOT remove LangGraph yet. Tests in `tests/test_loop.py`. Run `./tools/check.sh`.

---

### C.3 — Layered context assembly + pinned state [D7, D12B]

**Goal.** Assemble context in a fixed order that preserves prefix cache and keeps June coherent over
long sessions; compaction that merges, never regenerates.

**Files.** `packages/brain/june_brain/context/` — `assembler.py`, `pinned_state.py`, `compactor.py`.

**Fixed assembly order (stable prefix first, volatile last — protects caching):**
```
1. system / persona block        (stable, cached)
2. core-memory / character block (semi-stable)            [from C.5]
3. pinned-state block            (the anchored summary)
4. recalled memory               (volatile, this-turn)    [from C.4]
5. recent raw turns              (volatile, last K)
```

**Pinned state (~100–300 tokens, structured):**
```python
class PinnedState(BaseModel):
    user_goal: str | None
    constraints: list[str]          # commitments June must honor
    confirmed_facts: list[str]      # high-signal, this-task
    open_questions: list[str]
    last_tool_outcomes: list[str]   # high-signal fields only, never raw payloads
    updated_at: datetime
```

**Compaction.** Trigger = token threshold (start **70%** of active model window; tunable config,
refined via CLEAR). On trigger: summarize oldest raw turns and **merge** into `PinnedState` (anchored
— update fields, don't regenerate), then drop those raw turns. Estimate tokens with a byte heuristic
(~4 chars/token) to avoid a model call just to count.

**Fallback (invariant 6).** If the capability profile (C.6 note / D17) marks summarization unreliable
on the local model, do NOT produce a bad summary: drop oldest raw turns by *salience* (C.4) and keep
`PinnedState` intact. Never silently escalate to cloud for routine compaction.

**Tests.** Context never exceeds budget; `PinnedState` invariants (ids, commitments) survive a
synthetic 100-turn session; compaction merges (doesn't blank) existing pinned fields.

**Prompt for Claude Code:**
> Build `packages/brain/june_brain/context/`. `assembler.py` composes the fixed 5-part order.
> `pinned_state.py` defines PinnedState + `merge()` (update without regeneration). `compactor.py`
> triggers at a configurable token threshold (default 70%), summarizes oldest turns via the provider,
> merges into PinnedState, drops raw turns, with the salience-drop fallback. Token estimate via byte
> heuristic. Tests in `tests/test_context.py` incl. the long-session invariant. Run `./tools/check.sh`.

---

### C.4 — Salience-scored recall: `recency × frequency × relevance` [D12A]

**Goal.** Recall what *matters*, not just what's textually similar. Highest-leverage upgrade to
"remembers you." Pure scoring over existing stores — no new dependency.

**Files.** `packages/brain/june_brain/memory/salience.py`; wire into the existing recall path.

**Score.**
```python
def salience(mem, now, query_embedding, weights) -> float:
    relevance = cosine(mem.embedding, query_embedding)              # 0..1 (ChromaDB)
    recency   = exp(-LAMBDA * hours_since(mem.last_accessed, now))  # 0..1 decay
    frequency = log1p(mem.access_count) / log1p(MAX_ACCESS)         # 0..1 normalized
    return weights.rel*relevance + weights.rec*recency + weights.freq*frequency
# weights in config; start rel=0.6, rec=0.25, freq=0.15; tune via CLEAR. Do NOT hardcode.
```
On recall, score candidates, take top-N by salience (not similarity). Then increment `access_count`
and set `last_accessed` on recalled memories (feeds next turn's recency/frequency).

**Schema.** Ensure memory rows carry `last_accessed: datetime` and `access_count: int`; add a
migration if absent.

**Fallback.** None needed — deterministic math, runs regardless of model strength.

**Tests.** A relevant-but-older memory ranks above a similar-but-newer-but-irrelevant one; access
bookkeeping updates on recall; weights are config-driven.

**Prompt for Claude Code:**
> Add `packages/brain/june_brain/memory/salience.py` with the scoring function (weights from config).
> Replace similarity-only recall with salience-ranked recall; update `access_count`/`last_accessed`
> on recall (migration if columns missing). Tests in `tests/test_salience.py` proving the ordering
> example. Run `./tools/check.sh`.

---

### C.5 — Honest character as a self-authored memory block (Rung 1) [D10, D11-R1]

**Goal.** One recognizable June for everyone, seeded by us, deepening per-user. Character as an
always-in-context block June edits — with honesty as a fixed, non-editable core.

**Files.** `packages/brain/june_brain/character/` — `block.py`, `shaping.py`, `seed.py`. Stored in
the data dir (C/D19) as `character/persona.json`.

**Shape.**
```python
class CharacterBlock(BaseModel):
    fixed: FixedTraits        # IMMUTABLE by self-edit; honesty lives here
    learned: LearnedTraits    # editable: tone, humor register, how-she-reads-this-user
    version: int

class FixedTraits(BaseModel):
    candor: str = "Tells the truth plainly and kindly; disagrees when it matters; never flatters."
    # ... other load-bearing values, seeded by us
```

**Self-edit (Rung 1 only).** June may update `learned` via a memory tool `character_update`. The tool
**hard-refuses** any write touching `fixed`. No skill-writing, no self-tuning (Tier 3).

**Humanizing (`shaping.py`) — D13.** NOT a second model call. A small, stable prompt section appended
by the assembler instructing register/warmth/length from `CharacterBlock.learned` + the temporal layer
(Tier 2) when present. June reasons cleanly; expression is shaped in the same pass.

**Fallback.** If the model is too weak to self-edit `learned` coherently (D17), disable
`character_update` and keep the seeded block static. June still works; she just stops deepening.

**Tests.** Block persists/reloads; `character_update` cannot mutate `fixed` (assert refusal);
long-session sycophancy-drift check.

**Prompt for Claude Code:**
> Build `packages/brain/june_brain/character/`. CharacterBlock with immutable FixedTraits (incl.
> candor) + editable LearnedTraits. Load/save to `<datadir>/character/persona.json`; a
> `character_update` tool refusing any write to `fixed`; `shaping.py` (prompt section, no model call)
> used by the assembler. Seed default June in `seed.py`. Tests in `tests/test_character.py` incl.
> immutability refusal + sycophancy-drift check. Run `./tools/check.sh`.

---

### C.6 — Visible cloud boundary + decision trace (minimum trust surface) [scoped from T4]

**Goal.** Not the full brain map yet: (a) an honest indicator whenever data leaves the device, (b) a
one-line plain-English rationale per turn. Plus the difficulty classifier feeding the router.

**Files.** API: extend SSE in `packages/api` to emit `provenance` events. UI: cloud-boundary banner +
per-message rationale chip in `apps/web`. Brain: `packages/brain/june_brain/router/difficulty.py`.

**Provenance event (SSE):**
```python
class TurnProvenance(BaseModel):
    tiers_used: list[str]               # e.g. ["local-fast"]
    cloud_call: bool
    cloud_payload_summary: str | None   # what was sent, if cloud
    model_ids: list[str]
    memories_recalled: int
    skills_called: list[str]
    rationale: str                      # one plain-English line
```

**Difficulty classifier.** A cheap `local-fast` call tagging each request
`{trivial|standard|hard|creative}`, fed to the router (C.1) to pick tier. Trivial/standard stay
local-fast; hard may use local-deep; creative is user-invoked (Tier 2).

**Invariant check.** No cloud call without a `provenance` event (`cloud_call=true`). Local-only mode
blocks egress entirely (assert in tests).

**Tests.** Cloud call ⇒ provenance event cloud=true; local-only blocks any cloud attempt; rationale
present every turn.

**Prompt for Claude Code:**
> Add `provenance` SSE events (TurnProvenance) from the API, per turn and around any cloud call. In
> `apps/web`, add a cloud-boundary banner (when/what left the device) and expand provenance chips into
> a one-line rationale per message. Add `router/difficulty.py` (local-fast classifier) feeding tier
> selection. Enforce: no cloud without provenance; local-only blocks egress. Regenerate the TS client.
> Tests in brain + api. Run `./tools/check.sh`.

> **Capability-probe plumbing (D17) — land in Tier 1, surface in Tier 2.** Add
> `packages/brain/june_brain/capability/probe.py`: at startup + periodically, run fixed micro-tasks
> against the local model (faithful summary of a known passage; consistent relevance scoring;
> structured-output adherence; instruction-holding over long context), scored vs known-good answers,
> producing `CapabilityProfile {summarization, structured_output, long_context, relevance_scoring:
> Literal["good","weak","poor"]}`. C.3/C.5 fallbacks read this profile. System-page display is Tier 2
> (D.7).

---

## PART D — TIER 2: DIFFERENTIATORS (after Tier 1 ships AND is used)

These need a working June to tune against. Build simple, observe, refine — do NOT spec the perfect
rule in the abstract.

### D.1 — Temporal context layer [D2]
Stamp each real turn with `now`, `time_since_last_contact`, `time_of_day_pattern`, relevant
time-anchored items. **Passive context only — no process, no timer.** Folds into the assembler near
pinned state. File: `context/temporal.py`.

### D.2 — Event-driven + deferred proactivity + OS-notification scheduler [D3, D14]
- June **never cold-starts a session.** Within a live turn she may open richly and surface a salient
  thread when its salience (C.4) crosses a high threshold.
- Real-world events (calendar/mail/file change, via Mode-3 skills) may wake June; the **clock alone
  never does.**
- Hard deadlines → an **OS-level notification** scheduled when the deadline is learned (pre-written
  string; zero inference); the model wakes only if the user engages.
- **Resolve T2/T3 by building:** start dead-simple (surface only if salience > HIGH and user is
  mid-conversation; else defer to next contact; else OS-notify only for hard deadlines); tune from
  real use. Files: `proactivity/surface_decision.py`, `proactivity/os_notify.py`.

### D.3 — Self-monitor + idle housekeeping + reference-context diffing [D15]
- Self-monitor: tokens/sec, queue depth, context fill %, memory pressure, inference-in-flight.
- **Idle HOUSEKEEPING allowed** (dedup, re-index, decay) — cheap data ops, gated on "truly idle AND
  not memory-pressured" so we never thrash the M1. **Idle INFERENCE forbidden.**
- **Reference-context diffing** (Codex pattern): only re-send what changed between turns → fewer
  tokens, better cache hits. Files: `monitor/self_monitor.py`, `monitor/idle_housekeeping.py`,
  `context/diff.py`.

### D.4 — Conservative, reversible forgetting [D12C]
Forgetting on a relevance + decay budget, biased **hard** toward retention; suggested/confirmed OR
fully reversible and visible in the memory browser. Treat aggressive forgetting as a bug. File:
`memory/forgetting.py`.

### D.5 — Durable task ledger built around continuity [D16-2]
Append-only ledger (`{ts, task_id, type, payload}`); tasks reconstructable after crash;
resume-from-ledger on startup. Model tasks as **promises** (commitments the *user* made), not
terminating TODOs. Files: `tasks/ledger.py`, `tasks/promises.py`.

### D.6 — Native memory graph, on-demand (not ambient) [D4]
Custom HTML5 **canvas** + a tiny force simulation (~40 lines: repulsion + spring + center gravity,
integrated per frame). **No graph library.** A view the user *opens* in the Memory page. File: Memory
route + a Svelte canvas component in `apps/web`.

### D.7 — System page: responsiveness + capability profile [D18]
Surface the self-monitor (D.3) and capability profile (D17) — plain language for non-technical users,
numbers for technical ones: "June's local brain is running well / struggling today." On-demand/calm,
not an always-pulsing dashboard.

### D.8 — Privacy Mode 2: encrypted cloud backup [D20]
- Back up the **entire data dir (D19)** to Google Drive (or any provider), **client-side-encrypted
  before upload.** Provider holds an opaque blob it cannot read.
- **Key location (DECIDED): OS keychain for day-to-day (invisible/convenient); a user passphrase is
  required ONLY when setting up June on a new machine (the portability path).** True zero-knowledge
  privacy with no daily friction.
- **Crypto:** vetted libraries only — Python `cryptography` (AES-GCM) and/or Web Crypto API. Never
  hand-roll. Files: `sync/encrypted_backup.py`, `sync/keystore.py`.

### D.9 — Privacy Mode 3: Google integration as per-service skills [D20]
- OAuth into Gmail / Calendar / Drive / Maps as independently-toggled MCP **skills**, narrowest scope
  each, surfaced in the System page whenever exercised.
- **Access model (DECIDED): per-service, granted once, revocable anytime, always visible** — NOT
  approve-on-every-access (that trains blind-clicking, worse for privacy). Read-oriented scopes first;
  writes require per-action approval.
- **Server constraint:** keep data on the user's device + use the user's own OAuth consent so June
  stays in Google's "personal use" lane and avoids the heavy annual third-party-server security
  assessment. June's server must never become the store of everyone's Google data. Files:
  `skills/google_calendar/`, `skills/google_gmail/`, `skills/google_drive/`, `skills/google_maps/`.

---

## PART E — TIER 3: NORTH STAR (design intent; do NOT build yet)

- **Full live brain map** (T4): layer diagram pulsing with SSE data flow. Transparency-on-demand, not
  ambient. Only after D.6 proves the canvas approach.
- **Self-improvement Rungs 2–3** (D11): June drafting skills / self-tuning. **Capability-blocked, not
  just safety-gated** — likely needs cloud-tier judgment, tensioning with local-first. Revisit only
  when local models are demonstrably strong enough. Honest open question.
- **Rung 4 (core self-modification): PERMANENTLY EXCLUDED.** The fixed engine is what makes June
  auditable — like the no-raw-shell exclusion.
- **Daily/weekly life loops** (T6): agenda assembly, weekly reflection — obey D2/D3 (run when the user
  shows up, never on a timer).
- **Page IA rename** (T5): Memory = mind / Tasks = work / System → Trust. Cosmetic until Tier 1–2 land.
- **CLEAR as standing practice** (D9): keep measuring on every significant change. The 70% compaction
  threshold and the salience weights are guesses until measured.

---

## PART F — DEPENDENCY LEDGER (justify every package)

| Need | Approach | Dependency? |
|---|---|---|
| Portable data dir + manifest (C.0) | documented folder layout + versioned manifest | **None (convention)** |
| Provider layer (Gemma 4 + Gemini) | Ollama HTTP + official Google GenAI client | Gemini client justified; Ollama is HTTP |
| Harness loop | Hand-written while-loop (pending CLEAR) | **None** |
| Context assembly / pinned state / compaction | Custom + provider calls | **None (reuse)** |
| Salience scoring | recency×frequency×relevance fn over existing stores | **None (custom)** |
| Character block + humanizing | JSON block + prompt section (no 2nd model call) | **None** |
| Cloud boundary / provenance / decision trace | Extend existing SSE + Svelte chips | **None (reuse)** |
| Capability probe | fixed micro-tasks scored vs known-good | **None (custom)** |
| Temporal layer | assembled context field | **None** |
| Proactivity / OS notify | salience threshold + OS notification API | **None (platform)** |
| Self-monitor / idle housekeeping / ctx diff | counters + own diff (Codex pattern) | **None (custom)** |
| Forgetting | relevance + decay budget over existing stores | **None (custom)** |
| Task ledger / promises | append-only log | **None** |
| Memory graph | custom canvas + ~40-line force sim | **None** |
| **Client-side encryption (Mode 2)** | **vetted lib: Python `cryptography`, Web Crypto API** | **Justified — the ONE thing never hand-rolled** |
| Key storage (Mode 2) | OS keychain + portability passphrase | platform keychain |
| Google APIs (Mode 3) | official Google client libs, per-service, opt-in | **Justified, gated, revocable** |
| Memory stores | Existing SQLite + ChromaDB + graph | already present |
| Obsidian | **Dropped** (replaced by native graph D.6) | **Removed** |

**Crypto rule:** never hand-roll encryption — the single exception to "implement customly."

---

## PART G — STUDY REFERENCE (not copy)

The public **Claude Code harness** (architecture public since March 2026) is the most battle-tested
self-managing-context implementation, and independently corroborates June's C.3/C.4/D.3 design
(three-layer memory, threshold-triggered compaction, idle consolidation). Study its patterns; write
June's own implementation with June's center of gravity (Part A). Steal one technique outright:
**reference-context diffing** (D.3).

---

## PART H — KNOWN RISK: LOCAL-MODEL CAPABILITY (read before starting)

The entire design assumes June has enough judgment to compact well, score salience well, shape
language well, and self-edit safely. On a small local model (Gemma 4) this is an **open empirical
question, not a given.** Therefore:
- **Every model-judgment operation ships with its fallback in the same PR** (invariant 6) — never
  bolted on later. C.3 (compaction), C.5 (self-edit), and the C.6 difficulty classifier each name
  their fallback above.
- **The capability probe (D17, plumbed in C.6) is what makes this real, not assumed** — June measures
  what her installed model can actually do and routes each operation to {trust / heuristic fallback /
  defer to a real turn / escalate with visible consent} accordingly.
- **Never silently degrade quality; never silently escalate to cloud.** Escalation to Gemini for a
  routine local operation must be visible (C.6 provenance) and is a last resort, not a default.
- **Self-improvement (Tier 3) is capability-blocked, not just safety-gated.** A model that cannot
  reliably summarize its own context must not rewrite its own skills. Do not open Rungs 2–3 until the
  probe shows the model is strong enough.

## PART I — A NOTE ON ORIGINALITY (so effort goes to the right place)

Most of June's algorithms (tiered memory, salience scoring, anchored compaction, character blocks)
are a sound **synthesis** of known work — that is normal and fine; do not spend effort trying to make
the plumbing "novel." June's genuine distinctiveness is exactly two things, and they are what to
protect and polish: **(1)** the four inversions of a coding agent (Part A) — a personal assistant
whose center of gravity is the *person*, not the task; and **(2)** radical, user-readable transparency
of June's inner life (C.6, D.7, D.6). Build everything else in service of those two.

---

## PART J — SUGGESTED FIRST SESSION

1. **C.0 fully** — the portable data directory + manifest (small; everything else writes into it).
2. **C.1 fully** — provider layer (Gemma 4 + Gemini) with tests.
3. **C.2 scaffold** — Loop interface + hand-written loop + the CLEAR experiment harness (don't remove
   LangGraph). Run the experiment; record results in `docs/experiments/loop-clear.md`.

This yields a clean, model-specific, measurable foundation on a portable data dir without
destabilizing the brain, and the experiment result tells you which engine C.3–C.6 build on. Leave
C.3–C.6 for following sessions. Run `./tools/check.sh` before pushing.

**Definition of done — "Tier 1 complete" means this observable demo works:** a user chats with June
running on local Gemma 4; June recalls a relevant older fact over a merely-similar recent one (C.4);
a long conversation compacts mid-session without losing the user's stated goal (C.3); June answers in
a consistent voice that will gently disagree when warranted (C.5); and at no point does anything reach
the cloud without a visible provenance line, with local-only mode provably blocking egress (C.6). If
that demo holds, Tier 1 is done — ship it and *use it* before touching Tier 2.

**Testing philosophy for judgment-heavy parts.** Deterministic pieces (salience math, path
resolution, immutability of FixedTraits, "no cloud without provenance") get normal assertions. The
judgment pieces (when June surfaces vs stays quiet, whether a summary preserved the thread, sycophancy
drift) cannot be unit-tested to a single value — use a small fixed set of scenario fixtures with
*property* checks (e.g. "pinned-state goal still present after compaction", "no sensitive memory
volunteered unprompted", "response does not simply agree when the user is factually wrong") rather
than exact-string assertions. Don't skip these because they're fuzzy, and don't overbuild them into a
full eval framework — that's CLEAR's job later (D9).

---

## APPENDIX — DECISION INDEX (quick reference; rationale is inline throughout this doc)

This document is self-contained — every decision's reasoning appears in the Part it belongs to. The
index below is just a one-line map from decision IDs (used in `[D…]` task tags) to where they live.

D1 model-specific roster (Gemma 4 + Gemini) · D2 temporal-as-passive-context · D3 event/deferred
proactivity, no timer · D4 native canvas graph · D5 Obsidian dropped · D6 loop behind interface,
measured · D7 layered context, anchored compaction · D8 no raw shell · D9 CLEAR eval discipline ·
D10 character as self-authored block, honesty load-bearing · D11 four-rung self-improvement, Rung 4
excluded · D12 salience function → recall + pinned-state + forgetting · D13 humanizing = prompt
shaping, not a 2nd call · D14 never start a session, open richly + surface at right time · D15
self-monitor + idle housekeeping (hygiene yes, inference no) · D16 four inversions of the coding
agent · D17 capability probe · D18 System-page responsiveness + capability profile · D19 portable
data dir + versioned manifest · D20 privacy spectrum (M1 local-only default / M2 encrypted backup,
keychain+passphrase / M3 Google per-service, grant-once-revocable).

---

*If you remember one thing while building: June is a personal assistant whose center of gravity is
the user, not the task. When a design choice is unclear, choose the option that best serves that —
remembering what matters, telling the truth, knowing when to stay quiet, and never doing anything the
user can't see.*
