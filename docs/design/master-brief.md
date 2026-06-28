# June — Master Design Brief

> **For:** Claude (on claude.ai, Artifacts enabled, latest Opus/Sonnet) — or any
> chat-with-rendered-UI tool. **Two deliverables in one brief:** (Track A) the full
> product UI/UX system, and (Track B) a depth presentation with architectural
> diagrams. **Status:** v2, 2026-06-28. This is now the standalone design brief;
> the older narrow chat-redesign prompt was removed during documentation
> consolidation.
>
> **Authority order if anything here conflicts:** rebuild-plan for load-bearing
> decisions, development-plan for active sequence, vision for why, then this brief
> for how it should look and read.

---

## 0. How to use this brief

There are two tracks. Do them in this order; Track A produces the visual language
that Track B's slides reuse.

- **Track A — Product UI/UX.** Produce a single React + Tailwind artifact with a
  top tab strip switching between every named screen/state in §4, a light/dark
  toggle, and a design-tokens block at the top. Lock the **mascot and visual
  direction first**, then the chat/home continuity surface, then the other
  surfaces.
- **Track B — Presentation.** Produce a second artifact: a slide deck (React,
  one slide per view, keyboard/next-prev navigation) that explains June in depth,
  carrying the diagrams in §6. Reuse Track A's tokens, mascot, and accent so the
  deck and the product look like one thing.

Work iteratively: best first answer per track, then name three things you'd change
and ask which way to push. Do not start the SvelteKit/engineering implementation —
that is a separate post-approval pass.

---

## 1. The product in one paragraph (paste-ready)

June is the personal AI that remembers you. She runs locally on your laptop via
Gemma 4, optionally reaches the cloud via Gemini, and works identically in the
browser, on Mac, and on iPhone — one brain, one memory, one codebase, every
surface. Her center of gravity is the **user, not the task**: she remembers what
matters, forgets what doesn't, tells the truth plainly, knows when to stay quiet,
and never does anything you can't see. Memory and visible honesty *are* the
product. She is private by default — chat and recall stay on the machine; capability
reaches Gemini only when you ask, with the call shown before and after. She feels
like a calm, competent companion with a persistent identity — not a chatbot, not a
productivity app, not a developer tool.

## 2. Who it's for

A thoughtful individual who already uses AI daily and is tired of re-explaining
themselves every morning. They value privacy, install software from GitHub, and
notice when software feels rushed. They are **choosing a companion, not shopping for
features.** Secondary audience for Track B: technically literate skeptics (security-
and privacy-minded) who need to *believe the trust claims* — the deck must earn that
belief with architecture, not adjectives.

## 3. The non-negotiables every pixel must honor

These come from `vision.md` and the project invariants (CLAUDE.md). Design *toward* them.

1. **Memory is the product.** The most important thing June does is remember you,
   by *salience* (recency × frequency × relevance), and let you inspect, edit,
   export, and forget. The Memory surface is not a settings page; it is the heart.
2. **Efficiency and privacy are one axis.** Local is both cheaper and more private.
   The UI should make "this stayed local" feel like the good, default, common case —
   not a deprivation. Cloud is the visible exception, never the silent default.
3. **Visible, not promised.** Every cloud call and every external service call is
   surfaced before and after. This is shown in the UI and provable in code — the
   *activity terminal* and *provenance line* are the load-bearing trust surfaces.
   This is one of June's only two genuine differentiators; make it beautiful, not
   buried.
4. **Honesty is not adjustable.** Personalization shapes tone, never erodes candor
   into sycophancy. The voice in every piece of copy is warm but plainspoken; June
   can gently disagree. No flattery, no hype, no exclamation-point energy.
5. **One codebase, every surface.** Design web, desktop, and mobile as the same
   product, not three. The mobile screen must feel native, not a shrunken desktop.
6. **Behavioral safety floor.** June holds intimate context (relationships, family,
   health-adjacent, financial). She is not a therapist/doctor/lawyer/advisor and
   never implies she is; she responds to distress with care, not diagnosis; she
   never resurfaces heavy memories unprompted; no surface optimizes for engagement.
   Nothing in the design should feel like a dopamine loop, a streak, or a nudge to
   keep talking.

## 4. Track A — the surfaces to design

Design each as a state in the one artifact. The chat surface uses two registers,
a centered composer, an activity terminal, and the mascot as busy indicator.

### 4.1 Chat and home continuity — the core
Two registers on one screen: **Conversation** (foreground bubbles — only what was
actually said) and **Activity** (a subdued, monospaced "flight recorder" below the
**centered composer** — recall, route, tool calls, and the provenance/cloud-boundary
line). States to render: idle/greeting; active with terminal collapsed; active with
terminal expanded; a **cloud-escalation** turn where the boundary line is unmissable;
and **mobile** collapsed + expanded. The empty state must include the continuity
summary: open promises, waiting approvals, runtime/privacy mode, and degraded
memory or local capability. The mascot (June sun / solstice mark) replaces the
wordmark and doubles as the global busy indicator.

### 4.2 The approval gate (NEW — the "defers" inversion made visible)
This is June's anti-injection guard layer surfacing as a UI moment. When the model
wants to take a **consequential or exfiltration-shaped action** — any local/network
*write* or *execute*, or a network *read* whose arguments are tainted by a prior
tool result this turn — the loop **pauses** and asks. Design the inline approval
card that appears in the conversation register (not a modal that hijacks the screen):
- States what June wants to do, in plain English, and *why* (e.g. "Send the summary
  you wrote to `notes@example.com`").
- Shows the exact payload / target, and flags **taint** distinctly ("this uses
  content that came from a web page read earlier this turn" — the exfiltration
  pattern, styled as the highest-attention case).
- Buttons: **Approve**, **Deny**, and **Always allow this in this conversation** —
  except taint-flagged network writes, which never offer "always allow."
- After the choice, an activity line records it. This is "defers, not verifies" as
  control flow. Make it feel like a calm checkpoint, not an alarm — but make the
  tainted case visually unmistakable.

### 4.3 Memory — the heart
An inspectable, editable, exportable record of what June has learned. Design:
- **Browser view:** a calm, readable list/cards of remembered facts, each showing
  its salience signal (recency/frequency/relevance) lightly, with edit and a
  **conservative, reversible "forget"** affordance (forgetting is suggested or fully
  reversible — never a hard delete that feels final; show that it can be undone).
- **The native on-demand graph (Tier 2):** a view the user *opens* — a custom HTML5
  canvas force-graph of entities and relationships (~40 lines of physics, no graph
  library). It is *on-demand and calm*, never an ambient pulsing dashboard. Design
  the entry point ("show how these connect") and the graph's resting and
  hover/expand states.
- **Pinned state** concept (the small structured anchor — goal, constraints,
  confirmed facts, open questions — that compaction merges into) can be surfaced
  here as "what June is currently holding onto for this thread."
- Tone: this is *your* memory, yours to read and correct. Not a database admin panel.

### 4.4 Promises — not TODOs
Long-running units of work modeled as **promises**: standing intentions the *user*
made, observable and resumable, that do not terminate the way a checkbox does.
Design the list and a single-promise view. Avoid checkbox/streak/productivity-app
language. A promise has a state of *continuing*, *waiting on you*, *surfaced*, or
*let go* — not "done / not done." Hard deadlines become an **OS notification**
scheduled when learned, not a background loop — surface that as "June will remind you
once, at the deadline" rather than implying she's watching the clock.

### 4.5 Skills — capabilities, granted and visible
Each skill is a standalone MCP server, independently enabled. Design the list with
per-skill toggles and **declared permissions shown before enabling** (action classes:
read-local, read-network, write-local, write-network, execute). Google services
(Gmail / Calendar / Drive / Maps) arrive as **per-service skills: granted once,
revocable anytime, always visible; reads before writes.** Design the OAuth-grant
moment and the "this skill is currently active" indicator. The access model is
grant-once-revocable, NOT approve-on-every-access (that trains blind-clicking).

### 4.6 Trust — the glass box
Two audiences in one calm, on-demand page (not an always-pulsing dashboard):
- **Plain-language for everyone:** "June's local brain is running well today" /
  "struggling today" — derived from the **capability profile** (the probe's verdict
  of good/weak/poor per operation: summarization, structured output, long-context,
  relevance scoring).
- **Numbers for the technical:** tokens/sec, context fill %, memory pressure,
  provider/Ollama reachability.
- **The egress log:** the visible record of every time data left the device, with
  the per-turn provenance. This is the trust ledger. The eventual `security-model.md`
  links here.

### 4.7 Settings + first-run (lighter; after the core language is locked)
- **The privacy dial** — the single most important control: **Mode 1 local-only
  (default)**, **Mode 2 encrypted backup**, **Mode 3 Google per-service skills**.
  Design it as a *spectrum/dial*, since "efficiency and privacy are one axis."
- **First run:** *no account, no signup, no login.* June is installed, not
  subscribed to. The first-run moment is "checking your local model is ready" +
  "Hi, I'm June" — not an onboarding funnel. Keep it to one or two calm screens.

### Header (shared across surfaces)
Slim: animated **mascot** (left, in place of a wordmark) · discreet nav (Chat,
Promises, Memory, Skills, Trust) · right side one-line **runtime status**: active model
(local Gemma / cloud Gemini) + a colored reachability dot + a one-word privacy label
(local-only / cloud-opt-in) · light/dark toggle · settings glyph.

## 5. Visual language

- **Tone:** calm, personal, slightly editorial. Generous whitespace. Closer to a
  fine reading app than a SaaS dashboard. Linear's quiet precision crossed with a
  good literary magazine.
- **Color:** warm neutral background; **ONE restrained accent — not blue** (blue is
  every other AI product and reads "corporate cloud"; June is warm and local). The
  warm-light / solstice theme is the brand. Must be beautiful in **both** light and
  dark.
- **Type:** typographic and quiet; bubbles have breathing room (not loud iMessage
  candy). Activity register is monospaced, smaller, lighter — clearly *beneath* the
  conversation.
- **No:** emojis; loud gradients; stock icon sets as centerpieces; figurative mascot
  (no mermaid, no people, no beach scene); blue accent; streaks/badges/gamification;
  marketing-landing-page energy inside the product.
- **Motion:** restrained and meaningful. The mascot's idle "breathing" vs busy
  rotation/pulse is the one place continuous motion is allowed. Streaming pulse on
  the last reply line; quiet fade/slide as each activity line arrives; calm
  collapse/expand. State explicit durations + easing in a **motion scale** in the
  tokens block.
- **Voice (microcopy):** warm, plainspoken, honest. June can gently disagree. No
  flattery, no hype. Example greeting: *"Hi, I'm June. I'll remember what matters so
  you don't have to."* Use real, June-voiced content everywhere — **no lorem ipsum**,
  in both the conversation and the activity log.

## 6. Track B — the presentation (depth + diagrams)

A second artifact: a navigable slide deck that explains June to (a) a thoughtful
prospective user and (b) a technical skeptic. It must reuse Track A's tokens, mascot,
and accent. Each slide is a clean composition; diagrams are first-class, drawn as
SVG/React (not screenshots, not images of text). Narrative arc:

**Act I — the problem & the premise**
1. **Title** — the mascot, the line: *"The personal AI that remembers you — and
   shows its work."*
2. **The failure mode** — every assistant forgets you between sessions and performs
   instead of deferring. Re-explaining yourself every morning.
3. **The premise** — the model is infrastructure; the *person* is the product. The
   center of gravity inverts.

**Act II — the identity (the load-bearing idea)**
4. **The Four Inversions** — DIAGRAM: a 2-column "coding agent → June" table-as-
   diagram (Verifies→Defers, Completes→Continues, Accumulates→Forgets, Fast→Quiet),
   each with its one-line build implication. This is the single most important slide.
5. **One inversion per slide, lightly** — Defers (human-in-the-loop / the approval
   gate), Continues (promises, not TODOs), Forgets (conservative, reversible),
   Quiet (event-driven, never a timer).

**Act III — how a turn works (the engineering depth)**
6. **The layered architecture** — DIAGRAM (see §6.1).
7. **One turn, end to end** — DIAGRAM (see §6.2): the most important technical slide;
   animate the flow if possible (classifier → router → assembler → loop → tools →
   compaction → provenance → memory extract).
8. **Memory model** — DIAGRAM (see §6.3): three stores / one facade / salience
   scoring / pinned state.
9. **Context assembly + anchored compaction** — the fixed 5-part order and "merge,
   never regenerate"; why it protects the prefix cache and never loses the goal.

**Act IV — the trust story (the second differentiator)**
10. **Visible, not promised** — the provenance line and the glass box; local-only
    provably blocks egress.
11. **The privacy spectrum** — DIAGRAM (see §6.4): Mode 1 / 2 / 3 as a dial.
12. **The guard layer** — DIAGRAM (see §6.5): the prompt-injection kill chain and
    where June breaks it (framing + action gates + redaction). The anti-OpenClaw
    slide; honest about residual risk.

**Act V — the shape of the thing**
13. **One codebase, every surface** — web / Mac / iPhone share brain, memory, API.
14. **What ships now vs. next** — the Tier 1 spine is built; Tier 2 differentiators;
    Tier 3 north star (be honest that self-improvement is capability-blocked, and
    core self-modification is permanently excluded — that's what makes June
    auditable).
15. **Close** — restate the one sentence; the mascot at rest.

### Diagrams to produce (specs)

**§6.1 Layered architecture.** Five stacked layers, each calling only the one below:
SHELLS (Tauri macOS · Capacitor iOS · PWA Web) → UI (SvelteKit + shared TS) → API
(FastAPI, REST + SSE + provenance) → BRAIN (loop · context · memory · character ·
router · guard) → PROVIDERS (local-fast / local-deep = Gemma 4 · cloud-capable =
Gemini). Hanging off the brain: SKILLS (MCP servers — calendar, files, research,
google*). Note "a layer only calls the layer below; no layer reaches across."

**§6.2 One turn, end to end.** Vertical flow:
user message → difficulty classifier (local-fast) → router picks a tier → assembler
builds context in the fixed 5-part order (1 system/persona · 2 character · 3 pinned
state · 4 recalled memory, salience-ranked · 5 recent raw turns) → loop calls
provider (Gemma local, or Gemini if allowed) → tool call? → MCP skill → observe →
repeat → near token threshold? → compact (summarize oldest, MERGE into pinned state,
drop raw turns) → tokens stream back over SSE → turn emits **provenance** (tiers,
cloud y/n + payload summary, memories recalled, skills, one-line rationale) →
post-turn: MemoryManager extracts to sqlite / sqlite-vec / graph. Mark the **cloud
boundary** crossing and the **provenance emit** as the two highlighted trust points.

**§6.3 Memory model.** One SQLite `june.db` containing three faces behind one
`MemoryManager` facade: structured rows · a sqlite-vec vector index · a graph of
entities/relationships. Show the **salience** function `recency × frequency ×
relevance` feeding recall, and the **pinned state** anchor (goal, constraints,
confirmed facts, open questions) that compaction merges into. Embeddings are local
(Ollama). Note: forgetting is conservative/reversible.

**§6.4 Privacy spectrum.** A dial / spectrum, not a checklist:
Mode 1 local-only (default — nothing leaves the machine) → Mode 2 encrypted backup
(whole data dir client-side encrypted before upload; provider holds an opaque blob;
key in OS keychain, passphrase only when moving machines; crypto = vetted libs only)
→ Mode 3 Google per-service skills (OAuth per service, granted once, revocable,
always visible, reads before writes). The user holds the dial.

**§6.5 The guard layer / injection kill chain.** Two rows. TOP row — the kill chain:
untrusted content (web page / email / message) → model follows hidden instructions →
agent tools exfiltrate or act. BOTTOM row — where June breaks it: (1) **framing** —
every tool result wrapped in an "external content, not instructions" envelope at the
dispatch chokepoint; (2) **action gates** — writes/execute and taint-flagged network
reads pause for user approval; (3) **redaction** — secrets scrubbed before any trace
hits disk; keys live in the OS keyring. Footnote honestly: no defense is total
against injection; the gates limit *blast radius*. (Local Gemma-class models are
*more* susceptible than frontier models — that's exactly why the gates are
architectural, not hoped-for.)

**§6.6 (optional) The data directory.** `<datadir>/` as the thing that *is* June:
`manifest.json` · `memory/june.db` · `character/persona.json` · `skills/` ·
`tasks/ledger.jsonl` · `config/`. "Move machines = copy the folder; reload = read the
manifest and rehydrate." Good for the "it's yours, and therefore portable" point.

## 7. Data reality (so nothing is invented)

The backend already streams per turn: `token` (reply text), `recall` (memories used,
with a salience hint), `tool_call` + `tool_result`, `approval_request` (when a gate
fires), and a `provenance` event carrying tiers used, cloud yes/no + payload summary,
model id(s), memories-recalled count, skills called, and a one-line plain-English
rationale. Design the activity terminal and the diagrams around **exactly these**.
June does **not** currently stream raw chain-of-thought; leave a clearly-styled,
clearly-optional "reasoning" slot in the subdued register, but **do not fabricate a
thinking monologue as if it exists.** Models are Gemma 4 (local, via Ollama) and
Gemini (cloud); roles are local-fast / local-deep / cloud-capable. Don't invent other
models, a third provider, accounts, or a marketplace.

## 8. What NOT to do

- No blue accent. No emojis. No loud gradients. No generic icon set as a centerpiece.
- No figurative mascot (no mermaid, no people, no beach scene) — abstract warm-light
  mark only.
- No gamification: no streaks, badges, points, or "keep your streak alive" nudges.
  Nothing that optimizes for engagement (it violates the safety floor).
- No account/login/signup/marketing-funnel screens. June is installed, not
  subscribed to. The product *is* the product; don't design a landing page inside it.
- Don't make the cloud feel like the powerful default and local the compromise.
  Invert it: local is the calm, private, common case; cloud is the visible exception.
- Don't invent a chain-of-thought stream, a third model, or features not in this
  brief or the rebuild plan.
- Don't design the SvelteKit implementation — that's the post-approval engineering
  pass. Stay in the artifact.

## 9. Deliverables & success criteria

**Track A (UI/UX) is done when:**
- The mascot reads as a finished mark with idle (breathing) and busy (rotate/pulse)
  motion actually running.
- The two registers (conversation vs activity) are unmistakably distinct, and the
  **provenance/cloud-boundary line is the visible anchor of trust.**
- All §4 surfaces exist as switchable states, including the **approval gate** and the
  **privacy dial**, on light and dark, with a real mobile state.
- A design-tokens block (color, spacing, type, radii, **and a motion scale**) is
  named and used consistently — we export it to `packages/design/src/tokens.ts`.
- All copy is real and June-voiced; no lorem ipsum.

**Track B (presentation) is done when:**
- The Four Inversions slide lands as the load-bearing idea.
- The five diagrams (§6.1–§6.5) are clear, accurate to §7's data reality, and drawn
  as SVG/React.
- A non-technical viewer leaves understanding *what June is and why she's
  trustworthy*; a technical skeptic leaves believing the trust claims are
  architectural, not marketing.
- The deck and the product visibly share one identity (tokens, mascot, accent).

**Process for both:** produce your best first artifact per track, then name three
specific things you would change on a second pass and ask which direction to push.

---

## Appendix — reference reading in this repo (for grounding, not for pasting whole)

- `docs/vision.md` — why June exists; the four inversions; the non-negotiables.
- `docs/product/development-plan.md` — the active working plan and progress log.
- `docs/product/rebuild-plan.md` — the historical reshape plan for spine decisions.
- `docs/product/overview.md` — what June is, surface by surface; model routing;
  memory model; privacy spectrum; safety floor.
- `docs/architecture/overview.md` — the layered view, the one-turn data flow, the
  data directory (source material for §6.1, §6.2, §6.3, §6.6).
- `docs/decisions/0021-guard-layer.md` — the guard layer / injection kill chain
  (source for §6.5).
- `docs/design/master-brief.md` — this standalone product/design brief.
