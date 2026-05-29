# Designing June With Claude

## What This Is For

The fastest way to design June's UI is to iterate visually with Claude and its
Artifacts renderer. You paste the brief below into [claude.ai](https://claude.ai),
Claude produces a live React + Tailwind artifact in the side panel, you tell it
what to change, it produces a new version. Two or three rounds and you have a
coherent visual identity to hand to the engineering pass (SvelteKit).

A note on terminology: Anthropic does not ship a product named "Claude Design".
This document is written for Claude on claude.ai with Artifacts enabled, using the
latest Opus or Sonnet model. Substitute any chat-with-rendered-UI tool you prefer;
the prompt is written to work in any such context.

This brief defines a **redesign** (the "two-register" interface, 2026-05-29). It
supersedes the earlier single-column prose-chat brief. Where it conflicts with the
shipped UI, this brief is the target.

## How To Use This

1. Open a new conversation on claude.ai with a capable model (Opus 4.x / Sonnet 4.x).
2. Paste the entire prompt block below as the first message.
3. Review the artifact. Ask for the mascot variants first (see Deliverables), pick one, then iterate the screens.
4. Ask Claude to produce each named screen/state in the chosen direction.
5. Export or screenshot. Hand off to the engineering pass. Design tokens live at `packages/design/src/tokens.ts` (and compile to `packages/design/tokens.css`).

Sequencing note (read before building anything): lock the **visual direction and
the mascot** in the artifact first. Do not start the SvelteKit implementation
until the artifact reads as a finished product. The redesign touches the core
chat surface (`apps/web/src/routes/+page.svelte`), the shared components
(`packages/ui/src/components/{Composer,ChatBubble,MessageList}.svelte`), the
header (`apps/web/src/routes/+layout.svelte`), and a new activity-stream
component — that engineering pass is a separate, post-approval task.

## The Big Idea — Two Registers, One Screen

June does two different things at once, and today's UI conflates them: the words
she actually *says* and the work she *does* to say them share one column, so tool
calls and machinery clutter the conversation. The redesign splits them into two
visually distinct registers on one screen:

- **Conversation (foreground).** Only the actual exchanged messages — the user's
  messages and June's replies — rendered as **bubbles**. This is what a person
  said and what June said. Nothing else lives here.
- **Activity (background).** Everything June *did* to produce a reply — memory
  recall, tool calls and their results, the tier/route decision (local vs cloud),
  the per-turn provenance and cloud-boundary line, timing, and (when available)
  her reasoning. Rendered **subdued**: smaller, lighter gray, monospaced, clearly
  *beneath* the conversation. Never mixed into the bubbles.

The conversation reads like a calm chat. The activity reads like a quiet,
honest "flight recorder" you can glance at or ignore — June's transparency made
visible without shouting over the conversation.

## Layout — Composer At The Center

The composer (the user input) is the fulcrum of the screen, vertically centered,
not pinned to the bottom.

```
┌───────────────────────────────────────────────┐
│  HEADER:  [mascot]  Tasks Memory Skills System   runtime·privacy  ☼/☾  ⚙ │
├───────────────────────────────────────────────┤
│                                                 │
│   CONVERSATION (foreground bubbles)             │  ← scrolls; newest just
│     June:  ▢▢▢▢                                 │    above the composer.
│                       user: ▢▢  │              │    ~top half of the stage.
│     June:  ▢▢▢▢▢ (streaming, last line pulses)  │
│                                                 │
├──[‹ activity]──┬──────────────────────┬─────────┤  ← COMPOSER band, centered.
│  toggle button │  type to June…    [↵] │  stop  │    Button at its LEFT toggles
├────────────────┴──────────────────────┴─────────┤    the activity terminal.
│  ACTIVITY TERMINAL (subdued, real-time, scroll)  │
│   12:04:01  recall · 3 memories (salience)       │  ← collapsed by default to a
│   12:04:01  route · standard → local-fast        │    slim 1-line strip; expands
│   12:04:02  tool  · list_tasks {}                 │    to fill the lower area.
│   12:04:02         → 4 tasks                      │
│   12:04:05  cloud boundary · local · gemma4 · 0 ↑ │    Provenance/cloud line is
│   12:04:05  done · 1.2s · 320 tok                 │    the anchor of trust.
└───────────────────────────────────────────────┘
```

- Above the composer: the conversation, last few exchanged bubbles visible (about
  the top half of the stage), scrolling up into history.
- The composer sits in the center as a horizontal band. A **toggle button on its
  left** collapses or expands the activity terminal.
- Below the composer: the activity terminal. Collapsed = a slim one-line strip
  showing only the latest step (or nothing when idle). Expanded = a scrollable
  panel that fills the lower stage and streams steps in real time as June works.
- When the terminal is collapsed, the conversation gets the room. When expanded,
  the composer stays centered with conversation above and machinery below.

Validate the ergonomics in the artifact: where a new reply appears (it grows in
the conversation, just above the composer), and that the eye's reading order is
calm. If centering the composer hurts readability of long histories, propose the
adjustment in your notes rather than silently reverting to a bottom composer.

## The Mascot

Replace the "June" wordmark in the header with an **abstract animated mascot**.

Direction (design around this; explore variants, then we pick one): the **June
sun / solstice mark**. June is the month of the summer solstice — the longest day,
peak light, warmth. The mark is a minimal warm disc with a soft corona of short
rays. It is a *mark, not a character*: abstract, non-gendered, timeless, and it
scales from header to favicon.

Motion (the mascot doubles as the global busy indicator):
- **Idle:** still, or a very slow "breathing" glow (a 4–6s ease in/out on the corona opacity/scale).
- **Thinking / streaming / running a task:** the rays rotate slowly and the corona pulses — light shimmering, calm not frantic. This is the only place motion is allowed to be continuous.
- **Returned to rest:** settle back to idle over ~400ms.

Deliver it as inline SVG with CSS (or a tiny Lottie-style keyframe set) so the
motion is visible in the artifact. Explore 2–3 abstract variants on the
sun/solstice/warm-light theme (e.g. a solid disc + corona; a thin-ring sun; a
single warm "first-light" orb / horizon glow). Do **not** produce a literal
figurative character (no mermaid, no people, no beach scene) — keep it abstract
and brand-appropriate for a privacy- and trust-first assistant.

## The Prompt

Copy everything inside the fence. Paste into claude.ai.

```
You are designing the visual identity and UI for June, an open-source personal AI
that remembers you. Produce a coherent visual system as a SINGLE React + Tailwind
artifact that shows several linked screens/states, switchable in-artifact via a top
tab strip. Include a light/dark toggle and define design tokens at the top.

PRODUCT, IN ONE PARAGRAPH
June is the personal AI that remembers you. She runs locally on your laptop via
Gemma 4, optionally reaches the cloud via Gemini, and works identically in the
browser, on Mac, and on iPhone. Her center of gravity is the user, not the task:
she remembers what matters, forgets what doesn't, tells the truth plainly, knows
when to stay quiet, and never does anything the user can't see. Memory and visible
honesty are the product. She feels like a calm, competent companion with a
persistent identity — not a chatbot, not a productivity app, not a developer tool.

TARGET USER
A thoughtful individual who already uses AI daily but is tired of re-explaining
themselves every morning. They value privacy, install software from GitHub, and
notice when software feels rushed. They are choosing a companion, not shopping for
features.

VISUAL TONE
Calm, personal, slightly editorial. Generous whitespace. Closer to a fine reading
app than a SaaS dashboard. Warm neutral background, ONE restrained accent color
(not blue). Must work beautifully in both light and dark. No emojis. No loud
gradients. No stock icon sets as centerpieces. Think Linear's quiet precision
crossed with a good literary magazine.

THE CORE INTERACTION MODEL — TWO REGISTERS, ONE SCREEN
The screen separates what June SAYS from what June DOES:
- CONVERSATION (foreground): only the actual exchanged messages — the user's
  messages and June's replies — as bubbles. June's bubbles left-aligned, the
  user's right-aligned. Bubbles are quiet and typographic, with breathing room
  (not loud iMessage candy). A streaming reply shows a subtle pulse on its last line.
- ACTIVITY (background): everything June did to produce the reply — memory recall,
  tool calls and results, the route/tier decision (local vs cloud), the per-turn
  provenance + cloud-boundary line, timing. Rendered SUBDUED: smaller, lighter gray,
  monospaced, clearly beneath the conversation. Never inside the bubbles. It reads
  like a calm flight recorder — June's transparency made visible without shouting.

LAYOUT — COMPOSER AT THE CENTER
The composer (user input) is vertically centered, not pinned to the bottom.
- Above the composer: the conversation, last few bubbles visible (~top half),
  scrolling up into history; newest reply grows just above the composer.
- The composer is a centered horizontal band: a text input, a send button, a
  cancel/stop button that appears while streaming, and a visible Cmd+Enter hint.
- A TOGGLE BUTTON on the LEFT of the composer collapses/expands an ACTIVITY
  TERMINAL below it. Collapsed = a slim one-line strip showing the latest step (or
  empty when idle). Expanded = a scrollable panel filling the lower stage that
  streams steps in real time.
- Header (slim, top): an abstract animated MASCOT on the left in place of a
  wordmark; a discreet nav (Tasks, Memory, Skills, System); on the right a one-line
  runtime status (active model — local Gemma or cloud Gemini — a colored dot for
  reachability, and a one-word privacy label: local-only or cloud-opt-in), plus a
  light/dark toggle and a settings glyph.

THE MASCOT
Replace the wordmark with an abstract "June sun / solstice" mark: a minimal warm
disc with a soft corona of short rays. A mark, not a character — abstract,
non-gendered, scales to a favicon. Motion: idle = still or a slow ~5s breathing
glow; thinking/streaming/running = rays rotate slowly and the corona pulses (calm,
not frantic); it doubles as the global busy indicator and settles back to idle in
~400ms. Deliver as inline SVG + CSS so the motion is visible. Explore 2–3 abstract
variants on the warm-light theme. NO figurative character (no mermaid, no people,
no beach scene).

DESIGN THESE SCREENS/STATES IN ONE ARTIFACT (tab strip to switch)
1. Mascot board — the 2–3 abstract mascot variants, each shown idle AND animated,
   at header size and favicon size, on light and dark. This is the first thing to get right.
2. Chat — idle / greeting. Composer centered, activity terminal collapsed to its
   slim strip, an inviting empty state above ("Hi, I'm June. I'll remember what
   matters so you don't have to.").
3. Chat — active, terminal collapsed. Two or three exchanged bubbles above the
   composer (June left, user right), June's latest reply mid-stream with the
   last-line pulse. The collapsed strip shows the single latest activity step.
4. Chat — active, terminal EXPANDED. Same conversation above; below the composer,
   the activity terminal scrolling a real-time log: a recall line ("recall · 3
   memories"), a route line ("route · standard → local-fast"), a tool call and its
   result, the cloud-boundary/provenance line ("local · gemma4:e2b · 0 sent to
   cloud · 1.2s · 320 tok"), and a final "done" line. Make the provenance/
   cloud-boundary line the visual anchor of trust.
5. Mobile — the chat in both collapsed and expanded states. Must feel native, not
   a shrunken desktop. Decide how the centered composer + toggle behave on a phone.

MOTION SPEC (state it explicitly in prose under the artifact, with durations/easing)
- Streaming pulse on the last reply line.
- Activity-stream line entry (a quiet fade/slide as each step arrives).
- Terminal collapse/expand transition.
- Mascot idle breathing vs busy rotation/pulse, and the settle-to-rest.

DELIVERABLES
- One artifact: React + Tailwind, tab strip across the screens/states above, a
  light/dark toggle in the artifact's own corner.
- The mascot as inline SVG + CSS with its idle and busy animations actually running.
- A design-tokens block at the top (color scale, spacing scale, type scale, radii,
  AND a small motion scale: durations + easing). Use them consistently; we export
  to packages/design/src/tokens.ts.
- Real, June-voiced content — both the conversation AND the activity-log lines.
  No lorem ipsum.

DATA REALITY (so the activity terminal is honest, not invented)
June's backend already streams these per turn: token (the reply text), recall
(memories used, with a salience hint), tool_call + tool_result, and a provenance
event carrying tier(s) used, cloud yes/no, model id(s), memories-recalled count,
and a one-line plain-English rationale. Design the activity terminal around
exactly these. June does NOT currently stream raw chain-of-thought "thinking"
tokens; leave a clearly-styled slot for an optional future "reasoning" line in the
same subdued register, but do not fabricate a thinking monologue as if it exists.

WHAT NOT TO DO
- No settings page, setup flow, or onboarding screen — those come after the core
  visual language is locked.
- No marketing landing page. This is the product.
- No features not described here. No voice button, no attach button, no plus menu.
- No blue accent. No emojis. No generic icon set as a centerpiece. No figurative
  mascot.

PROCESS
Produce one artifact now with your best first answer, leading with the mascot
board. Then, in prose below the artifact, name three specific things you would
change on a second pass and ask me which direction to push.
```

## After The First Round

- **"Lock the mascot, variant N."** Commit to one mark; drop the others and apply it across the header in every screen.
- **"Tighten it."** Reduce decoration, increase typographic restraint, push closer to Linear.
- **"Warm it up."** More personality in type and microcopy, closer to a reading app.
- **"Stress the activity terminal."** Show a turn that escalates to the cloud (a Gemini route), so the cloud-boundary line is unmissable and the local-vs-cloud distinction is obvious at a glance.
- **"Show the dense history case."** A long conversation, to check the centered composer holds up when there is a lot of scrollback.

Hand off when: the mascot reads as a finished mark with its idle/busy motion;
the two registers are unmistakably distinct; the centered-composer + collapsible-
terminal layout is ergonomically sound on desktop and phone; and the design tokens
(including the motion scale) are named and consistent.
