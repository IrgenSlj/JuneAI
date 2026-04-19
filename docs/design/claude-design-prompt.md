# Designing June With Claude

## What This Is For

The fastest way to design June's UI is to iterate visually with Claude and its Artifacts renderer. You paste the brief below into [claude.ai](https://claude.ai), Claude produces a live React + Tailwind artifact in the side panel, you tell it what to change, it produces a new version. Two or three rounds and you have a coherent visual identity to hand to the engineering pass.

A note on terminology: Anthropic does not ship a product named "Claude Design". The workflow this document is written for is Claude on claude.ai with Artifacts enabled, using the latest Sonnet or Opus model. If you have a specific tool in mind by that name, substitute it; the prompt below is written so it works in any chat-with-rendered-UI context.

## How To Use This

1. Open a new conversation on claude.ai with a capable model (Claude Opus 4.x or Sonnet 4.x).
2. Paste the entire prompt block below as the first message.
3. Review the artifact. Ask for three variants, one tight, one warm, one editorial.
4. Pick the direction. Ask Claude to produce each named screen in that direction.
5. Export or screenshot. Hand off to the engineering pass; design tokens live at `packages/design/src/tokens.ts`.

## The Prompt

Copy everything inside the fence. Paste into claude.ai. Do not edit unless you have a reason — the prompt is tuned to produce a single, consistent artifact rather than a wall of text.

```
You are designing the visual identity and UI for June, an open-source personal AI that remembers you. I want a coherent visual system expressed as a single React + Tailwind artifact that shows three linked screens. Render the artifact with the screens switchable in-artifact via a top tab strip.

PRODUCT, IN ONE PARAGRAPH
June is the personal AI that remembers you. It runs locally on your laptop via Gemma 4, optionally reaches the cloud via Gemini, and works identically in the browser, on Mac, and on iPhone. Every conversation feeds a personal knowledge graph that the user owns. Memory is the product. Privacy is non-negotiable. The product feels like a calm, competent assistant with persistent identity — not a chatbot, not a productivity app, not a developer tool.

TARGET USER
A thoughtful individual who already uses AI daily but is tired of explaining themselves from scratch every morning. They value privacy, they install software from GitHub, and they notice when software feels rushed. They are not shopping for features. They are choosing a companion.

VISUAL TONE
Calm, personal, slightly editorial. Generous whitespace. A typographic feel closer to a reading app than a SaaS dashboard. Warm neutral background, one restrained accent color. Not dark-mode-first, but must work beautifully in both light and dark. No emojis. No gradients that scream. No stock-looking icons. Think Linear's quiet precision crossed with a good literary magazine, not Notion and not ChatGPT.

LAYOUT CONSTRAINT
Single-column, chat-first. 860px maximum content width on desktop. The chat is always the primary surface; memory and skills are secondary views reached from a discreet nav. No sidebars that eat the conversation. No modals that break the reading flow. Mobile must feel native, not a shrunken desktop.

DESIGN THE FOLLOWING THREE SCREENS IN ONE ARTIFACT

1. Chat (primary)
   - Centered message list with the user's messages right-aligned, June's left-aligned.
   - Messages are prose, not bubbles. Give them breathing room.
   - A streaming response shows a subtle pulse on the last line.
   - Tool calls render inline between messages as small, monospaced inline blocks (name and arguments on one line, result on a second line). Treat them as marginalia, not main content.
   - A fixed composer at the bottom with a send button, a cancel button that appears while streaming, and a visible shortcut hint (Cmd+Enter).
   - A slim top bar with: "June" wordmark on the left; a one-line status on the right showing active model (local Gemma or cloud Gemini), a colored dot for reachability, and a one-word privacy label (local-only or cloud-opt-in).
   - Include two example exchanges so the typography and rhythm are visible.

2. Memory
   - Three sections stacked vertically: Facts (structured rows), Semantic memories (prose cards), People & places (chips with relationship hints).
   - Each memory shows its source, the date it was learned, and a delete affordance that appears on hover.
   - A single search input at the top that filters across all three.
   - An empty state for each section that teaches the user what will eventually live there.
   - The screen must feel like browsing a well-kept notebook, not a CRM.

3. Skills
   - A list of skill cards. Each card shows the skill's name, one sentence of description, a status badge (running, stopped, crashed, disabled), an enable/disable toggle, and a collapsible list of the tools that skill exposes.
   - At least five skills visible: Calendar, Health, Research, Files, Daily.
   - The card for a crashed skill should show a restrained error state, never a red-alert banner.
   - A small summary at the top: "4 of 5 skills enabled".

DELIVERABLES
- One artifact, React + Tailwind, tab strip to switch between Chat, Memory, Skills.
- Include a dark mode toggle in the artifact's own corner so I can see both.
- Define design tokens at the top of the artifact (color scale, spacing scale, type scale, radii). Use them consistently. I will export these to `packages/design/src/tokens.ts`.
- Use real-sounding content, not lorem ipsum. The product's voice is reflected through what is on screen.

WHAT NOT TO DO
- Do not design a settings page, a setup flow, or an onboarding screen. Those come after the core visual language is locked.
- Do not design a marketing landing page. This is the product.
- Do not invent features that are not in the brief above. No voice button, no attach button, no plus menu. If it is not described, it is not there.
- Do not use a blue accent. Pick something more considered.
- Do not use generic icon sets (Heroicons, Lucide) as visual centerpieces. A logomark and any required glyphs should feel bespoke, even if simple.

PROCESS
Produce one artifact now with your best first answer. Then, in prose below the artifact, tell me three specific things you would change on a second pass, and ask me which direction to push.
```

## After The First Round

When you have an artifact you like, follow up with one of these:

- **"Tighten it."** Asks Claude to reduce decoration, increase typographic restraint, push closer to Linear's feel.
- **"Warm it up."** Asks Claude to add more personality in the typography and the microcopy, push closer to a reading-app feel.
- **"Show the same three screens on mobile."** Produces the responsive answer in the same artifact.
- **"Produce the first-run setup flow in the same language."** Extends the system to the onboarding surface once the core is locked.

Hand off when the artifact reads as a finished product at a glance, the design tokens are named and consistent, and the three screens feel like the same app rather than three separate designs.
