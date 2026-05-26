# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## What's New

- **v0.1.1 personal operating layer** — The active development track is quick
  capture, Daily Home, durable events, action approvals, promises, and agenda
  suggestions. See [ADR 0014](decisions/0014-personal-operating-layer.md) and
  the [v0.1.1 scheduled development plan](plans/v0.1.1-scheduled-development.md).
- **Research memo** — The technical and product research behind the plan lives
  at [`product/personal-operating-layer-research.md`](product/personal-operating-layer-research.md).
- **Desktop shell builds** — The Tauri shell has produced a v0.1.0 Apple Silicon
  DMG. The current artifact is ad-hoc signed and not notarized; signed
  distribution is deferred until external users justify the cost.
- **Open-source readiness** — The hardening plan remains useful backlog, but it
  is no longer the active roadmap. Keep its correctness checks in mind when
  touching providers, setup, memory, or local API safety.
- **Light mode default** — June defaults to light theme. Click the moon icon in the header to switch to dark mode.
- **Black J branding** — Clean black "J" on transparent for favicon and PWA icons.

- [**Vision**](vision.md) — the product premise and the three non-negotiables
- [**Product overview**](product/overview.md) — what June is
- [**v0.1.1 scheduled development plan**](plans/v0.1.1-scheduled-development.md) — the active implementation schedule
- [**Personal operating layer research**](product/personal-operating-layer-research.md) — product/technical research memo
- [**Open-source readiness plan**](product/open-source-readiness-plan.md) — historical hardening plan and backlog
- [**Roadmap**](product/roadmap.md) — what ships next and when the next surface is worth planning
- [**Desktop shell plan**](product/desktop-shell-plan.md) — status and remaining distribution work for the Tauri desktop shell
- [**Responsive and touch plan**](product/responsive-plan.md) — how the UI works on every screen size and input method
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the ADRs that justify the design
- [**Environment**](setup/environment.md) — runtime configuration reference
- [**Desktop setup**](setup/desktop.md) — Rust toolchain, run/build commands for the desktop shell
- [**Design brief**](design/claude-design-prompt.md) — the prompt for iterating on June's UI with Claude

## Structure

```
docs/
├── vision.md                      # product north star
├── product/
│   ├── overview.md                # what June is
│   ├── personal-operating-layer-research.md # product/technical research memo
│   ├── open-source-readiness-plan.md # historical hardening plan and backlog
│   ├── roadmap.md                 # scaling map, trigger-gated
│   ├── desktop-shell-plan.md      # historical/current status plan for the Tauri desktop shell
│   └── responsive-plan.md         # touch and tablet hardening shipped alongside the shell
├── architecture/
│   └── overview.md                # layered system architecture
├── decisions/                     # Architecture Decision Records
│   ├── README.md                  # ADR index
│   ├── 0001-monorepo-structure.md
│   ├── 0002-gemma-gemini-only.md
│   ├── 0003-streamlit-to-sveltekit.md
│   ├── 0004-memory-architecture.md
│   ├── 0005-skills-as-mcp.md
│   ├── 0006-desktop-and-mobile-shells.md
│   ├── 0007-sse-over-websockets.md
│   ├── 0008-ollama-supervision.md
│   └── 0014-personal-operating-layer.md
├── design/
│   └── claude-design-prompt.md    # UI design brief
├── setup/
│   ├── environment.md             # env vars, .env template
│   └── desktop.md                 # Rust toolchain, dev/build commands for the desktop shell
└── README.md                      # this file
```

## Documentation Rules

1. The vision document is the tiebreaker. When documentation conflicts, the vision wins and the other document is updated.
2. Every architectural decision gets an ADR. If it is worth debating, it is worth recording.
3. The product overview describes what June is. The roadmap describes what ships next. The v0.1.1 plan is the scheduled development document for the active release.
4. No emojis in documentation.
5. Complete sentences. Two-page maximum per document unless the content genuinely requires more.

## How the Docs Evolve

- The vision evolves rarely. Material changes require deliberate discussion.
- ADRs are append-only. Superseding an ADR means writing a new one that references and deprecates the old one.
- The product overview is updated when the shipped surface area changes.
- The roadmap is updated when a trigger fires, a feature ships, or priorities shift.
- The environment reference is updated whenever runtime configuration changes.
