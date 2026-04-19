# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## Start Here

- [**Vision**](vision.md) — the product premise and the three non-negotiables
- [**Product overview**](product/overview.md) — what June is
- [**Roadmap**](product/roadmap.md) — what ships next and when the next surface is worth planning
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the ADRs that justify the design
- [**Environment**](setup/environment.md) — runtime configuration reference
- [**Design brief**](design/claude-design-prompt.md) — the prompt for iterating on June's UI with Claude

## Structure

```
docs/
├── vision.md                      # product north star
├── product/
│   ├── overview.md                # what June is
│   └── roadmap.md                 # scaling map + remaining prototype work
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
│   └── 0007-sse-over-websockets.md
├── design/
│   └── claude-design-prompt.md    # UI design brief
├── setup/
│   └── environment.md             # env vars, .env template
└── README.md                      # this file
```

## Documentation Rules

1. The vision document is the tiebreaker. When documentation conflicts, the vision wins and the other document is updated.
2. Every architectural decision gets an ADR. If it is worth debating, it is worth recording.
3. The product overview describes what June is. The roadmap describes what ships next. Neither doc records week numbers; surfaces are planned by trigger, not schedule.
4. No emojis in documentation.
5. Complete sentences. Two-page maximum per document unless the content genuinely requires more.

## How the Docs Evolve

- The vision evolves rarely. Material changes require deliberate discussion.
- ADRs are append-only. Superseding an ADR means writing a new one that references and deprecates the old one.
- The product overview is updated when the shipped surface area changes.
- The roadmap is updated when a trigger fires, a feature ships, or priorities shift.
- The environment reference is updated whenever runtime configuration changes.
