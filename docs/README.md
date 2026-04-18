# June Documentation

This directory is the documentation root. Everything you need to understand, contribute to, or operate June lives here.

## Start Here

- [**Vision**](vision.md) — the product premise and the three non-negotiables
- [**Architecture overview**](architecture/overview.md) — how the system is layered
- [**Architecture decisions**](decisions/README.md) — the seven ADRs that justify the design
- [**8-week plan**](product/plan.md) — the canonical development plan
- [**Environment**](setup/environment.md) — runtime configuration reference

## Structure

```
docs/
├── vision.md                      # product north star
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
├── product/
│   └── plan.md                    # canonical 8-week plan
├── setup/
│   └── environment.md             # env vars, .env template
└── README.md                      # this file
```

## Documentation Rules

1. The vision document is the tiebreaker. When documentation conflicts, the vision wins and the other document is updated.
2. Every architectural decision gets an ADR. If it's worth debating, it's worth recording.
3. The 8-week plan is the single source of truth for what we're building and when. Older planning documents are deleted, not kept as compatibility shims.
4. No emojis in documentation.
5. Complete sentences. Two-page maximum per document unless the content genuinely requires more.

## How the Docs Evolve

- The vision evolves rarely. Material changes require deliberate discussion.
- ADRs are append-only. Superseding an ADR means writing a new one that references and deprecates the old one.
- The 8-week plan is updated each week with the current state and lessons learned.
- The environment reference is updated whenever runtime configuration changes.
