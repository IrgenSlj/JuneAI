# June

A private, EU-first AI assistant that lives on your machine.

June's first product is a local vault for every AI conversation you have:
ChatGPT, Claude and others, imported from their official exports into plain
Markdown files in a folder you choose, and drawn as one Map that shows what was
said, what was remembered and what was planned. The files stay readable without
June. Nothing leaves your machine unless you see where it is going first.

Status: early rebuild. Nothing to install yet. See [PLAN.md](PLAN.md) for the
plan and [lab/](lab/README.md) for the experiments behind it. The previous
version of June is on the `legacy` branch.

## Develop

```
pnpm install
pnpm test:watch   # inner loop
pnpm check        # typecheck + tests, the gate
```

Requires Node 22 or newer and pnpm 10.

## License

MIT
