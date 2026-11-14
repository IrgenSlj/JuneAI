# June

Your AI, your files. One app for every model, where you can see and keep
everything.

June is a Mac app (and the same interface in your browser) for talking to any
AI model you bring: an API key, a model running on your own machine, a free
gateway, or a subscription you already pay for, through official routes only.
Every conversation is saved as a plain Markdown file in a folder you choose,
alongside the history you import from ChatGPT and Claude. Before anything
leaves your Mac, June can show you exactly what is going where, and it keeps a
tamper-evident record of every send. June is a wrapper and says so: it hosts no
models and has no accounts.

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
