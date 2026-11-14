# June

**Your AI, your files. One app for every model, where you can see and keep everything.**

June is a private AI app for the Mac. You bring the model: an API key you
already have, a model running on your own machine, or a free gateway. June
brings everything around it: the conversation, a memory you can read, and an
honest account of what left your computer and where it went.

Every conversation is a plain Markdown file in a folder you choose. Close June,
open the folder, and it is all still there, readable in any text editor and in
Obsidian. Nothing is locked in an app, an account or someone else's server.

> **Status: early development.** The foundation runs (the local engine, the
> Mac window, choosing a vault), but you cannot chat with a model yet. That
> arrives in the next milestone. Watch or star the repo to follow along; there
> is nothing to install for everyday use yet.

## Why June exists

Most people now use more than one AI assistant, and each keeps its own copy of
your history behind its own login. What you told one, the others never know.
What any of them keeps, you cannot easily see.

June is being built to take the other side of that trade:

- **Your files.** Conversations, notes and plans are Markdown in a folder on
  your Mac. The folder is the product; the app is a way to use it.
- **Any model.** OpenAI, Anthropic, Mistral, OpenRouter, or models running on
  your own machine through Ollama or LM Studio. Switch per conversation, or
  continue an old thread on a different model.
- **See what leaves.** Before a message is sent, June knows exactly which
  model, which company, which country and which parts of your context are in
  it. You choose how much of that to see, from a quiet chip to the full detail.
- **Proof, not promises.** Every send, memory read and memory write leaves a
  receipt in a tamper-evident log on your Mac. One edited or deleted line breaks
  the chain, and `verify` tells you where.
- **Memory you decide.** June never claims to have remembered something unless
  a file was actually written. You can read, edit or delete every note.

June is a wrapper and says so. It hosts no models, has no accounts, sends no
telemetry and has no server of its own.

## What it will do

| | |
| --- | --- |
| **Chat with any model** | Your keys, local models, or a free gateway. Streams like any chat app, saved as you go. |
| **Import your history** | Bring in your ChatGPT and Claude exports. Original dates, titles and projects are kept, and re-importing never duplicates. |
| **Continue anywhere** | Pick up any conversation, imported or not, on any model you have connected. |
| **Transparency levels** | Quiet, Glance or Full: from nothing on screen, to a one-line chip, to the exact context being sent, which you can edit before it goes. |
| **Notes and trace** | Save a fact or a decision as a note, then see where it came from and which answers it shaped. |
| **The Map** | One timeline of every conversation, note and plan, across every assistant you have used. |

The order, and what "done" means for each step, is in [PLAN.md](PLAN.md).

## How it works

```
 June.app window  ─┐
                   ├──  http://127.0.0.1:<random port>  ──  june (local engine)
 or a browser tab ─┘        one origin, per-launch token        │
                                                                ├── your vault folder (Markdown, receipts)
                                                                └── the model you chose
```

- A small local engine (`june serve`, Node) owns your vault and talks to model
  providers. It listens only on your own machine, on a random port, and every
  request must carry a token created fresh at each launch.
- The interface is a web app served by that engine. The Mac app is a thin
  window around it, built with [Tauri](https://tauri.app); the same interface
  also runs in your browser.
- The core (file formats, receipts, ids) is plain TypeScript with no access to
  the network or the operating system. Tests enforce that boundary.

## Principles

- **Files over app.** The vault must stay readable without June. Everything
  June derives from it (search index, layout) can be deleted and rebuilt.
- **Nothing leaves without a label.** Every call to a model is labelled and
  receipted first. If the receipt cannot be written, the call does not happen.
- **The model never decides whether something was saved.** Code writes files
  and reports the result; model text is never the confirmation.
- **Official routes only.** June uses providers' public APIs and documented
  integrations, never scraped logins or borrowed tokens.
- **Small, measured steps.** Ideas start as experiments in [lab/](lab/README.md)
  with a question, a hypothesis and a result before they become features.

## Develop

Requirements: macOS, Node 26 (see `.node-version`), pnpm 10. Rust is needed only
for the Mac app.

```sh
pnpm install
pnpm dev          # the engine plus the interface with hot reload; prints a link to open
pnpm dev:app      # the Mac app around them (Tauri, needs Rust)
pnpm test:watch   # unit tests on every save
pnpm check        # the gate: typecheck, unit tests, end-to-end tests
```

| Path | What lives there |
| --- | --- |
| `packages/core` | Vault file formats, ids, receipts. Platform-free TypeScript. |
| `packages/providers` | Model adapters. Network only through an injected `fetch`. |
| `packages/ui`, `packages/design` | Svelte components, the typed API client, design tokens. |
| `apps/server` | The local engine (`june serve`). |
| `apps/web` | The interface (SvelteKit, built to static files). |
| `apps/desktop` | The Mac app (Tauri 2). |
| `lab/` | Experiments: question, hypothesis, result, decision. |

[PLAN.md](PLAN.md) is the single plan of record and [CLAUDE.md](CLAUDE.md)
holds the working rules for the repo, including for AI coding agents. The
previous version of June, a Python assistant, is preserved on the `legacy`
branch.

## Contributing

June is early and moving fast, so the most useful contributions right now are
questions, ideas and bug reports in the issue tracker. If you want to write
code, open an issue first so we can agree on the approach; small, tested pull
requests are easiest to merge.

To report a security problem, please do not open a public issue. Use GitHub's
private vulnerability reporting on this repository instead.

## License

[MIT](LICENSE)
