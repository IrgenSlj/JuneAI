# June — plan of record

As of 2026-11-14. This is the only plan. Edit it in place when the plan is
wrong; do not write a second one. Experiments that change it live in `lab/`.

June is the user's own interface to AI: a Mac app (and the same UI in a browser
at `localhost`) that they open instead of ChatGPT or Claude. They bring the
inference: API keys, local models, a free gateway, or their own subscription
through official routes. June owns everything else: the conversation history,
the memory, the plans, the transparency, the look and feel. June is a wrapper
and says so: the neutral, private layer on top of whichever models the user
already pays for.

**Your AI, your files. One app for every model, where you can see and keep everything.**

History: rebuilt on 2026-11-14 from the founder's brief, then redirected the
same day (chat first, Mac only, Tauri, legacy UI reused). The Python June is on
branch `legacy` / tag `v0-final`.

## Status

| Milestone | State |
| --- | --- |
| M0 Shell and daemon | Done 2026-11-14. Design and ui restored; `june serve` with token, Host check, strict CSP, vault endpoints; web app with setup and an empty chat; `pnpm dev` and `pnpm dev:app`; the Mac app starts the daemon, opens its URL, and the daemon dies with the app even on a crash (checked by kill -9). Gaps: the Playwright suite runs on Chromium because Playwright's WebKit crashes on macOS 26.0, and the Mac window itself is checked by hand (WebDriver for Tauri does not run on macOS); the release build does not bundle Node yet (gate work); no watchdog restarts a crashed daemon yet. Decided: Node 26 LTS, default vault `~/June`, no mascot, `node:sqlite` with trigram search. |
| M1 Chat on any model | Started. `packages/providers`: the SSE parser (9 edge cases), the OpenAI-compatible adapter (errors and broken streams become events, never throws), local presets for Ollama and LM Studio, and the no-Node boundary test. Next: the Anthropic API adapter, then `packages/node` (vault files, keychain), then `engine.runTurn`. |
| M2 onwards | Not started. |

## Decisions

| # | Decision | Choice |
| --- | --- | --- |
| 1 | Product centre | A chat app with a vault underneath. Chat is the daily surface; the vault is what makes it worth choosing |
| 2 | Platforms | macOS app, plus the same UI in a browser at `localhost`. Windows after the gate; the daemon and UI already run anywhere Node does. CI runs `packages/*` tests on macOS, Windows and Linux because it is cheap |
| 3 | Architecture | A local daemon (`june serve`, Node) on loopback owns the vault and talks to providers. The UI is a static SvelteKit app over HTTP and SSE. The Mac app is a thin window around both |
| 4 | Shell | Tauri 2, restored from `legacy` (signing, notarisation, updater, sidecar supervisor). The sidecar is Node instead of Python. Electron is dropped |
| 5 | UI | `packages/design`, `packages/ui` and `apps/web` restored from `legacy`, cut to what remains; Claude Design restyles on top |
| 6 | Inference | Five route kinds, all optional, none hosted by us: API key, local, free gateway, ChatGPT plan, Claude CLI. Official routes only |
| 7 | Transparency | Always recorded, optionally shown: Quiet, Glance (default), Full |
| 8 | History exchange | Import (Claude, ChatGPT, then Le Chat exports), export (the vault is the export, plus per-thread Markdown and JSON), live for everything sent through June or written to disk by a local CLI agent |
| 9 | Storage | Plain files in a folder the user picks, by default `~/June` (outside iCloud's Desktop and Documents sync, which can corrupt a live index or git repo); the index is rebuildable. Raw provider JSON is kept beside each imported thread |
| 10 | The Map | June's signature view, built after chat works. A second screen; home is chat |
| 11 | Obsidian | Borrow files over app, `[[links]]`, backlinks, a readable graph, command palette, quick switcher, daily note. Never depend on Obsidian; a vault opens cleanly in it as a side effect |
| 12 | Agent scope | Plans are files; actions are vault operations and approved tool calls. No background agents |
| 13 | Licence | MIT. It also matters for the ChatGPT plan route, reported as documented for open-source, locally hosted apps |
| 14 | Gate | Ten outside users before M4, measured by talking to them. No telemetry |
| 15 | Audience | EU professionals and knowledge workers who use two or more assistants; family use later |

Deliberately out: hosted inference, accounts, sync, telemetry, a skills
marketplace, scheduled or background agents, voice, mobile, Linux. Windows comes after the gate.

Ideas carried over from the old June as design, not code: the hash-chained
receipts log, taint tracking for untrusted content, consent-gated MCP access,
reversible forgetting, and the loopback daemon rules below.

## Repository layout

```
packages/
  core/        pure TypeScript: vault formats, importers, receipts, trace, index queries. No Node built-ins
  node/        Node adapters for core: files, SQLite index, git, keychain, child processes, file locks
  providers/   model adapters and route metadata. Network only through an injected fetch, processes only through an injected spawner
  engine/      the turn loop: assemble context, build the label, send, stream, write the thread file, write receipts
  design/      tokens (restored from legacy)
  ui/          Svelte 5 components, typed API client, platform layer (restored from legacy, client rewritten)
apps/
  server/      the daemon: HTTP + SSE API over engine, plus the `june` CLI
  web/         SvelteKit static app (restored from legacy)
  desktop/     Tauri 2 shell (restored from legacy)
```

Dependency direction, enforced by boundary tests:
`core <- node`, `core <- providers <- engine <- server <- (web over HTTP) <- desktop`.
`core` and `providers` never import Node built-ins. `engine` receives its
adapters by injection. Only `node` and `server` touch the machine. Packages are
created when their milestone needs them.

## Runtime

```
Tauri window or browser tab, both at http://127.0.0.1:<port>/#token=<token>
  SvelteKit static UI (served by the daemon) --same-origin HTTP + SSE, X-June-Token--> june serve (Node)
                                                      engine
                                                      |-- core + node adapters --> vault folder
                                                      |-- providers: api-key | local | gateway | chatgpt-plan | claude-cli
                                                      '-- receipts log
```

Daemon rules:

- Bind to `127.0.0.1` only, on a port picked at launch. Never a fixed port, never "adopt whatever answers".
- **The daemon serves the UI.** The Mac window and a browser tab open the same URL, `http://127.0.0.1:<port>/#token=<token>`, so there is one origin in every mode: no CORS, no port baked into the build, and the browser test exercises exactly what the app runs. The token travels in the URL fragment, which browsers never send to a server, so it stays out of logs and referrers; the UI moves it to session storage and clears it from the address bar. In development, Vite proxies `/api` to the daemon so hot reload still works.
- Every API request carries `X-June-Token`, generated per launch (by the shell, or by `june serve` in browser mode, which prints the URL). Static files need no token; every `/api` route does.
- Check `Host` against loopback names (DNS rebinding). No CORS headers at all.
- Every response sets a strict CSP with `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`, so no other site can frame the UI or read the token.
- HTTP on `node:http` directly; the API is small enough that a framework would be the larger thing to audit.
- Sidecar: in development Tauri spawns the `node` on PATH. The release build bundles the official Node binary with one esbuild-bundled `server.mjs`; that is gate work, with signing and notarisation. Pinned to Node 26 LTS (`.node-version`).
- The folder picker is Tauri's native dialog in the app (granted to the daemon's origin through `capabilities.remote`), and a typed path with `~/June` prefilled in the browser.
- Index: `node:sqlite` (FTS5 confirmed in Node 26, lab 004). Search uses the `trigram` tokenizer so compound words like "Küchenbudget" match "budget"; queries under three characters fall back to a word index. Recheck on the bundled sidecar binary.
- A real CSP in `tauri.conf.json` (legacy had `"csp": null`).
- All model output and imported text render as sanitised Markdown. No raw HTML.

### HTTP API (first version)

Hand-written types in `packages/ui/src/api/types.ts` (or a shared `api-types`
file). Never regenerate the legacy types.

All under `/api`.

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | liveness and version (no token; returns nothing private) |
| `GET /api/vault`, `POST /api/vault/open` | current vault; open or create one |
| `GET /api/threads?lane=&q=&before=` | list or search threads |
| `GET /api/threads/:id` | one thread with turns |
| `POST /api/threads` | new thread |
| `POST /api/threads/:id/branch` | continue from a turn as a new thread, any model |
| `POST /api/threads/:id/label` | compute the send label for a draft; nothing is sent |
| `POST /api/threads/:id/turns` (SSE) | send a turn and stream the reply |
| `GET /api/providers`, `PUT /api/providers/:id` | configured routes; keys go to the keychain, never to disk |
| `GET /api/providers/:id/models` | models for a route |
| `POST /api/import` (multipart) | import an export zip; returns a count report |
| `GET /api/threads/:id/export?format=md\|json` | one thread out |
| `GET /api/receipts?after=`, `POST /api/receipts/verify` | the log and its check |
| `GET /api/settings`, `PUT /api/settings` | includes the transparency level |

SSE frames for a turn, one JSON object per `data:` line, in this order:
`label` (always first: what is about to leave), `text` (zero or more), `usage`,
`saved` (only after the file is written), `error`, `done`. The UI shows saved
state only from the `saved` frame.

## Providers

Policies moved several times in 2026. Before building each subscription
adapter, open the provider's own page, record the clause and date in `lab/`, and
stop if it no longer says what this table says.

| Kind | Examples | How | Status |
| --- | --- | --- | --- |
| API key | OpenAI, Anthropic, Mistral, OpenRouter | HTTPS with the user's key from the keychain | Safe. Build first |
| Local | Ollama, LM Studio | OpenAI-compatible endpoint on localhost | Safe. Build first. Nothing leaves the Mac |
| Free gateway | OpenCode Zen | OpenAI-compatible at `https://opencode.ai/zen/v1`, key from an OpenCode account | Free models are time-limited and some may train on prompts; the label says so |
| ChatGPT plan | Plus, Pro | OpenAI's "Sign in with ChatGPT" for open-source, locally hosted apps, official SDK and flow only | Verify at developers.openai.com. Never copy Codex tokens or imitate the Codex client |
| Claude CLI | Pro, Max | Spawn the user's own installed `claude` in print mode and read its stream | Verify at code.claude.com legal and Agent SDK pages. Ship last, labelled experimental, ready to remove |

Hard rules: June never implements an Anthropic login, handles Anthropic tokens
or reads `~/.claude` credentials. June never proxies one user's subscription to
anyone else. A route that cannot be verified as permitted does not ship; the UI
says so and points to the API-key route.

One OpenAI-compatible adapter covers OpenAI, Mistral, OpenRouter, Ollama, LM
Studio and OpenCode Zen. Anthropic's API gets its own small adapter. The Claude
CLI adapter takes an injected spawner; read `claude --help` for current flags.
`sse.ts` is written test-first: chunks split mid-line, `\r\n`, multi-line
`data:`, comments, a stream that ends without `[DONE]`.

Contracts (starting point):

```ts
type RouteKind = "api-key" | "local" | "gateway" | "chatgpt-plan" | "claude-cli";

interface Route {
  id: string; name: string; kind: RouteKind;
  host: string;                 // "api.openai.com", "localhost:11434", "local process"
  leavesDevice: boolean;
  billing: "your API key" | "your subscription" | "free" | "none (local)";
  dataNote: string;             // one honest sentence; "unknown" if unknown, never guessed
  dataNoteSource?: { url: string; checked: string };
}

interface Provider {
  readonly route: Route;
  listModels(): Promise<string[]>;
  stream(request: ChatRequest): AsyncIterable<StreamEvent>;  // never throws once started
}

interface SendLabel {            // built by engine, never by a model
  route: Route; model: string;
  parts: { kind: "system" | "history" | "note" | "attachment" | "draft"; id?: string; chars: number }[];
  noteIds: string[]; totalChars: number;
}
```

Route presets hold only checked facts.

## Transparency

Receipts are written for every send, memory read, memory write, tool call,
import and export, whatever the level. The level changes only what is shown.

| Level | Before sending | On each turn | Elsewhere |
| --- | --- | --- | --- |
| Quiet | nothing | nothing | receipts page in settings |
| Glance (default) | one-line chip under the composer: model, route, "leaves this Mac" or "stays on this Mac" | small chip: model and route | same |
| Full | the label opens before the first send to a new route and whenever attached context changes; notes can be removed from it | gutter with notes in, notes out, route, receipt number | receipts strip on the Map |

## The turn loop

`engine.runTurn`, plain code, each step a place where it can stop:

1. Load the thread; assemble context (which notes attach; pure, returns ids).
2. Build the send label and yield it.
3. Append the send receipt. If that fails, nothing has left the machine.
4. Append the user turn to the thread file, so a crash loses nothing.
5. Stream the provider; on error, yield it and stop. No assistant turn is invented.
6. Append the assistant turn with route, receipt and note ids; yield `saved`, then `done`.

Memory in the first version is explicit only: "Remember this" on any turn and a
`/remember` command write a note file through code and return a result. No
model-decided saving and no background extraction until M5.

## The vault

The vault is a folder of plain files and the only source of truth. Deleting
`.june/` must lose nothing; the app rebuilds it.

```
MyVault/
  threads/2026/09/2026-09-02-kitchen-quotes-5t1wb.md   one conversation per file
  threads/.../*.raw.json                                provider's original JSON beside imports
  notes/budget-kitchen.md                               one fact or decision per file
  plans/kitchen-renovation.md                           one plan per file
  lanes.md                                              lane names, order, colours
  packs/claude-desktop.md                               what one assistant may read
  attachments/                                          files referenced by threads
  receipts/2026-11.jsonl                                append-only, hash-chained
  .june/index.sqlite                                    derived: search, links, layout
```

**Thread file** (`packages/core/src/thread.ts`). Frontmatter values are JSON,
which is valid YAML. Each turn starts at an invisible marker comment; the
heading under it is for human readers and ignored when parsing, so message text
containing headings, `---` or a fake marker round-trips exactly.

```markdown
---
id: "th_01JD3..."
title: "Kitchen quotes"
source: "claude-export"
sourceId: "6734ab..."
started: "2026-09-02T18:04:11Z"
parent: "th_01JC9...#m7"
---

<!-- june:turn {"id":"m1","role":"user","at":"2026-09-02T18:04:11Z"} -->
## User

What should a 12 m2 kitchen cost?
```

**Note file.** One claim, its status and its sources. A note with no source link is invalid.

```markdown
---
id: nt_01JD4...
kind: fact                    # fact | decision | preference | question
status: accepted              # proposed | accepted | retired
trust: user                   # user | model | untrusted
sources: ["th_01JD3...#m5"]
---
Kitchen budget is 18k maximum, including appliances.
```

**Plan file.** A goal and a checklist; each step carries a state and the notes it depends on.

```markdown
---
id: pl_01JD5...
lane: kitchen-renovation
status: active
---
# Kitchen renovation
- [ ] Get three quotes ^s1  {state: approved, needs: [nt_01JD4...]}
- [ ] Pick a contractor ^s2 {state: proposed}
```

**Rules that hold everywhere**

- Paths obey `vaultPath()`; filenames are ASCII slugs plus a date and an id tail (`threadPath()`); the real title lives in frontmatter.
- Ids are ULIDs with a type prefix (`th_`, `nt_`, `pl_`). Imported items use `stableId()`, so a re-import on any machine yields the same id. Links are `[[id]]` or `[[id#anchor]]`.
- Imports are idempotent; dedupe on `source` plus `sourceId`. Timestamps are the provider's originals.
- The vault is a git repository managed by June; every accepted change to a note or plan is one commit.
- A receipt line holds kind, time, actor, target ids, a payload hash and the previous line's hash. Never content.
- Keys live in the macOS keychain. Nothing secret is written into the vault.

Known gaps in core: `VaultFs` is text-only (add bytes, `stat`, atomic write);
`receipts` re-reads the last month file on every append (cache it once the
cross-process lock exists); turn ids for June-native turns are not generated yet.

## Reusing the old app

From `legacy` with `git checkout legacy -- <path>`, then cut. Components and
tokens come back as they are; never redesign and port in one step. Pages do not:
the legacy chat page (1,238 lines) is wired to tasks, holdings, greetings,
recall votes and tool approval through the 4,781-line generated types, so it
cannot run against the new daemon without being rewritten anyway. New pages are
written thin, from the restored components, with the legacy page open as the
reference. Nothing from the Python tree comes back.

| Path | Keep | Remove |
| --- | --- | --- |
| `packages/design` | all tokens | nothing |
| `packages/ui/src/components` | `ChatBubble`, `Composer`, `MessageList`, `ConfirmDialog`, `EgressLine` (becomes the Glance chip), `ReasonChip`, `VerifyAffordance`, `OfflineNotice`, `TraceEventList` (becomes the gutter) | `SkillRegistry`, `ActivityStream`, `Mascot` (dropped by the founder) |
| `packages/ui/src/platform` | `types.ts`, `web.ts`, `tauri.ts` | `capacitor.ts`, unused Ollama-install and tray methods |
| `packages/ui/src/api` | the shape of `client.ts`: base URL, token header, timeout, SSE-over-POST reader | the generated `types.ts` and every method for a removed route |
| `apps/web` | `+layout.svelte`, `chat/`, `settings/`, `setup/`, `system/receipts/` | `skills/`, `tasks/`, `memory/` (rebuilt as Notes in M5), `system/silence/`, `system/glass/`, `help/ollama/`, the PWA service worker |
| `apps/desktop/src-tauri` | `sidecar.rs` (new binary and arguments), `lib.rs`, updater and signing config, entitlements | `ollama.rs`, the fixed port 8000, port adoption, `"csp": null` |

## Milestones

One at a time. Each "done when" becomes tests before the code.

1. **M0 Shell and daemon (M).** Slices: restore design and ui; `apps/server` with health, vault, token and host checks, serving the UI; `apps/web` layout and an empty chat page against it; Tauri (dev mode) starting the daemon on a random port and opening its URL.
    - Done when: `pnpm dev` opens the UI in a browser against a running daemon; the Mac app opens, picks a vault folder and shows the chat page empty; one Playwright test covers both; a request without the token is refused.
2. **M1 Chat on any model (M).** `providers` (OpenAI-compatible, Anthropic API, presets for Ollama, LM Studio, OpenAI, Mistral, OpenRouter, OpenCode Zen), `engine.runTurn`, thread files, receipts, branch to another model, the Glance chip, keys in the keychain.
    - Done when: a conversation with a local model and one with an API-key model both stream, survive a restart, and exist as readable Markdown in the vault; killing the daemon mid-reply loses no user turn; receipts verify; with the network off, a local model still works and a remote one fails with a clear message.
3. **M2 Import and export (M).** Lab 001 on the founder's real Claude export, then the Claude importer, then ChatGPT. Raw JSON kept. Search. Per-thread export. Watch a folder for new export zips.
    - Done when: the founder's full Claude history imports with a count report that matches the export; a second import changes zero files; any imported thread can be continued on any configured model as a branch; deleting `.june/` and reopening rebuilds an identical index; search answers in under 100 ms on that history.
4. **M3 Transparency levels (S).** Quiet, Glance, Full; the label sheet; the turn gutter; the receipts page with verify.
    - Done when: switching level changes only the display, never what is recorded; in Full, removing a note from the label removes it from the request, proven by inspecting the outgoing body.
5. **Gate: ten outside users.** Signed, notarised Mac build. Ten people who are not the founder use it as their chat app for a week. Talk to each.
6. **M4 Subscriptions (M).** The ChatGPT plan route, then the Claude CLI route, each after its policy check is written in `lab/`.
    - Done when: each route streams a reply; the label shows "your subscription"; June holds no Anthropic credential anywhere; removing a route leaves no token behind.
7. **M5 Notes and trace (L).** Note files with revisions, backlinks, "Remember this", model-proposed notes the user accepts, backward and forward trace.
8. **M6 The Map (L).** Lanes, threads, notes band, plans as track past Now. Canvas 2D with `d3-zoom` and a quadtree.
9. **M7 Plans and MCP (M).** Plan files with approval; a read-only MCP server so other assistants can read the vault (lab 002).

## The Map

The Map is June's signature view: a single time canvas where conversations,
memory and plans are the same picture. Read it left to right: a message in an
old ChatGPT thread wrote a budget note, a later Claude turn read that note, the
first planned step depends on it. Selecting any one lights the whole path.

- **Lanes** are projects or topics, not providers. Colour carries the assistant.
- **Threads** are lines, turns are dots. Continuing a thread on another model draws a branch.
- **Notes** sit in a band underneath, each a file with its revisions.
- **Plans** continue past Now as dashed track; done steps turn solid and slide into the past.
- **Receipts** are a strip along the bottom.

Zoom levels: Map, Thread (the reader with its gutter), Note or Plan (the file,
its history, every use). **The trace**: backward (where did this note come from)
and forward (which answers did it shape). Forward trace exists only for turns
sent through June; imported threads get backward trace only, and the UI says so.

## Risks and kill criteria

| Risk | What it would look like | Response |
| --- | --- | --- |
| No one switches their daily chat app | Fewer than ten outside users through a week, or they drift back to ChatGPT | Find out why in the calls; the vault and import may still stand alone as a viewer |
| Crowded chat-client market | Many BYOK clients already exist on the Mac (local and API-key chat apps) | Compete on what they lack: the vault as plain files, imported history from every provider, receipts, subscription routes |
| Mac only shrinks the market | EU professionals mostly on Windows ask for it | The daemon and browser UI already run anywhere Node does; revisit after the gate |
| A subscription route is withdrawn | A provider changes its terms again | Each route is optional, labelled and removable; the API-key route always works |
| Importing is already free | Obsidian plugins import ChatGPT, Claude, Le Chat, Perplexity and Grok exports ([Nexus](https://github.com/Superkikim/nexus-ai-chat-importer)) | Import is a door into June, not the product |
| Providers ship cross-provider history | Claude already imports memory from ChatGPT, Gemini and Copilot (March 2026), one way | Lean on what they will not do: files you own and receipts across competitors |
| Providers change export formats | An importer fails on a fresh export | Fixture tests per provider, a clear error naming the format, raw JSON kept for re-parsing |
| Daemon or webview security mistake | Another site or model output reaches the daemon or a privileged API | Loopback, token, host check, CSP and sanitised Markdown are release blockers, tested in CI |
| Scope creep back into agents | Requests for background tasks, skills, scheduling | Refuse until the gate numbers hold |

One honest limit to state in the product: a local copy gives ownership, not
erasure. What a provider retains is outside June's control, and the label is
where the user learns that.

## Design brief (Claude Design)

Design the chat app first, then the Map, starting from the tokens in
`packages/design` and the legacy chat page. Frames, in order:

1. Chat, Glance level: conversation, composer, route chip, a model switcher grouping routes by "on this Mac", "your keys", "your subscriptions", "free".
2. Chat, Full level: the same conversation with the gutter open on one turn.
3. The send label sheet, with one attached note being removed.
4. First run: pick a vault folder, connect a first route, import history.
5. Import result: the count report and the imported threads arriving.
6. Thread list and search, with source badges for imported threads.
7. The Map with density, then with one trace selected.

Direction: calm and editorial, closer to a writing tool than a chatbot.
Transparency elements are a quiet part of the page, not warnings. Dark and
light, both designed. No sparkle icons, no gradient mesh. Every colour that
carries meaning has a second cue for colour-blind users.

## Open questions for the founder

- How June makes money (lab 003).
