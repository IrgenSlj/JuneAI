# 002 — Running on the user's subscriptions

## Question
Many people already pay for ChatGPT, Claude or Mistral. Can June use that
inference, legitimately, instead of asking for an API key and a second bill,
and can it give users a reason to use June every day?

## Hypothesis
Driving a consumer chat app programmatically breaks its terms and is fragile;
that route is closed. The legitimate route is the reverse: the subscription apps
attach to MCP servers, so June serves the vault **read-only** over MCP and the
user's own assistant does the inference.

Revised 2026-11-14 after research: use MCP for **reads, not capture**. Capture
through MCP would depend on the model choosing to call a save tool, which is
exactly old June's 73% failure. Reads are also what the platforms reliably allow.
If this holds, "your vault follows you into every assistant" is the daily-use
reason the M2 gate needs, and a read-only slice of M6 should move before the Map.

## What research found (2026-11-14, secondary sources, verify in the experiment)
- ChatGPT Plus and Pro can add custom MCP connectors in Developer Mode (Settings, Apps & Connectors, Advanced). Some guides say Plus/Pro are limited to read/fetch tools, and a forum report says ChatGPT memory is off in Developer Mode. [InfoQ](https://www.infoq.com/news/2025/10/chat-gpt-mcp), [Boomi](https://help.boomi.com/docs/Atomsphere/Platform/Connect_chatgpt_setup), [OpenAI forum](https://community.openai.com/t/mcp-server-tools-now-in-chatgpt-developer-mode/1357233?page=4). Authority to read: help.openai.com/en/articles/12584461.
- ChatGPT connectors are reached over the network, not as a local process. A local-only June needs a way in (a tunnel defeats the point); check whether a local server is usable at all. Claude Desktop runs local MCP servers directly.
- Mistral Le Chat: MCP connector support not yet checked.

## Cheapest experiment
1. Read the current consumer terms and help pages of OpenAI, Anthropic and Mistral on custom connectors. Record clauses and dates.
2. Table: app x plan x platform x (local server possible? read tools? write tools?).
3. Prototype: a 50-line MCP server with two read-only tools (`search_vault`, `read_note`) over a fixture vault, connected to Claude Desktop. Does the model call it when it should, and what does the user see?

## Result
Pending.

## Decision
Pending. Candidate plan edit (founder's call): a read-only "vault in every assistant" milestone before the Map.
