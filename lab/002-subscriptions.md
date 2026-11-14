# 002 — Running on the user's subscriptions

## Question
Many people already pay for ChatGPT, Claude or Mistral subscriptions. Can June
use that inference, legitimately, instead of asking them for an API key and a
second bill?

## Hypothesis
Driving a consumer chat product programmatically is not permitted by its terms
and is fragile, so that route is closed. The legitimate route is the reverse
direction: the subscription apps already connect to MCP servers, so June
exposes the vault as an MCP server (M6) and the user's own assistant does the
inference while June supplies memory, packs and receipts. If that holds, M6 may
deserve to move earlier, because it turns every existing subscription into a
June client.

## Cheapest experiment
1. Read the current consumer and API terms of Anthropic, OpenAI and Mistral on third-party access. Record the exact clauses and dates; terms change.
2. List which subscription apps can connect to a local MCP server today, on which platforms and plans.
3. Prototype: a 50-line MCP server exposing two read-only tools over a fixture vault, connected to one subscription app. Does it work, and what does the user see?

## Result
Pending.

## Decision
Pending. Possible plan edit: move part of M6 before M3.
