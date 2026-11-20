# 007 May June answer with the user's Claude subscription by running their own `claude` CLI?

## Question

Rule 12 requires a written policy check before the Claude CLI route (M4). Do Anthropic's current terms allow a third-party app to spawn the user's installed `claude -p` so their Pro or Max plan answers in June?

## Hypothesis

Allowed, because June never handles credentials: the user's own unmodified binary signs in through Anthropic's flow.

## Cheapest experiment

Read Anthropic's own pages on 2026-11-20 and quote them.

## Result

Surprise first: **the hypothesis does not hold as stated.** https://code.claude.com/docs/en/legal-and-compliance (no date shown), "Authentication and credential use":

- "Anthropic does not permit third-party developers to offer Claude.ai login into their own applications, or to route requests through Free, Pro, Max, or Ultra plan credentials on behalf of their users."
- "OAuth authentication is intended exclusively for purchasers of Claude Free, Pro, Max, Ultra, Team, and Enterprise subscription plans and is designed to support ordinary use of Claude Code and other native Anthropic applications."
- "Developers building products or services that interact with Claude's capabilities, including those using the Agent SDK, should use API key authentication".
- The one carve-out: it does not prevent "an end user from signing in to the unmodified Claude Code binary with their own Claude subscription". The page points to sales for specific cases.

https://code.claude.com/docs/en/agent-sdk/overview: "Unless previously approved, Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products". It documents running "the CLI as a subprocess with the -p flag", but under the same authentication rules.

https://www.anthropic.com/legal/consumer-terms (effective Oct 8, 2025), section 3: no access "through automated or non-human means, whether through a bot, script, or otherwise" except via an API key "or where we otherwise explicitly permit it".

## Decision

The Claude CLI route cannot be verified as permitted, so under PLAN.md it does not ship. M4 builds the ChatGPT plan route only; the Claude CLI route stays out unless Anthropic gives written permission (the founder may ask through Anthropic sales). Claude models remain fully available through the Anthropic API-key route. June keeps its hard rules: no Anthropic login, no Anthropic tokens, no reading `~/.claude`.
