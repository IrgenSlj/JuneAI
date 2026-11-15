# 005 What does each remote route do with what June sends?

## Question

Can each remote route's label say, in one honest sentence from the provider's own page, whether it trains on what is sent and how long it keeps it?

## Hypothesis

The API-key providers (OpenAI, Anthropic, Mistral) publish a clear no-training default and a retention period; the gateways (OpenRouter, OpenCode Zen) depend on the model behind them.

## Cheapest experiment

Read each provider's own data page on 2026-11-15 and quote it. Nothing else counts as a source.

## Result

Surprise first: **OpenCode Zen's free models cannot be used from June.** Its docs say "Free-tier chat models require OpenCode; use paid models or Go from other clients." The base URL in PLAN.md was also stale: chat completions now live at `https://opencode.ai/inference/openai/v1/chat/completions`.

| Route | Training | Retention | Source (checked 2026-11-15) |
| --- | --- | --- | --- |
| OpenAI | "data sent to the OpenAI API is not used to train or improve OpenAI models" | abuse-monitoring logs "retained for up to 30 days" | https://developers.openai.com/api/docs/guides/your-data |
| Anthropic | "By default, we will not use your inputs or outputs from our commercial products" (API included); feedback you submit may be used | "delete inputs and outputs on our backend within 30 days", with exceptions (policy enforcement up to 2 years, legal, ZDR) | https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training, https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data |
| Mistral | Commercial terms: no training "except" when opted in, or not opted out of a product set to opt-in by default; an opt-out toggle exists in the Admin panel; the API default is not stated | not stated outside Zero Data Retention, which Mistral grants at its discretion | https://legal.mistral.ai/terms/commercial-terms-of-service, https://help.mistral.ai/en/articles/455207-can-i-opt-out-of-my-input-or-output-data-being-used-for-training |
| OpenRouter | "OpenRouter does not use your Inputs or Outputs for model training"; but "Each AI provider on OpenRouter has its own data handling policies", and some may train; an account setting can exclude them | not stated for text prompts | https://openrouter.ai/privacy, https://openrouter.ai/docs/features/privacy-and-logging |
| OpenCode Zen | varies by model; most providers zero-retention and no training, but several free models retain data and may train | varies by model (some 30 days) | https://opencode.ai/docs/zen |

## Decision

Each preset gets a one-sentence data note with its source and date; where the page does not say, the note says so instead of guessing. Zen's base URL is corrected and its billing is "your API key", since only paid models work outside OpenCode. Whether June needs a free route now goes to the founder (PLAN.md, open questions). Recheck these pages before the gate.
