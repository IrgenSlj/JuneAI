// Anthropic's Messages API over raw HTTP (POST /v1/messages with stream:true).
// Raw HTTP rather than the SDK because providers reach the network only
// through the injected FetchLike function; see the dependency rule in PLAN.md.
//
// Choices that matter for honesty:
// - API version 2026-10-21. It rejects temperature/top_p/top_k with a 400, so
//   they are never sent, whatever the caller asks.
// - No server-side model fallbacks. If the model declines, the turn ends with
//   finishReason "refusal" instead of another model answering under the label
//   of the one the user chose.
import { sseData } from "./sse.js";
import type { ChatMessage, FetchLike, Provider, Route, StreamEvent } from "./types.js";

export const ANTHROPIC_API_VERSION = "2026-10-21";
const DEFAULT_BASE_URL = "https://api.anthropic.com/v1";
const DEFAULT_MAX_TOKENS = 64_000;

interface AnthropicEvent {
  type?: string;
  message?: { usage?: { input_tokens?: unknown } };
  delta?: { type?: string; text?: unknown; stop_reason?: unknown };
  usage?: { output_tokens?: unknown };
  error?: { type?: unknown; message?: unknown };
}

const asNumber = (value: unknown): number | undefined => (typeof value === "number" ? value : undefined);

/** Leading system messages go to the top-level `system` field; the rest stay in order. */
export function toAnthropicMessages(messages: ChatMessage[]): {
  system?: string;
  messages: { role: "user" | "assistant"; content: string }[];
} {
  const system: string[] = [];
  let i = 0;
  while (i < messages.length && messages[i]?.role === "system") system.push(messages[i++]?.content ?? "");
  const rest = messages.slice(i).map((m) => ({
    // A system message later in the history is sent as user text, labelled, so
    // nothing is dropped silently on models without mid-conversation system turns.
    role: m.role === "assistant" ? ("assistant" as const) : ("user" as const),
    content: m.role === "system" ? `[System note] ${m.content}` : m.content,
  }));
  return { ...(system.length > 0 ? { system: system.join("\n\n") } : {}), messages: rest };
}

export function anthropicMessages(opts: { route: Route; apiKey: string; fetch: FetchLike; baseUrl?: string }): Provider {
  const baseUrl = (opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  const headers = (): Record<string, string> => ({
    "content-type": "application/json",
    "x-api-key": opts.apiKey,
    "anthropic-version": ANTHROPIC_API_VERSION,
  });

  return {
    route: opts.route,

    async listModels() {
      try {
        const res = await opts.fetch(`${baseUrl}/models?limit=1000`, { method: "GET", headers: headers() });
        if (!res.ok) return [];
        const json = JSON.parse(await res.text()) as { data?: { id?: unknown }[] };
        return (json.data ?? []).flatMap((m) => (typeof m.id === "string" ? [m.id] : [])).sort();
      } catch {
        return [];
      }
    },

    async *stream(request): AsyncIterable<StreamEvent> {
      const where = opts.route.host;
      let res: Awaited<ReturnType<FetchLike>>;
      try {
        res = await opts.fetch(`${baseUrl}/messages`, {
          method: "POST",
          headers: headers(),
          signal: request.signal,
          body: JSON.stringify({
            model: request.model,
            max_tokens: request.maxTokens ?? DEFAULT_MAX_TOKENS,
            stream: true,
            ...toAnthropicMessages(request.messages),
          }),
        });
      } catch (cause) {
        yield { type: "error", message: `could not reach ${where}: ${String(cause)}` };
        return;
      }
      if (!res.ok || res.body === null) {
        const text = await res.text().catch(() => "");
        yield { type: "error", status: res.status, message: errorMessage(text) || `${where} answered ${res.status}` };
        return;
      }

      let inputTokens: number | undefined;
      let outputTokens: number | undefined;
      let finishReason: string | undefined;
      try {
        for await (const data of sseData(res.body)) {
          let event: AnthropicEvent;
          try {
            event = JSON.parse(data) as AnthropicEvent;
          } catch {
            continue;
          }
          switch (event.type) {
            case "message_start":
              inputTokens = asNumber(event.message?.usage?.input_tokens);
              break;
            case "content_block_delta":
              // Only visible text. Thinking deltas are the model's working, not its answer.
              if (event.delta?.type === "text_delta" && typeof event.delta.text === "string" && event.delta.text !== "") {
                yield { type: "text", text: event.delta.text };
              }
              break;
            case "message_delta":
              if (typeof event.delta?.stop_reason === "string") finishReason = event.delta.stop_reason;
              outputTokens = asNumber(event.usage?.output_tokens) ?? outputTokens;
              break;
            case "error":
              yield { type: "error", message: typeof event.error?.message === "string" ? event.error.message : "provider error" };
              return;
            case "message_stop":
              break;
            default:
              // ping, content_block_start/stop and future event types carry nothing to show.
              break;
          }
        }
      } catch (cause) {
        yield { type: "error", message: `connection to ${where} broke: ${String(cause)}` };
        return;
      }
      if (inputTokens !== undefined || outputTokens !== undefined) {
        yield {
          type: "usage",
          ...(inputTokens === undefined ? {} : { inputTokens }),
          ...(outputTokens === undefined ? {} : { outputTokens }),
        };
      }
      yield { type: "done", ...(finishReason === undefined ? {} : { finishReason }) };
    },
  };
}

function errorMessage(body: string): string {
  try {
    const json = JSON.parse(body) as { error?: { message?: unknown } };
    if (typeof json.error?.message === "string") return json.error.message;
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return body.slice(0, 500);
}
