// One adapter for every endpoint that speaks OpenAI's /chat/completions with
// streaming: OpenAI, Mistral, OpenRouter, Ollama, LM Studio, OpenCode Zen.
import { sseData } from "./sse.js";
import type { FetchLike, Provider, Route, StreamEvent } from "./types.js";

interface Chunk {
  choices?: { delta?: { content?: unknown }; finish_reason?: unknown }[];
  usage?: { prompt_tokens?: unknown; completion_tokens?: unknown } | null;
  error?: { message?: unknown };
}

const asNumber = (value: unknown): number | undefined => (typeof value === "number" ? value : undefined);

export function openAICompatible(opts: { route: Route; baseUrl: string; apiKey?: string; fetch: FetchLike }): Provider {
  const baseUrl = opts.baseUrl.replace(/\/+$/, "");
  const headers = (): Record<string, string> => ({
    "content-type": "application/json",
    ...(opts.apiKey ? { authorization: `Bearer ${opts.apiKey}` } : {}),
  });

  return {
    route: opts.route,

    async listModels() {
      try {
        const res = await opts.fetch(`${baseUrl}/models`, { method: "GET", headers: headers() });
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
        res = await opts.fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: headers(),
          signal: request.signal,
          body: JSON.stringify({
            model: request.model,
            messages: request.messages,
            stream: true,
            stream_options: { include_usage: true },
            ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
            ...(request.maxTokens === undefined ? {} : { max_tokens: request.maxTokens }),
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

      let finishReason: string | undefined;
      try {
        for await (const data of sseData(res.body)) {
          if (data === "[DONE]") break;
          let chunk: Chunk;
          try {
            chunk = JSON.parse(data) as Chunk;
          } catch {
            continue;
          }
          if (chunk.error !== undefined) {
            yield { type: "error", message: typeof chunk.error.message === "string" ? chunk.error.message : "provider error" };
            return;
          }
          const choice = chunk.choices?.[0];
          const content = choice?.delta?.content;
          if (typeof content === "string" && content !== "") yield { type: "text", text: content };
          if (typeof choice?.finish_reason === "string") finishReason = choice.finish_reason;
          if (chunk.usage) {
            const inputTokens = asNumber(chunk.usage.prompt_tokens);
            const outputTokens = asNumber(chunk.usage.completion_tokens);
            yield {
              type: "usage",
              ...(inputTokens === undefined ? {} : { inputTokens }),
              ...(outputTokens === undefined ? {} : { outputTokens }),
            };
          }
        }
      } catch (cause) {
        yield { type: "error", message: `connection to ${where} broke: ${String(cause)}` };
        return;
      }
      yield { type: "done", ...(finishReason === undefined ? {} : { finishReason }) };
    },
  };
}

/** Pull a readable message out of an error body, which is JSON on most providers. */
function errorMessage(body: string): string {
  try {
    const json = JSON.parse(body) as { error?: { message?: unknown } | string; message?: unknown };
    if (typeof json.error === "string") return json.error;
    if (typeof json.error?.message === "string") return json.error.message;
    if (typeof json.message === "string") return json.message;
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return body.slice(0, 500);
}
