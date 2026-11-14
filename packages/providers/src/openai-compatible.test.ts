import { describe, expect, it } from "vitest";
import { openAICompatible } from "./openai-compatible.js";
import { OLLAMA } from "./presets.js";
import type { FetchLike, StreamEvent } from "./types.js";

const enc = new TextEncoder();

function streamOf(...parts: string[]): AsyncIterable<Uint8Array> {
  return (async function* () {
    for (const part of parts) yield enc.encode(part);
  })();
}

function fakeFetch(respond: (url: string, body: unknown) => Awaited<ReturnType<FetchLike>>) {
  const calls: { url: string; headers: Record<string, string>; body: unknown }[] = [];
  const fetch: FetchLike = async (url, init) => {
    const body = init.body === undefined ? undefined : JSON.parse(init.body);
    calls.push({ url, headers: init.headers, body });
    return respond(url, body);
  };
  return { fetch, calls };
}

const ok = (body: AsyncIterable<Uint8Array>) => ({ ok: true, status: 200, text: async () => "", body });

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const out: StreamEvent[] = [];
  for await (const e of events) out.push(e);
  return out;
}

const request = { model: "m", messages: [{ role: "user" as const, content: "hi" }] };

describe("openAICompatible", () => {
  it("streams text, usage and done, and sends a streaming request", async () => {
    const { fetch, calls } = fakeFetch(() =>
      ok(
        streamOf(
          'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
          'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
          "data: [DONE]\n\n",
        ),
      ),
    );
    const provider = openAICompatible({ route: OLLAMA, baseUrl: "http://localhost:11434/v1/", apiKey: "k", fetch });
    expect(await collect(provider.stream(request))).toEqual([
      { type: "text", text: "Hel" },
      { type: "text", text: "lo" },
      { type: "usage", inputTokens: 5, outputTokens: 2 },
      { type: "done", finishReason: "stop" },
    ]);
    expect(calls[0]?.url).toBe("http://localhost:11434/v1/chat/completions");
    expect(calls[0]?.headers.authorization).toBe("Bearer k");
    expect(calls[0]?.body).toMatchObject({ model: "m", stream: true, messages: request.messages });
  });

  it("sends no authorization header without a key", async () => {
    const { fetch, calls } = fakeFetch(() => ok(streamOf("data: [DONE]\n\n")));
    await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(calls[0]?.headers.authorization).toBeUndefined();
  });

  it("finishes with done when the stream ends without [DONE]", async () => {
    const { fetch } = fakeFetch(() => ok(streamOf('data: {"choices":[{"delta":{"content":"x"}}]}\n\n')));
    const events = await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(events.at(-1)).toEqual({ type: "done" });
  });

  it("turns an HTTP error into an error event with the provider's message", async () => {
    const { fetch } = fakeFetch(() => ({
      ok: false,
      status: 429,
      text: async () => '{"error":{"message":"Rate limit reached"}}',
      body: null,
    }));
    const events = await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(events).toEqual([{ type: "error", status: 429, message: "Rate limit reached" }]);
  });

  it("turns an unreachable host into an error event, never a throw", async () => {
    const fetch: FetchLike = async () => {
      throw new Error("ECONNREFUSED");
    };
    const events = await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "error" });
    expect((events[0] as { message: string }).message).toContain("could not reach localhost:11434");
  });

  it("reports a connection that breaks mid-stream, after the text so far", async () => {
    const broken: AsyncIterable<Uint8Array> = (async function* () {
      yield enc.encode('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n');
      throw new Error("socket hang up");
    })();
    const { fetch } = fakeFetch(() => ok(broken));
    const events = await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(events[0]).toEqual({ type: "text", text: "partial" });
    expect(events[1]).toMatchObject({ type: "error" });
    expect(events.some((e) => e.type === "done")).toBe(false);
  });

  it("surfaces an error object sent inside the stream", async () => {
    const { fetch } = fakeFetch(() => ok(streamOf('data: {"error":{"message":"model not found"}}\n\n')));
    const events = await collect(openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).stream(request));
    expect(events).toEqual([{ type: "error", message: "model not found" }]);
  });

  it("lists models sorted, and returns none on failure", async () => {
    const { fetch } = fakeFetch(() => ({
      ok: true,
      status: 200,
      text: async () => '{"data":[{"id":"zeta"},{"id":"alpha"},{"bad":1}]}',
      body: null,
    }));
    expect(await openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch }).listModels()).toEqual(["alpha", "zeta"]);
    const down: FetchLike = async () => {
      throw new Error("down");
    };
    expect(await openAICompatible({ route: OLLAMA, baseUrl: "http://x/v1", fetch: down }).listModels()).toEqual([]);
  });
});
