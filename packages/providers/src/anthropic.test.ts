import { describe, expect, it } from "vitest";
import { ANTHROPIC_API_VERSION, anthropicMessages, toAnthropicMessages } from "./anthropic.js";
import type { FetchLike, Route, StreamEvent } from "./types.js";

const ROUTE: Route = {
  id: "anthropic",
  name: "Anthropic (test)",
  kind: "api-key",
  host: "api.anthropic.com",
  leavesDevice: true,
  billing: "your API key",
  dataNote: "unknown",
};

const enc = new TextEncoder();
const sse = (...events: object[]) =>
  (async function* () {
    for (const e of events) yield enc.encode(`event: ${(e as { type: string }).type}\ndata: ${JSON.stringify(e)}\n\n`);
  })();

function fakeFetch(respond: () => Awaited<ReturnType<FetchLike>>) {
  const calls: { url: string; headers: Record<string, string>; body: Record<string, unknown> }[] = [];
  const fetch: FetchLike = async (url, init) => {
    calls.push({ url, headers: init.headers, body: init.body === undefined ? {} : JSON.parse(init.body) });
    return respond();
  };
  return { fetch, calls };
}

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const out: StreamEvent[] = [];
  for await (const e of events) out.push(e);
  return out;
}

const HAPPY = [
  { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 21, output_tokens: 1 } } },
  { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "private working" } },
  { type: "content_block_stop", index: 0 },
  { type: "ping" },
  { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Bon" } },
  { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "jour" } },
  { type: "content_block_stop", index: 1 },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 9 } },
  { type: "message_stop" },
];

describe("anthropicMessages", () => {
  it("streams only the visible text, then usage and done", async () => {
    const { fetch } = fakeFetch(() => ({ ok: true, status: 200, text: async () => "", body: sse(...HAPPY) }));
    const events = await collect(anthropicMessages({ route: ROUTE, apiKey: "sk", fetch }).stream({ model: "claude-opus-5-5", messages: [{ role: "user", content: "hi" }] }));
    expect(events).toEqual([
      { type: "text", text: "Bon" },
      { type: "text", text: "jour" },
      { type: "usage", inputTokens: 21, outputTokens: 9 },
      { type: "done", finishReason: "end_turn" },
    ]);
  });

  it("sends the key, the pinned API version, a streaming body, and never temperature", async () => {
    const { fetch, calls } = fakeFetch(() => ({ ok: true, status: 200, text: async () => "", body: sse({ type: "message_stop" }) }));
    await collect(
      anthropicMessages({ route: ROUTE, apiKey: "sk-test", fetch }).stream({
        model: "claude-opus-5-5",
        temperature: 0.2,
        messages: [
          { role: "system", content: "Be brief." },
          { role: "user", content: "hi" },
        ],
      }),
    );
    expect(calls[0]?.url).toBe("https://api.anthropic.com/v1/messages");
    expect(calls[0]?.headers).toMatchObject({ "x-api-key": "sk-test", "anthropic-version": ANTHROPIC_API_VERSION });
    expect(calls[0]?.body).toEqual({
      model: "claude-opus-5-5",
      max_tokens: 64_000,
      stream: true,
      system: "Be brief.",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(calls[0]?.body).not.toHaveProperty("fallbacks");
  });

  it("reports a refusal as how the turn finished, with no other model answering", async () => {
    const { fetch } = fakeFetch(() => ({
      ok: true,
      status: 200,
      text: async () => "",
      body: sse({ type: "message_start", message: { usage: { input_tokens: 3 } } }, { type: "message_delta", delta: { stop_reason: "refusal" }, usage: { output_tokens: 0 } }),
    }));
    const events = await collect(anthropicMessages({ route: ROUTE, apiKey: "sk", fetch }).stream({ model: "m", messages: [{ role: "user", content: "x" }] }));
    expect(events.at(-1)).toEqual({ type: "done", finishReason: "refusal" });
  });

  it("turns an error event mid-stream into an error, after the text so far", async () => {
    const { fetch } = fakeFetch(() => ({
      ok: true,
      status: 200,
      text: async () => "",
      body: sse(
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial" } },
        { type: "error", error: { type: "overloaded_error", message: "Overloaded" } },
      ),
    }));
    const events = await collect(anthropicMessages({ route: ROUTE, apiKey: "sk", fetch }).stream({ model: "m", messages: [{ role: "user", content: "x" }] }));
    expect(events).toEqual([
      { type: "text", text: "partial" },
      { type: "error", message: "Overloaded" },
    ]);
  });

  it("turns an HTTP error into an error event with the API's message", async () => {
    const { fetch } = fakeFetch(() => ({
      ok: false,
      status: 401,
      text: async () => '{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}',
      body: null,
    }));
    const events = await collect(anthropicMessages({ route: ROUTE, apiKey: "bad", fetch }).stream({ model: "m", messages: [{ role: "user", content: "x" }] }));
    expect(events).toEqual([{ type: "error", status: 401, message: "invalid x-api-key" }]);
  });

  it("lists models", async () => {
    const { fetch, calls } = fakeFetch(() => ({
      ok: true,
      status: 200,
      text: async () => '{"data":[{"id":"claude-sonnet-5-5"},{"id":"claude-opus-5-5"}],"has_more":false}',
      body: null,
    }));
    expect(await anthropicMessages({ route: ROUTE, apiKey: "sk", fetch }).listModels()).toEqual(["claude-opus-5-5", "claude-sonnet-5-5"]);
    expect(calls[0]?.url).toBe("https://api.anthropic.com/v1/models?limit=1000");
  });
});

describe("toAnthropicMessages", () => {
  it("lifts leading system messages and keeps later ones visible as user notes", () => {
    expect(
      toAnthropicMessages([
        { role: "system", content: "A" },
        { role: "system", content: "B" },
        { role: "user", content: "q" },
        { role: "assistant", content: "a" },
        { role: "system", content: "late" },
      ]),
    ).toEqual({
      system: "A\n\nB",
      messages: [
        { role: "user", content: "q" },
        { role: "assistant", content: "a" },
        { role: "user", content: "[System note] late" },
      ],
    });
  });
});
