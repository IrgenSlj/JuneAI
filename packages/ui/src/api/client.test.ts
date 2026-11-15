import { describe, expect, it } from "vitest";
import { ApiError, createJuneClient } from "./client.js";
import { takeApiToken } from "./token.js";

function fakeFetch(status: number, body: unknown) {
  const calls: { url: string; init: RequestInit }[] = [];
  const impl = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }) as unknown as typeof fetch;
  return { impl, calls };
}

describe("client", () => {
  it("calls /api on the same origin with the token header", async () => {
    const { impl, calls } = fakeFetch(200, { path: null, defaultPath: "/Users/a/June" });
    const client = createJuneClient({ token: "t0k", fetchImpl: impl });
    expect(await client.getVault()).toEqual({ path: null, defaultPath: "/Users/a/June" });
    expect(calls[0]?.url).toBe("/api/vault");
    expect(new Headers(calls[0]?.init.headers).get("X-June-Token")).toBe("t0k");
  });

  it("surfaces the daemon's error message", async () => {
    const { impl } = fakeFetch(400, { error: "not a folder" });
    const error = await createJuneClient({ fetchImpl: impl })
      .openVault({ path: "/x" })
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(400);
    expect((error as ApiError).reason).toBe("not a folder");
  });
});

describe("streamed turns", () => {
  function streaming(chunks: string[], status = 200) {
    const calls: { url: string; init: RequestInit }[] = [];
    const impl = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
          controller.close();
        },
      });
      return new Response(body, { status, headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    return { impl, calls };
  }

  it("posts the turn and yields each frame, even when a frame is split across chunks", async () => {
    const { impl, calls } = streaming(['data: {"type":"te', 'xt","text":"Hel"}\n\ndata: {"type":"text","text":"lo"}\n', '\ndata: {"type":"done"}\n\n']);
    const client = createJuneClient({ token: "t0k", fetchImpl: impl });
    const frames = [];
    for await (const frame of client.sendTurn("th_1", { route: "ollama", model: "m", text: "hi" })) frames.push(frame);
    expect(frames).toEqual([{ type: "text", text: "Hel" }, { type: "text", text: "lo" }, { type: "done" }]);
    expect(calls[0]?.url).toBe("/api/threads/th_1/turns");
    expect(calls[0]?.init.method).toBe("POST");
    expect(new Headers(calls[0]?.init.headers).get("X-June-Token")).toBe("t0k");
  });

  it("ends with an error frame, not silence, when the stream stops without done", async () => {
    const { impl } = streaming(['data: {"type":"text","text":"cut"}\n\n']);
    const frames = [];
    for await (const frame of createJuneClient({ fetchImpl: impl }).sendTurn("th_1", { route: "r", model: "m", text: "x" })) frames.push(frame);
    expect(frames.map((f) => f.type)).toEqual(["text", "error", "done"]);
  });

  it("throws the daemon's reason when the send is refused before streaming", async () => {
    const { impl } = fakeFetch(409, { error: "a reply is still arriving in this thread" });
    const iterator = createJuneClient({ fetchImpl: impl }).sendTurn("th_1", { route: "r", model: "m", text: "x" });
    const error = await iterator.next().catch((e: unknown) => e);
    expect((error as ApiError).reason).toBe("a reply is still arriving in this thread");
  });
});

describe("token handoff", () => {
  function env(hash: string, stored?: string) {
    const store = new Map<string, string>(stored ? [["june.token", stored]] : []);
    const state = { cleared: false };
    return {
      state,
      store,
      env: {
        hash,
        storage: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
        clearHash: () => void (state.cleared = true),
      },
    };
  }

  it("moves the token from the fragment into session storage and clears the URL", () => {
    const t = env("#token=abc123");
    expect(takeApiToken(t.env)).toBe("abc123");
    expect(t.store.get("june.token")).toBe("abc123");
    expect(t.state.cleared).toBe(true);
  });

  it("falls back to the stored token after a reload", () => {
    const t = env("", "abc123");
    expect(takeApiToken(t.env)).toBe("abc123");
    expect(t.state.cleared).toBe(false);
  });

  it("has no token when neither is present", () => {
    expect(takeApiToken(env("").env)).toBeUndefined();
  });
});
