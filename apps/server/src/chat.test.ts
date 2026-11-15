// The chat API end to end against the real daemon, a real vault on disk and
// a stand-in for the network (an injected fetch), so no test touches the
// internet or the real keychain.
import { mkdtemp, readFile } from "node:fs/promises";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createReceiptLog, memorySecretStore, parseThread, type SecretStore } from "@june/core";
import { nodeVaultFs } from "@june/node";
import type { FetchLike } from "@june/providers";
import type { ProviderInfo, ThreadDetail, ThreadSummary, TurnFrame } from "@june/ui/api/types";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type RunningServer, startServer } from "./server.js";

const TOKEN = "test-token-0123456789";
let root: string;
let vault: string;
let secrets: SecretStore;
let server: RunningServer;
let network: { online: boolean; sent: { url: string; headers: Record<string, string>; body?: string }[]; hold?: Promise<void> };

const encoder = new TextEncoder();
function sse(...frames: unknown[]): AsyncIterable<Uint8Array> {
  return (async function* () {
    for (const frame of frames) yield encoder.encode(`data: ${typeof frame === "string" ? frame : JSON.stringify(frame)}\n\n`);
  })();
}

/** Speaks just enough OpenAI-compatible HTTP for Ollama and OpenAI. */
const fakeFetch: FetchLike = async (url, init) => {
  network.sent.push({ url, headers: init.headers, ...(init.body === undefined ? {} : { body: init.body }) });
  const local = new URL(url).hostname === "localhost";
  if (!network.online && !local) throw new TypeError("fetch failed");
  if (url.endsWith("/models")) {
    return { ok: true, status: 200, text: async () => JSON.stringify({ data: [{ id: "model-b" }, { id: "model-a" }] }), body: null };
  }
  const hold = network.hold;
  return {
    ok: true,
    status: 200,
    text: async () => "",
    body: (async function* () {
      yield* sse({ choices: [{ delta: { content: "Hello" } }] });
      if (hold !== undefined) await hold;
      yield* sse({ choices: [{ delta: { content: " there." }, finish_reason: "stop" }] }, { choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } }, "[DONE]");
    })(),
  };
};

async function start(): Promise<void> {
  server = await startServer({ token: TOKEN, configDir: join(root, "config"), home: join(root, "home"), version: "test", secrets, fetch: fakeFetch });
}

function call(path: string, opts: { method?: string; body?: unknown; token?: string | null } = {}): Promise<{ status: number; body: string; type: string }> {
  return new Promise((resolve, reject) => {
    const payload = opts.body === undefined ? undefined : JSON.stringify(opts.body);
    const req = request(
      {
        host: "127.0.0.1",
        port: server.port,
        path,
        method: opts.method ?? "GET",
        headers: {
          host: `127.0.0.1:${server.port}`,
          ...(opts.token === null ? {} : { "x-june-token": opts.token ?? TOKEN }),
          ...(payload === undefined ? {} : { "content-type": "application/json", "content-length": Buffer.byteLength(payload) }),
        },
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => (body += chunk));
        res.on("end", () => resolve({ status: res.statusCode ?? 0, body, type: String(res.headers["content-type"] ?? "") }));
      },
    );
    req.on("error", reject);
    req.end(payload);
  });
}

const json = async <T>(path: string, opts: Parameters<typeof call>[1] = {}): Promise<T> => {
  const res = await call(path, opts);
  if (res.status >= 300) throw new Error(`${res.status} ${res.body}`);
  return JSON.parse(res.body) as T;
};

const frames = (body: string): TurnFrame[] =>
  body
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => JSON.parse(line.slice(6)) as TurnFrame);

async function newThread(): Promise<ThreadSummary> {
  return json<ThreadSummary>("/api/threads", { method: "POST", body: { title: "Kitchen quotes" } });
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "june-chat-"));
  vault = join(root, "vault");
  secrets = memorySecretStore();
  network = { online: true, sent: [] };
  await start();
  await json("/api/vault/open", { method: "POST", body: { path: vault } });
});

afterEach(async () => {
  await server.close();
});

describe("providers API", () => {
  it("lists the routes, with local ones ready and keyed ones waiting for a key", async () => {
    const list = await json<ProviderInfo[]>("/api/providers");
    const byId = Object.fromEntries(list.map((p) => [p.route.id, p]));
    expect(byId.ollama).toMatchObject({ needsKey: false, ready: true, route: { leavesDevice: false } });
    expect(byId.openai).toMatchObject({ needsKey: true, ready: false, route: { leavesDevice: true, host: "api.openai.com" } });
  });

  it("stores a key in the secret store, never echoes it, and removes it", async () => {
    const put = await call("/api/providers/openai", { method: "PUT", body: { apiKey: "sk-live-abc" } });
    expect(put.status).toBe(200);
    expect(put.body).not.toContain("sk-live-abc");
    expect(await secrets.get("route:openai")).toBe("sk-live-abc");
    const list = await call("/api/providers");
    expect(list.body).not.toContain("sk-live-abc");
    expect((JSON.parse(list.body) as ProviderInfo[]).find((p) => p.route.id === "openai")?.ready).toBe(true);
    expect((await call("/api/providers/openai", { method: "PUT", body: { apiKey: null } })).status).toBe(200);
    expect(await secrets.get("route:openai")).toBeUndefined();
  });

  it("writes no key anywhere in the vault or the config folder", async () => {
    await call("/api/providers/openai", { method: "PUT", body: { apiKey: "sk-live-abc" } });
    const fs = await nodeVaultFs(root);
    for (const path of await fs.list("")) expect(await fs.read(path), path).not.toContain("sk-live-abc");
  });

  it("refuses an unknown route and a malformed key", async () => {
    expect((await call("/api/providers/nope", { method: "PUT", body: { apiKey: "x" } })).status).toBe(404);
    expect((await call("/api/providers/openai", { method: "PUT", body: { apiKey: "sk-1\n" } })).status).toBe(400);
  });

  it("lists models for a route, and requires the token", async () => {
    expect(await json("/api/providers/ollama/models")).toEqual({ models: ["model-a", "model-b"] });
    expect((await call("/api/providers", { token: null })).status).toBe(401);
  });
});

describe("threads and turns", () => {
  it("needs an open vault", async () => {
    await server.close();
    root = await mkdtemp(join(tmpdir(), "june-chat-"));
    await start();
    expect((await call("/api/threads", { method: "POST", body: {} })).status).toBe(409);
  });

  it("streams a reply as label, text, usage, saved, done, and keeps it as readable Markdown", async () => {
    const thread = await newThread();
    const res = await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "What should a kitchen cost?" } });
    expect(res.status).toBe(200);
    expect(res.type).toContain("text/event-stream");
    const got = frames(res.body);
    expect(got.map((f) => f.type)).toEqual(["label", "text", "text", "usage", "saved", "done"]);
    expect(got[0]).toMatchObject({ type: "label", label: { route: { id: "ollama", leavesDevice: false }, model: "model-a" } });

    const detail = await json<ThreadDetail>(`/api/threads/${thread.id}`);
    expect(detail.turns.map((t) => [t.role, t.text])).toEqual([
      ["user", "What should a kitchen cost?"],
      ["assistant", "Hello there."],
    ]);
    expect(detail.turns[1]).toMatchObject({ model: "model-a", route: "ollama", receipt: 1 });

    const fs = await nodeVaultFs(vault);
    const [path] = (await fs.list("threads")).filter((p) => p.endsWith(".md"));
    const markdown = await readFile(join(vault, path ?? ""), "utf8");
    expect(markdown).toContain("## User\n\nWhat should a kitchen cost?");
    expect(markdown).toContain("## Assistant\n\nHello there.");
    expect(parseThread(markdown).id).toBe(thread.id);
  });

  it("survives a restart: the thread, its turns and a verifying receipt chain", async () => {
    const thread = await newThread();
    await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "one" } });
    await server.close();
    await start();
    const list = await json<ThreadSummary[]>("/api/threads");
    expect(list).toMatchObject([{ id: thread.id, title: "Kitchen quotes", turns: 2 }]);
    await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "two" } });
    expect((await json<ThreadDetail>(`/api/threads/${thread.id}`)).turns).toHaveLength(4);
    const receipts = createReceiptLog(await nodeVaultFs(vault));
    expect(await receipts.verify()).toEqual({ ok: true, count: 2 });
  });

  it("sends the key from the secret store to a keyed route, and never the key into a frame", async () => {
    await secrets.set("route:openai", "sk-live-abc");
    const thread = await newThread();
    const res = await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "openai", model: "gpt-x", text: "hi" } });
    expect(frames(res.body).at(-2)?.type).toBe("saved");
    expect(res.body).not.toContain("sk-live-abc");
    const sent = network.sent.find((s) => s.url === "https://api.openai.com/v1/chat/completions");
    expect(sent?.headers.authorization).toBe("Bearer sk-live-abc");
  });

  it("with the network off, a local model still answers and a remote one fails with a clear message", async () => {
    network.online = false;
    await secrets.set("route:openai", "sk-live-abc");
    const thread = await newThread();
    const local = frames((await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "hi" } })).body);
    expect(local.map((f) => f.type)).toContain("saved");
    const remote = frames((await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "openai", model: "gpt-x", text: "hi again" } })).body);
    expect(remote.map((f) => f.type)).toEqual(["label", "error", "done"]);
    expect(remote[1]).toMatchObject({ type: "error", stage: "provider" });
    expect((remote[1] as { message: string }).message).toMatch(/could not reach api\.openai\.com/);
    // The user's message is kept even though no reply came.
    expect((await json<ThreadDetail>(`/api/threads/${thread.id}`)).turns.map((t) => t.text)).toEqual(["hi", "Hello there.", "hi again"]);
  });

  it("refuses a keyed route with no key before anything is written", async () => {
    const thread = await newThread();
    const res = await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "openai", model: "gpt-x", text: "hi" } });
    expect(res.status).toBe(400);
    expect(res.body).toMatch(/needs an API key/);
    expect((await json<ThreadDetail>(`/api/threads/${thread.id}`)).turns).toEqual([]);
  });

  it("answers 409 to a second send while a reply is still streaming in the same thread", async () => {
    let release = () => {};
    network.hold = new Promise((r) => (release = r));
    const thread = await newThread();
    const first = call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "first" } });
    await new Promise((r) => setTimeout(r, 50));
    const second = await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama", model: "model-a", text: "second" } });
    expect(second.status).toBe(409);
    release();
    expect(frames((await first).body).map((f) => f.type)).toContain("saved");
  });

  it("validates ids and bodies", async () => {
    expect((await call("/api/threads/not-an-id")).status).toBe(404);
    expect((await call("/api/threads/th_01JD3K9V2Q8XN4M6P0R7S5T1WB")).status).toBe(404);
    const thread = await newThread();
    expect((await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "ollama" } })).status).toBe(400);
    expect((await call(`/api/threads/${thread.id}/turns`, { method: "POST", body: { route: "nope", model: "m", text: "x" } })).status).toBe(404);
  });
});
