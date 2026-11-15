import { type ReceiptLog, canonicalJson, createReceiptLog, memoryLock, memoryVaultFs, parseThread, serializeThread, sha256Hex } from "@june/core";
import type { ChatRequest, Provider, Route, StreamEvent } from "@june/providers";
import { describe, expect, it } from "vitest";
import { type TurnEvent, branchThread, createThread, runTurn } from "./index.js";

const ROUTE: Route = {
  id: "test",
  name: "Test route",
  kind: "api-key",
  host: "api.test.example",
  leavesDevice: true,
  billing: "your API key",
  dataNote: "unknown",
};

function fakeProvider(script: (request: ChatRequest) => StreamEvent[] | AsyncIterable<StreamEvent>) {
  const requests: ChatRequest[] = [];
  const provider: Provider = {
    route: ROUTE,
    listModels: async () => ["m-1"],
    async *stream(request) {
      requests.push(request);
      yield* script(request);
    },
  };
  return { provider, requests };
}

const reply = (...parts: string[]): StreamEvent[] => [
  ...parts.map((text) => ({ type: "text" as const, text })),
  { type: "usage", inputTokens: 10, outputTokens: 4 },
  { type: "done", finishReason: "stop" },
];

async function collect(events: AsyncIterable<TurnEvent>): Promise<TurnEvent[]> {
  const out: TurnEvent[] = [];
  for await (const event of events) out.push(event);
  return out;
}

function clock(start = "2026-11-15T10:00:00Z") {
  let t = Date.parse(start);
  return () => new Date((t += 1000));
}

async function setup() {
  const fs = memoryVaultFs();
  const now = clock();
  const receipts = createReceiptLog(fs, { clock: now });
  const lock = memoryLock();
  const { path, thread } = await createThread(fs, { title: "Kitchen quotes", clock: now });
  return { fs, receipts, path, thread, now, lock };
}

describe("createThread", () => {
  it("writes an empty, readable thread file named by its date and title", async () => {
    const { fs, path, thread } = await setup();
    expect(path).toMatch(/^threads\/2026\/11\/2026-11-15-kitchen-quotes-[a-z0-9]{5}\.md$/);
    expect(thread.id).toMatch(/^th_/);
    expect(parseThread((await fs.read(path)) ?? "")).toMatchObject({ id: thread.id, title: "Kitchen quotes", source: "june", turns: [] });
  });
});

describe("runTurn", () => {
  it("yields the label first, then the reply, then saved and done", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(() => reply("Around ", "12k."));
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "What should a kitchen cost?" }));
    expect(events.map((e) => e.type)).toEqual(["label", "text", "text", "usage", "saved", "done"]);
    const label = events[0];
    expect(label).toMatchObject({
      type: "label",
      label: { route: ROUTE, model: "m-1", noteIds: [], totalChars: "What should a kitchen cost?".length },
    });
  });

  it("writes both turns to the thread file, with route, model and receipt on the reply", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(() => reply("Around 12k."));
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "What should a kitchen cost?" }));
    const thread = parseThread((await fs.read(path)) ?? "");
    expect(thread.turns.map((t) => [t.role, t.text])).toEqual([
      ["user", "What should a kitchen cost?"],
      ["assistant", "Around 12k."],
    ]);
    const [user, assistant] = thread.turns;
    expect(user?.id).not.toBe(assistant?.id);
    expect(assistant).toMatchObject({ model: "m-1", meta: { route: "test", receipt: 1, notes: [] } });
    expect(events.find((e) => e.type === "saved")).toMatchObject({ type: "saved", path, userTurn: user?.id, assistantTurn: assistant?.id, receipt: 1 });
  });

  it("records a send receipt before calling the provider, hashing exactly what was sent and storing none of it", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    let receiptsAtSend = -1;
    const { provider, requests } = fakeProvider(async function* () {
      receiptsAtSend = (await receipts.verify()).count;
      yield* reply("ok");
    });
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "my private question" }));
    expect(receiptsAtSend).toBe(1);
    const log = (await fs.read("receipts/2026-11.jsonl")) ?? "";
    expect(log).not.toContain("my private question");
    const receipt = JSON.parse(log.trim()) as { kind: string; payloadHash: string; meta: Record<string, unknown> };
    const sent = requests[0];
    expect(receipt.kind).toBe("send");
    expect(receipt.payloadHash).toBe(await sha256Hex(canonicalJson({ model: sent?.model ?? "", messages: sent?.messages.map((m) => ({ ...m })) ?? [] })));
    expect(receipt.meta).toMatchObject({ route: "test", host: "api.test.example", model: "m-1", leavesDevice: true });
    expect(await receipts.verify()).toEqual({ ok: true, count: 1 });
  });

  it("sends nothing and writes nothing when the receipt cannot be written", async () => {
    const { fs, path, now, lock } = await setup();
    const before = await fs.read(path);
    const broken = { append: async () => { throw new Error("disk full"); } } as unknown as ReceiptLog;
    const { provider, requests } = fakeProvider(() => reply("never"));
    const events = await collect(runTurn({ fs, receipts: broken, provider, lock, clock: now }, { path, model: "m-1", text: "hello" }));
    expect(requests).toEqual([]);
    expect(await fs.read(path)).toBe(before);
    expect(events.map((e) => e.type)).toEqual(["label", "error", "done"]);
    expect(events[1]).toMatchObject({ type: "error", stage: "receipt" });
  });

  it("has the user turn on disk before the provider streams, so a crash mid-reply loses nothing typed", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    let onDisk: string[] = [];
    const { provider } = fakeProvider(async function* () {
      onDisk = parseThread((await fs.read(path)) ?? "").turns.map((t) => t.text);
      yield* reply("ok");
    });
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "keep me" }));
    expect(onDisk).toEqual(["keep me"]);
  });

  it("on a provider error keeps the user turn, invents no reply, and says why", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(() => [{ type: "text", text: "half an ans" }, { type: "error", message: "connection reset" }]);
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "hello" }));
    expect(events.map((e) => e.type)).toEqual(["label", "text", "error", "done"]);
    expect(events[2]).toMatchObject({ type: "error", stage: "provider", message: "connection reset" });
    expect(parseThread((await fs.read(path)) ?? "").turns.map((t) => t.role)).toEqual(["user"]);
  });

  it("treats a stream that ends without done or error as an error, not a reply", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(() => [{ type: "text", text: "cut off" }]);
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "hello" }));
    expect(events.map((e) => e.type)).toEqual(["label", "text", "error", "done"]);
    expect(parseThread((await fs.read(path)) ?? "").turns).toHaveLength(1);
  });

  it("sends the history of the thread with each new turn, and labels its size", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider, requests } = fakeProvider(() => reply("first answer"));
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "first" }));
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "second" }));
    expect(requests[1]?.messages).toEqual([
      { role: "user", content: "first" },
      { role: "assistant", content: "first answer" },
      { role: "user", content: "second" },
    ]);
    const label = events[0]?.type === "label" ? events[0].label : undefined;
    expect(label?.parts).toEqual([
      { kind: "history", chars: "first".length + "first answer".length },
      { kind: "draft", chars: "second".length },
    ]);
    expect(await receipts.verify()).toEqual({ ok: true, count: 2 });
    const turns = parseThread((await fs.read(path)) ?? "").turns;
    expect(new Set(turns.map((t) => t.id)).size).toBe(4);
  });

  it("refuses an empty message before anything happens", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider, requests } = fakeProvider(() => reply("x"));
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "   " }));
    expect(events.map((e) => e.type)).toEqual(["error", "done"]);
    expect(requests).toEqual([]);
    expect(await receipts.verify()).toEqual({ ok: true, count: 0 });
  });

  it("ends cleanly when an adapter breaks its promise and throws", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(async function* () {
      yield { type: "text", text: "a" };
      throw new Error("socket hang up");
    });
    const events = await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "hello" }));
    expect(events.map((e) => e.type)).toEqual(["label", "text", "error", "done"]);
    expect(events[2]).toMatchObject({ message: "socket hang up" });
  });

  it("keeps an edit made to the file while the reply streams (say, in Obsidian)", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider } = fakeProvider(async function* () {
      const thread = parseThread((await fs.read(path)) ?? "");
      await fs.write(path, serializeThread({ ...thread, title: "Renamed by hand" }));
      yield* reply("ok");
    });
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "hi" }));
    const thread = parseThread((await fs.read(path)) ?? "");
    expect(thread.title).toBe("Renamed by hand");
    expect(thread.turns.map((t) => t.role)).toEqual(["user", "assistant"]);
  });

  it("refuses to send, with no receipt, if the thread changed after the label was built", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider, requests } = fakeProvider(() => reply("x"));
    const events: TurnEvent[] = [];
    for await (const event of runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "hi" })) {
      events.push(event);
      if (event.type === "label") {
        const thread = parseThread((await fs.read(path)) ?? "");
        await fs.write(path, serializeThread({ ...thread, turns: [{ id: "x1", role: "user", text: "written elsewhere" }] }));
      }
    }
    expect(events.map((e) => e.type)).toEqual(["label", "error", "done"]);
    expect(events[1]).toMatchObject({ stage: "thread" });
    expect(requests).toEqual([]);
    expect(await receipts.verify()).toEqual({ ok: true, count: 0 });
  });

  it("never loses a turn when two sends race on one thread: one goes through, the other is refused", async () => {
    const { fs, receipts, path, now, lock } = await setup();
    const { provider, requests } = fakeProvider(() => reply("answer"));
    const deps = { fs, receipts, provider, lock, clock: now };
    const [a, b] = await Promise.all([collect(runTurn(deps, { path, model: "m-1", text: "from tab A" })), collect(runTurn(deps, { path, model: "m-1", text: "from tab B" }))]);
    const saved = [a, b].filter((events) => events.some((e) => e.type === "saved"));
    const refused = [a, b].filter((events) => events.some((e) => e.type === "error" && e.stage === "thread"));
    expect(saved).toHaveLength(1);
    expect(refused).toHaveLength(1);
    expect(requests).toHaveLength(1);
    const turns = parseThread((await fs.read(path)) ?? "").turns;
    expect(turns.map((t) => t.role)).toEqual(["user", "assistant"]);
    expect(await receipts.verify()).toEqual({ ok: true, count: 1 });
  });
});

describe("branchThread", () => {
  it("copies the turns up to the chosen one into a new thread that names its parent", async () => {
    const { fs, receipts, path, now, lock, thread } = await setup();
    const { provider } = fakeProvider(() => reply("answer"));
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "first" }));
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path, model: "m-1", text: "second" }));
    const original = parseThread((await fs.read(path)) ?? "");
    const at = original.turns[1]?.id ?? "";

    const branch = await branchThread(fs, { path, turnId: at, clock: now });
    expect(branch.thread.parent).toBe(`${thread.id}#${at}`);
    expect(branch.thread.id).not.toBe(thread.id);
    expect(branch.thread.title).toBe("Kitchen quotes (branch)");
    const saved = parseThread((await fs.read(branch.path)) ?? "");
    expect(saved.turns.map((t) => t.text)).toEqual(["first", "answer"]);
    expect(saved.turns[1]).toMatchObject({ model: "m-1", meta: { route: "test" } });
    // The original is untouched, and the branch continues on its own.
    expect(await fs.read(path)).toBe(serializeThread(original));
    await collect(runTurn({ fs, receipts, provider, lock, clock: now }, { path: branch.path, model: "other-model", text: "on another model" }));
    expect(parseThread((await fs.read(branch.path)) ?? "").turns.map((t) => t.id)).toEqual(["m1", "m2", "m3", "m4"]);
  });

  it("refuses a turn the thread does not have", async () => {
    const { fs, path, now } = await setup();
    await expect(branchThread(fs, { path, turnId: "m9", clock: now })).rejects.toThrow(/no turn m9/);
  });
});
