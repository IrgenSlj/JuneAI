import { describe, expect, it } from "vitest";
import { memoryVaultFs } from "./adapters/memory.js";
import { canonicalJson } from "./hash.js";
import { createReceiptLog } from "./receipts.js";

function clockFrom(start: string, stepMs = 1000) {
  let t = Date.parse(start);
  return () => {
    const now = new Date(t);
    t += stepMs;
    return now;
  };
}

describe("canonical JSON", () => {
  it("does not depend on key order", () => {
    expect(canonicalJson({ b: 1, a: [{ d: null, c: "x" }] })).toBe(canonicalJson({ a: [{ c: "x", d: null }], b: 1 }));
  });
});

describe("receipts log", () => {
  it("chains receipts, across month files, and verifies", async () => {
    const fs = memoryVaultFs();
    const log = createReceiptLog(fs, clockFrom("2026-11-30T23:59:59Z"));
    const first = await log.append({ kind: "send", actor: "june", targets: ["th_1#m2"], payload: { text: "secret" }, meta: { host: "api.example" } });
    const second = await log.append({ kind: "memory-write", actor: "user", targets: ["nt_1"], payload: "x" });
    expect(first.seq).toBe(1);
    expect(second.prev).toBe(first.hash);
    expect(Object.keys(fs.snapshot())).toEqual(["receipts/2026-11.jsonl", "receipts/2026-12.jsonl"]);
    expect(await log.verify()).toEqual({ ok: true, count: 2 });
  });

  it("never writes the payload", async () => {
    const fs = memoryVaultFs();
    await createReceiptLog(fs).append({ kind: "send", actor: "june", targets: [], payload: { text: "my-secret-sentence" } });
    expect(JSON.stringify(fs.snapshot())).not.toContain("my-secret-sentence");
  });

  it("loses nothing when many appends race in one process", async () => {
    const fs = memoryVaultFs();
    const log = createReceiptLog(fs);
    await Promise.all(Array.from({ length: 200 }, (_, i) => log.append({ kind: "tool-call", actor: "june", targets: [], payload: i })));
    expect(await log.verify()).toEqual({ ok: true, count: 200 });
  });

  it("continues the chain from a second log on the same vault", async () => {
    const fs = memoryVaultFs();
    await createReceiptLog(fs).append({ kind: "import", actor: "user", targets: [], payload: 1 });
    const again = await createReceiptLog(fs).append({ kind: "import", actor: "user", targets: [], payload: 2 });
    expect(again.seq).toBe(2);
  });

  it("detects an edited, removed or reordered line", async () => {
    const make = async () => {
      const fs = memoryVaultFs();
      const log = createReceiptLog(fs, clockFrom("2026-11-14T10:00:00Z"));
      for (const i of [1, 2, 3]) await log.append({ kind: "send", actor: "june", targets: [`t${i}`], payload: i });
      return { fs, log, lines: (fs.snapshot()["receipts/2026-11.jsonl"] ?? "").trimEnd().split("\n") };
    };
    const edited = await make();
    await edited.fs.write("receipts/2026-11.jsonl", `${edited.lines.join("\n").replace('"t2"', '"t9"')}\n`);
    expect((await edited.log.verify()).brokenAt).toMatchObject({ line: 2, reason: "content does not match its hash" });

    const removed = await make();
    await removed.fs.write("receipts/2026-11.jsonl", `${[removed.lines[0], removed.lines[2]].join("\n")}\n`);
    expect((await removed.log.verify()).ok).toBe(false);

    const failing = await make();
    await expect(
      createReceiptLog({ ...failing.fs, append: async () => { throw new Error("disk full"); } }).append({ kind: "send", actor: "june", targets: [], payload: 0 }),
    ).rejects.toThrow("disk full");
  });
});
