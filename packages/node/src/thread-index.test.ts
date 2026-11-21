import { mkdtemp, rm, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Thread, serializeThread, threadPath } from "@june/core";
import { describe, expect, it } from "vitest";
import { makeSnippet, openThreadIndex, queryTerms } from "./thread-index.js";
import { nodeVaultFs } from "./vault-fs.js";

const freshRoot = () => mkdtemp(join(tmpdir(), "june-index-"));
const ULID_TAIL = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const idFor = (n: number) => `th_01JD3${String(n).padStart(5, "0").replace(/\d/g, (d) => ULID_TAIL[Number(d)] ?? "0")}AAAAAAAAAAAA`.slice(0, 29);

function thread(n: number, title: string, texts: string[], day = 10 + (n % 15)): Thread {
  return {
    id: idFor(n),
    title,
    source: n % 2 === 0 ? "june" : "claude-export",
    started: `2025-03-${day}T10:00:00Z`,
    turns: texts.map((text, i) => ({ id: `m${i + 1}`, role: i % 2 === 0 ? "user" : "assistant", at: `2025-03-${day}T10:${String(i).padStart(2, "0")}:00Z`, text })),
  };
}

async function vaultWith(threads: Thread[]) {
  const root = await freshRoot();
  const fs = await nodeVaultFs(root);
  for (const t of threads) await fs.write(threadPath(t), serializeThread(t));
  return { root, fs };
}

const sample = [
  thread(1, "Kitchen quotes", ["What should a 12 m2 kitchen cost?", "Das Küchenbudget liegt bei 18k."]),
  thread(2, "Trip to Lisbon", ["Plan a café crawl", "Day one: Alfama."]),
  thread(3, "日本語のテキスト", ["このテキストを翻訳して", "Here is the translation."]),
  thread(4, "Vue or Svelte", ["vu at a glance", "Svelte 5 runes."]),
];

describe("thread index", () => {
  it("lists threads newest first and pages with before", async () => {
    const { root, fs } = await vaultWith(sample);
    const index = await openThreadIndex(root, fs);
    expect(await index.sync()).toEqual({ indexed: 4, removed: 0, unreadable: [] });
    const all = index.list();
    expect(all.map((t) => t.title)).toEqual(["Vue or Svelte", "日本語のテキスト", "Trip to Lisbon", "Kitchen quotes"]);
    expect(index.list({ limit: 2, before: all[1]!.updated ?? "2025-03-13T10:01:00Z" }).map((t) => t.title)).toEqual(["Trip to Lisbon", "Kitchen quotes"]);
    expect(index.get(idFor(2))).toMatchObject({ title: "Trip to Lisbon", source: "june", turns: 2 });
    index.close();
  });

  it("finds compound words, folds accents, handles Japanese and short queries", async () => {
    const { root, fs } = await vaultWith(sample);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    const titles = (q: string) => index.search(q).map((h) => h.title);
    expect(titles("budget")).toEqual(["Kitchen quotes"]);
    expect(titles("kuchen")).toEqual(["Kitchen quotes"]);
    expect(titles("cafe")).toEqual(["Trip to Lisbon"]);
    expect(titles("テキスト")).toEqual(["日本語のテキスト"]);
    expect(titles("vu")).toEqual(["Vue or Svelte"]);
    expect(titles("lisbon alfama")).toEqual([]); // terms must meet in one turn or the title
    expect(titles("\"café crawl\"")).toEqual(["Trip to Lisbon"]);
    expect(titles("x\" OR 1")).toEqual([]); // FTS syntax in a query is just text
    const [hit] = index.search("budget");
    expect(hit).toMatchObject({ turn: "m2" });
    expect(hit?.snippet).toBe("Das Küchen\u0002budget\u0003 liegt bei 18k.");
    index.close();
  });

  it("picks up files written, edited or deleted outside June on the next sync", async () => {
    const { root, fs } = await vaultWith(sample);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    const edited = { ...sample[1]!, title: "Trip to Porto" };
    await fs.write(threadPath(sample[1]!), serializeThread(edited));
    await utimes(join(root, threadPath(sample[1]!)), new Date(), new Date(Date.now() + 5000));
    await fs.remove(threadPath(sample[0]!));
    await fs.write("threads/broken.md", "not a thread");
    expect(await index.sync()).toEqual({ indexed: 1, removed: 1, unreadable: ["threads/broken.md"] });
    expect(index.get(idFor(2))?.title).toBe("Trip to Porto");
    expect(index.search("budget")).toEqual([]);
    expect(await index.sync()).toMatchObject({ indexed: 0, removed: 0 });
    index.close();
  });

  it("rebuilds an identical index after .june/ is deleted", async () => {
    const { root, fs } = await vaultWith(sample);
    let index = await openThreadIndex(root, fs);
    await index.sync();
    const answer = () => ({ list: index.list(), searches: ["budget", "cafe", "vu", "テキスト", "svelte"].map((q) => index.search(q)) });
    const before = answer();
    index.close();
    await rm(join(root, ".june"), { recursive: true });
    index = await openThreadIndex(root, fs);
    await index.sync();
    expect(answer()).toEqual(before);
    index.close();
  });

  it("answers a search in under 100 ms on a large history", async () => {
    const words = ["budget", "kitchen", "travel", "invoice", "garden", "python", "contract", "recipe", "meeting", "Lisbon"];
    const many = Array.from({ length: 3000 }, (_, n) =>
      thread(n + 10, `Conversation ${n}`, Array.from({ length: 12 }, (_, i) => `Turn ${i} about ${words[(n + i) % words.length]} and ${words[(n * 7 + i) % words.length]} `.repeat(8))),
    );
    many.push(thread(9999, "Needle", ["the word Zwetschgenkuchen appears once"]));
    const { root, fs } = await vaultWith(many);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    for (const q of ["kitchen", "Zwetschgen", "invoice python", "py"]) {
      const start = performance.now();
      const hits = index.search(q);
      const ms = performance.now() - start;
      expect(hits.length, q).toBeGreaterThan(0);
      expect(ms, `${q} took ${ms.toFixed(1)} ms`).toBeLessThan(100);
    }
    index.close();
  }, 120_000);
});

describe("snippets and terms", () => {
  it("marks the first match and keeps the window short", () => {
    expect(makeSnippet("A café near the river", ["cafe"])).toBe("A \u0002café\u0003 near the river");
    const long = `${"x ".repeat(200)}needle${" y".repeat(200)}`;
    const snippet = makeSnippet(long, ["needle"], 40);
    expect(snippet.startsWith("…") && snippet.endsWith("…") && snippet.includes("\u0002needle\u0003")).toBe(true);
    expect(queryTerms(' budget "two words" budget ')).toEqual(["budget", "two words"]);
  });
});

describe("thread index edge cases", () => {
  it("pages through threads that share a timestamp without skipping any", async () => {
    const same = Array.from({ length: 7 }, (_, i) => ({ ...thread(100 + i, `Same ${i}`, ["x"], 12) }));
    const { root, fs } = await vaultWith(same);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    const seen: string[] = [];
    let page = index.list({ limit: 3 });
    while (page.length > 0) {
      seen.push(...page.map((t) => t.id));
      const last = page.at(-1)!;
      page = index.list({ limit: 3, before: last.activity, beforeId: last.id });
    }
    expect(seen.sort()).toEqual(same.map((t) => t.id).sort());
    index.close();
  });

  it("drops a thread from the index when its file stops parsing", async () => {
    const { root, fs } = await vaultWith(sample);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    await fs.write(threadPath(sample[0]!), "broken now");
    await utimes(join(root, threadPath(sample[0]!)), new Date(), new Date(Date.now() + 5000));
    expect((await index.sync()).unreadable).toEqual([threadPath(sample[0]!)]);
    expect(index.get(idFor(1))).toBeUndefined();
    expect(index.search("budget")).toEqual([]);
    index.close();
  });

  it("short queries point at the turn that matched", async () => {
    const { root, fs } = await vaultWith([thread(50, "Go notes", ["first turn", "second turn says go"])]);
    const index = await openThreadIndex(root, fs);
    await index.sync();
    const [hit] = index.search("go");
    expect(hit?.turn).toBe("m2");
    expect(hit?.snippet).toContain("\u0002go\u0003");
    index.close();
  });
});
