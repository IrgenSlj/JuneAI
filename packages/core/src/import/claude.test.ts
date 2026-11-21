import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { memoryLock, memoryVaultFs } from "../index.js";
import { createReceiptLog } from "../receipts.js";
import { parseThread, serializeThread, threadPath } from "../thread.js";
import { looksLikeClaudeConversations, looksLikeClaudeProjects, parseClaudeExport } from "./claude.js";
import { rawPathFor, writeImported } from "./write.js";

const fixture = (name: string): unknown[] => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8")) as unknown[];
const conversations = fixture("claude-sample.json");
const projects = fixture("claude-projects-sample.json");

describe("parseClaudeExport", () => {
  it("reports counts that add up to the export", async () => {
    const { threads, report } = await parseClaudeExport({ conversations, projects });
    expect(report).toEqual({ conversations: 4, threads: 2, empty: 1, unreadable: 1, messages: 4, offPath: 2, attachments: 1, filesNotIncluded: 1 });
    expect(report.threads + report.empty + report.unreadable).toBe(report.conversations);
    expect(threads.map((t) => t.thread.title)).toEqual(["Kitchen quotes", "Plan a trip to Lisbon"]);
  });

  it("keeps the provider's ids, times and model, and maps projects to lanes", async () => {
    const [kitchen] = (await parseClaudeExport({ conversations, projects })).threads;
    const thread = kitchen?.thread;
    expect(thread).toMatchObject({ source: "claude-export", sourceId: "conv-plain", started: "2025-03-02T18:04:11.000000Z", lane: "home-renovation" });
    expect(thread?.id).toMatch(/^th_/);
    expect(thread?.turns.map((t) => [t.id, t.role, t.at, t.model, t.meta?.sourceId])).toEqual([
      ["m1", "user", "2025-03-02T18:04:11.000000Z", undefined, "m-1"],
      ["m2", "assistant", "2025-03-02T18:04:30.000000Z", "claude-sonnet-4-5", "m-2"],
    ]);
  });

  it("gives the same id on every import", async () => {
    const a = await parseClaudeExport({ conversations });
    const b = await parseClaudeExport({ conversations: structuredClone(conversations) });
    expect(a.threads.map((t) => t.thread.id)).toEqual(b.threads.map((t) => t.thread.id));
  });

  it("follows the current path of an edited conversation", async () => {
    const edited = (await parseClaudeExport({ conversations })).threads[1]?.thread;
    expect(edited?.turns.map((t) => t.text)).toEqual(["Plan a trip to Lisbon\nwith two days", "Day one: Alfama."]);
  });

  it("renders artifacts and attachments readably and leaves thinking and tool results out", async () => {
    const kitchen = (await parseClaudeExport({ conversations })).threads[0]?.thread;
    const [user, assistant] = kitchen?.turns ?? [];
    expect(user?.text).toContain("**Attached: quote.txt**\n\n````\nLine with ``` fences inside");
    expect(user?.text).toContain("*Attached photo.jpg (not included in the export).*");
    expect(assistant?.text).toContain("**Budget table**\n\n```markdown\n| item | cost |");
    expect(assistant?.text).toContain("*Used web_search.*");
    expect(assistant?.text).not.toContain("private reasoning");
    expect(assistant?.text).not.toContain("result");
  });

  it("round-trips hostile text through the thread file exactly", async () => {
    for (const { thread } of (await parseClaudeExport({ conversations })).threads) {
      expect(parseThread(serializeThread(thread))).toEqual(thread);
    }
  });

  it("recognises conversation and project files by shape", () => {
    expect(looksLikeClaudeConversations(conversations)).toBe(true);
    expect(looksLikeClaudeConversations(projects)).toBe(false);
    expect(looksLikeClaudeProjects(projects)).toBe(true);
    expect(looksLikeClaudeProjects(conversations)).toBe(false);
    expect(looksLikeClaudeConversations({ fake: true })).toBe(false);
  });
});

describe("writeImported", () => {
  const setup = () => {
    const fs = memoryVaultFs();
    return { fs, receipts: createReceiptLog(fs, { lock: memoryLock() }) };
  };

  it("writes each thread with its raw JSON beside it, and a second import changes zero files", async () => {
    const { fs, receipts } = setup();
    const { threads } = await parseClaudeExport({ conversations, projects });
    expect(await writeImported(fs, receipts, threads, { source: "claude-export" })).toMatchObject({ created: 2, updated: 0, unchanged: 0 });
    const files = await fs.list("threads");
    expect(files).toHaveLength(4);
    const snapshot = await Promise.all(files.map(async (f) => [f, await fs.read(f)]));

    const again = await parseClaudeExport({ conversations: structuredClone(conversations), projects });
    expect(await writeImported(fs, receipts, again.threads, { source: "claude-export" })).toMatchObject({ created: 0, updated: 0, unchanged: 2 });
    expect(await Promise.all(files.map(async (f) => [f, await fs.read(f)]))).toEqual(snapshot);
    expect((await receipts.list()).map((r) => [r.kind, r.targets.length])).toEqual([["import", 2], ["import", 0]]);
    expect((await receipts.verify()).ok).toBe(true);
  });

  it("keeps a user's edit when the export did not change, and moves a renamed conversation", async () => {
    const { fs, receipts } = setup();
    const { threads } = await parseClaudeExport({ conversations });
    await writeImported(fs, receipts, threads, { source: "claude-export" });
    const kitchen = threads[0]!.thread;
    const path = threadPath(kitchen);
    await fs.write(path, `${(await fs.read(path)) ?? ""}\nMy own note.\n`);
    await writeImported(fs, receipts, (await parseClaudeExport({ conversations })).threads, { source: "claude-export" });
    expect(await fs.read(path)).toContain("My own note.");

    const renamed = structuredClone(conversations) as { name?: string }[];
    renamed[0]!.name = "Kitchen quotes, final";
    const next = await parseClaudeExport({ conversations: renamed });
    expect(await writeImported(fs, receipts, next.threads, { source: "claude-export" })).toMatchObject({ updated: 1, unchanged: 1 });
    const moved = threadPath(next.threads[0]!.thread);
    expect(moved).not.toBe(path);
    expect(await fs.read(path)).toBeUndefined();
    expect(await fs.read(rawPathFor(path))).toBeUndefined();
    expect(parseThread((await fs.read(moved)) ?? "").id).toBe(kitchen.id);
  });
});
