import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseThread, serializeThread } from "../thread.js";
import { looksLikeChatGptConversations, parseChatGptExport } from "./chatgpt.js";
import { looksLikeClaudeConversations } from "./claude.js";

const conversations = JSON.parse(readFileSync(new URL("./fixtures/chatgpt-sample.json", import.meta.url), "utf8")) as unknown[];

describe("parseChatGptExport", () => {
  it("reports counts that add up to the export", async () => {
    const { threads, report } = await parseChatGptExport(conversations);
    expect(report).toMatchObject({ conversations: 4, threads: 2, empty: 1, unreadable: 1, filesNotIncluded: 1 });
    expect(report.threads + report.empty + report.unreadable).toBe(report.conversations);
    expect(threads.map((t) => t.thread.title)).toEqual(["Sourdough schedule", "Sum 2 and 3 in Python"]);
  });

  it("follows the current node, keeps provider times and models, and skips hidden system messages", async () => {
    const thread = (await parseChatGptExport(conversations)).threads[0]!.thread;
    expect(thread).toMatchObject({ source: "chatgpt-export", sourceId: "gpt-plain", started: "2025-03-03T11:06:40.500Z" });
    expect(thread.turns.map((t) => [t.id, t.role, t.model, t.text])).toEqual([
      ["m1", "user", undefined, "When should I feed the starter?"],
      ["m2", "assistant", "gpt-4o", "Feed it 8 to 12 hours before baking."],
      ["m3", "user", undefined, "*Attached an image (not included in the conversation text).*\n\nIs this crumb right?"],
      ["m4", "assistant", "gpt-4o", "The crumb looks open; good."],
    ]);
    expect(thread.turns[0]?.meta).toEqual({ sourceId: "n1" });
  });

  it("joins a tool call and its answer into one reply, leaving reasoning and tool output out", async () => {
    const thread = (await parseChatGptExport(conversations)).threads[1]!.thread;
    expect(thread.turns.map((t) => [t.role, t.text])).toEqual([
      ["user", "Sum 2 and 3 in Python"],
      ["assistant", "*Used python.*\n\nThe sum is 5."],
    ]);
    expect(thread.turns[1]?.model).toBe("o3");
  });

  it("gives stable ids, round-trips, and is told apart from Claude by shape", async () => {
    const a = await parseChatGptExport(conversations);
    const b = await parseChatGptExport(structuredClone(conversations));
    expect(a.threads.map((t) => t.thread.id)).toEqual(b.threads.map((t) => t.thread.id));
    for (const { thread } of a.threads) expect(parseThread(serializeThread(thread))).toEqual(thread);
    expect(looksLikeChatGptConversations(conversations)).toBe(true);
    expect(looksLikeClaudeConversations(conversations)).toBe(false);
  });
});
