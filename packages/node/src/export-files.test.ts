import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { findClaudeData } from "@june/core";
import { strToU8, zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { ExportReadError, documentsIn, readExportPath } from "./export-files.js";

const conversation = { uuid: "c1", name: "Hello", created_at: "2025-01-01T00:00:00Z", chat_messages: [] };
const project = { uuid: "p1", name: "Home", docs: [] };
const json = (value: unknown) => strToU8(JSON.stringify(value));

describe("export files", () => {
  it("reads the classic single zip and finds conversations and projects by shape", () => {
    const zip = zipSync({ "conversations.json": json([conversation]), "projects.json": json([project]), "users.json": json([{ uuid: "u" }]), "__MACOSX/._x.json": json([]), "readme.txt": strToU8("hi") });
    const docs = documentsIn("data-2026.zip", zip);
    expect(docs.map((d) => d.name)).toEqual(["data-2026.zip/conversations.json", "data-2026.zip/projects.json", "data-2026.zip/users.json"]);
    expect(findClaudeData(docs)).toMatchObject({ conversations: [conversation], projects: [project], manifestOnly: false, ignored: ["data-2026.zip/users.json"] });
  });

  it("reads the 2026 multi-part export from a folder, including JSON Lines and nested zips", async () => {
    const dir = await mkdtemp(join(tmpdir(), "june-export-"));
    await writeFile(join(dir, "conversations-000.zip"), zipSync({ "conversations.jsonl": strToU8(`${JSON.stringify(conversation)}\n${JSON.stringify({ ...conversation, uuid: "c2" })}\n`) }));
    await writeFile(join(dir, "projects-000.zip"), zipSync({ "inner.zip": zipSync({ "projects.json": json([project]) }) }));
    await writeFile(join(dir, "manifest.json"), JSON.stringify({ data_files: [] }));
    await mkdir(join(dir, ".hidden"));
    const found = findClaudeData(await readExportPath(dir));
    expect(found.conversations.map((c) => (c as { uuid: string }).uuid)).toEqual(["c1", "c2"]);
    expect(found.projects).toEqual([project]);
    expect(found.manifestOnly).toBe(false);
  });

  it("recognises a manifest with no conversations", () => {
    expect(findClaudeData(documentsIn("is_claude_history.json", json({ data_files: [{ filename: "conversations-000.zip" }] }))).manifestOnly).toBe(true);
  });

  it("says plainly when a file is not JSON or not a zip", async () => {
    expect(() => documentsIn("conversations.json", strToU8("{nope"))).toThrow(ExportReadError);
    expect(() => documentsIn("broken.zip", new Uint8Array([0x50, 0x4b, 1, 2, 3]))).toThrow("not a readable zip");
    await expect(readExportPath("/nonexistent/june")).rejects.toThrow("nothing found");
  });
});
