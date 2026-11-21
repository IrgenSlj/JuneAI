import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { check, contentProblems, problemsWith } from "./privacy-check.mjs";

describe("privacy check", () => {
  it("refuses real data, exports, archives and vault files", () => {
    for (const path of [
      "lab/private/notes.md",
      "conversations.json",
      "lab/scripts/users.json",
      "data-2026-11-15.zip",
      "receipts/2026-11.jsonl",
      "vault/.june/index.sqlite",
      "threads/2026/09/a.raw.json",
    ]) {
      expect(problemsWith(path, 10), path).not.toEqual([]);
    }
    expect(problemsWith("packages/core/src/big.ts", 2_000_000)).not.toEqual([]);
  });

  it("allows code, docs and small redacted fixtures", () => {
    for (const path of ["packages/core/src/thread.ts", "lab/001-claude-export.md", "packages/core/src/fixtures/receipts/2026-11.jsonl", "pnpm-lock.yaml"]) {
      expect(problemsWith(path, 10), path).toEqual([]);
    }
  });

  it("finds nothing private in this repository", () => {
    expect(check()).toEqual([]);
  });

  it("reads staged content: export fields and the personal denylist", () => {
    const repo = mkdtempSync(join(tmpdir(), "june-privacy-"));
    try {
      const git = (...args) => execFileSync("git", args, { cwd: repo, encoding: "utf8" });
      git("init", "-q");
      mkdirSync(join(repo, "lab/private"), { recursive: true });
      mkdirSync(join(repo, "src/fixtures"), { recursive: true });
      writeFileSync(join(repo, "lab/private/denylist.txt"), "# comment\nAda Example\n");
      writeFileSync(join(repo, "notes.md"), "Met ADA EXAMPLE today.\n");
      writeFileSync(join(repo, "dump.json"), '{"chat_messages": []}');
      writeFileSync(join(repo, "src/fixtures/claude.json"), '{"chat_messages": []}');
      writeFileSync(join(repo, "importer.ts"), 'const key = "chat_messages";\n');
      git("add", "notes.md", "dump.json", "src/fixtures/claude.json", "importer.ts");
      const found = contentProblems(repo);
      expect(found).toContain("notes.md contains a string from lab/private/denylist.txt");
      expect(found).toContain("dump.json contains fields from a real provider export");
      expect(found.join("\n")).not.toContain("Ada");
      expect(found.some((line) => line.startsWith("src/fixtures/") || line.startsWith("importer.ts"))).toBe(false);
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });
});
