import { describe, expect, it } from "vitest";
import { check, problemsWith } from "./privacy-check.mjs";

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
});
