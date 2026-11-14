import { describe, expect, it } from "vitest";
import { VaultPathError, vaultPath } from "./adapters.js";
import { memoryVaultFs } from "./adapters/memory.js";

describe("vaultPath", () => {
  it("accepts and normalises vault-relative paths", () => {
    expect(vaultPath("notes/budget.md")).toBe("notes/budget.md");
    expect(vaultPath("threads/2026/")).toBe("threads/2026");
    expect(vaultPath("")).toBe("");
  });

  it("normalises accents so a Mac and a Windows machine name the same file", () => {
    const decomposed = "notes/cafe\u0301.md"; // e + combining accent, as macOS may store it
    const composed = "notes/caf\u00e9.md";
    expect(vaultPath(decomposed)).toBe(composed);
  });

  it.each([
    "/etc/passwd",
    "../outside.md",
    "notes/../../x",
    "C:/x",
    "notes\\x",
    "a//b",
    "./a",
    "notes/CON",
    "notes/aux.md",
    "notes/lpt1.txt",
    "notes/what?.md",
    "notes/a:b.md",
    "notes/trailing.",
    "notes/trailing ",
    "notes/tab\there.md",
  ])(
    "rejects %s",
    (path) => {
      expect(() => vaultPath(path)).toThrow(VaultPathError);
    },
  );
});

describe("memoryVaultFs", () => {
  it("reads, writes, appends, lists and removes", async () => {
    const fs = memoryVaultFs({ "notes/a.md": "A" });
    await fs.write("threads/2026/11/t.md", "T");
    await fs.append("receipts/2026-11.jsonl", "1\n");
    await fs.append("receipts/2026-11.jsonl", "2\n");
    expect(await fs.read("notes/a.md")).toBe("A");
    expect(await fs.read("missing.md")).toBeUndefined();
    expect(await fs.read("receipts/2026-11.jsonl")).toBe("1\n2\n");
    expect(await fs.list("threads")).toEqual(["threads/2026/11/t.md"]);
    expect(await fs.list("")).toHaveLength(3);
    await fs.remove("notes/a.md");
    await fs.remove("notes/a.md");
    expect(await fs.list("notes")).toEqual([]);
  });

  it("fails on a case-only collision instead of silently overwriting", async () => {
    const fs = memoryVaultFs({ "notes/Budget.md": "A" });
    await expect(fs.write("notes/budget.md", "B")).rejects.toThrow(VaultPathError);
    await fs.write("notes/Budget.md", "C");
    expect(await fs.read("notes/Budget.md")).toBe("C");
    await fs.remove("notes/Budget.md");
    await fs.write("notes/budget.md", "D");
    expect(await fs.list("notes")).toEqual(["notes/budget.md"]);
  });

  it("refuses paths that escape the vault", async () => {
    const fs = memoryVaultFs();
    await expect(fs.write("../x.md", "x")).rejects.toThrow(VaultPathError);
  });
});
