import { describe, expect, it } from "vitest";
import { VaultPathError, vaultPath } from "./adapters.js";
import { memoryVaultFs } from "./adapters/memory.js";

describe("vaultPath", () => {
  it("accepts and normalises vault-relative paths", () => {
    expect(vaultPath("notes/budget.md")).toBe("notes/budget.md");
    expect(vaultPath("threads/2026/")).toBe("threads/2026");
    expect(vaultPath("")).toBe("");
  });

  it.each(["/etc/passwd", "../outside.md", "notes/../../x", "C:/x", "notes\\x", "a//b", "./a"])(
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

  it("refuses paths that escape the vault", async () => {
    const fs = memoryVaultFs();
    await expect(fs.write("../x.md", "x")).rejects.toThrow(VaultPathError);
  });
});
