// The behaviour every VaultFs must share. Run it against each adapter, so the
// in-memory vault used in tests can never drift from the real disk.
import { describe, expect, it } from "vitest";
import { type VaultFs, VaultPathError } from "../adapters.js";

export function vaultFsContract(name: string, make: () => Promise<VaultFs>): void {
  describe(`VaultFs contract: ${name}`, () => {
    it("reads what it wrote, and undefined for a missing file", async () => {
      const fs = await make();
      await fs.write("threads/2026/11/a.md", "hello\nworld");
      expect(await fs.read("threads/2026/11/a.md")).toBe("hello\nworld");
      expect(await fs.read("threads/2026/11/missing.md")).toBeUndefined();
      expect(await fs.read("nowhere/at/all.md")).toBeUndefined();
    });

    it("replaces a file on write and grows it on append", async () => {
      const fs = await make();
      await fs.write("notes/n.md", "one");
      await fs.write("notes/n.md", "two");
      await fs.append("receipts/2026-11.jsonl", "a\n");
      await fs.append("receipts/2026-11.jsonl", "b\n");
      expect(await fs.read("notes/n.md")).toBe("two");
      expect(await fs.read("receipts/2026-11.jsonl")).toBe("a\nb\n");
    });

    it("keeps UTF-8 exactly, including accents and emoji in content", async () => {
      const fs = await make();
      const text = "Küchenbudget: 18 000 €, café, 日本語, \u{1F600}";
      await fs.write("notes/utf8.md", text);
      expect(await fs.read("notes/utf8.md")).toBe(text);
    });

    it("lists files recursively, sorted, with vault-relative paths", async () => {
      const fs = await make();
      await fs.write("threads/b.md", "");
      await fs.write("threads/2026/a.md", "");
      await fs.write("notes/x.md", "");
      expect(await fs.list("threads")).toEqual(["threads/2026/a.md", "threads/b.md"]);
      expect(await fs.list("empty")).toEqual([]);
    });

    it("removes files, and removing a missing file is fine", async () => {
      const fs = await make();
      await fs.write("notes/gone.md", "x");
      await fs.remove("notes/gone.md");
      await fs.remove("notes/gone.md");
      expect(await fs.read("notes/gone.md")).toBeUndefined();
    });

    it("refuses paths that escape the vault or that Windows cannot store", async () => {
      const fs = await make();
      for (const bad of ["../x.md", "/etc/passwd", "notes/../../x.md", "notes/CON", "notes/a:b.md"]) {
        await expect(fs.write(bad, "x")).rejects.toThrow(VaultPathError);
        await expect(fs.read(bad)).rejects.toThrow(VaultPathError);
      }
    });

    it("refuses a second file that differs only by case", async () => {
      const fs = await make();
      await fs.write("notes/Budget.md", "A");
      await expect(fs.write("notes/budget.md", "B")).rejects.toThrow(VaultPathError);
      expect(await fs.read("notes/Budget.md")).toBe("A");
    });
  });
}
