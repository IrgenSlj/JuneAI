import { mkdtemp, readdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { VaultPathError } from "@june/core";
import { vaultFsContract } from "@june/core/testing";
import { describe, expect, it } from "vitest";
import { nodeVaultFs } from "./vault-fs.js";

const freshRoot = () => mkdtemp(join(tmpdir(), "june-vault-"));

vaultFsContract("node", async () => nodeVaultFs(await freshRoot()));

describe("node vault, disk-only behaviour", () => {
  it("writes atomically and leaves no temporary files behind", async () => {
    const root = await freshRoot();
    const fs = await nodeVaultFs(root);
    await Promise.all(Array.from({ length: 50 }, (_, i) => fs.write("notes/same.md", `version ${i}`)));
    expect(await fs.read("notes/same.md")).toMatch(/^version \d+$/);
    expect(await readdir(join(root, "notes"))).toEqual(["same.md"]);
  });

  it("refuses to follow a symlink out of the vault", async () => {
    const root = await freshRoot();
    const outside = await freshRoot();
    await writeFile(join(outside, "secret.md"), "outside");
    await symlink(outside, join(root, "escape"));
    const fs = await nodeVaultFs(root);
    await expect(fs.read("escape/secret.md")).rejects.toThrow(VaultPathError);
    await expect(fs.write("escape/new.md", "x")).rejects.toThrow(VaultPathError);
  });

  it("finds files written by other programs, as plain files", async () => {
    const root = await freshRoot();
    await writeFile(join(root, "edited-in-obsidian.md"), "by hand");
    const fs = await nodeVaultFs(root);
    expect(await fs.list("")).toEqual(["edited-in-obsidian.md"]);
  });
});
