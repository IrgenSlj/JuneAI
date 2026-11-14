import { type VaultFs, vaultPath } from "../adapters.js";

/** An in-memory vault for tests and experiments. */
export function memoryVaultFs(initial: Record<string, string> = {}): VaultFs & {
  snapshot(): Record<string, string>;
} {
  const files = new Map<string, string>();
  for (const [path, content] of Object.entries(initial)) files.set(vaultPath(path), content);

  return {
    async read(path) {
      return files.get(vaultPath(path));
    },
    async write(path, content) {
      const p = vaultPath(path);
      if (p === "") throw new Error("cannot write the vault root");
      files.set(p, content);
    },
    async append(path, content) {
      const p = vaultPath(path);
      if (p === "") throw new Error("cannot write the vault root");
      files.set(p, (files.get(p) ?? "") + content);
    },
    async list(dir) {
      const d = vaultPath(dir);
      const prefix = d === "" ? "" : `${d}/`;
      return [...files.keys()].filter((p) => p.startsWith(prefix)).sort();
    },
    async remove(path) {
      files.delete(vaultPath(path));
    },
    snapshot() {
      return Object.fromEntries([...files.entries()].sort(([a], [b]) => (a < b ? -1 : 1)));
    },
  };
}
