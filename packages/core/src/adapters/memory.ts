import { type VaultFs, VaultPathError, pathIdentity, vaultPath } from "../adapters.js";

/**
 * An in-memory vault for tests and experiments. It is deliberately as strict
 * as the strictest real disk: writing a path that differs from an existing one
 * only by case fails, where macOS and Windows would silently overwrite.
 */
export function memoryVaultFs(initial: Record<string, string> = {}): VaultFs & {
  snapshot(): Record<string, string>;
} {
  const files = new Map<string, string>();
  const identities = new Map<string, string>();

  function claim(path: string): string {
    const p = vaultPath(path);
    if (p === "") throw new VaultPathError(path);
    const existing = identities.get(pathIdentity(p));
    if (existing !== undefined && existing !== p) {
      throw new VaultPathError(`${path} (same file as ${existing} on a case-insensitive disk)`);
    }
    identities.set(pathIdentity(p), p);
    return p;
  }

  for (const [path, content] of Object.entries(initial)) files.set(claim(path), content);

  return {
    async read(path) {
      return files.get(vaultPath(path));
    },
    async write(path, content) {
      files.set(claim(path), content);
    },
    async append(path, content) {
      const p = claim(path);
      files.set(p, (files.get(p) ?? "") + content);
    },
    async list(dir) {
      const d = vaultPath(dir);
      const prefix = d === "" ? "" : `${d}/`;
      return [...files.keys()].filter((p) => p.startsWith(prefix)).sort();
    },
    async remove(path) {
      const p = vaultPath(path);
      files.delete(p);
      if (identities.get(pathIdentity(p)) === p) identities.delete(pathIdentity(p));
    },
    snapshot() {
      return Object.fromEntries([...files.entries()].sort(([a], [b]) => (a < b ? -1 : 1)));
    },
  };
}
