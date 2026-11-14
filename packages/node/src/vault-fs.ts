// The vault on disk. Same contract as the in-memory vault (core's
// vault-fs-contract), plus what only a real disk needs:
// - Writes are atomic: a temporary file in the same folder, then a rename, so
//   a crash leaves the old file or the new one, never half of one.
// - Nothing resolves outside the vault, including through a symlink that
//   points elsewhere.
// - A name that differs from an existing one only by case is refused on every
//   OS, so a vault made on Linux still opens on a Mac.
import { randomBytes } from "node:crypto";
import { appendFile, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import { type VaultFs, VaultPathError, vaultPath } from "@june/core";

function isMissing(error: unknown): boolean {
  return (error as NodeJS.ErrnoException)?.code === "ENOENT" || (error as NodeJS.ErrnoException)?.code === "ENOTDIR";
}

/**
 * Windows refuses to rename over a file another handle still has open (EPERM,
 * EACCES, EBUSY): a concurrent write, an antivirus scan, the search indexer.
 * Those locks are brief, so retry with a short backoff before giving up.
 */
async function renameWithRetry(from: string, to: string): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await rename(from, to);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (attempt >= 8 || (code !== "EPERM" && code !== "EACCES" && code !== "EBUSY")) throw error;
      await new Promise((resolve) => setTimeout(resolve, 10 * 2 ** attempt));
    }
  }
}

export async function nodeVaultFs(root: string): Promise<VaultFs> {
  await mkdir(root, { recursive: true });
  const realRoot = await realpath(root);
  // Writes to one path run one after another within this process, so June
  // never races itself on a file (the case Windows handles worst).
  const queues = new Map<string, Promise<unknown>>();
  function serialized<T>(key: string, task: () => Promise<T>): Promise<T> {
    const result = (queues.get(key) ?? Promise.resolve()).then(task);
    const settled = result.catch(() => undefined);
    queues.set(key, settled);
    void settled.then(() => {
      if (queues.get(key) === settled) queues.delete(key);
    });
    return result;
  }

  /** Absolute path for a vault path, after checking that what exists of it stays inside the vault. */
  async function resolveInside(path: string): Promise<string> {
    const relativePath = vaultPath(path);
    const absolute = relativePath === "" ? realRoot : join(realRoot, ...relativePath.split("/"));
    // Walk up to the deepest part that exists and make sure its real location is inside.
    let probe = absolute;
    for (;;) {
      try {
        const real = await realpath(probe);
        if (real !== realRoot && !real.startsWith(realRoot + sep)) throw new VaultPathError(`${path} (leaves the vault)`);
        return absolute;
      } catch (error) {
        if (!isMissing(error)) throw error;
        const parent = dirname(probe);
        if (parent === probe) return absolute;
        probe = parent;
      }
    }
  }

  /** Refuse a new name that only differs by case from one already in its folder. */
  async function checkCase(absolute: string, path: string): Promise<void> {
    const name = absolute.slice(dirname(absolute).length + 1);
    let siblings: string[];
    try {
      siblings = await readdir(dirname(absolute));
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    const clash = siblings.find((s) => s !== name && s.toLowerCase() === name.toLowerCase());
    if (clash !== undefined) throw new VaultPathError(`${path} (same file as ${clash} on a case-insensitive disk)`);
  }

  return {
    async read(path) {
      const absolute = await resolveInside(path);
      try {
        return await readFile(absolute, "utf8");
      } catch (error) {
        if (isMissing(error) || (error as NodeJS.ErrnoException).code === "EISDIR") return undefined;
        throw error;
      }
    },

    async write(path, content) {
      const absolute = await resolveInside(path);
      if (absolute === realRoot) throw new VaultPathError(path);
      await mkdir(dirname(absolute), { recursive: true });
      await checkCase(absolute, path);
      await serialized(absolute.toLowerCase(), async () => {
        const tmp = join(dirname(absolute), `.${randomBytes(6).toString("hex")}.june-tmp`);
        try {
          await writeFile(tmp, content, "utf8");
          await renameWithRetry(tmp, absolute);
        } catch (error) {
          await rm(tmp, { force: true });
          throw error;
        }
      });
    },

    async append(path, content) {
      const absolute = await resolveInside(path);
      if (absolute === realRoot) throw new VaultPathError(path);
      await mkdir(dirname(absolute), { recursive: true });
      await checkCase(absolute, path);
      await serialized(absolute.toLowerCase(), () => appendFile(absolute, content, "utf8"));
    },

    async list(dir) {
      const absolute = await resolveInside(dir);
      let entries;
      try {
        entries = await readdir(absolute, { recursive: true, withFileTypes: true });
      } catch (error) {
        if (isMissing(error)) return [];
        throw error;
      }
      return entries
        .filter((e) => e.isFile() && !e.name.endsWith(".june-tmp"))
        .map((e) => relative(realRoot, join(e.parentPath, e.name)).split(sep).join("/"))
        .sort();
    },

    async remove(path) {
      const absolute = await resolveInside(path);
      if (absolute === realRoot) throw new VaultPathError(path);
      await rm(absolute, { force: true });
    },
  };
}
