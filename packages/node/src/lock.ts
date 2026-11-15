// A lock shared by every process that opens the same vault, so the daemon, a
// CLI and an importer cannot interleave a read-then-append (the receipts chain
// would fork). The lock is a file in .june/locks/, created with O_EXCL, which
// every OS makes atomic. It holds the owner's pid, so a lock left by a crashed
// process is taken over instead of blocking June forever.
//
// Known limit: if two processes find the same dead lock at the same moment,
// one can remove the other's fresh lock. That needs a crash plus a three-way
// race inside a few milliseconds; the receipts verify would then report it.
import { randomBytes } from "node:crypto";
import { mkdir, open, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import type { VaultLock } from "@june/core";

export class LockTimeoutError extends Error {
  constructor(name: string, holder: string) {
    super(`the vault lock "${name}" is held by ${holder}; gave up waiting`);
    this.name = "LockTimeoutError";
  }
}

interface Holder {
  pid: number;
  at: string;
  token: string;
}

export interface FileLockOptions {
  /** How long to wait for another process before failing. */
  timeoutMs?: number;
  /** A lock older than this is taken over even if its pid looks alive (pids get reused). */
  staleMs?: number;
}

function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: the process exists but belongs to someone else.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

async function readHolder(path: string): Promise<Holder | undefined> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as Partial<Holder>;
    if (typeof parsed.pid === "number" && typeof parsed.at === "string" && typeof parsed.token === "string") {
      return parsed as Holder;
    }
  } catch {
    // Missing, or written by a process that died before finishing the write.
  }
  return undefined;
}

export function nodeFileLock(root: string, options: FileLockOptions = {}): VaultLock {
  const timeoutMs = options.timeoutMs ?? 10_000;
  const staleMs = options.staleMs ?? 30_000;
  const dir = join(root, ".june", "locks");
  // Callers in this process queue here first, so they never spin against each other on disk.
  const queues = new Map<string, Promise<unknown>>();

  async function acquire(name: string): Promise<{ path: string; token: string }> {
    if (!/^[a-z0-9-]+$/.test(name)) throw new Error(`invalid lock name: ${name}`);
    await mkdir(dir, { recursive: true });
    const path = join(dir, `${name}.lock`);
    const token = randomBytes(8).toString("hex");
    const deadline = Date.now() + timeoutMs;
    let unreadableSince: number | undefined;
    for (let attempt = 0; ; attempt++) {
      try {
        const handle = await open(path, "wx");
        try {
          await handle.writeFile(JSON.stringify({ pid: process.pid, at: new Date().toISOString(), token } satisfies Holder));
        } finally {
          await handle.close();
        }
        return { path, token };
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
      const holder = await readHolder(path);
      if (holder === undefined) {
        // Being written right now, or left half-written by a crash: give it a moment.
        unreadableSince ??= Date.now();
        if (Date.now() - unreadableSince > 1_000) await rm(path, { force: true });
      } else {
        unreadableSince = undefined;
        if (!isAlive(holder.pid) || Date.now() - Date.parse(holder.at) > staleMs) {
          await rm(path, { force: true });
          continue;
        }
      }
      if (Date.now() > deadline) throw new LockTimeoutError(name, holder === undefined ? "an unknown process" : `process ${holder.pid}`);
      await new Promise((resolve) => setTimeout(resolve, Math.min(50, 2 + attempt * 2) + Math.random() * 5));
    }
  }

  async function release(path: string, token: string): Promise<void> {
    // Remove it only if it is still ours (it may have been taken over as stale).
    if ((await readHolder(path))?.token === token) await rm(path, { force: true });
  }

  return {
    exclusive<T>(name: string, task: () => Promise<T>): Promise<T> {
      const run = async () => {
        const { path, token } = await acquire(name);
        try {
          return await task();
        } finally {
          await release(path, token);
        }
      };
      const result = (queues.get(name) ?? Promise.resolve()).then(run);
      const settled = result.catch(() => undefined);
      queues.set(name, settled);
      void settled.then(() => {
        if (queues.get(name) === settled) queues.delete(name);
      });
      return result;
    },
  };
}
