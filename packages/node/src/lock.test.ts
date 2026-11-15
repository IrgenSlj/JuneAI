import { spawn } from "node:child_process";
import { mkdtemp, readdir, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createReceiptLog } from "@june/core";
import { build } from "esbuild";
import { beforeAll, describe, expect, it } from "vitest";
import { LockTimeoutError, nodeFileLock } from "./lock.js";
import { nodeVaultFs } from "./vault-fs.js";

const freshRoot = () => mkdtemp(join(tmpdir(), "june-lock-"));

function run(script: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script, ...args], { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    child.stderr.on("data", (chunk) => (err += chunk));
    child.once("error", reject);
    child.once("exit", (code) => (code === 0 ? resolve() : reject(new Error(`writer exited ${code}: ${err}`))));
  });
}

describe("node file lock", () => {
  let writer: string;
  beforeAll(async () => {
    writer = join(await mkdtemp(join(tmpdir(), "june-writer-")), "receipt-writer.mjs");
    await build({
      entryPoints: [join(import.meta.dirname, "fixtures/receipt-writer.ts")],
      bundle: true,
      platform: "node",
      format: "esm",
      outfile: writer,
      logLevel: "silent",
    });
  });

  it("keeps one receipt chain when three processes append at once", { timeout: 60_000 }, async () => {
    const root = await freshRoot();
    await Promise.all(["a", "b", "c"].map((actor) => run(writer, [root, "40", actor])));
    const log = createReceiptLog(await nodeVaultFs(root), { lock: nodeFileLock(root) });
    expect(await log.verify()).toEqual({ ok: true, count: 120 });
    expect(await readdir(join(root, ".june", "locks"))).toEqual([]);
  });

  it("runs tasks one at a time within a process, and releases after a failure", async () => {
    const lock = nodeFileLock(await freshRoot());
    let inside = 0;
    let most = 0;
    const task = async () => {
      inside++;
      most = Math.max(most, inside);
      await new Promise((r) => setTimeout(r, 2));
      inside--;
    };
    await expect(lock.exclusive("x", async () => { throw new Error("boom"); })).rejects.toThrow("boom");
    await Promise.all(Array.from({ length: 20 }, () => lock.exclusive("x", task)));
    expect(most).toBe(1);
  });

  it("takes over a lock left by a process that died", async () => {
    const root = await freshRoot();
    await mkdir(join(root, ".june", "locks"), { recursive: true });
    // A pid far above any real one stands in for a crashed process.
    await writeFile(join(root, ".june", "locks", "receipts.lock"), JSON.stringify({ pid: 2 ** 22 + 7, at: new Date().toISOString(), token: "old" }));
    expect(await nodeFileLock(root).exclusive("receipts", async () => "ran")).toBe("ran");
  });

  it("gives up with a clear error when a live process holds the lock too long", async () => {
    const root = await freshRoot();
    await mkdir(join(root, ".june", "locks"), { recursive: true });
    await writeFile(join(root, ".june", "locks", "receipts.lock"), JSON.stringify({ pid: process.pid, at: new Date().toISOString(), token: "held" }));
    await expect(nodeFileLock(root, { timeoutMs: 200 }).exclusive("receipts", async () => "ran")).rejects.toThrow(LockTimeoutError);
  });
});
