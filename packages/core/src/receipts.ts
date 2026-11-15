// The receipts log: an append-only, hash-chained record of everything June
// did on the user's behalf (a send, a memory read or write, a tool call).
// A line proves that something happened and to what; it never holds content.
//
// One JSON object per line in receipts/YYYY-MM.jsonl. Each line's hash covers
// the previous line's hash, so removing or editing a line breaks the chain,
// and the chain runs across month files.
import type { VaultFs, VaultLock } from "./adapters.js";
import { type Json, canonicalJson, sha256Hex } from "./hash.js";

export type ReceiptKind = "send" | "memory-read" | "memory-write" | "tool-call" | "import" | "export" | "mcp-read";

export interface ReceiptInput {
  kind: ReceiptKind;
  /** Who acted: "june", "user", or a client name such as "mcp:claude-desktop". */
  actor: string;
  /** Ids this receipt is about (threads, turns, notes, plans). */
  targets: string[];
  /** Hashed, then discarded. Put the content here; it is never written. */
  payload: Json;
  /** Shape only: model, host, byte counts. Never message text. */
  meta?: { [key: string]: string | number | boolean };
}

export interface Receipt {
  seq: number;
  at: string;
  kind: ReceiptKind;
  actor: string;
  targets: string[];
  payloadHash: string;
  meta?: { [key: string]: string | number | boolean };
  prev: string;
  hash: string;
}

export interface VerifyResult {
  ok: boolean;
  count: number;
  /** The first line that fails, when ok is false. */
  brokenAt?: { file: string; line: number; reason: string };
}

const GENESIS = "0".repeat(64);
const DIR = "receipts";

async function hashOf(receipt: Omit<Receipt, "hash">): Promise<string> {
  return sha256Hex(canonicalJson(receipt as unknown as Json));
}

export interface ReceiptLogOptions {
  clock?: () => Date;
  /**
   * Held around reading the tail and appending, so two processes cannot both
   * link to the same previous line. Without it, appends are serialised only
   * inside this process, which is enough for tests and the in-memory vault.
   */
  lock?: VaultLock;
}

export function createReceiptLog(fs: VaultFs, options: ReceiptLogOptions = {}) {
  const clock = options.clock ?? (() => new Date());
  const lock = options.lock;
  let queue: Promise<unknown> = Promise.resolve();

  async function files(): Promise<string[]> {
    return (await fs.list(DIR)).filter((p) => /^receipts\/\d{4}-\d{2}\.jsonl$/.test(p)).sort();
  }

  async function tail(): Promise<{ seq: number; hash: string }> {
    const all = await files();
    for (const file of all.reverse()) {
      const lines = ((await fs.read(file)) ?? "").split("\n").filter((l) => l !== "");
      const last = lines.at(-1);
      if (last !== undefined) {
        const receipt = JSON.parse(last) as Receipt;
        return { seq: receipt.seq, hash: receipt.hash };
      }
    }
    return { seq: 0, hash: GENESIS };
  }

  async function appendNow(input: ReceiptInput): Promise<Receipt> {
    const previous = await tail();
    const now = clock();
    const body: Omit<Receipt, "hash"> = {
      seq: previous.seq + 1,
      at: now.toISOString(),
      kind: input.kind,
      actor: input.actor,
      targets: input.targets,
      payloadHash: await sha256Hex(canonicalJson(input.payload)),
      ...(input.meta === undefined ? {} : { meta: input.meta }),
      prev: previous.hash,
    };
    const receipt: Receipt = { ...body, hash: await hashOf(body) };
    await fs.append(`${DIR}/${receipt.at.slice(0, 7)}.jsonl`, `${JSON.stringify(receipt)}\n`);
    return receipt;
  }

  return {
    /** Write one receipt. Rejects if it cannot be written; callers must then not act. */
    append(input: ReceiptInput): Promise<Receipt> {
      const result = queue.then(() => (lock === undefined ? appendNow(input) : lock.exclusive(DIR, () => appendNow(input))));
      queue = result.catch(() => undefined);
      return result;
    },

    /** Receipts with seq above `after`, oldest first, at most `limit`. Lines that do not parse are left to verify(). */
    async list(options: { after?: number; limit?: number } = {}): Promise<Receipt[]> {
      const after = options.after ?? 0;
      const limit = options.limit ?? 500;
      const out: Receipt[] = [];
      for (const file of await files()) {
        for (const line of ((await fs.read(file)) ?? "").split("\n")) {
          if (line === "") continue;
          try {
            const receipt = JSON.parse(line) as Receipt;
            if (receipt.seq > after) out.push(receipt);
          } catch {
            // verify() reports it.
          }
          if (out.length >= limit) return out;
        }
      }
      return out;
    },

    /** Walk every line of every month and check order, links and hashes. */
    async verify(): Promise<VerifyResult> {
      let prev = GENESIS;
      let seq = 0;
      for (const file of await files()) {
        const lines = ((await fs.read(file)) ?? "").split("\n").filter((l) => l !== "");
        for (const [index, line] of lines.entries()) {
          const fail = (reason: string): VerifyResult => ({ ok: false, count: seq, brokenAt: { file, line: index + 1, reason } });
          let receipt: Receipt;
          try {
            receipt = JSON.parse(line) as Receipt;
          } catch {
            return fail("not JSON");
          }
          const { hash, ...body } = receipt;
          if (receipt.seq !== seq + 1) return fail(`expected seq ${seq + 1}, found ${receipt.seq}`);
          if (receipt.prev !== prev) return fail("does not link to the previous line");
          if (hash !== (await hashOf(body))) return fail("content does not match its hash");
          prev = hash;
          seq = receipt.seq;
        }
      }
      return { ok: true, count: seq };
    },
  };
}

export type ReceiptLog = ReturnType<typeof createReceiptLog>;
