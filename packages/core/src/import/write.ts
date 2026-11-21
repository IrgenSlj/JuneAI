// Writes imported threads into the vault, idempotently (PLAN.md: "a second
// import changes zero files"). Each thread file has its provider's original
// JSON beside it as .raw.json. A thread is rewritten only when that raw JSON
// changed, so an edit the user made in another editor survives a re-import of
// the same export. A renamed conversation moves to its new filename.
import type { VaultFs } from "../adapters.js";
import type { ReceiptLog } from "../receipts.js";
import { serializeThread, threadPath } from "../thread.js";
import type { ImportedThread } from "./claude.js";

export interface WriteReport {
  created: number;
  updated: number;
  unchanged: number;
  /** Seq of the import receipt. */
  receipt: number;
}

export const rawPathFor = (threadFile: string): string => threadFile.replace(/\.md$/, ".raw.json");

/** Thread id to file, read from frontmatter only, for every thread in the vault. */
export async function threadIdsOnDisk(fs: VaultFs): Promise<Map<string, string>> {
  const byId = new Map<string, string>();
  for (const path of await fs.list("threads")) {
    if (!path.endsWith(".md")) continue;
    const head = ((await fs.read(path)) ?? "").slice(0, 400);
    const id = /^id: "([^"]+)"$/m.exec(head)?.[1];
    if (id !== undefined) byId.set(id, path);
  }
  return byId;
}

export async function writeImported(
  fs: VaultFs,
  receipts: ReceiptLog,
  imported: ImportedThread[],
  options: { source: string; onProgress?: (done: number, total: number) => void },
): Promise<WriteReport> {
  const existing = await threadIdsOnDisk(fs);
  const report = { created: 0, updated: 0, unchanged: 0 };
  const changed: string[] = [];
  let done = 0;
  for (const { thread, raw } of imported) {
    const path = threadPath(thread);
    const before = existing.get(thread.id);
    if (before !== undefined && (await fs.read(rawPathFor(before))) === raw) {
      report.unchanged++;
    } else {
      // Raw first, thread file last: a crash in between leaves no thread without its original.
      await fs.write(rawPathFor(path), raw);
      await fs.write(path, serializeThread(thread));
      if (before !== undefined && before !== path) {
        await fs.remove(before);
        await fs.remove(rawPathFor(before));
      }
      if (before === undefined) report.created++;
      else report.updated++;
      changed.push(thread.id);
    }
    options.onProgress?.(++done, imported.length);
  }
  const receipt = await receipts.append({
    kind: "import",
    actor: "user",
    targets: changed,
    payload: imported.map(({ thread, raw }) => [thread.id, raw.length]),
    meta: { source: options.source, ...report },
  });
  return { ...report, receipt: receipt.seq };
}
