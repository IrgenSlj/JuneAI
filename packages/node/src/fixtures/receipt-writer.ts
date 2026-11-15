// A separate process that appends receipts to a vault, for the lock tests.
// Usage: node receipt-writer.mjs <vault root> <count> <actor>
import { createReceiptLog } from "@june/core";
import { nodeFileLock, nodeVaultFs } from "../index.js";

const [root, count, actor] = process.argv.slice(2);
if (root === undefined || count === undefined || actor === undefined) throw new Error("usage: <root> <count> <actor>");
const log = createReceiptLog(await nodeVaultFs(root), { lock: nodeFileLock(root) });
for (let i = 0; i < Number(count); i++) {
  await log.append({ kind: "send", actor, targets: [`${actor}-${i}`], payload: i });
}
