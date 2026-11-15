// Refuses to let private data into git. Run by the pre-commit hook and by the
// gate (tools/privacy.test.mjs), so a commit made without `pnpm check` is
// still caught, and CI catches a commit made with --no-verify.
//
// It looks at every file git tracks or has staged. Real data belongs in
// lab/private/ (ignored) or in a vault outside the repo (default ~/June).
import { execFileSync } from "node:child_process";
import { statSync } from "node:fs";
import { pathToFileURL } from "node:url";

const MAX_BYTES = 1_000_000;
// Big files that are known to be ours. Add here only with a reason.
const SIZE_ALLOWED = new Set(["pnpm-lock.yaml", "apps/desktop/src-tauri/Cargo.lock"]);
// What provider exports and June vaults are made of.
const EXPORT_NAMES = /(^|\/)(conversations|users|projects|memories|chat\.html|message_feedback|shared_conversations)\.(json|html)$/i;
const ARCHIVES = /\.(zip|tar|tgz|gz|7z|rar)$/i;
const VAULT_DATA = /(^|\/)(receipts\/\d{4}-\d{2}\.jsonl|\.june\/|[^/]+\.raw\.json$)/;
// Redacted fixtures are the one place vault-shaped files may live.
const FIXTURE = /(^|\/)fixtures\//;

/** Reasons a path must not be committed; empty when it is fine. */
export function problemsWith(path, bytes) {
  const problems = [];
  if (path.startsWith("lab/private/")) problems.push("is in lab/private/, which holds real data");
  if (EXPORT_NAMES.test(path)) problems.push("is named like a provider export file");
  if (ARCHIVES.test(path)) problems.push("is an archive (exports arrive as zips)");
  if (VAULT_DATA.test(path) && !FIXTURE.test(path)) problems.push("is vault data outside a fixtures/ folder");
  if (bytes > MAX_BYTES && !SIZE_ALLOWED.has(path)) problems.push(`is ${Math.round(bytes / 1000)} KB; fixtures should be small and redacted`);
  return problems;
}

/** Every tracked or staged path, with its size on disk (0 if deleted in the work tree). */
export function trackedFiles() {
  const cwd = execFileSync("git", ["rev-parse", "--show-toplevel"], { encoding: "utf8" }).trim();
  const out = execFileSync("git", ["ls-files", "--cached", "-z"], { cwd, encoding: "utf8" });
  return out
    .split("\0")
    .filter((p) => p !== "")
    .map((path) => {
      let bytes = 0;
      try {
        bytes = statSync(`${cwd}/${path}`).size;
      } catch {}
      return { path, bytes };
    });
}

export function check() {
  return trackedFiles().flatMap(({ path, bytes }) => problemsWith(path, bytes).map((reason) => `${path} ${reason}`));
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const found = check();
  if (found.length > 0) {
    console.error("June privacy check: these files must not be committed:\n");
    for (const line of found) console.error(`  ${line}`);
    console.error("\nUnstage them with: git rm --cached <path>");
    process.exit(1);
  }
}
