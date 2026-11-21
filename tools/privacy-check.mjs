// Refuses to let private data into git. Run by the pre-commit hook and by the
// gate (tools/privacy.test.mjs), so a commit made without `pnpm check` is
// still caught, and CI catches a commit made with --no-verify.
//
// It looks at every file git tracks or has staged. Real data belongs in
// lab/private/ (ignored) or in a vault outside the repo (default ~/June).
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
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
  const cwd = repoRoot();
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

// Fields that only appear in real provider exports and manifests. Code may
// name them; data and docs outside a redacted fixtures/ folder may not.
const SIGNATURES = String.raw`"(chat_messages|export_url|email_address|full_name|verified_phone_number)"[[:space:]]*:`;
const DATA_FILES = ["*.json", "*.jsonl", "*.ndjson", "*.md", "*.html", "*.txt", "*.csv", "*.yaml", "*.yml", "*.xml"];

/** Paths in the index whose content matches, via git grep (exit 1 means no match). */
function grepIndex(args, cwd) {
  try {
    return execFileSync("git", ["grep", "--cached", "-I", "-l", ...args], { cwd, encoding: "utf8" })
      .split("\n")
      .filter((p) => p !== "");
  } catch (error) {
    if (error.status === 1) return [];
    throw error;
  }
}

/** Strings personal to this user, one per line, kept in git-ignored lab/private/denylist.txt. */
export function readDenylist(cwd) {
  const file = `${cwd}/lab/private/denylist.txt`;
  if (!existsSync(file)) return [];
  return readFileSync(file, "utf8")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length >= 3 && !line.startsWith("#"));
}

/** Staged or tracked files whose content looks private. Reads the index, so it checks exactly what will be committed. */
export function contentProblems(cwd = repoRoot()) {
  const found = grepIndex(["-E", SIGNATURES, "--", ...DATA_FILES, ":(exclude)**/fixtures/**"], cwd).map(
    (path) => `${path} contains fields from a real provider export`,
  );
  const deny = readDenylist(cwd);
  if (deny.length > 0) {
    const args = ["-i", "-F", ...deny.flatMap((line) => ["-e", line]), "--", "."];
    // Never print which string matched: the denylist itself is private.
    found.push(...grepIndex(args, cwd).map((path) => `${path} contains a string from lab/private/denylist.txt`));
  }
  return found;
}

function repoRoot() {
  return execFileSync("git", ["rev-parse", "--show-toplevel"], { encoding: "utf8" }).trim();
}

export function check() {
  return [
    ...trackedFiles().flatMap(({ path, bytes }) => problemsWith(path, bytes).map((reason) => `${path} ${reason}`)),
    ...contentProblems(),
  ];
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
