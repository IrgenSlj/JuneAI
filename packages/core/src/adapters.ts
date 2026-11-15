// The only doors out of the core. Core code reaches storage through these
// interfaces and nothing else, so the same core runs in the daemon and in
// tests. Index and git adapters join here when their milestone needs them.

/**
 * A vault's files. Paths are vault-relative with "/" separators, validated by
 * vaultPath(). Implementations must never resolve a path outside the vault.
 */
export interface VaultFs {
  /** File contents, or undefined if there is no such file. */
  read(path: string): Promise<string | undefined>;
  /** Create or replace a file, creating parent folders as needed. */
  write(path: string, content: string): Promise<void>;
  /** Append to a file, creating it if needed. Used for receipts. */
  append(path: string, content: string): Promise<void>;
  /** Every file under a folder, recursively, as vault-relative paths, sorted. */
  list(dir: string): Promise<string[]>;
  /** Delete a file. Deleting a missing file is not an error. */
  remove(path: string): Promise<void>;
}

/**
 * Mutual exclusion across every process that opens the same vault (the daemon,
 * a CLI, an importer). Within one process an in-memory queue is enough; on disk
 * the Node adapter uses a lock file. The task runs only while the lock is held,
 * and the lock is released whether it resolves or rejects.
 */
export interface VaultLock {
  exclusive<T>(name: string, task: () => Promise<T>): Promise<T>;
}

/**
 * Validate and normalise a vault-relative path. The vault must open unchanged
 * on macOS, Windows and Linux, so this enforces the strictest of the three:
 * - nothing that escapes the vault (absolute, drive letters, "..", "\\");
 * - nothing Windows refuses (reserved names like CON or aux.md, <>:"|?*,
 *   control characters, a trailing dot or space);
 * - Unicode NFC, so "é" typed on a Mac and on Windows is the same file.
 * Case is preserved here; adapters must treat paths that differ only in case
 * as the same file, because macOS and Windows disks do.
 */
export function vaultPath(path: string): string {
  const trimmed = path.normalize("NFC").replace(/\/+$/, "");
  if (trimmed === "") return "";
  if (trimmed.startsWith("/") || /^[A-Za-z]:/.test(trimmed) || trimmed.includes("\\")) {
    throw new VaultPathError(path);
  }
  const segments = trimmed.split("/");
  for (const segment of segments) {
    if (!isPortableSegment(segment)) throw new VaultPathError(path);
  }
  return segments.join("/");
}

const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$/i;
const FORBIDDEN_CHARS = /[<>:"|?*\u0000-\u001f]/;

function isPortableSegment(segment: string): boolean {
  return (
    segment !== "" &&
    segment !== "." &&
    segment !== ".." &&
    segment.length <= 255 &&
    !FORBIDDEN_CHARS.test(segment) &&
    !WINDOWS_RESERVED.test(segment) &&
    !/[. ]$/.test(segment)
  );
}

/** The key two paths share when a case-insensitive disk sees them as one file. */
export function pathIdentity(path: string): string {
  return vaultPath(path).toLowerCase();
}

export class VaultPathError extends Error {
  constructor(path: string) {
    super(`not a safe vault-relative path: ${JSON.stringify(path)}`);
    this.name = "VaultPathError";
  }
}
