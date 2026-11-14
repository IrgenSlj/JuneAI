// The only doors out of the core. Core code reaches storage through these
// interfaces and nothing else, so the same core runs under Electron, in tests,
// and later in a phone shell. Index and git adapters join here in M1.

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
 * Validate and normalise a vault-relative path. Rejects anything that could
 * escape the vault: absolute paths, drive letters, backslashes, "..", empty
 * segments. Every adapter calls this before touching storage.
 */
export function vaultPath(path: string): string {
  const trimmed = path.replace(/\/+$/, "");
  if (trimmed === "") return "";
  if (trimmed.startsWith("/") || /^[A-Za-z]:/.test(trimmed) || trimmed.includes("\\")) {
    throw new VaultPathError(path);
  }
  const segments = trimmed.split("/");
  for (const segment of segments) {
    if (segment === "" || segment === "." || segment === ".." || segment.includes("\0")) {
      throw new VaultPathError(path);
    }
  }
  return segments.join("/");
}

export class VaultPathError extends Error {
  constructor(path: string) {
    super(`not a safe vault-relative path: ${JSON.stringify(path)}`);
    this.name = "VaultPathError";
  }
}
