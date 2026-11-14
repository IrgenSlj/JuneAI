// The daemon's HTTP API, written by hand. apps/server imports these same types,
// so a change here that the server does not follow fails its typecheck.
// Contract: PLAN.md, "HTTP API".

export interface Health {
  ok: true;
  name: "june";
  version: string;
}

export interface VaultInfo {
  /** Absolute path of the open vault, or null when none is open yet. */
  path: string | null;
  /** Suggested location for a new vault (~/June expanded). */
  defaultPath: string;
}

export interface OpenVaultRequest {
  /** Absolute path, or a path starting with "~/". Created if missing. */
  path: string;
}

export interface ApiErrorBody {
  error: string;
}
