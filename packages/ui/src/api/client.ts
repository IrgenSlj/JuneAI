/**
 * Typed client for the June daemon. Same origin in every mode, so the default
 * base URL is "" and requests go to /api/... on the page's own host.
 *
 * Streaming turns arrive in M1 as SSE over POST: EventSource cannot POST or
 * set headers, so the body is read through fetch and parsed frame by frame.
 */
import type { ApiErrorBody, Health, OpenVaultRequest, VaultInfo } from "./types.js";

export interface JuneClientOptions {
  /** Origin of the daemon; "" (same origin) unless testing. No trailing slash. */
  baseUrl?: string;
  /** The per-launch token; sent as X-June-Token on every request. */
  token?: string;
  fetchImpl?: typeof fetch;
  /** Applied to every non-streaming request so the UI never hangs on a stuck daemon. */
  requestTimeoutMs?: number;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
  ) {
    super(`June API ${status}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
  }

  /** The daemon's own error message, when the body is its JSON error shape. */
  get reason(): string {
    try {
      return (JSON.parse(this.body) as ApiErrorBody).error ?? this.body;
    } catch {
      return this.body;
    }
  }
}

export function createJuneClient(options: JuneClientOptions = {}) {
  const baseUrl = (options.baseUrl ?? "").replace(/\/+$/, "");
  const fetchImpl = options.fetchImpl ?? ((input, init) => fetch(input, init));
  const timeoutMs = options.requestTimeoutMs ?? 30_000;
  let token = options.token;

  async function request<T>(method: "GET" | "POST" | "PUT", path: string, body?: unknown): Promise<T> {
    const headers = new Headers({ Accept: "application/json" });
    if (token) headers.set("X-June-Token", token);
    if (body !== undefined) headers.set("Content-Type", "application/json");
    const response = await fetchImpl(`${baseUrl}/api${path}`, {
      method,
      headers,
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) throw new ApiError(response.status, await response.text());
    return (await response.json()) as T;
  }

  return {
    setToken(next: string | undefined) {
      token = next;
    },
    health: () => request<Health>("GET", "/health"),
    getVault: () => request<VaultInfo>("GET", "/vault"),
    openVault: (body: OpenVaultRequest) => request<VaultInfo>("POST", "/vault/open", body),
  };
}

export type JuneClient = ReturnType<typeof createJuneClient>;
