/**
 * Typed client for the June daemon. Same origin in every mode, so the default
 * base URL is "" and requests go to /api/... on the page's own host.
 *
 * Streaming turns arrive in M1 as SSE over POST: EventSource cannot POST or
 * set headers, so the body is read through fetch and parsed frame by frame.
 */
import type {
  ApiErrorBody,
  BranchRequest,
  Health,
  ModelList,
  NewThreadRequest,
  OpenVaultRequest,
  ProviderInfo,
  PutProviderRequest,
  SendTurnRequest,
  ThreadDetail,
  ThreadSummary,
  TurnFrame,
  VaultInfo,
} from "./types.js";

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

  /**
   * Send a turn and yield its frames as they arrive. A refusal before the
   * stream starts (no key, busy thread) throws ApiError; once streaming, every
   * failure arrives as an error frame and the last frame is always done.
   */
  async function* sendTurn(threadId: string, body: SendTurnRequest, signal?: AbortSignal): AsyncGenerator<TurnFrame> {
    const headers = new Headers({ Accept: "text/event-stream", "Content-Type": "application/json" });
    if (token) headers.set("X-June-Token", token);
    const response = await fetchImpl(`${baseUrl}/api/threads/${encodeURIComponent(threadId)}/turns`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      ...(signal === undefined ? {} : { signal }),
    });
    if (!response.ok || response.body === null) throw new ApiError(response.status, await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;
    let failure: string | undefined;
    try {
      for (;;) {
        const chunk = await reader.read();
        buffer += decoder.decode(chunk.value, { stream: !chunk.done });
        let end = buffer.indexOf("\n\n");
        while (end !== -1) {
          const block = buffer.slice(0, end);
          buffer = buffer.slice(end + 2);
          const data = block
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (data !== "") {
            const frame = JSON.parse(data) as TurnFrame;
            if (frame.type === "done") done = true;
            yield frame;
          }
          end = buffer.indexOf("\n\n");
        }
        if (chunk.done) break;
      }
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error);
    } finally {
      reader.releaseLock();
    }
    if (!done) {
      yield { type: "error", stage: "connection", message: failure ?? "the connection to June closed before the reply finished" };
      yield { type: "done" };
    }
  }

  return {
    sendTurn,
    providers: () => request<ProviderInfo[]>("GET", "/providers"),
    putProvider: (id: string, body: PutProviderRequest) => request<ProviderInfo[]>("PUT", `/providers/${encodeURIComponent(id)}`, body),
    models: (id: string) => request<ModelList>("GET", `/providers/${encodeURIComponent(id)}/models`),
    threads: () => request<ThreadSummary[]>("GET", "/threads"),
    thread: (id: string) => request<ThreadDetail>("GET", `/threads/${encodeURIComponent(id)}`),
    newThread: (body: NewThreadRequest = {}) => request<ThreadSummary>("POST", "/threads", body),
    branch: (id: string, body: BranchRequest) => request<ThreadSummary>("POST", `/threads/${encodeURIComponent(id)}/branch`, body),
    setToken(next: string | undefined) {
      token = next;
    },
    health: () => request<Health>("GET", "/health"),
    getVault: () => request<VaultInfo>("GET", "/vault"),
    openVault: (body: OpenVaultRequest) => request<VaultInfo>("POST", "/vault/open", body),
  };
}

export type JuneClient = ReturnType<typeof createJuneClient>;
