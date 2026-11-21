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
  ExportToPathRequest,
  Health,
  ImportReport,
  ModelList,
  NewThreadRequest,
  OpenVaultRequest,
  ProviderInfo,
  PutProviderRequest,
  ReceiptInfo,
  ReceiptsVerify,
  SendTurnRequest,
  Settings,
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

  /** A request whose body or answer is not JSON (an upload, a download). */
  async function raw(method: "GET" | "POST", path: string, init: { body?: Blob; headers?: Record<string, string>; timeoutMs?: number } = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    if (token) headers.set("X-June-Token", token);
    const response = await fetchImpl(`${baseUrl}/api${path}`, {
      method,
      headers,
      ...(init.body === undefined ? {} : { body: init.body }),
      signal: AbortSignal.timeout(init.timeoutMs ?? timeoutMs),
    });
    if (!response.ok) throw new ApiError(response.status, await response.text());
    return response;
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
    /** Newest first; with `q`, search results carrying `match`. */
    threads: (options: { q?: string; before?: string; limit?: number } = {}) => {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(options)) if (value !== undefined && value !== "") params.set(key, String(value));
      const qs = params.toString();
      return request<ThreadSummary[]>("GET", `/threads${qs === "" ? "" : `?${qs}`}`);
    },
    /** Import from a file or folder on this Mac (the Mac app's pickers give real paths). */
    importPath: (path: string) => request<ImportReport>("POST", "/import", { path }),
    /** Import an uploaded export file (the browser cannot hand June a path). Large exports take a while. */
    importFile: async (file: File) =>
      (await (
        await raw("POST", "/import", { body: file, headers: { "Content-Type": "application/octet-stream", "X-June-Filename": encodeURIComponent(file.name) }, timeoutMs: 15 * 60_000 })
      ).json()) as ImportReport,
    /** One thread as Markdown or JSON text, for a download or the clipboard. */
    exportText: async (id: string, format: "md" | "json") => (await raw("GET", `/threads/${encodeURIComponent(id)}/export?format=${format}`)).text(),
    exportToPath: (id: string, body: ExportToPathRequest) => request<{ path: string }>("POST", `/threads/${encodeURIComponent(id)}/export`, body),
    thread: (id: string) => request<ThreadDetail>("GET", `/threads/${encodeURIComponent(id)}`),
    newThread: (body: NewThreadRequest = {}) => request<ThreadSummary>("POST", "/threads", body),
    settings: () => request<Settings>("GET", "/settings"),
    receipts: (after = 0) => request<ReceiptInfo[]>("GET", `/receipts?after=${after}`),
    verifyReceipts: () => request<ReceiptsVerify>("POST", "/receipts/verify"),
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
