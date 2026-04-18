/**
 * Typed client for the June API.
 *
 * All request/response shapes are sourced from `types.ts`, which is
 * generated from the FastAPI OpenAPI spec via `tools/codegen.sh`.
 * Hand-written shapes are forbidden — if a route needs something new,
 * add it to the Python schema and regenerate.
 *
 * `streamChat` does SSE over POST. EventSource can't POST or send
 * headers, so we read the response body through the Fetch ReadableStream
 * and parse `data:` frames manually.
 */

import type { components } from "./types.js";

export type ChatRequest = components["schemas"]["ChatRequest"];
export type ChatEvent = components["schemas"]["ChatEvent"];
export type MemorySnapshot = components["schemas"]["MemorySnapshot"];
export type MemoryFact = components["schemas"]["MemoryFact"];
export type SkillInfo = components["schemas"]["SkillInfo"];
export type SkillsResponse = components["schemas"]["SkillsResponse"];
export type SystemStatus = components["schemas"]["SystemStatus"];

export interface JuneClientOptions {
  /** Base URL for the API, e.g. "http://localhost:8000". No trailing slash. */
  baseUrl: string;
  /** Optional fetch override for testing. */
  fetchImpl?: typeof fetch;
}

export interface StreamChatOptions extends ChatRequest {
  /** Abort signal from the caller. Stops the stream when the user cancels. */
  signal?: AbortSignal;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly body: string,
  ) {
    super(`June API ${status} ${statusText}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
  }
}

export function createJuneClient(options: JuneClientOptions) {
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const fetchImpl = options.fetchImpl ?? fetch;

  async function getJson<T>(path: string): Promise<T> {
    const response = await fetchImpl(`${baseUrl}${path}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new ApiError(response.status, response.statusText, await response.text());
    }
    return (await response.json()) as T;
  }

  return {
    baseUrl,

    /** GET /system — runtime indicator (provider, model, mode). */
    getSystem(): Promise<SystemStatus> {
      return getJson<SystemStatus>("/system");
    },

    /** GET /skills — tools exposed to the agent right now. */
    getSkills(): Promise<SkillsResponse> {
      return getJson<SkillsResponse>("/skills");
    },

    /** GET /memory/{user_id} — structured highlights of what June remembers. */
    getMemory(userId: string): Promise<MemorySnapshot> {
      return getJson<MemorySnapshot>(`/memory/${encodeURIComponent(userId)}`);
    },

    /**
     * POST /chat as an async iterator of ChatEvent frames.
     *
     * Yields each parsed SSE frame. Terminates naturally on a `done`
     * frame, or when the signal aborts. Errors in the stream surface
     * as a final frame with `type: "error"` — the caller decides what
     * to do with it.
     */
    async *streamChat(
      options: StreamChatOptions,
    ): AsyncGenerator<ChatEvent, void, void> {
      const { signal, ...body } = options;
      const response = await fetchImpl(`${baseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal,
      });

      if (!response.ok || !response.body) {
        const text = response.body ? await response.text() : "";
        throw new ApiError(response.status, response.statusText, text);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // SSE frames are separated by blank lines. Process each
          // complete frame and keep any partial trailing frame in
          // the buffer for the next chunk.
          let frameEnd = buffer.indexOf("\n\n");
          while (frameEnd !== -1) {
            const frame = buffer.slice(0, frameEnd);
            buffer = buffer.slice(frameEnd + 2);
            const event = parseSseFrame(frame);
            if (event) {
              yield event;
              if (event.type === "done") return;
            }
            frameEnd = buffer.indexOf("\n\n");
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}

function parseSseFrame(frame: string): ChatEvent | null {
  // A frame is one or more lines. Only the `data:` lines matter for
  // our schema — `event:`, `id:`, and comments are ignored.
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  try {
    return JSON.parse(dataLines.join("\n")) as ChatEvent;
  } catch {
    return null;
  }
}

export type JuneClient = ReturnType<typeof createJuneClient>;
