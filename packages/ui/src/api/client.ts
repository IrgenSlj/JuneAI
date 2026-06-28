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
export type RecallHit = components["schemas"]["RecallHit"];
export type MemorySnapshot = components["schemas"]["MemorySnapshot"];
export type MemoryFact = components["schemas"]["MemoryFact"];
export type MemoryStats = components["schemas"]["MemoryStats"];
export type GreetingResponse = components["schemas"]["GreetingResponse"];
export type MemoryStoreCount = components["schemas"]["MemoryStoreCount"];
export type MemoryDeleteResponse = components["schemas"]["MemoryDeleteResponse"];
export type MemoryUpdateRequest = components["schemas"]["MemoryUpdateRequest"];
export type MemoryUpdateResponse = components["schemas"]["MemoryUpdateResponse"];
export type MemoryFeedbackRequest = components["schemas"]["MemoryFeedbackRequest"];
export type MemoryFeedbackResponse = components["schemas"]["MemoryFeedbackResponse"];
export type MemoryWriteRequest = components["schemas"]["MemoryWriteRequest"];
export type MemoryWriteResponse = components["schemas"]["MemoryWriteResponse"];
export type SkillInfo = components["schemas"]["SkillInfo"];
export type SkillToolInfo = components["schemas"]["SkillToolInfo"];
export type SkillsResponse = components["schemas"]["SkillsResponse"];
export type SkillToggleRequest = components["schemas"]["SkillToggleRequest"];
export type SkillToggleResponse = components["schemas"]["SkillToggleResponse"];
export type SkillToolToggleResponse = components["schemas"]["SkillToolToggleResponse"];
export type SkillWriteRecord = components["schemas"]["SkillWriteRecord"];
export type SkillWritesResponse = components["schemas"]["SkillWritesResponse"];
export type SkillToolInvokeRequest = components["schemas"]["SkillToolInvokeRequest"];
export type SkillToolInvokeResponse = components["schemas"]["SkillToolInvokeResponse"];
export type RegistryEntry = components["schemas"]["RegistryEntry"];
export type RegistryResponse = components["schemas"]["RegistryResponse"];
export type RegistryInstallResponse = components["schemas"]["RegistryInstallResponse"];
export type RegistryUninstallResponse = components["schemas"]["RegistryUninstallResponse"];
export type SystemStatus = components["schemas"]["SystemStatus"];
export type CapabilityProfileView = components["schemas"]["CapabilityProfileView"];
export type ActivityEntryView = components["schemas"]["ActivityEntryView"];
export type ActivityResponse = components["schemas"]["ActivityResponse"];
export type TraceEventView = components["schemas"]["TraceEventView"];
export type TraceSummary = components["schemas"]["TraceSummary"];
export type TraceListResponse = components["schemas"]["TraceListResponse"];
export type TurnTraceView = components["schemas"]["TurnTraceView"];
export type SetupStatus = components["schemas"]["SetupStatus"];
export type SetupApplyRequest = components["schemas"]["SetupApplyRequest"];
export type SetupApplyResponse = components["schemas"]["SetupApplyResponse"];
export type SettingsView = components["schemas"]["SettingsView"];
export type ForgetKeyResponse = components["schemas"]["ForgetKeyResponse"];
// PrivacyDial is a Pydantic Literal alias, inlined by openapi-typescript rather
// than emitted as a standalone schema entry. We re-derive the union here so
// callers have a single source of truth for the three values.
export type PrivacyDial = "local_only" | "private_by_default" | "cloud_first";
export type PrivacyDialUpdateRequest = components["schemas"]["PrivacyDialUpdateRequest"];
export type PrivacyDialUpdateResponse = components["schemas"]["PrivacyDialUpdateResponse"];
export type TaskView = components["schemas"]["TaskView"];
export type TaskStepView = components["schemas"]["TaskStepView"];
export type TaskEventFrame = components["schemas"]["TaskEventFrame"];
export type TaskListResponse = components["schemas"]["TaskListResponse"];
export type TaskCreateRequest = components["schemas"]["TaskCreateRequest"];
export type TaskPatchRequest = components["schemas"]["TaskPatchRequest"];
export type TaskDeleteResponse = components["schemas"]["TaskDeleteResponse"];
export type ChatHistory = components["schemas"]["ChatHistory"];

export interface JuneClientOptions {
  /** Base URL for the API, e.g. "http://localhost:8000". No trailing slash. */
  baseUrl: string;
  /** Optional fetch override for testing. */
  fetchImpl?: typeof fetch;
  /**
   * Per-request timeout in milliseconds. Applied to every non-streaming
   * request so the UI never hangs forever on a stuck brain. SSE
   * (`streamChat`) is intentionally exempt.
   */
  requestTimeoutMs?: number;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

/** Combine an optional caller signal with a timeout signal. */
function withTimeout(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  if (!signal) return timeoutSignal;
  // AbortSignal.any short-circuits on the first signal to abort.
  return AbortSignal.any([signal, timeoutSignal]);
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
  const timeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;

  /** Issue a JSON request with a default timeout; throw ApiError on non-2xx. */
  async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
    const headers = new Headers(init?.headers);
    headers.set("Accept", "application/json");
    if (init?.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    const response = await fetchImpl(`${baseUrl}${path}`, {
      ...init,
      headers,
      signal: withTimeout(init?.signal ?? undefined, timeoutMs),
    });
    if (!response.ok) {
      throw new ApiError(response.status, response.statusText, await response.text());
    }
    return (await response.json()) as T;
  }

  function getJson<T>(path: string): Promise<T> {
    return requestJson<T>(path);
  }

  return {
    baseUrl,

    /** GET /system — runtime indicator (provider, model, mode). */
    getSystem(): Promise<SystemStatus> {
      return getJson<SystemStatus>("/system");
    },

    /** GET /system/capability — cached local-model capability profile. */
    getCapability(): Promise<CapabilityProfileView> {
      return getJson<CapabilityProfileView>("/system/capability");
    },

    /** GET /system/activity — reverse-chronological log of recent API requests and tool calls. */
    getActivity(limit = 100, kind?: string): Promise<ActivityResponse> {
      const params = new URLSearchParams();
      params.set("limit", String(limit));
      if (kind) params.set("kind", kind);
      return getJson<ActivityResponse>(`/system/activity?${params.toString()}`);
    },

    /** DELETE /system/activity — clear the activity log. */
    clearActivity(): Promise<ActivityResponse> {
      return requestJson<ActivityResponse>("/system/activity", { method: "DELETE" });
    },

    /** GET /system/traces — recent turn traces, newest first (summaries only). */
    getTraces(limit = 50): Promise<TraceListResponse> {
      const params = new URLSearchParams();
      params.set("limit", String(limit));
      return getJson<TraceListResponse>(`/system/traces?${params.toString()}`);
    },

    /** GET /system/traces/{turn_id} — the full glass-box trace for one turn. */
    getTrace(turnId: string): Promise<TurnTraceView> {
      return getJson<TurnTraceView>(`/system/traces/${encodeURIComponent(turnId)}`);
    },

    /** DELETE /system/traces — clear persisted turn traces. */
    clearTraces(): Promise<TraceListResponse> {
      return requestJson<TraceListResponse>("/system/traces", { method: "DELETE" });
    },

    /** GET /setup/status — whether the active provider is usable end to end. */
    getSetupStatus(): Promise<SetupStatus> {
      return getJson<SetupStatus>("/setup/status");
    },

    /** GET /settings — non-secret snapshot of the active configuration. */
    getSettings(): Promise<SettingsView> {
      return getJson<SettingsView>("/settings");
    },

    /** POST /settings/forget-key — delete the Gemini key from the credential store. */
    forgetGeminiKey(): Promise<ForgetKeyResponse> {
      return requestJson<ForgetKeyResponse>("/settings/forget-key", { method: "POST" });
    },

    /** PUT /settings/privacy-dial — persist the user's privacy dial (ADR 0009). */
    updatePrivacyDial(dial: PrivacyDial): Promise<PrivacyDialUpdateResponse> {
      return requestJson<PrivacyDialUpdateResponse>("/settings/privacy-dial", {
        method: "PUT",
        body: JSON.stringify({ dial }),
      });
    },

    /** POST /setup/apply — persist provider + key choice and verify with a round-trip. */
    applySetup(request: SetupApplyRequest): Promise<SetupApplyResponse> {
      // Setup apply hits a real model round-trip; give it a longer budget than
      // the default so a cold Ollama start or slow Gemini ping doesn't time out.
      return requestJson<SetupApplyResponse>("/setup/apply", {
        method: "POST",
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(60_000),
      });
    },

    /** GET /skills — MCP skill servers and the tools they expose. */
    getSkills(): Promise<SkillsResponse> {
      return getJson<SkillsResponse>("/skills");
    },

    /** GET /skills/registry — curated list of installable MCP servers. */
    getSkillRegistry(): Promise<RegistryResponse> {
      return getJson<RegistryResponse>("/skills/registry");
    },

    /** POST /skills/registry/{key}/install — install a third-party MCP server. */
    installRegistryEntry(key: string): Promise<RegistryInstallResponse> {
      // Spawning a new MCP subprocess and reloading the agent can take a while.
      return requestJson<RegistryInstallResponse>(
        `/skills/registry/${encodeURIComponent(key)}/install`,
        { method: "POST", signal: AbortSignal.timeout(60_000) },
      );
    },

    /** DELETE /skills/registry/{key} — remove an installed registry skill. */
    uninstallRegistryEntry(key: string): Promise<RegistryUninstallResponse> {
      return requestJson<RegistryUninstallResponse>(
        `/skills/registry/${encodeURIComponent(key)}`,
        { method: "DELETE" },
      );
    },

    /**
     * POST /skills/{key}/toggle — enable or disable a skill.
     *
     * The API persists the flip to the on-disk manifest and rebuilds the
     * agent so the next chat turn sees the updated tool surface.
     */
    toggleSkill(
      key: string,
      enabled: boolean,
    ): Promise<SkillToggleResponse> {
      return requestJson<SkillToggleResponse>(
        `/skills/${encodeURIComponent(key)}/toggle`,
        { method: "POST", body: JSON.stringify({ enabled }) },
      );
    },

    /**
     * GET /skills/{key}/writes/{user_id} — list paraphrased facts this skill
     * has written into the vector store. Useful for showing "what each skill
     * remembered" without scanning all of memory.
     */
    getSkillWrites(
      key: string,
      userId: string,
      limit = 30,
    ): Promise<SkillWritesResponse> {
      const path =
        `/skills/${encodeURIComponent(key)}/writes/${encodeURIComponent(userId)}` +
        `?limit=${encodeURIComponent(String(limit))}`;
      return getJson<SkillWritesResponse>(path);
    },

    /** POST /skills/{key}/tools/{tool}/invoke — run a tool with arbitrary args (playground). */
    invokeSkillTool(
      key: string,
      tool: string,
      args: Record<string, unknown>,
    ): Promise<SkillToolInvokeResponse> {
      // Skill tools can do real work (HTTP, file I/O, model calls); give them
      // a generous window before timing out.
      return requestJson<SkillToolInvokeResponse>(
        `/skills/${encodeURIComponent(key)}/tools/${encodeURIComponent(tool)}/invoke`,
        {
          method: "POST",
          body: JSON.stringify({ arguments: args }),
          signal: AbortSignal.timeout(60_000),
        },
      );
    },

    /** POST /skills/{key}/tools/{tool}/toggle — enable or disable a single tool. */
    toggleSkillTool(
      key: string,
      tool: string,
      enabled: boolean,
    ): Promise<SkillToolToggleResponse> {
      return requestJson<SkillToolToggleResponse>(
        `/skills/${encodeURIComponent(key)}/tools/${encodeURIComponent(tool)}/toggle`,
        { method: "POST", body: JSON.stringify({ enabled }) },
      );
    },

    /** GET /memory/{user_id} — structured highlights of what June remembers. */
    getMemory(userId: string, query = ""): Promise<MemorySnapshot> {
      const params = new URLSearchParams();
      const q = query.trim();
      if (q) params.set("q", q);
      const qs = params.toString();
      return getJson<MemorySnapshot>(
        `/memory/${encodeURIComponent(userId)}${qs ? `?${qs}` : ""}`,
      );
    },

    /** GET /memory/{user_id}/stats — per-store totals and last-write timestamp. */
    getMemoryStats(userId: string): Promise<MemoryStats> {
      return getJson<MemoryStats>(`/memory/${encodeURIComponent(userId)}/stats`);
    },

    /** GET /greeting/{user_id} — a short, local greeting for the empty chat. */
    getGreeting(userId: string, name = ""): Promise<GreetingResponse> {
      const params = new URLSearchParams();
      if (name) params.set("name", name);
      const qs = params.toString();
      return getJson<GreetingResponse>(
        `/greeting/${encodeURIComponent(userId)}${qs ? `?${qs}` : ""}`,
      );
    },

    /** GET /chat/history/{user_id} — persisted chat transcript, oldest first. */
    getChatHistory(userId: string): Promise<ChatHistory> {
      return getJson<ChatHistory>(`/chat/history/${encodeURIComponent(userId)}`);
    },

    /** GET /tasks/{user_id} — list this user's tasks, newest first. */
    getTasks(userId: string, status?: string, limit = 50): Promise<TaskListResponse> {
      const params = new URLSearchParams();
      if (status) params.set("status", status);
      params.set("limit", String(limit));
      const path = `/tasks/${encodeURIComponent(userId)}?${params.toString()}`;
      return getJson<TaskListResponse>(path);
    },

    /** GET /tasks/{user_id}/{task_id} — one task with its full step trace. */
    getTask(userId: string, taskId: string): Promise<TaskView> {
      return getJson<TaskView>(
        `/tasks/${encodeURIComponent(userId)}/${encodeURIComponent(taskId)}`,
      );
    },

    /** POST /tasks/{user_id} — create a new task in the planning state. */
    createTask(userId: string, request: TaskCreateRequest): Promise<TaskView> {
      return requestJson<TaskView>(`/tasks/${encodeURIComponent(userId)}`, {
        method: "POST",
        body: JSON.stringify(request),
      });
    },

    /** PATCH /tasks/{user_id}/{task_id} — pause, resume, cancel, complete. */
    patchTask(
      userId: string,
      taskId: string,
      request: TaskPatchRequest,
    ): Promise<TaskView> {
      return requestJson<TaskView>(
        `/tasks/${encodeURIComponent(userId)}/${encodeURIComponent(taskId)}`,
        { method: "PATCH", body: JSON.stringify(request) },
      );
    },

    /** DELETE /tasks/{user_id}/{task_id} — remove a task. */
    deleteTask(userId: string, taskId: string): Promise<TaskDeleteResponse> {
      return requestJson<TaskDeleteResponse>(
        `/tasks/${encodeURIComponent(userId)}/${encodeURIComponent(taskId)}`,
        { method: "DELETE" },
      );
    },

    /**
     * GET /tasks/{user_id}/{task_id}/events as an async iterator of TaskEventFrame objects.
     *
     * Yields each parsed SSE frame. Terminates naturally on a `done`
     * frame, or when the signal aborts. Mirrors the streamChat pattern.
     */
    async *streamTaskEvents(
      userId: string,
      taskId: string,
      signal?: AbortSignal,
    ): AsyncGenerator<TaskEventFrame, void, void> {
      const response = await fetchImpl(
        `${baseUrl}/tasks/${encodeURIComponent(userId)}/${encodeURIComponent(taskId)}/events`,
        {
          headers: { Accept: "text/event-stream" },
          signal,
        },
      );

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

          let frameEnd = buffer.indexOf("\n\n");
          while (frameEnd !== -1) {
            const frame = buffer.slice(0, frameEnd);
            buffer = buffer.slice(frameEnd + 2);
            const event = parseTaskSseFrame(frame);
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

    /** POST /memory/{user_id}/fact — manually write a structured or semantic fact. */
    writeMemoryFact(
      userId: string,
      request: MemoryWriteRequest,
    ): Promise<MemoryWriteResponse> {
      return requestJson<MemoryWriteResponse>(
        `/memory/${encodeURIComponent(userId)}/fact`,
        { method: "POST", body: JSON.stringify(request) },
      );
    },

    /**
     * DELETE /memory/{user_id}/fact/{ref} — remove a fact across stores.
     *
     * `ref` is the opaque identifier from a MemoryFact (e.g. "semantic:abc",
     * "node:person:ana"). Colons inside the ref are fine — the route is
     * declared with `{ref:path}` on the server.
     */
    deleteMemoryFact(userId: string, ref: string): Promise<MemoryDeleteResponse> {
      return requestJson<MemoryDeleteResponse>(
        `/memory/${encodeURIComponent(userId)}/fact/${encodeURI(ref)}`,
        { method: "DELETE" },
      );
    },

    /**
     * PATCH /memory/{user_id}/fact/{ref} — update structured memory fields.
     *
     * The returned `ref` may differ from the request ref when the primary
     * key changed (rename a goal, reschedule a calendar item). Use the
     * returned ref for any follow-up edit/delete.
     */
    patchMemoryFact(
      userId: string,
      ref: string,
      fields: Record<string, string>,
    ): Promise<MemoryUpdateResponse> {
      return requestJson<MemoryUpdateResponse>(
        `/memory/${encodeURIComponent(userId)}/fact/${encodeURI(ref)}`,
        { method: "PATCH", body: JSON.stringify({ fields }) },
      );
    },

    /**
     * POST /memory/{user_id}/feedback — vote on a recalled memory.
     *
     * Pass `vote: "up" | "down"` to record a vote, or `"clear"` to drop a
     * prior vote. The same ref can be re-voted at any time; the latest
     * value wins. Server returns the recorded vote (empty string when cleared).
     */
    postMemoryFeedback(
      userId: string,
      ref: string,
      vote: "up" | "down" | "clear",
    ): Promise<MemoryFeedbackResponse> {
      return requestJson<MemoryFeedbackResponse>(
        `/memory/${encodeURIComponent(userId)}/feedback`,
        { method: "POST", body: JSON.stringify({ ref, vote }) },
      );
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

function parseTaskSseFrame(frame: string): TaskEventFrame | null {
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  try {
    return JSON.parse(dataLines.join("\n")) as TaskEventFrame;
  } catch {
    return null;
  }
}

export type JuneClient = ReturnType<typeof createJuneClient>;
