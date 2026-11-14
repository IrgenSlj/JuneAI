// Contracts every provider adapter meets. PLAN.md, "Providers".

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  signal?: unknown;
}

export type StreamEvent =
  | { type: "text"; text: string }
  | { type: "usage"; inputTokens?: number; outputTokens?: number }
  | { type: "error"; message: string; status?: number }
  | { type: "done"; finishReason?: string };

export type RouteKind = "api-key" | "local" | "gateway" | "chatgpt-plan" | "claude-cli";

/** Everything the send label needs to tell the truth about a route. */
export interface Route {
  /** "openai", "ollama", "opencode-zen", ... */
  id: string;
  /** Shown to the user. */
  name: string;
  kind: RouteKind;
  /** "api.openai.com", "localhost:11434", "local process". */
  host: string;
  leavesDevice: boolean;
  billing: "your API key" | "your subscription" | "free" | "none (local)";
  /** One honest sentence. If we do not know, it says "unknown". Never guessed. */
  dataNote: string;
  /** Where the dataNote comes from, so it can be checked and dated. */
  dataNoteSource?: { url: string; checked: string };
}

export interface Provider {
  readonly route: Route;
  listModels(): Promise<string[]>;
  /** Never throws once called: every failure arrives as an "error" event. */
  stream(request: ChatRequest): AsyncIterable<StreamEvent>;
}

/** The slice of fetch an adapter needs; injected so providers never touch the network on their own. */
export type FetchLike = (
  url: string,
  init: { method: "GET" | "POST"; headers: Record<string, string>; body?: string; signal?: unknown },
) => Promise<{ ok: boolean; status: number; text(): Promise<string>; body: AsyncIterable<Uint8Array> | null }>;
