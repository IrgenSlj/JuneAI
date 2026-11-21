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

/** The facts the label shows about a route. Mirrors providers' Route. */
export interface RouteInfo {
  id: string;
  name: string;
  kind: "api-key" | "local" | "gateway" | "chatgpt-plan" | "claude-cli";
  host: string;
  leavesDevice: boolean;
  billing: string;
  dataNote: string;
  dataNoteSource?: { url: string; checked: string };
}

/** One route as the settings page sees it. The key itself never comes back. */
export interface ProviderInfo {
  route: RouteInfo;
  needsKey: boolean;
  /** Usable now: a key is stored, or (for a local route) its server answers. */
  ready: boolean;
  /** Local routes only: whether the server on this Mac answered just now. */
  running?: boolean;
}

/** PUT /api/providers/:id. A string stores the key in the keychain; null removes it. */
export interface PutProviderRequest {
  apiKey: string | null;
}

export interface ModelList {
  models: string[];
}

export interface ThreadSummary {
  id: string;
  title: string;
  source: string;
  started?: string;
  updated?: string;
  parent?: string;
  /** Project or topic, from frontmatter (imported Claude projects become lanes). */
  lane?: string;
  turns: number;
  /** Present on search results: the best matching turn ("title" for a title match) and text around it, the match wrapped in \u0002 and \u0003. */
  match?: { turn: string; snippet: string };
}

export interface TurnInfo {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  at?: string;
  model?: string;
  route?: string;
  receipt?: number;
  text: string;
}

export interface ThreadDetail extends Omit<ThreadSummary, "turns"> {
  turns: TurnInfo[];
}

export interface NewThreadRequest {
  title?: string;
}

/** POST /api/threads/:id/turns; the reply is an SSE stream of TurnFrame. */
export interface SendTurnRequest {
  route: string;
  model: string;
  text: string;
}

export interface SendLabelInfo {
  route: RouteInfo;
  model: string;
  parts: { kind: "system" | "history" | "note" | "attachment" | "draft"; id?: string; chars: number }[];
  noteIds: string[];
  totalChars: number;
}

/** SSE frames, in order: label, text*, usage, saved | error, done. Saved state is shown only from `saved`. */
export type TurnFrame =
  | { type: "label"; label: SendLabelInfo }
  | { type: "text"; text: string }
  | { type: "usage"; inputTokens?: number; outputTokens?: number }
  | { type: "saved"; threadId: string; userTurn: string; assistantTurn: string; receipt: number }
  | { type: "error"; stage: string; message: string }
  | { type: "done" };

/** POST /api/threads/:id/branch: continue from `turn` as a new thread. */
export interface BranchRequest {
  turn: string;
}

/** One line of the receipts log: what happened and to what, never content. */
export interface ReceiptInfo {
  seq: number;
  at: string;
  kind: string;
  actor: string;
  targets: string[];
  payloadHash: string;
  meta?: { [key: string]: string | number | boolean };
  prev: string;
  hash: string;
}

export interface ReceiptsVerify {
  ok: boolean;
  count: number;
  brokenAt?: { file: string; line: number; reason: string };
}

/** GET /api/settings. The transparency level joins in M3. */
export interface Settings {
  lastChoice?: { route: string; model: string };
}

/** POST /api/import with this JSON body imports from a file or folder on this Mac; a raw zip or JSON body imports the uploaded file. */
export interface ImportRequest {
  path: string;
}

/** The count report: conversations = threads + empty + unreadable, and created + updated + unchanged = threads. */
export interface ImportReport {
  source: "claude-export";
  conversations: number;
  threads: number;
  empty: number;
  unreadable: number;
  messages: number;
  offPath: number;
  attachments: number;
  filesNotIncluded: number;
  created: number;
  updated: number;
  unchanged: number;
  receipt: number;
  /** Files in the export June does not import yet (memories, users, design chats ...). */
  ignored: string[];
}
