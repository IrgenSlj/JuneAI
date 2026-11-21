// The chat screen's state: routes and models, the thread list, the open
// thread, and one streaming turn at a time. The page renders this; nothing
// here touches the DOM.
//
// A turn is shown as saved only when the daemon's `saved` frame arrives; text
// alone never means saved (PLAN.md, rule 4).
import { type ChatMessage, sourceName } from "@june/ui";
import { ApiError, type ProviderInfo, type RouteInfo, type SendLabelInfo, type ThreadDetail, type ThreadSummary, type Transparency, type TurnInfo } from "@june/ui/api";
import { getPlatform } from "@june/ui/platform";
import { session } from "./session.svelte.js";

interface Choice {
  route: string;
  model: string;
}

/** The route and model of the conversation's last reply, if any. */
function lastChoiceIn(thread: ThreadDetail): Choice | undefined {
  const reply = thread.turns.findLast((t) => t.role === "assistant" && t.route !== undefined && t.model !== undefined);
  return reply?.route !== undefined && reply.model !== undefined ? { route: reply.route, model: reply.model } : undefined;
}

const errorText = (error: unknown): string => (error instanceof ApiError ? error.reason : error instanceof Error ? error.message : String(error));

/** Routes grouped the way the model switcher shows them (PLAN.md, design brief frame 1). */
export function routeGroup(route: RouteInfo): string {
  if (route.kind === "local") return "On this Mac";
  if (route.kind === "gateway") return "Free";
  if (route.kind === "chatgpt-plan" || route.kind === "claude-cli") return "Your subscriptions";
  return "Your keys";
}

const PAGE = 200;

/**
 * What makes a send need the label sheet again in Full: the route, and any kind
 * of context beyond June's own history and the draft (imported text now; notes
 * and attachments from M5).
 */
export function labelSignature(label: SendLabelInfo): string {
  const extra = label.parts.filter((p) => p.kind !== "history" && p.kind !== "draft").map((p) => p.kind);
  return [label.route.id, ...[...new Set(extra)].sort()].join("|");
}

function toMessage(turn: TurnInfo, routes: ProviderInfo[], source: string, level: Transparency): ChatMessage {
  const route = routes.find((p) => p.route.id === turn.route)?.route;
  const imported = sourceName(source);
  const where = imported === undefined ? (route?.name ?? turn.route ?? "") : `imported from ${imported}`;
  const tier = level === "full" && turn.receipt !== undefined ? `${where} · receipt #${turn.receipt}` : where;
  return {
    id: turn.id,
    role: turn.role === "user" ? "user" : turn.role === "tool" ? "tool" : "assistant",
    content: turn.text,
    ...(turn.role === "assistant" && turn.model !== undefined && level !== "quiet"
      ? { provenance: { model: turn.model, tier, cloud_call: route?.leavesDevice ?? true } }
      : {}),
  };
}

class Chat {
  routes: ProviderInfo[] = $state([]);
  routeId: string | undefined = $state();
  models: string[] = $state([]);
  model = $state("");
  threads: ThreadSummary[] = $state([]);
  /** Whether the daemon has older threads than the ones listed. */
  moreThreads = $state(false);
  /** The search box; results replace the list while it is not empty. */
  query = $state("");
  results: ThreadSummary[] | undefined = $state();
  private searchSeq = 0;
  thread: ThreadDetail | undefined = $state();
  messages: ChatMessage[] = $state([]);
  streaming = $state(false);
  /** What left (or was about to) on the last send. */
  label: SendLabelInfo | undefined = $state();
  /** How much of what leaves is shown; receipts are written the same at every level. */
  transparency: Transparency = $state("glance");
  /** Full level: the label waiting for the user's yes, with the draft it describes. */
  pending: { label: SendLabelInfo; text: string; threadId: string | undefined } | undefined = $state();
  /** Full level: label signatures already approved, per thread ("new" before the thread exists). */
  private approved = new Map<string, Set<string>>();
  private labelSeq = 0;
  /** The last reply that the daemon confirmed written to the vault. */
  savedTurn: string | undefined = $state();
  error: string | undefined = $state();
  private abort: AbortController | undefined;

  /** True once providers have been asked at least once, so the page does not flash "no model". */
  loaded = $state(false);
  readonly route = $derived(this.routes.find((p) => p.route.id === this.routeId)?.route);
  readonly ready = $derived(this.routes.filter((p) => p.ready));
  /** Imported threads stay as the export had them; they continue as a branch. */
  readonly readOnly = $derived(this.thread !== undefined && this.thread.source !== "june");

  private async refreshThreads(): Promise<void> {
    const threads = await session.client.threads({ limit: PAGE });
    this.threads = threads;
    this.moreThreads = threads.length === PAGE;
  }

  async loadMore(): Promise<void> {
    const last = this.threads.at(-1);
    if (last?.activity === undefined) return;
    try {
      const older = await session.client.threads({ before: last.activity, beforeId: last.id, limit: PAGE });
      const known = new Set(this.threads.map((t) => t.id));
      this.threads = [...this.threads, ...older.filter((t) => !known.has(t.id))];
      this.moreThreads = older.length === PAGE;
    } catch (error) {
      this.error = errorText(error);
    }
  }

  /** Search as the user types; a stale answer never replaces a newer one. */
  async search(query: string): Promise<void> {
    this.query = query;
    const seq = ++this.searchSeq;
    if (query.trim() === "") {
      this.results = undefined;
      return;
    }
    try {
      const results = await session.client.threads({ q: query.trim(), limit: 100 });
      if (seq === this.searchSeq) this.results = results;
    } catch (error) {
      if (seq === this.searchSeq) this.error = errorText(error);
    }
  }

  /** Continue an imported thread on any model: a branch from its last turn. */
  async continueThread(): Promise<void> {
    const last = this.thread?.turns.at(-1);
    if (last !== undefined) await this.branchAt(last.id);
  }

  /** Save the open thread as Markdown or JSON: a save dialog in the Mac app, a download in the browser. */
  async exportThread(format: "md" | "json"): Promise<string | undefined> {
    const thread = this.thread;
    if (thread === undefined) return undefined;
    const name = `${thread.title.normalize("NFKD").replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "conversation"}.${format}`;
    try {
      const platform = getPlatform();
      if (platform.runtime === "tauri") {
        const path = await platform.pickSavePath({ title: "Export conversation", defaultName: name, extensions: [format] });
        if (path === null) return undefined;
        return (await session.client.exportToPath(thread.id, { format, path })).path;
      }
      const text = await session.client.exportText(thread.id, format);
      const url = URL.createObjectURL(new Blob([text], { type: format === "md" ? "text/markdown" : "application/json" }));
      const link = Object.assign(document.createElement("a"), { href: url, download: name });
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
      return name;
    } catch (error) {
      this.error = errorText(error);
      return undefined;
    }
  }

  /** Move the open thread to .trash/ in the vault; returns where it went. */
  async trashThread(): Promise<string | undefined> {
    const thread = this.thread;
    if (thread === undefined || this.streaming) return undefined;
    try {
      const { trashedTo } = await session.client.trashThread(thread.id);
      this.startNew();
      await this.refreshThreads();
      if (this.query.trim() !== "") await this.search(this.query);
      return trashedTo;
    } catch (error) {
      this.error = errorText(error);
      return undefined;
    }
  }

  /** The open thread as Markdown on the clipboard. */
  async copyThread(): Promise<boolean> {
    if (this.thread === undefined) return false;
    try {
      await navigator.clipboard.writeText(await session.client.exportText(this.thread.id, "md"));
      return true;
    } catch (error) {
      this.error = errorText(error);
      return false;
    }
  }

  async load(): Promise<void> {
    try {
      const [routes, settings] = await Promise.all([session.client.providers(), session.client.settings(), this.refreshThreads()]);
      this.routes = routes;
      this.transparency = settings.transparency;
      if (!(await this.choose(settings.lastChoice))) {
        const first = routes.find((p) => p.ready && p.route.kind === "local") ?? routes.find((p) => p.ready);
        if (first !== undefined) await this.pickRoute(first.route.id);
      }
    } catch (error) {
      this.error = errorText(error);
    } finally {
      this.loaded = true;
    }
  }

  /** Ask again which routes are usable, after the user starts a local model. */
  async recheck(): Promise<void> {
    try {
      this.routes = await session.client.providers();
      if (this.route === undefined || !this.ready.some((p) => p.route.id === this.routeId)) {
        const first = this.ready.find((p) => p.route.kind === "local") ?? this.ready[0];
        if (first !== undefined) await this.pickRoute(first.route.id);
      }
    } catch (error) {
      this.error = errorText(error);
    }
  }

  /** Switch to a route, and to `wanted` if that route offers it (or lists nothing, so any name is typed). */
  async pickRoute(id: string, wanted?: string): Promise<void> {
    if (this.routeId !== id) {
      this.routeId = id;
      this.models = [];
      try {
        this.models = (await session.client.models(id)).models;
      } catch {
        // No list (the route is down, or lists nothing): the model can still be typed.
      }
    }
    if (wanted !== undefined && (this.models.length === 0 || this.models.includes(wanted))) this.model = wanted;
    else if (!this.models.includes(this.model)) this.model = this.models[0] ?? "";
  }

  /** Use a remembered choice if its route is usable now. Returns whether it did. */
  private async choose(choice: Choice | undefined): Promise<boolean> {
    if (choice === undefined || !this.ready.some((p) => p.route.id === choice.route)) return false;
    await this.pickRoute(choice.route, choice.model);
    return true;
  }

  async open(id: string): Promise<void> {
    if (this.streaming) return;
    // A label belongs to the thread it was built for; switching threads drops it.
    this.pending = undefined;
    this.labelSeq++;
    try {
      this.thread = await session.client.thread(id);
      this.messages = this.thread.turns.map((t) => toMessage(t, this.routes, this.thread?.source ?? "june", this.transparency));
      if (!this.streaming) await this.choose(lastChoiceIn(this.thread));
      this.label = undefined;
      this.savedTurn = undefined;
      this.error = undefined;
    } catch (error) {
      this.error = errorText(error);
    }
  }

  startNew(): void {
    if (this.streaming) return;
    this.pending = undefined;
    this.labelSeq++;
    this.thread = undefined;
    this.messages = [];
    this.label = undefined;
    this.savedTurn = undefined;
    this.error = undefined;
  }

  /** Continue the open thread from one of its turns as a new thread; the route switcher then picks any model. */
  async branchAt(turnId: string): Promise<void> {
    const thread = this.thread;
    if (this.streaming || thread === undefined) return;
    try {
      const branch = await session.client.branch(thread.id, { turn: turnId });
      await this.refreshThreads();
      await this.open(branch.id);
    } catch (error) {
      this.error = errorText(error);
    }
  }

  async setTransparency(level: Transparency): Promise<void> {
    try {
      this.transparency = (await session.client.putSettings({ transparency: level })).transparency;
      if (this.thread !== undefined && !this.streaming) await this.open(this.thread.id);
    } catch (error) {
      this.error = errorText(error);
    }
  }

  /** In Full, a send first shows its label; the user's yes sends exactly that draft. */
  async submit(text: string): Promise<void> {
    const route = this.route;
    if (this.transparency !== "full" || route === undefined || this.model === "" || this.streaming || this.readOnly || text.trim() === "") return this.send(text);
    const seq = ++this.labelSeq;
    const threadId = this.thread?.id;
    try {
      const label = await session.client.label(threadId, { route: route.id, model: this.model, text });
      if (seq !== this.labelSeq || this.thread?.id !== threadId) return; // the user moved on; this label describes another thread
      const seen = this.approved.get(threadId ?? "new");
      if (seen?.has(labelSignature(label)) === true) return this.send(text);
      this.pending = { label, text, threadId };
    } catch (error) {
      this.error = errorText(error);
    }
  }

  approvePending(): Promise<void> {
    const pending = this.pending;
    this.pending = undefined;
    if (pending === undefined || pending.threadId !== this.thread?.id) return Promise.resolve();
    const key = pending.threadId ?? "new";
    this.approved.set(key, new Set([...(this.approved.get(key) ?? []), labelSignature(pending.label)]));
    return this.send(pending.text);
  }

  cancelPending(): string | undefined {
    const text = this.pending?.text;
    this.pending = undefined;
    return text;
  }

  stop(): void {
    this.abort?.abort();
  }

  async send(text: string): Promise<void> {
    const route = this.route;
    if (this.streaming || this.readOnly || text.trim() === "" || route === undefined || this.model === "") return;
    this.error = undefined;
    this.savedTurn = undefined;
    this.streaming = true;
    this.abort = new AbortController();
    const userId = `pending-user-${Date.now()}`;
    const replyId = `pending-reply-${Date.now()}`;
    this.messages = [...this.messages, { id: userId, role: "user", content: text }, { id: replyId, role: "assistant", content: "" }];
    const reply = () => this.messages.find((m) => m.id === replyId);
    try {
      if (this.thread === undefined) {
        const created = await session.client.newThread({ title: text.trim().split("\n")[0]?.slice(0, 80) ?? "" });
        this.thread = { ...created, turns: [] };
        // What was approved before the thread existed applies to it.
        const before = this.approved.get("new");
        if (before !== undefined) this.approved.set(created.id, before);
        this.approved.delete("new");
      }
      const threadId = this.thread.id;
      for await (const frame of session.client.sendTurn(threadId, { route: route.id, model: this.model, text }, this.abort.signal)) {
        if (frame.type === "label") this.label = frame.label;
        else if (frame.type === "text") {
          const message = reply();
          if (message !== undefined) message.content += frame.text;
        } else if (frame.type === "saved") this.savedTurn = frame.assistantTurn;
        else if (frame.type === "error") this.error = frame.message;
      }
      // The file is the truth: show what the vault now holds, not what streamed.
      this.streaming = false;
      await this.open(threadId);
      this.savedTurn = this.error === undefined ? this.messages.at(-1)?.id : undefined;
    } catch (error) {
      this.error = errorText(error);
      this.messages = this.messages.filter((m) => m.id !== replyId || m.content !== "");
    } finally {
      this.streaming = false;
      this.abort = undefined;
      const error = this.error;
      await this.refreshThreads().catch(() => undefined);
      this.error = error;
    }
  }
}

export const chat = new Chat();
