// The chat screen's state: routes and models, the thread list, the open
// thread, and one streaming turn at a time. The page renders this; nothing
// here touches the DOM.
//
// A turn is shown as saved only when the daemon's `saved` frame arrives; text
// alone never means saved (PLAN.md, rule 4).
import type { ChatMessage } from "@june/ui";
import { ApiError, type ProviderInfo, type RouteInfo, type SendLabelInfo, type ThreadDetail, type ThreadSummary, type TurnInfo } from "@june/ui/api";
import { session } from "./session.svelte.js";

interface Choice {
  route: string;
  model: string;
}

// The last route and model used, kept in this browser as a convenience. The
// conversation files hold the real record; losing this only means a default.
const LAST_CHOICE = "june.lastModel";

function readLastChoice(): Choice | undefined {
  try {
    const parsed = JSON.parse(localStorage.getItem(LAST_CHOICE) ?? "null") as Partial<Choice> | null;
    return typeof parsed?.route === "string" && typeof parsed.model === "string" ? { route: parsed.route, model: parsed.model } : undefined;
  } catch {
    return undefined;
  }
}

function writeLastChoice(choice: Choice): void {
  try {
    localStorage.setItem(LAST_CHOICE, JSON.stringify(choice));
  } catch {
    // Storage blocked (private window): the next load starts on the default.
  }
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

function toMessage(turn: TurnInfo, routes: ProviderInfo[]): ChatMessage {
  const route = routes.find((p) => p.route.id === turn.route)?.route;
  return {
    id: turn.id,
    role: turn.role === "user" ? "user" : turn.role === "tool" ? "tool" : "assistant",
    content: turn.text,
    ...(turn.role === "assistant" && turn.model !== undefined
      ? { provenance: { model: turn.model, tier: route?.name ?? turn.route ?? "", cloud_call: route?.leavesDevice ?? true } }
      : {}),
  };
}

class Chat {
  routes: ProviderInfo[] = $state([]);
  routeId: string | undefined = $state();
  models: string[] = $state([]);
  model = $state("");
  threads: ThreadSummary[] = $state([]);
  thread: ThreadDetail | undefined = $state();
  messages: ChatMessage[] = $state([]);
  streaming = $state(false);
  /** What left (or was about to) on the last send. */
  label: SendLabelInfo | undefined = $state();
  /** The last reply that the daemon confirmed written to the vault. */
  savedTurn: string | undefined = $state();
  error: string | undefined = $state();
  private abort: AbortController | undefined;

  readonly route = $derived(this.routes.find((p) => p.route.id === this.routeId)?.route);
  readonly ready = $derived(this.routes.filter((p) => p.ready));

  async load(): Promise<void> {
    try {
      const [routes, threads] = await Promise.all([session.client.providers(), session.client.threads()]);
      this.routes = routes;
      this.threads = threads;
      if (!(await this.choose(readLastChoice()))) {
        const first = routes.find((p) => p.ready && p.route.kind === "local") ?? routes.find((p) => p.ready);
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
    try {
      this.thread = await session.client.thread(id);
      this.messages = this.thread.turns.map((t) => toMessage(t, this.routes));
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
      this.threads = await session.client.threads();
      await this.open(branch.id);
    } catch (error) {
      this.error = errorText(error);
    }
  }

  stop(): void {
    this.abort?.abort();
  }

  async send(text: string): Promise<void> {
    const route = this.route;
    if (this.streaming || text.trim() === "" || route === undefined || this.model === "") return;
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
      if (this.savedTurn !== undefined) writeLastChoice({ route: route.id, model: this.model });
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
      this.threads = await session.client.threads().catch(() => this.threads);
      this.error = error;
    }
  }
}

export const chat = new Chat();
