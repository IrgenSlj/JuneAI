/**
 * Chat state that outlives the chat route.
 *
 * Lives at module scope so navigating to /memory, /skills, or /settings
 * does not tear down the transcript or abort an in-flight stream. The
 * async generator runs inside a detached promise; as long as this module
 * stays loaded the stream keeps appending tokens into `chat.messages`,
 * and when the user returns to / the chat page re-renders from the same
 * state.
 */
import { type ChatEvent, type ChatMessage, type ActivityStep } from "@june/ui";
import { client } from "$lib/api.js";
import { profileName } from "$lib/stores/user.svelte.js";

export const chat = $state({
  messages: [] as ChatMessage[],
  activity: [] as ActivityStep[],
  activityOpen: false,
  streaming: false,
  abortController: null as AbortController | null,
  /** When a stream is running, the timestamp it started — used to render
   * an elapsed-time hint if the model takes a while to emit its first token. */
  streamStartedAt: null as number | null,
});

/** Tracks the activity step id used to accumulate reasoning for the current turn. */
let currentReasoningStepId: string | null = null;

function appendToken(id: string, token: string) {
  chat.messages = chat.messages.map((m) =>
    m.id === id ? { ...m, content: m.content + token } : m,
  );
}

function attachRecallHits(id: string, hits: ChatEvent["recall_hits"]) {
  chat.messages = chat.messages.map((m) =>
    m.id === id ? { ...m, recallHits: hits } : m,
  );
}

function attachProvenance(id: string, provenance: ChatEvent["provenance"]) {
  const p = provenance as ChatMessage["provenance"];
  chat.messages = chat.messages.map((m) =>
    m.id === id ? { ...m, provenance: p ?? undefined } : m,
  );
}

function formatToolCall(
  name: string,
  args: Record<string, unknown> | undefined,
): string {
  const argText = args && Object.keys(args).length ? JSON.stringify(args) : "(no args)";
  return `${name} ${argText}`;
}

export function toggleActivity(): void {
  chat.activityOpen = !chat.activityOpen;
}

function pushActivity(step: Omit<ActivityStep, "id" | "ts">) {
  chat.activity = [
    ...chat.activity,
    { ...step, id: `act-${Date.now()}-${Math.random()}`, ts: Date.now() },
  ];
}

export async function loadHistory(userId: string): Promise<void> {
  if (chat.streaming || chat.messages.length > 0) return;
  try {
    const history = await client.getChatHistory(userId);
    const mapped = (history.messages ?? []).map((m, i) => ({
      id: `h-${i}`,
      role: m.role as "user" | "assistant",
      content: m.content,
    }));
    if (mapped.length > 0) {
      chat.messages = mapped;
    }
  } catch {
    // Best-effort: brain-down state leaves the empty greeting.
  }
}

export async function sendMessage(text: string): Promise<void> {
  if (chat.streaming) return;

  const userMsg: ChatMessage = {
    id: `u-${Date.now()}`,
    role: "user",
    content: text,
  };
  const assistantId = `a-${Date.now()}`;
  const assistantMsg: ChatMessage = {
    id: assistantId,
    role: "assistant",
    content: "",
  };

  chat.messages = [...chat.messages, userMsg, assistantMsg];
  chat.activity = [];
  chat.streaming = true;
  chat.streamStartedAt = Date.now();
  chat.abortController = new AbortController();
  currentReasoningStepId = null;

  try {
    for await (const event of client.streamChat({
      user_id: profileName.value,
      message: text,
      skill: "assistant",
      signal: chat.abortController.signal,
    })) {
      handleEvent(event, assistantId);
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      appendToken(assistantId, " [stopped]");
    } else {
      appendToken(assistantId, `\n\n[stream failed: ${String(err)}]`);
    }
  } finally {
    chat.streaming = false;
    chat.streamStartedAt = null;
    chat.abortController = null;
  }
}

function handleEvent(event: ChatEvent, assistantId: string) {
  switch (event.type) {
    case "token":
      appendToken(assistantId, event.content);
      break;
    case "recall": {
      attachRecallHits(assistantId, event.recall_hits);
      const hits = event.recall_hits ?? [];
      const n = hits.length;
      pushActivity({
        kind: "recall",
        label: `recall · ${n} ${n === 1 ? "memory" : "memories"}`,
      });
      break;
    }
    case "provenance": {
      attachProvenance(assistantId, event.provenance);
      const p = event.provenance;
      const isCloud = !!p?.cloud_call;
      const modelPart = p?.model ?? "";
      const recalledPart = p?.memories_recalled ? ` · ${p.memories_recalled} recalled` : "";
      const latencyPart = p?.latency_ms ? ` · ${p.latency_ms}ms` : "";
      pushActivity({
        kind: "provenance",
        cloud: isCloud,
        label: `${isCloud ? "cloud" : "local"} · ${modelPart}${recalledPart}${latencyPart}`,
        detail: p?.rationale != null ? String(p.rationale) : undefined,
      });
      break;
    }
    case "tool_call":
      pushActivity({
        kind: "tool",
        label: event.tool_name,
        detail: formatToolCall(event.tool_name, event.tool_args),
      });
      break;
    case "tool_result": {
      const resultSnippet = String(event.tool_result ?? "").slice(0, 120);
      pushActivity({
        kind: "tool_result",
        label: `→ ${resultSnippet}`,
      });
      break;
    }
    case "reasoning": {
      if (currentReasoningStepId !== null) {
        // Append to the existing reasoning step for this turn.
        chat.activity = chat.activity.map((s) =>
          s.id === currentReasoningStepId
            ? { ...s, detail: (s.detail ?? "") + event.content }
            : s,
        );
      } else {
        const newStep: ActivityStep = {
          id: `act-${Date.now()}-${Math.random()}`,
          ts: Date.now(),
          kind: "reasoning",
          label: "thinking",
          detail: event.content,
        };
        chat.activity = [...chat.activity, newStep];
        currentReasoningStepId = newStep.id;
      }
      break;
    }
    case "error":
      pushActivity({ kind: "error", label: "error", detail: event.content });
      appendToken(assistantId, `\n\n[error: ${event.content}]`);
      break;
    case "done":
      pushActivity({ kind: "done", label: "done" });
      break;
  }
}

export function cancelStream(): void {
  chat.abortController?.abort();
}

export async function voteRecall(
  ref: string,
  vote: "up" | "down" | "clear",
): Promise<void> {
  if (!ref) return;
  await client.postMemoryFeedback(profileName.value, ref, vote);
}

export function regenerateLast(): void {
  // Find the most recent user message, drop everything after it, resend it.
  const lastUserIdx = [...chat.messages]
    .reverse()
    .findIndex((m) => m.role === "user");
  if (lastUserIdx < 0) return;
  const realIdx = chat.messages.length - 1 - lastUserIdx;
  const userText = chat.messages[realIdx].content;
  chat.messages = chat.messages.slice(0, realIdx);
  void sendMessage(userText);
}
