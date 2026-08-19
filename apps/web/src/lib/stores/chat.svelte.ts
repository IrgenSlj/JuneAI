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
import { loadSystem } from "$lib/stores/system.svelte.js";
import { syncLiveTurn, endLiveTurn } from "$lib/stores/glass.svelte.js";

export const chat = $state({
  messages: [] as ChatMessage[],
  activity: [] as ActivityStep[],
  activityOpen: false,
  streaming: false,
  abortController: null as AbortController | null,
  /** When a stream is running, the timestamp it started — used to render
   * an elapsed-time hint if the model takes a while to emit its first token. */
  streamStartedAt: null as number | null,
  /** Set when a networked tool was blocked by Local-only mode this turn, so the
   * UI can offer a one-click switch + retry. Cleared on the next send. */
  blockedTool: null as { name: string } | null,
  /** Set when the guard withheld a consequential action pending the user's
   * explicit approval (network egress, code execution). Distinct from
   * blockedTool: resolved by approving the one action, not changing the dial. */
  pendingApproval: null as { name: string; actionClass: string; reason: string } | null,
  /** Tools the user approved for this conversation (the guard's allow-list).
   * Sent with every turn so an approved action runs without asking again. */
  approvedTools: [] as string[],
  /** The last user message, kept so a mode-switch can retry the same request. */
  lastUserMessage: "" as string,
});

/** Tracks the activity step id used to accumulate reasoning for the current turn. */
let currentReasoningStepId: string | null = null;

/** Tracks the turn_id of the current in-flight stream for glass-store mirroring. */
let currentTurnId: string | null = null;

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

/**
 * Accept June's offer: switch the privacy dial to Private-by-default and retry
 * the last request. Surfaced as a one-click button when a tool was blocked.
 */
export async function switchToPrivateAndRetry(): Promise<void> {
  const retry = chat.lastUserMessage;
  try {
    await client.updatePrivacyDial("private_by_default");
  } catch {
    // If the switch fails, leave the offer in place rather than silently retrying.
    return;
  }
  chat.blockedTool = null;
  void loadSystem(); // refresh the header's mode chip
  if (retry) await sendMessage(retry);
}

/**
 * Approve the single consequential action June asked about and retry the last
 * request. The tool joins this conversation's allow-list, so the guard waives
 * it from here on (taint-flagged network actions still always ask).
 */
export async function approveAndRetry(): Promise<void> {
  const pending = chat.pendingApproval;
  if (!pending) return;
  if (!chat.approvedTools.includes(pending.name)) {
    chat.approvedTools = [...chat.approvedTools, pending.name];
  }
  chat.pendingApproval = null;
  const retry = chat.lastUserMessage;
  if (retry) await sendMessage(retry);
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
  chat.blockedTool = null;
  chat.pendingApproval = null;
  chat.lastUserMessage = text;
  currentReasoningStepId = null;

  try {
    for await (const event of client.streamChat({
      user_id: profileName.value,
      message: text,
      skill: "assistant",
      approved_tools: chat.approvedTools,
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
  if (event.turn_id) currentTurnId = event.turn_id;
  switch (event.type) {
    case "token":
      appendToken(assistantId, event.content);
      break;
    case "recall": {
      attachRecallHits(assistantId, event.recall_hits);
      const hits = event.recall_hits ?? [];
      const n = hits.length;
      const recallDetail = hits
        .map((h) => {
          const score = (h as { score?: number }).score;
          const recency = (h as { recency?: number }).recency;
          const frequency = (h as { frequency?: number }).frequency;
          const relevance = (h as { relevance?: number }).relevance;
          const snippet = ((h as { text?: string }).text ?? "").slice(0, 80);
          const scorePart = score != null ? score.toFixed(2) : "?";
          const recPart = recency != null ? ` rec ${recency.toFixed(2)}` : "";
          const freqPart = frequency != null ? ` freq ${frequency.toFixed(2)}` : "";
          const relPart = relevance != null ? ` rel ${relevance.toFixed(2)}` : "";
          return `${scorePart}${recPart}${freqPart}${relPart} · ${snippet}`;
        })
        .join("\n");
      pushActivity({
        kind: "recall",
        label: `recall · ${n} ${n === 1 ? "memory" : "memories"}`,
        detail: recallDetail || undefined,
      });
      break;
    }
    case "provenance": {
      attachProvenance(assistantId, event.provenance);
      const p = event.provenance;
      const isCloud = !!p?.cloud_call;
      const modelPart = p?.model ?? "";
      const recalledPart = p?.memories_recalled ? ` · ${p.memories_recalled} recalled` : "";
      // Writes are shown as well as reads. June has always said what it took
      // out of memory and never what it put in, so storing or deleting one of
      // the user's memories was the one action the turn frame could not see.
      const written = (p as { memories_written?: number })?.memories_written ?? 0;
      const writtenPart = written ? ` · ${written} remembered` : "";
      const latencyPart = p?.latency_ms ? ` · ${p.latency_ms}ms` : "";
      const egress = (p as { egress?: string[] })?.egress ?? [];
      const egressPart = egress.length ? ` · egress: ${egress.join(", ")}` : "";
      const _fmtTok = (n: number | undefined) =>
        n != null && n > 0 ? (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)) : "";
      const _inTok = _fmtTok((p as { input_tokens?: number })?.input_tokens);
      const _outTok = _fmtTok((p as { output_tokens?: number })?.output_tokens);
      const tokenPart = _inTok && _outTok ? ` · ${_inTok}/${_outTok} tok` : "";
      pushActivity({
        kind: "provenance",
        cloud: isCloud,
        network: egress.length > 0,
        label: `${isCloud ? "cloud" : "local"} · ${modelPart}${recalledPart}${writtenPart}${latencyPart}${tokenPart}${egressPart}`,
        detail: p?.rationale != null ? String(p.rationale) : undefined,
      });
      break;
    }
    case "tool_call":
      pushActivity({
        kind: "tool",
        label: `tool · ${event.tool_name}${event.network ? " · egress" : ""}`,
        network: event.network,
        // Prefer the brain's full args body; fall back to the formatted args.
        detail: event.detail || formatToolCall(event.tool_name, event.tool_args),
      });
      break;
    case "tool_result": {
      const resultSnippet = String(event.tool_result ?? "").slice(0, 120);
      pushActivity({
        kind: "tool_result",
        label: `→ ${resultSnippet}`,
        detail: event.detail || String(event.tool_result ?? ""),
      });
      break;
    }
    case "prompt":
      pushActivity({
        kind: "prompt",
        label: event.content || "prompt assembled",
        detail: event.detail,
      });
      break;
    case "iteration":
      pushActivity({
        kind: "iteration",
        label: event.content || "iteration",
        detail: event.detail,
      });
      break;
    case "compaction":
      pushActivity({
        kind: "compaction",
        label: event.content || "conversation compacted",
        detail: event.detail,
      });
      break;
    case "model_call":
      pushActivity({ kind: "model_call", label: event.content || "model call", detail: event.detail });
      break;
    case "tool_blocked":
      if (event.needs_approval) {
        // A consequential action June won't take without explicit approval.
        chat.pendingApproval = {
          name: event.tool_name,
          actionClass: event.action_class ?? "",
          reason: event.detail ?? "",
        };
        pushActivity({
          kind: "tool_blocked",
          label: `blocked · ${event.tool_name} (needs approval)`,
          detail: event.detail,
        });
      } else {
        chat.blockedTool = { name: event.tool_name };
        pushActivity({
          kind: "tool_blocked",
          label: `blocked · ${event.tool_name} (local-only)`,
          network: true,
          detail: event.detail,
        });
      }
      break;
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
      if (currentTurnId) endLiveTurn(currentTurnId);
      currentTurnId = null;
      break;
  }
  // Mirror the live turn into the glass store after every event.
  // Guard so a glass-store error can never break the chat stream path.
  if (currentTurnId) {
    try {
      syncLiveTurn(
        currentTurnId,
        chat.activity,
        Math.floor((chat.streamStartedAt ?? Date.now()) / 1000),
      );
    } catch {
      // Best-effort: glass-store error must not break chat.
    }
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
