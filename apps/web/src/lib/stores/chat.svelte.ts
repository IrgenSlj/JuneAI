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
import { type ChatEvent, type ChatMessage } from "@june/ui";
import { client } from "$lib/api.js";

const USER_ID = "local";

export const chat = $state({
  messages: [] as ChatMessage[],
  streaming: false,
  abortController: null as AbortController | null,
  /** When a stream is running, the timestamp it started — used to render
   * an elapsed-time hint if the model takes a while to emit its first token. */
  streamStartedAt: null as number | null,
});

function pushMessage(msg: ChatMessage) {
  chat.messages = [...chat.messages, msg];
}

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

function formatToolCall(
  name: string,
  args: Record<string, unknown> | undefined,
): string {
  const argText = args && Object.keys(args).length ? JSON.stringify(args) : "(no args)";
  return `${name} ${argText}`;
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
  chat.streaming = true;
  chat.streamStartedAt = Date.now();
  chat.abortController = new AbortController();

  try {
    for await (const event of client.streamChat({
      user_id: USER_ID,
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
    case "recall":
      attachRecallHits(assistantId, event.recall_hits);
      break;
    case "tool_call":
      pushMessage({
        id: `t-${Date.now()}-${Math.random()}`,
        role: "tool",
        content: formatToolCall(event.tool_name, event.tool_args),
        toolName: event.tool_name,
      });
      break;
    case "tool_result":
      // Attach onto the most recent tool bubble with the same name that
      // hasn't already been answered, preserving transcript order.
      chat.messages = chat.messages.map((m) =>
        m.role === "tool" &&
        m.toolName === event.tool_name &&
        !m.content.includes("→")
          ? { ...m, content: `${m.content}\n  → ${event.tool_result}` }
          : m,
      );
      break;
    case "error":
      appendToken(assistantId, `\n\n[error: ${event.content}]`);
      break;
    case "done":
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
  await client.postMemoryFeedback(USER_ID, ref, vote);
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
