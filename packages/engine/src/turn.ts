// The turn loop (PLAN.md, "The turn loop"). Plain code, one step after
// another, each a place where it can stop:
//   1. load the thread and assemble context (which notes attach: none until M5)
//   2. build the send label and yield it, before anything leaves
//   3. append the send receipt; if that fails, nothing is sent
//   4. append the user turn to the thread file, so a crash loses nothing typed
//
// Steps 3-4 and 6 hold the thread's lock and re-read the file, because the
// daemon is not its only writer (a second tab, Obsidian, an importer). If the
// thread changed after the label was built, the send is refused before the
// receipt: the label must describe exactly what leaves.
//   5. stream the provider; on error, say so and stop. No reply is invented
//   6. append the assistant turn, then yield `saved` (the only proof of saving) and `done`
import { type Json, type ReceiptLog, type Thread, type Turn, type VaultFs, type VaultLock, newId, parseThread, serializeThread, threadPath } from "@june/core";
import type { ChatMessage, Provider, Route } from "@june/providers";

/** What is about to leave, built by code from the request itself, never by a model. */
export interface SendLabel {
  route: Route;
  model: string;
  parts: { kind: "system" | "history" | "note" | "attachment" | "draft"; id?: string; chars: number }[];
  noteIds: string[];
  totalChars: number;
}

export type TurnEvent =
  | { type: "label"; label: SendLabel }
  | { type: "text"; text: string }
  | { type: "usage"; inputTokens?: number; outputTokens?: number }
  | { type: "saved"; path: string; threadId: string; userTurn: string; assistantTurn: string; receipt: number }
  | { type: "error"; stage: "input" | "thread" | "receipt" | "provider" | "save"; message: string }
  | { type: "done" };

export interface TurnDeps {
  fs: VaultFs;
  receipts: ReceiptLog;
  provider: Provider;
  /** Serialises writes to one thread across tabs and processes. */
  lock: VaultLock;
  clock?: () => Date;
}

export interface TurnInput {
  /** Vault path of the thread file. */
  path: string;
  model: string;
  text: string;
  signal?: unknown;
}

const message = (error: unknown): string => (error instanceof Error ? error.message : String(error));

async function readThreadFile(fs: VaultFs, path: string): Promise<{ file: string; thread: Thread }> {
  const file = await fs.read(path);
  if (file === undefined) throw new Error(`no thread at ${path}`);
  return { file, thread: parseThread(file) };
}

/** Lock names are plain lowercase; thread ids are ULIDs with an underscore. */
const threadLock = (thread: Thread): string => `thread-${thread.id.toLowerCase().replace(/[^a-z0-9]/g, "-")}`;

class ThreadChanged extends Error {}

/** The next free turn id in this thread: m1, m2, ... skipping any already used (imports keep their own ids). */
function nextTurnId(thread: Thread): string {
  const used = new Set(thread.turns.map((t) => t.id));
  let n = thread.turns.length + 1;
  while (used.has(`m${n}`)) n++;
  return `m${n}`;
}

function toMessages(turns: Turn[]): ChatMessage[] {
  return turns.flatMap((t) => (t.role === "tool" ? [] : [{ role: t.role, content: t.text }]));
}

export async function createThread(fs: VaultFs, options: { title: string; clock?: () => Date }): Promise<{ path: string; thread: Thread }> {
  const now = (options.clock ?? (() => new Date()))().toISOString();
  const thread: Thread = { id: newId("thread"), title: options.title, source: "june", started: now, updated: now, turns: [] };
  const path = threadPath(thread);
  if ((await fs.read(path)) !== undefined) throw new Error(`a thread already exists at ${path}`);
  await fs.write(path, serializeThread(thread));
  return { path, thread };
}

/**
 * Continue a thread from one of its turns as a new thread, on any model. The
 * new file holds copies of the turns up to and including that one, and names
 * its origin in `parent` ("th_...#m3"); the original file is not touched.
 */
export async function branchThread(
  fs: VaultFs,
  options: { path: string; turnId: string; clock?: () => Date },
): Promise<{ path: string; thread: Thread }> {
  const { thread: source } = await readThreadFile(fs, options.path);
  const index = source.turns.findIndex((t) => t.id === options.turnId);
  if (index === -1) throw new Error(`no turn ${options.turnId} in this thread`);
  const now = (options.clock ?? (() => new Date()))().toISOString();
  const thread: Thread = {
    id: newId("thread"),
    title: `${source.title} (branch)`,
    source: "june",
    started: now,
    updated: now,
    parent: `${source.id}#${options.turnId}`,
    ...(source.lane === undefined ? {} : { lane: source.lane }),
    turns: source.turns.slice(0, index + 1).map((t) => ({ ...t })),
  };
  const path = threadPath(thread);
  await fs.write(path, serializeThread(thread));
  return { path, thread };
}

export async function* runTurn(deps: TurnDeps, input: TurnInput): AsyncGenerator<TurnEvent> {
  const clock = deps.clock ?? (() => new Date());
  const { route } = deps.provider;
  if (input.text.trim() === "") {
    yield { type: "error", stage: "input", message: "the message is empty" };
    yield { type: "done" };
    return;
  }

  // 1. Load and assemble.
  let thread: Thread;
  let snapshot: string;
  try {
    ({ file: snapshot, thread } = await readThreadFile(deps.fs, input.path));
  } catch (error) {
    yield { type: "error", stage: "thread", message: message(error) };
    yield { type: "done" };
    return;
  }
  const history = toMessages(thread.turns);
  const messages: ChatMessage[] = [...history, { role: "user", content: input.text }];

  // 2. The label, from the exact messages that will be sent.
  const historyChars = history.reduce((sum, m) => sum + m.content.length, 0);
  const parts: SendLabel["parts"] = [...(historyChars > 0 ? [{ kind: "history" as const, chars: historyChars }] : []), { kind: "draft", chars: input.text.length }];
  const label: SendLabel = { route, model: input.model, parts, noteIds: [], totalChars: historyChars + input.text.length };
  yield { type: "label", label };

  // 3-4. Under the thread's lock: check nothing changed, write the receipt, then the user turn.
  const userTurnId = nextTurnId(thread);
  const sentAt = clock().toISOString();
  const userTurn: Turn = { id: userTurnId, role: "user", at: sentAt, text: input.text };
  let receiptSeq = 0;
  let failure: TurnEvent | undefined;
  try {
    await deps.lock.exclusive(threadLock(thread), async () => {
      if ((await deps.fs.read(input.path)) !== snapshot) throw new ThreadChanged();
      try {
        const receipt = await deps.receipts.append({
          kind: "send",
          actor: "june",
          targets: [thread.id, `${thread.id}#${userTurnId}`],
          payload: { model: input.model, messages: messages.map((m) => ({ ...m })) } as unknown as Json,
          meta: { route: route.id, host: route.host, model: input.model, leavesDevice: route.leavesDevice, chars: label.totalChars },
        });
        receiptSeq = receipt.seq;
      } catch (error) {
        failure = { type: "error", stage: "receipt", message: `nothing was sent: the receipt could not be written (${message(error)})` };
        return;
      }
      try {
        await deps.fs.write(input.path, serializeThread({ ...thread, updated: sentAt, turns: [...thread.turns, userTurn] }));
      } catch (error) {
        failure = { type: "error", stage: "save", message: `nothing was sent: your message could not be saved (${message(error)})` };
      }
    });
  } catch (error) {
    failure =
      error instanceof ThreadChanged
        ? { type: "error", stage: "thread", message: "nothing was sent: this thread changed while the message was being prepared; send it again" }
        : { type: "error", stage: "thread", message: `nothing was sent: ${message(error)}` };
  }
  if (failure !== undefined) {
    yield failure;
    yield { type: "done" };
    return;
  }

  // 5. Stream. Only a "done" from the provider makes a reply.
  let text = "";
  let usage: { inputTokens?: number; outputTokens?: number } = {};
  let finished = false;
  let streamFailure: string | undefined;
  try {
    for await (const event of deps.provider.stream({ model: input.model, messages, signal: input.signal })) {
      if (event.type === "text") {
        text += event.text;
        yield event;
      } else if (event.type === "usage") {
        usage = { ...(event.inputTokens === undefined ? {} : { inputTokens: event.inputTokens }), ...(event.outputTokens === undefined ? {} : { outputTokens: event.outputTokens }) };
        yield { type: "usage", ...usage };
      } else if (event.type === "error") {
        streamFailure = event.message;
        break;
      } else {
        finished = true;
        break;
      }
    }
  } catch (error) {
    // Adapters promise never to throw; one that does still ends the turn cleanly.
    streamFailure = message(error);
  }
  if (!finished) {
    yield { type: "error", stage: "provider", message: streamFailure ?? "the reply stopped before it finished; nothing was saved for it" };
    yield { type: "done" };
    return;
  }

  // 6. The assistant turn, under the lock and on a fresh read, so an edit made meanwhile (in Obsidian, say) is kept.
  try {
    const repliedAt = clock().toISOString();
    const assistantTurnId = await deps.lock.exclusive(threadLock(thread), async () => {
      const { thread: current } = await readThreadFile(deps.fs, input.path);
      const assistantTurn: Turn = {
        id: nextTurnId(current),
        role: "assistant",
        at: repliedAt,
        model: input.model,
        meta: { route: route.id, receipt: receiptSeq, notes: [], ...usage },
        text,
      };
      await deps.fs.write(input.path, serializeThread({ ...current, updated: repliedAt, turns: [...current.turns, assistantTurn] }));
      return assistantTurn.id;
    });
    yield { type: "saved", path: input.path, threadId: thread.id, userTurn: userTurnId, assistantTurn: assistantTurnId, receipt: receiptSeq };
  } catch (error) {
    yield { type: "error", stage: "save", message: `the reply arrived but could not be saved (${message(error)})` };
  }
  yield { type: "done" };
}
