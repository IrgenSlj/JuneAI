// A stand-in for Ollama's OpenAI-compatible API on a random loopback port, so
// the chat test runs the real daemon end to end without a real model.
import { type Server, createServer } from "node:http";

export interface FakeOllama {
  host: string;
  /** While set, replies stop after their first piece and wait (to test a crash mid-reply). */
  hold: Promise<void> | undefined;
  requests: { path: string; body: string }[];
  close(): Promise<void>;
}

export async function startFakeOllama(reply: string[]): Promise<FakeOllama> {
  const requests: FakeOllama["requests"] = [];
  const state: { hold: Promise<void> | undefined } = { hold: undefined };
  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk: Buffer) => (body += chunk.toString()));
    req.on("end", () => {
      requests.push({ path: req.url ?? "", body });
      if (req.url === "/v1/models") {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({ data: [{ id: "llama-test" }] }));
      }
      if (req.url === "/v1/chat/completions") {
        res.writeHead(200, { "content-type": "text/event-stream" });
        void (async () => {
          for (const [i, text] of reply.entries()) {
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`);
            if (i === 0 && state.hold !== undefined) await state.hold;
          }
          res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`);
          res.end("data: [DONE]\n\n");
        })();
        return;
      }
      res.writeHead(404).end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("fake ollama did not bind");
  return {
    host: `127.0.0.1:${address.port}`,
    get hold() {
      return state.hold;
    },
    set hold(value) {
      state.hold = value;
    },
    requests,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
