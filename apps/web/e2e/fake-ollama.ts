// A stand-in for Ollama's OpenAI-compatible API on a random loopback port, so
// the chat test runs the real daemon end to end without a real model.
import { type Server, createServer } from "node:http";

export interface FakeOllama {
  host: string;
  requests: { path: string; body: string }[];
  close(): Promise<void>;
}

export async function startFakeOllama(reply: string[]): Promise<FakeOllama> {
  const requests: FakeOllama["requests"] = [];
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
        for (const text of reply) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`);
        res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`);
        return res.end("data: [DONE]\n\n");
      }
      res.writeHead(404).end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("fake ollama did not bind");
  return {
    host: `127.0.0.1:${address.port}`,
    requests,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}
