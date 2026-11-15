// M1 "done when": killing the daemon mid-reply loses no user turn, invents no
// reply, and leaves a receipt chain that still verifies.
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { type Daemon, startDaemon } from "./daemon.js";
import { type FakeOllama, startFakeOllama } from "./fake-ollama.js";

let ollama: FakeOllama;
let daemon: Daemon;

test.beforeAll(async () => {
  ollama = await startFakeOllama(["First words", " and the rest."]);
});
test.afterAll(async () => {
  daemon.stop();
  await ollama.close();
});

test("kill -9 mid-reply: the user turn is on disk, no half reply is saved, receipts verify", async ({ request }) => {
  daemon = await startDaemon({ env: { OLLAMA_HOST: ollama.host } });
  const headers = { "x-june-token": daemon.token };
  const vault = join(daemon.root, "Vault");
  await request.post(`${daemon.origin}/api/vault/open`, { headers, data: { path: vault } });
  const thread = (await (await request.post(`${daemon.origin}/api/threads`, { headers, data: { title: "Crash test" } })).json()) as { id: string };

  let release = () => {};
  ollama.hold = new Promise((resolve) => (release = resolve));
  const turn = fetch(`${daemon.origin}/api/threads/${thread.id}/turns`, {
    method: "POST",
    headers: { ...headers, "content-type": "application/json" },
    body: JSON.stringify({ route: "ollama", model: "llama-test", text: "Do not lose this message" }),
  }).then(async (res) => {
    const reader = res.body?.getReader();
    const first = await reader?.read();
    return new TextDecoder().decode(first?.value);
  });
  // Wait until the reply has started streaming, then pull the plug.
  expect(await turn).toContain('"type":"label"');
  await expect.poll(() => ollama.requests.filter((r) => r.path === "/v1/chat/completions").length).toBe(1);
  await daemon.crash();
  release();

  const files = (await readdir(join(vault, "threads"), { recursive: true })).filter((f) => f.endsWith(".md"));
  const markdown = await readFile(join(vault, "threads", files[0] ?? ""), "utf8");
  expect(markdown).toContain("Do not lose this message");
  expect(markdown).not.toContain("First words");

  ollama.hold = undefined;
  daemon = await startDaemon({ root: daemon.root, env: { OLLAMA_HOST: ollama.host } });
  const after = { "x-june-token": daemon.token };
  const detail = (await (await request.get(`${daemon.origin}/api/threads/${thread.id}`, { headers: after })).json()) as { turns: { role: string }[] };
  expect(detail.turns.map((t) => t.role)).toEqual(["user"]);
  const verify = await request.post(`${daemon.origin}/api/receipts/verify`, { headers: after });
  expect(verify.status()).toBe(200);
  expect(await verify.json()).toEqual({ ok: true, count: 1 });
});
