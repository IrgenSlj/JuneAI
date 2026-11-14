import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type RunningServer, startServer } from "./server.js";

const TOKEN = "test-token-0123456789";
let root: string;
let server: RunningServer;

/** Raw HTTP so tests can send any Host header, which fetch() forbids. */
function call(
  path: string,
  opts: { method?: string; token?: string | null; host?: string; body?: unknown } = {},
): Promise<{ status: number; headers: Record<string, string | string[] | undefined>; body: string }> {
  return new Promise((resolve, reject) => {
    const payload = opts.body === undefined ? undefined : JSON.stringify(opts.body);
    const req = request(
      {
        host: "127.0.0.1",
        port: server.port,
        path,
        method: opts.method ?? "GET",
        headers: {
          host: opts.host ?? `127.0.0.1:${server.port}`,
          ...(opts.token === null ? {} : { "x-june-token": opts.token ?? TOKEN }),
          ...(payload === undefined ? {} : { "content-type": "application/json", "content-length": Buffer.byteLength(payload) }),
        },
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => (body += chunk));
        res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body }));
      },
    );
    req.on("error", reject);
    req.end(payload);
  });
}

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "june-server-"));
  await mkdir(join(root, "static"));
  await writeFile(join(root, "static", "index.html"), "<!doctype html><title>June</title><script>window.boot=1</script>");
  await writeFile(join(root, "secret.txt"), "outside the static folder");
  server = await startServer({
    token: TOKEN,
    staticDir: join(root, "static"),
    configDir: join(root, "config"),
    home: join(root, "home"),
    version: "test",
  });
});

afterEach(async () => {
  await server.close();
});

describe("daemon security", () => {
  it("listens on loopback only, on a random port", () => {
    expect(server.host).toBe("127.0.0.1");
    expect(server.port).toBeGreaterThan(0);
  });

  it("answers health without a token, and nothing private in it", async () => {
    const res = await call("/api/health", { token: null });
    expect(res.status).toBe(200);
    expect(JSON.parse(res.body)).toEqual({ ok: true, name: "june", version: "test" });
  });

  it("refuses an API request without the token, or with a wrong one", async () => {
    expect((await call("/api/vault", { token: null })).status).toBe(401);
    expect((await call("/api/vault", { token: "wrong" })).status).toBe(401);
    expect((await call("/api/vault")).status).toBe(200);
  });

  it("refuses a foreign Host header (DNS rebinding), even with the token", async () => {
    expect((await call("/api/vault", { host: "evil.example:80" })).status).toBe(403);
    expect((await call("/", { host: "evil.example" })).status).toBe(403);
    expect((await call("/api/health", { host: `localhost:${server.port}` })).status).toBe(200);
  });

  it("sends no CORS headers and forbids framing", async () => {
    const res = await call("/");
    expect(res.headers["access-control-allow-origin"]).toBeUndefined();
    expect(res.headers["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(res.headers["x-content-type-options"]).toBe("nosniff");
    expect(res.headers["referrer-policy"]).toBe("no-referrer");
  });

  it("allows the page's own inline boot script by hash, and nothing else inline", async () => {
    const res = await call("/");
    const csp = String(res.headers["content-security-policy"]);
    expect(csp).toMatch(/script-src 'self' 'sha256-[A-Za-z0-9+/=]+'/);
    expect(csp.split("script-src")[1]?.split(";")[0]).not.toContain("unsafe-inline");
  });

  it("never serves files outside the static folder", async () => {
    for (const path of ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/static/../../secret.txt"]) {
      const res = await call(path);
      expect(res.body).not.toContain("outside the static folder");
    }
  });
});

describe("static UI", () => {
  it("serves index.html for the app's routes", async () => {
    for (const path of ["/", "/chat", "/settings/models"]) {
      const res = await call(path, { token: null });
      expect(res.status).toBe(200);
      expect(res.headers["content-type"]).toContain("text/html");
      expect(res.body).toContain("<title>June</title>");
    }
  });

  it("answers unknown API paths with JSON 404, not the page", async () => {
    const res = await call("/api/nope");
    expect(res.status).toBe(404);
    expect(JSON.parse(res.body)).toEqual({ error: "not found" });
  });
});

describe("vault", () => {
  it("starts with no vault and suggests ~/June", async () => {
    const res = await call("/api/vault");
    expect(JSON.parse(res.body)).toEqual({ path: null, defaultPath: join(root, "home", "June") });
  });

  it("opens (creating) a vault, expands ~, and remembers it across restarts", async () => {
    const res = await call("/api/vault/open", { method: "POST", body: { path: "~/June" } });
    expect(res.status).toBe(200);
    const vault = join(root, "home", "June");
    expect(JSON.parse(res.body).path).toBe(vault);
    expect(JSON.parse(await readFile(join(root, "config", "config.json"), "utf8"))).toEqual({ vault });

    await server.close();
    server = await startServer({ token: TOKEN, configDir: join(root, "config"), home: join(root, "home"), version: "test" });
    expect(JSON.parse((await call("/api/vault")).body).path).toBe(vault);
  });

  it("rejects a relative path or a file", async () => {
    const relative = await call("/api/vault/open", { method: "POST", body: { path: "June" } });
    expect(relative.status).toBe(400);
    const file = await call("/api/vault/open", { method: "POST", body: { path: join(root, "secret.txt") } });
    expect(file.status).toBe(400);
    expect(JSON.parse(file.body).error).toContain("not a folder");
  });

  it("rejects malformed and oversized bodies", async () => {
    expect((await call("/api/vault/open", { method: "POST", body: "nope" })).status).toBe(400);
    expect((await call("/api/vault/open", { method: "POST", body: { path: "x".repeat(2_000_000) } })).status).toBe(413);
  });
});
