// Starts the real daemon (the built bundle) serving the built UI, as the Mac
// app will, and hands the test its URL.
import { type ChildProcess, spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export interface Daemon {
  origin: string;
  token: string;
  root: string;
  stop(): void;
  /** SIGKILL: no shutdown code runs, as in a crash or a force quit. */
  crash(): Promise<void>;
}

/** Pass `root` to restart on the same config and vault; `env` adds variables such as OLLAMA_HOST. */
export async function startDaemon(options: { root?: string; env?: Record<string, string> } = {}): Promise<Daemon> {
  const root = options.root ?? (await mkdtemp(join(tmpdir(), "june-e2e-")));
  const token = `e2e-${Math.random().toString(36).slice(2)}`;
  const child: ChildProcess = spawn(
    process.execPath,
    ["../server/dist/server.mjs", "serve", "--static", "build", "--config-dir", join(root, "config"), "--memory-secrets"],
    { env: { ...process.env, ...options.env, JUNE_TOKEN: token }, stdio: ["ignore", "pipe", "inherit"] },
  );
  const port = await new Promise<number>((resolve, reject) => {
    let out = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      out += chunk.toString();
      const match = /JUNE_READY (\{.*\})/.exec(out);
      if (match?.[1]) resolve((JSON.parse(match[1]) as { port: number }).port);
    });
    child.once("exit", (code) => reject(new Error(`daemon exited early with ${code}`)));
  });
  return {
    origin: `http://127.0.0.1:${port}`,
    token,
    root,
    stop: () => void child.kill(),
    crash: () =>
      new Promise((resolve) => {
        child.once("exit", () => resolve());
        child.kill("SIGKILL");
      }),
  };
}
