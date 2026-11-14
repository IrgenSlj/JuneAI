// `pnpm dev`: the daemon on a random port plus Vite with hot reload in front of
// it, then one URL to open. Dev state lives in .dev/ (git-ignored), so it never
// touches the real app's settings.
import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";

const token = randomBytes(24).toString("base64url");
const build = spawnSync("pnpm", ["--filter", "@june/server", "build"], { stdio: "inherit" });
if (build.status !== 0) process.exit(build.status ?? 1);

const daemon = spawn(process.execPath, ["apps/server/dist/server.mjs", "serve", "--config-dir", ".dev/config"], {
  env: { ...process.env, JUNE_TOKEN: token },
  stdio: ["pipe", "pipe", "inherit"],
});

const port = await new Promise((resolve, reject) => {
  let out = "";
  daemon.stdout.on("data", (chunk) => {
    out += chunk;
    const match = /JUNE_READY (\{.*\})/.exec(out);
    if (match) resolve(JSON.parse(match[1]).port);
  });
  daemon.once("exit", (code) => reject(new Error(`daemon exited with ${code}`)));
});

const vite = spawn("pnpm", ["--filter", "@june/web", "dev"], {
  env: { ...process.env, JUNE_DAEMON_URL: `http://127.0.0.1:${port}` },
  stdio: "inherit",
});

setTimeout(() => console.log(`\n  June (dev): http://127.0.0.1:5173/#token=${token}\n`), 1500);

const stop = () => {
  vite.kill();
  daemon.kill();
  process.exit(0);
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
vite.on("exit", stop);
