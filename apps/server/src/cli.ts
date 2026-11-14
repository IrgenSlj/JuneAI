// `june serve`: start the daemon and print where to open it.
//
//   june serve [--port N] [--static DIR] [--config-dir DIR] [--exit-with-stdin]
//
// The token comes from JUNE_TOKEN (how the Mac app passes it: environment, not
// argv, because any local user can read another process's arguments) or is
// generated here. The first stdout line is for machines:
//   JUNE_READY {"port":53121}
// then a line for people with the URL to open.
import { randomBytes } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { defaultConfigDir } from "./config.js";
import { startServer } from "./server.js";

declare const __JUNE_VERSION__: string | undefined;
const VERSION = typeof __JUNE_VERSION__ === "string" ? __JUNE_VERSION__ : "0.0.0-dev";

async function main(argv: string[]): Promise<void> {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      port: { type: "string" },
      static: { type: "string" },
      "config-dir": { type: "string" },
      "exit-with-stdin": { type: "boolean", default: false },
    },
  });
  const command = positionals[0] ?? "serve";
  if (command !== "serve") {
    console.error(`unknown command: ${command}\nusage: june serve [--port N] [--static DIR] [--config-dir DIR] [--exit-with-stdin]`);
    process.exit(2);
  }

  const token = process.env.JUNE_TOKEN || randomBytes(32).toString("base64url");
  const here = dirname(fileURLToPath(import.meta.url));
  const staticDir = values.static === undefined ? join(here, "web") : resolve(values.static);
  const server = await startServer({
    token,
    staticDir,
    configDir: values["config-dir"] === undefined ? defaultConfigDir() : resolve(values["config-dir"]),
    version: VERSION,
    port: values.port === undefined ? 0 : Number(values.port),
  });

  console.log(`JUNE_READY ${JSON.stringify({ port: server.port })}`);
  if (!process.env.JUNE_TOKEN) console.log(`June is running. Open: http://127.0.0.1:${server.port}/#token=${token}`);
  else console.log(`June is running on 127.0.0.1:${server.port}`);

  const stop = async (): Promise<void> => {
    await server.close();
    process.exit(0);
  };
  process.on("SIGINT", () => void stop());
  process.on("SIGTERM", () => void stop());
  // The Mac app holds our stdin open; if it dies in any way, the pipe closes
  // and we exit instead of living on as an orphan holding the vault.
  if (values["exit-with-stdin"]) {
    process.stdin.on("end", () => void stop());
    process.stdin.on("close", () => void stop());
    process.stdin.resume();
  }
}

main(process.argv.slice(2)).catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
