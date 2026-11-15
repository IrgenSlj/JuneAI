// API keys in the macOS keychain, through Apple's own /usr/bin/security tool,
// so June links no native code and needs no keychain library.
//
// The secret never appears in a command's arguments, where any local process
// could read it with `ps`: writes go through `security -i`, which reads its
// command from stdin. `security -i` exits 0 even for a malformed command, so
// every write is read back and compared before it counts as saved.
import { spawn } from "node:child_process";
import { type SecretStore, SecretStoreError, checkSecretAccount, checkSecretValue } from "@june/core";

export interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
}

/** Runs a program without a shell. Injected so tests never touch the real keychain. */
export type Runner = (program: string, args: string[], stdin?: string) => Promise<RunResult>;

const SECURITY = "/usr/bin/security";
const NOT_FOUND = 44;

export const runProgram: Runner = (program, args, stdin) =>
  new Promise((resolve, reject) => {
    const child = spawn(program, args, { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.once("error", reject);
    child.once("close", (code) => resolve({ code: code ?? 1, stdout, stderr }));
    child.stdin.end(stdin ?? "");
  });

/** Quote for `security -i`, which splits on spaces and honours "..." with backslash escapes. */
function quote(value: string): string {
  return `"${value.replace(/[\\"]/g, (c) => `\\${c}`)}"`;
}

function failed(action: string, result: RunResult): SecretStoreError {
  // stderr from security names the call and an OS status; it never echoes the secret.
  return new SecretStoreError(`keychain ${action} failed (exit ${result.code}): ${result.stderr.trim().split("\n").at(-1) ?? ""}`);
}

export function macKeychain(options: { service?: string; run?: Runner } = {}): SecretStore {
  const service = options.service ?? "ai.june.app";
  const run = options.run ?? runProgram;

  async function get(account: string): Promise<string | undefined> {
    const result = await run(SECURITY, ["find-generic-password", "-s", service, "-a", checkSecretAccount(account), "-w"]);
    if (result.code === NOT_FOUND) return undefined;
    if (result.code !== 0) throw failed("read", result);
    return result.stdout.replace(/\n$/, "");
  }

  return {
    get,

    async set(account, secret) {
      checkSecretAccount(account);
      checkSecretValue(secret);
      const command = ["add-generic-password", "-U", "-s", quote(service), "-a", quote(account), "-l", quote(`June: ${account}`), "-w", quote(secret)].join(" ");
      const result = await run(SECURITY, ["-i"], `${command}\n`);
      if (result.code !== 0) throw failed("write", result);
      if ((await get(account)) !== secret) throw new SecretStoreError("keychain write did not read back as written; the key was not saved");
    },

    async remove(account) {
      const result = await run(SECURITY, ["delete-generic-password", "-s", service, "-a", checkSecretAccount(account)]);
      if (result.code !== 0 && result.code !== NOT_FOUND) throw failed("delete", result);
    },
  };
}

/** Where keys go on this machine. Only macOS for now (PLAN.md: Windows after the gate). */
export function platformSecretStore(): SecretStore {
  if (process.platform === "darwin") return macKeychain();
  const refuse = async (): Promise<never> => {
    throw new SecretStoreError("June stores keys only in the macOS keychain for now, and never in a file");
  };
  return { get: refuse, set: refuse, remove: refuse };
}
