import { randomBytes } from "node:crypto";
import { SecretStoreError, memorySecretStore } from "@june/core";
import { secretStoreContract } from "@june/core/testing";
import { afterAll, describe, expect, it } from "vitest";
import { type RunResult, type Runner, macKeychain, runProgram } from "./keychain.js";

// A stand-in for /usr/bin/security that keeps items in memory and records every call.
function fakeSecurity(options: { corruptWrites?: boolean } = {}) {
  const items = memorySecretStore();
  const calls: { args: string[]; stdin: string | undefined }[] = [];
  const ok = (stdout = ""): RunResult => ({ code: 0, stdout, stderr: "" });
  const run: Runner = async (_program, args, stdin) => {
    calls.push({ args, stdin });
    const flag = (list: string[], name: string) => list[list.indexOf(name) + 1] ?? "";
    if (args[0] === "find-generic-password") {
      const value = await items.get(flag(args, "-a"));
      return value === undefined ? { code: 44, stdout: "", stderr: "security: item could not be found" } : ok(`${value}\n`);
    }
    if (args[0] === "delete-generic-password") {
      const had = (await items.get(flag(args, "-a"))) !== undefined;
      await items.remove(flag(args, "-a"));
      return had ? ok() : { code: 44, stdout: "", stderr: "not found" };
    }
    // -i: parse the one quoted command line the way security does.
    const words = [...(stdin ?? "").matchAll(/"((?:[^"\\]|\\.)*)"|(\S+)/g)].map((m) => (m[1] !== undefined ? m[1].replace(/\\(.)/g, "$1") : (m[2] ?? "")));
    const secret = flag(words, "-w");
    await items.set(flag(words, "-a"), options.corruptWrites === true ? `${secret}x` : secret);
    return ok();
  };
  return { run, calls };
}

secretStoreContract("macOS keychain over a fake security tool", async () => macKeychain({ run: fakeSecurity().run }));

describe("macOS keychain adapter", () => {
  it("never puts the secret in a command's arguments", async () => {
    const fake = fakeSecurity();
    const store = macKeychain({ run: fake.run });
    await store.set("route:openai", "sk-very-secret");
    await store.get("route:openai");
    await store.remove("route:openai");
    for (const call of fake.calls) expect(call.args.join(" ")).not.toContain("sk-very-secret");
    expect(fake.calls.some((c) => c.stdin?.includes("sk-very-secret"))).toBe(true);
  });

  it("does not report a write as saved unless it reads back exactly", async () => {
    const store = macKeychain({ run: fakeSecurity({ corruptWrites: true }).run });
    await expect(store.set("route:openai", "sk-1")).rejects.toThrow(SecretStoreError);
  });

  it("surfaces a keychain failure as an error, not as a missing key", async () => {
    const store = macKeychain({ run: async () => ({ code: 51, stdout: "", stderr: "security: user interaction is not allowed." }) });
    await expect(store.get("route:openai")).rejects.toThrow(/exit 51.*interaction/);
  });
});

// The real keychain, under a throwaway service name. Skipped in CI, whose
// runners have no unlocked login keychain to rely on.
const realKeychain = process.platform === "darwin" && process.env.CI === undefined;
describe.runIf(realKeychain)("the real macOS keychain", () => {
  const service = `ai.june.test-${randomBytes(4).toString("hex")}`;
  afterAll(async () => {
    for (const account of ["route:missing", "route:openai", "route:tricky", "route:a", "route:b", "route:gone", "route:nl", "route:empty", "route:uni"]) {
      await runProgram("/usr/bin/security", ["delete-generic-password", "-s", service, "-a", account]);
    }
  });
  secretStoreContract("real macOS keychain", async () => macKeychain({ service }));
});
