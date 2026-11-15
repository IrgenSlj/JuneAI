// The behaviour every SecretStore must share: the in-memory one in tests and
// the macOS keychain in the app.
import { describe, expect, it } from "vitest";
import { type SecretStore, SecretStoreError } from "../adapters.js";

export function secretStoreContract(name: string, make: () => Promise<SecretStore>): void {
  describe(`SecretStore contract: ${name}`, () => {
    it("returns undefined for a missing secret, and the secret once set", async () => {
      const store = await make();
      expect(await store.get("route:missing")).toBeUndefined();
      await store.set("route:openai", "sk-test-123");
      expect(await store.get("route:openai")).toBe("sk-test-123");
    });

    it("keeps quotes, backslashes, spaces and shell characters exactly", async () => {
      const store = await make();
      const tricky = `sk-"a" \\b $HOME \`c\` 'd' ;e|f`;
      await store.set("route:tricky", tricky);
      expect(await store.get("route:tricky")).toBe(tricky);
    });

    it("replaces on a second set and keeps accounts apart", async () => {
      const store = await make();
      await store.set("route:a", "first");
      await store.set("route:b", "other");
      await store.set("route:a", "second");
      expect(await store.get("route:a")).toBe("second");
      expect(await store.get("route:b")).toBe("other");
    });

    it("removes, and removing a missing secret is fine", async () => {
      const store = await make();
      await store.set("route:gone", "x");
      await store.remove("route:gone");
      await store.remove("route:gone");
      expect(await store.get("route:gone")).toBeUndefined();
    });

    it("refuses odd account names and values that are not a one-line ASCII key", async () => {
      const store = await make();
      await expect(store.set("", "x")).rejects.toThrow(SecretStoreError);
      await expect(store.set('route:"x"', "x")).rejects.toThrow(SecretStoreError);
      await expect(store.get("Route:Caps")).rejects.toThrow(SecretStoreError);
      await expect(store.set("route:nl", "sk-abc\n")).rejects.toThrow(SecretStoreError);
      await expect(store.set("route:empty", "")).rejects.toThrow(SecretStoreError);
      await expect(store.set("route:uni", "clé")).rejects.toThrow(SecretStoreError);
      expect(await store.get("route:nl")).toBeUndefined();
    });
  });
}
