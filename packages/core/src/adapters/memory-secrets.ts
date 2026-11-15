import { type SecretStore, checkSecretAccount, checkSecretValue } from "../adapters.js";

/** A secret store for tests, as strict as the keychain about names and values. */
export function memorySecretStore(): SecretStore {
  const secrets = new Map<string, string>();
  return {
    async get(account) {
      return secrets.get(checkSecretAccount(account));
    },
    async set(account, secret) {
      secrets.set(checkSecretAccount(account), checkSecretValue(secret));
    },
    async remove(account) {
      secrets.delete(checkSecretAccount(account));
    },
  };
}
