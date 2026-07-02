import { createJuneClient, type JuneClientOptions } from "@june/ui";
import { getPlatform } from "@june/ui/platform";
import { env } from "$env/dynamic/public";

export const DEFAULT_API = "http://localhost:8000";

export const apiUrl = env.PUBLIC_JUNE_API_URL || DEFAULT_API;

export function createClient(opts?: Partial<JuneClientOptions>) {
  return createJuneClient({ baseUrl: apiUrl, ...opts });
}

export const client = createClient();

/**
 * Fetch the loopback token from the host shell (Tauri only) and install it on
 * the client so every request carries `X-June-Token`. In a browser / dev build
 * `getApiToken()` resolves to undefined and this is a no-op — no header, and
 * the server middleware stays a pass-through. The root `load()` awaits this
 * promise before its first request, so the token is installed before any call
 * that the desktop sidecar would enforce. Guarded so it never throws.
 */
export const apiTokenReady: Promise<void> = (async () => {
  try {
    const token = await getPlatform().getApiToken();
    if (token) client.setApiToken(token);
  } catch {
    // Browser/dev, or a shell without the command: leave the client tokenless.
  }
})();
