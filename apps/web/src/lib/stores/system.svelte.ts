/**
 * System status (provider, model, reachability) loaded once per SPA
 * lifetime so every page can render the runtime badge without each
 * doing its own fetch.
 */
import { type SystemStatus } from "@june/ui";
import { client } from "$lib/api.js";

export const system = $state({
  data: null as SystemStatus | null,
  error: null as string | null,
  loading: false,
});

const MAX_RETRIES = 3;
const RETRY_BASE_MS = 1000;

export async function loadSystem(): Promise<void> {
  system.loading = true;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      system.data = await client.getSystem();
      system.error = null;
      return;
    } catch (err) {
      if (attempt < MAX_RETRIES) {
        await new Promise((r) => setTimeout(r, RETRY_BASE_MS * Math.pow(2, attempt)));
        continue;
      }
      system.error = err instanceof Error ? err.message : String(err);
      console.warn("June: /system unreachable", err);
    } finally {
      system.loading = false;
    }
  }
}
