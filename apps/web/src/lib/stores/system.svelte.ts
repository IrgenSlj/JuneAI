/**
 * System status (provider, model, reachability) loaded once per SPA
 * lifetime so every page can render the runtime badge without each
 * doing its own fetch.
 */
import { type SystemStatus } from "@june/ui";
import { client } from "./chat.svelte.js";

export const system = $state({
  data: null as SystemStatus | null,
  error: null as string | null,
  loading: false,
});

export async function loadSystem(): Promise<void> {
  system.loading = true;
  try {
    system.data = await client.getSystem();
    system.error = null;
  } catch (err) {
    system.error = err instanceof Error ? err.message : String(err);
    console.warn("June: /system unreachable", err);
  } finally {
    system.loading = false;
  }
}
