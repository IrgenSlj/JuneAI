/**
 * Quick Capture: submit a thought and hold the classified result so the home
 * screen can show what June understood and what it would do. A seed of Daily
 * Home — classification is local (see the capture backend).
 */
import { type CaptureResponse } from "@june/ui";
import { client } from "$lib/api.js";
import { profileName } from "$lib/stores/user.svelte.js";

export const capture = $state({
  result: null as CaptureResponse | null,
  loading: false,
  error: null as string | null,
});

export async function submitCapture(text: string): Promise<boolean> {
  const trimmed = text.trim();
  if (!trimmed || capture.loading) return false;
  capture.loading = true;
  capture.error = null;
  try {
    capture.result = await client.createCapture(profileName.value, trimmed);
    return true;
  } catch (err) {
    capture.error = err instanceof Error ? err.message : String(err);
    return false;
  } finally {
    capture.loading = false;
  }
}
