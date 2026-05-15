import { createJuneClient, type JuneClientOptions } from "@june/ui";
import { env } from "$env/dynamic/public";

export const DEFAULT_API = "http://localhost:8000";

export const apiUrl = env.PUBLIC_JUNE_API_URL || DEFAULT_API;

export function createClient(opts?: Partial<JuneClientOptions>) {
  return createJuneClient({ baseUrl: apiUrl, ...opts });
}

export const client = createClient();
