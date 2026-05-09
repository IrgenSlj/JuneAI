import { createJuneClient, type JuneClientOptions } from "@june/ui";

export const DEFAULT_API = "http://localhost:8000";

export const apiUrl =
  (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;

export function createClient(opts?: Partial<JuneClientOptions>) {
  return createJuneClient({ baseUrl: apiUrl, ...opts });
}

export const client = createClient();
