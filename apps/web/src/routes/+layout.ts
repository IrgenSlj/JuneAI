import { redirect } from "@sveltejs/kit";
import { createJuneClient } from "@june/ui";

// The entire app is a streaming chat surface. SSE against the local
// API won't work under SSR (no fetch body ReadableStream on the server
// side of SvelteKit), so we render client-only. This also simplifies
// hydration for the Tauri/Capacitor shells that load the built bundle
// as static assets.
export const ssr = false;
export const prerender = false;

const DEFAULT_API = "http://localhost:8000";

export const load = async ({ url, fetch }) => {
  if (url.pathname.startsWith("/setup")) return {};

  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl, fetchImpl: fetch });

  try {
    const status = await client.getSetupStatus();
    if (!status.is_configured) throw redirect(307, "/setup");
  } catch (err) {
    // A redirect from the try block is a SvelteKit control-flow exception and
    // must propagate. Any other failure (API down, CORS, etc.) leaves the user
    // on whatever screen they requested — the chat page already renders a
    // "couldn't reach June" state.
    if (err && typeof err === "object" && "status" in err && "location" in err) {
      throw err;
    }
    console.warn("June: setup status check failed", err);
  }

  return {};
};
