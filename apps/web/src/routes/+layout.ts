import { redirect } from "@sveltejs/kit";
import { client } from "$lib/api.js";

// The entire app is a streaming chat surface. SSE against the local
// API won't work under SSR (no fetch body ReadableStream on the server
// side of SvelteKit), so we render client-only. This also simplifies
// hydration for the Tauri/Capacitor shells that load the built bundle
// as static assets.
export const ssr = false;
export const prerender = false;

export const load = async ({ url }) => {
  if (url.pathname.startsWith("/setup")) return {};

  try {
    const status = await client.getSetupStatus();
    if (!status.is_configured) throw redirect(307, "/setup");
  } catch (err) {
    if (err && typeof err === "object" && "status" in err && "location" in err) {
      throw err;
    }
    console.warn("June: setup status check failed", err);
  }

  return {};
};
