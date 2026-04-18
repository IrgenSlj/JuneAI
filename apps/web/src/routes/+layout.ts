// The entire app is a streaming chat surface. SSE against the local
// API won't work under SSR (no fetch body ReadableStream on the server
// side of SvelteKit), so we render client-only. This also simplifies
// hydration for the Tauri/Capacitor shells that load the built bundle
// as static assets.
export const ssr = false;
export const prerender = false;
