import { sveltekit } from "@sveltejs/kit/vite";
import { SvelteKitPWA } from "@vite-pwa/sveltekit";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    sveltekit(),
    SvelteKitPWA({
      strategies: "generateSW",
      registerType: "autoUpdate",
      injectRegister: "auto",
      manifest: {
        name: "June",
        short_name: "June",
        description: "Your personal AI, running locally.",
        theme_color: "#fdfcfa",
        background_color: "#fdfcfa",
        display: "standalone",
        start_url: "/",
        scope: "/",
        icons: [
          {
            src: "/icon-192.png",
            sizes: "192x192",
            type: "image/png",
            purpose: "any",
          },
          {
            src: "/icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "any",
          },
          {
            src: "/icon-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        globPatterns: ["client/**/*.{js,css,ico,png,svg,webp,woff2}"],
        navigateFallback: "/",
        // The API lives on a separate origin (see PUBLIC_JUNE_API_URL), so the
        // service worker only ever intercepts navigations for the web app's
        // own origin — every one of those is a SvelteKit SPA route and should
        // fall back to the cached shell. We keep /api/* denylisted only as a
        // forward-looking hedge: if a future reverse-proxy setup ever colocates
        // the API under /api on this origin, those requests must bypass the
        // shell and hit the network.
        navigateFallbackDenylist: [/^\/api\//],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  server: {
    port: 5173,
    strictPort: false,
  },
});
