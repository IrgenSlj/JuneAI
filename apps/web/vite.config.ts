import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

// In development, Vite serves the UI with hot reload and forwards /api to a
// running daemon (tools/dev.mjs sets JUNE_DAEMON_URL), so the page still talks
// to a single origin, as it does when the daemon serves the built files.
const daemon = process.env.JUNE_DAEMON_URL ?? "http://127.0.0.1:4747";

export default defineConfig({
  plugins: [
    sveltekit({
      // A client-rendered app (see +layout.ts) built to plain files, with one
      // index.html fallback. The daemon serves it; any unknown path gets the
      // fallback and the app routes on the client.
      adapter: adapter({ pages: "build", assets: "build", fallback: "index.html", strict: false }),
    }),
  ],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: daemon, changeOrigin: true } },
  },
});
