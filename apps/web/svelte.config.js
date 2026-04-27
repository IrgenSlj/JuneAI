import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // Static SPA build. The whole app is client-rendered (see +layout.ts:
    // ssr=false, prerender=false), so adapter-static with a single
    // index.html fallback is the right shape for both the PWA and the
    // Tauri / Capacitor shells, all of which load the build as plain files.
    adapter: adapter({
      pages: "build",
      assets: "build",
      fallback: "index.html",
      precompress: false,
      strict: false,
    }),
  },
};

export default config;
