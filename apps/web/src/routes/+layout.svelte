<script lang="ts">
  import { onMount } from "svelte";
  import "@june/design/tokens.css";
  import "./app.css";

  const { children } = $props();

  onMount(async () => {
    if (typeof window === "undefined") return;
    try {
      const { registerSW } = await import("virtual:pwa-register");
      registerSW({ immediate: true });
    } catch {
      // SW module only exists in prod builds; ignore in dev.
    }
  });
</script>

{@render children()}
