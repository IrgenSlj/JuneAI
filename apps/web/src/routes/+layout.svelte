<script lang="ts">
  import "@june/design/tokens.css";
  import "./app.css";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import { onMount } from "svelte";
  import { session } from "#lib/session.svelte.js";

  let { children } = $props();
  let ready = $state(false);

  // Routing is decided here, once, before any page renders: no vault means
  // setup, and the bare root means chat. Pages never redirect on their own,
  // so two redirects cannot race.
  onMount(async () => {
    await session.start();
    const path = page.url.pathname;
    if (session.vault?.path === null && path !== "/setup") await goto("/setup", { replaceState: true });
    else if (session.vault?.path && (path === "/" || path === "/setup")) await goto("/chat", { replaceState: true });
    ready = true;
  });
</script>

{#if !ready}
  <main class="center" aria-busy="true"></main>
{:else if session.token === undefined}
  <main class="center">
    <h1>June</h1>
    <p>This page needs the link June printed when it started (it ends in <code>#token=…</code>), or open June from the Mac app.</p>
  </main>
{:else if session.error}
  <main class="center">
    <h1>June cannot reach its daemon</h1>
    <p>{session.error}</p>
  </main>
{:else}
  <div class="shell">
    <header>
      <a class="wordmark" href="/chat">June</a>
      {#if session.vault?.path}
        <span class="vault" title={session.vault.path}>{session.vault.path.split(/[\\/]/).filter(Boolean).at(-1)}</span>
        <a class="nav" href="/settings">Settings</a>
      {/if}
    </header>
    {@render children()}
  </div>
{/if}

<style>
  .center {
    max-width: 32rem;
    margin: 20vh auto 0;
    padding: 0 1.5rem;
    color: var(--color-text, inherit);
  }
  .shell {
    display: grid;
    grid-template-rows: auto 1fr;
    height: 100dvh;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--color-border, rgba(127, 127, 127, 0.2));
  }
  .wordmark {
    font-weight: 600;
    text-decoration: none;
    color: inherit;
  }
  .nav {
    margin-left: auto;
    font-size: 0.9rem;
  }
  .vault {
    font-family: var(--font-mono, monospace);
    font-size: 0.8rem;
    opacity: 0.6;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
