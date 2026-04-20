<script lang="ts">
  import { onMount } from "svelte";
  import { createJuneClient, type SystemStatus } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";
  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl });

  type OS = "mac" | "linux" | "windows";
  let status = $state<SystemStatus | null>(null);
  let detectedOS: OS = $state("mac");
  let checking = $state(false);

  onMount(async () => {
    detectedOS = detectOS();
    await recheck();
  });

  function detectOS(): OS {
    if (typeof navigator === "undefined") return "mac";
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("mac")) return "mac";
    if (ua.includes("win")) return "windows";
    return "linux";
  }

  async function recheck() {
    checking = true;
    try {
      status = await client.getSystem();
    } catch (err) {
      console.warn("June: /system unreachable", err);
    }
    checking = false;
  }

  const reachable = $derived(status ? status.ollama_reachable : false);
  const hasModel = $derived(status ? status.ollama_has_model : false);
</script>

<svelte:head>
  <title>Ollama setup — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header>
    <a class="back" href="/"><span aria-hidden="true">←</span> Chat</a>
    <h1>Get Ollama running</h1>
    <p class="lede">
      June uses Ollama to run Gemma 4 locally. Three short commands and you're
      done. Pick your OS below.
    </p>
  </header>

  <section class="status" aria-live="polite">
    <div class="status-row">
      <span class="dot" class:ok={reachable} class:warn={!reachable}></span>
      <span>
        {#if reachable}
          Ollama is reachable at {status?.base_url || "http://localhost:11434"}.
        {:else}
          Ollama is not reachable. Follow the steps below, then press Re-check.
        {/if}
      </span>
    </div>
    <div class="status-row">
      <span class="dot" class:ok={reachable && hasModel} class:warn={!(reachable && hasModel)}></span>
      <span>
        {#if reachable && hasModel}
          {status?.model || "gemma4:e4b"} is pulled.
        {:else if reachable}
          The model tag isn't pulled yet. Run <code>ollama pull gemma4:e4b</code>.
        {:else}
          Model check waits for Ollama to come online.
        {/if}
      </span>
    </div>
    <button type="button" class="ghost" onclick={recheck} disabled={checking}>
      {checking ? "Checking…" : "Re-check"}
    </button>
  </section>

  <nav class="os-tabs" aria-label="Operating system">
    <button
      type="button"
      class:active={detectedOS === "mac"}
      onclick={() => (detectedOS = "mac")}
    >macOS</button>
    <button
      type="button"
      class:active={detectedOS === "linux"}
      onclick={() => (detectedOS = "linux")}
    >Linux</button>
    <button
      type="button"
      class:active={detectedOS === "windows"}
      onclick={() => (detectedOS = "windows")}
    >Windows</button>
  </nav>

  <section class="steps">
    {#if detectedOS === "mac"}
      <article class="step">
        <h2>1. Install Ollama</h2>
        <pre><code>brew install ollama</code></pre>
        <p class="hint">
          No Homebrew? Download the .dmg from
          <a href="https://ollama.com/download" target="_blank" rel="noreferrer">ollama.com/download</a>.
        </p>
      </article>
      <article class="step">
        <h2>2. Start the server</h2>
        <pre><code>ollama serve</code></pre>
        <p class="hint">Leave this running in its own terminal, or enable the menu-bar app.</p>
      </article>
      <article class="step">
        <h2>3. Pull Gemma 4</h2>
        <pre><code>ollama pull gemma4:e4b</code></pre>
        <p class="hint">One-time download — a few gigabytes.</p>
      </article>
    {:else if detectedOS === "linux"}
      <article class="step">
        <h2>1. Install Ollama</h2>
        <pre><code>curl -fsSL https://ollama.com/install.sh | sh</code></pre>
        <p class="hint">
          The install script sets up a systemd service on most distros. Details:
          <a href="https://github.com/ollama/ollama/blob/main/docs/linux.md" target="_blank" rel="noreferrer">Linux install guide</a>.
        </p>
      </article>
      <article class="step">
        <h2>2. Start the service</h2>
        <pre><code>systemctl --user start ollama
# or, in the foreground:
ollama serve</code></pre>
      </article>
      <article class="step">
        <h2>3. Pull Gemma 4</h2>
        <pre><code>ollama pull gemma4:e4b</code></pre>
      </article>
    {:else}
      <article class="step">
        <h2>1. Install Ollama</h2>
        <pre><code>winget install Ollama.Ollama</code></pre>
        <p class="hint">
          No winget? Grab the .exe from
          <a href="https://ollama.com/download" target="_blank" rel="noreferrer">ollama.com/download</a>.
        </p>
      </article>
      <article class="step">
        <h2>2. Launch Ollama</h2>
        <p>Open the Ollama app from the Start menu. It runs in the system tray and exposes the API on
          <code>http://localhost:11434</code>.
        </p>
      </article>
      <article class="step">
        <h2>3. Pull Gemma 4</h2>
        <pre><code>ollama pull gemma4:e4b</code></pre>
        <p class="hint">Run this from PowerShell or the Ollama app.</p>
      </article>
    {/if}
  </section>

  <section class="card">
    <h2>Still stuck?</h2>
    <ul>
      <li>
        Confirm Ollama is listening with
        <code>curl http://localhost:11434/api/tags</code>. A JSON response means it's up.
      </li>
      <li>
        If you're on a custom port, set
        <code>OLLAMA_BASE_URL=http://localhost:PORT/v1</code> in your <code>.env</code>,
        restart June, and re-check.
      </li>
      <li>
        Prefer not to run a local model? Switch to Gemini in
        <a href="/settings">Settings</a>.
      </li>
    </ul>
  </section>
</main>

<style>
  .page {
    max-width: 640px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }

  header {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .back {
    color: var(--color-fg-muted);
    text-decoration: none;
    font-size: var(--size-sm);
    width: max-content;
  }
  .back:hover {
    color: var(--color-accent);
  }
  h1 {
    margin: 0;
    font-size: calc(var(--size-xl) * 1.15);
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .lede {
    margin: 0;
    color: var(--color-fg-muted);
    line-height: var(--leading-relaxed);
  }

  .status {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .status-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    flex-shrink: 0;
  }
  .dot.ok {
    background: var(--color-success);
  }
  .dot.warn {
    background: var(--color-accent);
  }

  .os-tabs {
    display: flex;
    gap: var(--space-2);
    border-bottom: 1px solid var(--color-border);
  }
  .os-tabs button {
    background: transparent;
    border: 0;
    color: var(--color-fg-muted);
    padding: var(--space-2) var(--space-4);
    font-size: var(--size-sm);
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .os-tabs button:hover {
    color: var(--color-fg-primary);
  }
  .os-tabs button.active {
    color: var(--color-fg-primary);
    border-bottom-color: var(--color-accent);
  }

  .steps {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .step {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .step h2 {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
  }

  pre {
    margin: 0;
    padding: var(--space-3) var(--space-4);
    background: var(--color-bg-sunken);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    overflow-x: auto;
    font-size: var(--size-sm);
  }
  pre code {
    font-family: var(--font-mono);
    background: transparent;
    padding: 0;
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.92em;
    padding: 0.1em 0.35em;
    background: var(--color-bg-sunken);
    border-radius: var(--radius-sm);
  }

  .hint {
    margin: 0;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    line-height: var(--leading-relaxed);
  }
  .hint a {
    color: var(--color-accent);
  }

  .card {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .card h2 {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
  }
  .card ul {
    margin: 0;
    padding: 0 0 0 var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-relaxed);
  }
  .card a {
    color: var(--color-accent);
  }

  button.ghost {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    font-size: var(--size-sm);
    cursor: pointer;
    width: max-content;
  }
  button.ghost:hover:not(:disabled) {
    color: var(--color-fg-primary);
    border-color: var(--color-border-strong);
  }
  button.ghost:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
