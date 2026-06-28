<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import { type SetupApplyResponse, type SetupStatus } from "@june/ui";
  import { client, apiUrl } from "$lib/api.js";

  type Provider = "gemma" | "gemini";
  type Step = "loading" | "choose" | "verifying" | "error" | "done";

  let status: SetupStatus | null = $state(null);
  let provider: Provider = $state("gemma");
  let geminiKey = $state("");
  let step: Step = $state("loading");
  let result: SetupApplyResponse | null = $state(null);

  onMount(async () => {
    await refreshStatus();
    if (status?.is_configured) {
      step = "done";
    } else {
      provider = (status?.provider as Provider) || "gemma";
      step = "choose";
    }
  });

  async function refreshStatus() {
    try {
      status = await client.getSetupStatus();
    } catch (err) {
      console.warn("June: /setup/status unreachable", err);
    }
  }

  async function handleApply() {
    if (provider === "gemini" && !geminiKey.trim()) return;
    step = "verifying";
    result = null;
    try {
      result = await client.applySetup({
        provider,
        gemini_api_key: provider === "gemini" ? geminiKey.trim() : null,
      });
    } catch (err) {
      console.error(err);
      result = {
        ok: false,
        provider,
        model: "",
        verified: false,
        message: `Couldn't reach the API at ${apiUrl}.`,
        hint: "Start it with ./tools/dev.sh and reload this page.",
      };
    }
    await refreshStatus();
    step = result?.ok ? "done" : "error";
  }

  function continueToChat() {
    goto("/").catch(() => {});
  }
</script>

<svelte:head>
  <title>Set up — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header>
    <p class="eyebrow">June</p>
    <h1>Set up your assistant</h1>
    <p class="lede">
      Pick a model provider. Your conversations and memories stay on this
      machine either way.
    </p>
  </header>

  {#if step === "loading"}
    <p class="hint">Checking your environment…</p>
  {:else}
    <section class="card">
      <fieldset class="providers">
        <legend>Provider</legend>

        <label class="option" class:selected={provider === "gemma"}>
          <input type="radio" value="gemma" bind:group={provider} />
          <span class="label">
            <span class="name">Gemma 4 (local)</span>
            <span class="desc">
              Runs entirely on this machine via Ollama. No network calls for
              inference.
            </span>
            {#if status}
              <span class="status-line">
                {#if status.ollama_reachable && status.ollama_has_model}
                  <span class="ok">Ollama is ready with {status.model || "gemma4:e4b"}.</span>
                  {#if status.semantic_recall_status === "degraded"}
                    <span class="warn">{status.semantic_recall_detail}</span>
                  {/if}
                {:else if status.ollama_reachable}
                  <span class="warn">
                    Ollama is running but the model isn't pulled.
                    <a href="/help/ollama">See the three-step guide.</a>
                  </span>
                {:else}
                  <span class="warn">
                    Ollama isn't running.
                    <a href="/help/ollama">See the three-step guide.</a>
                  </span>
                {/if}
              </span>
            {/if}
          </span>
        </label>

        <label class="option" class:selected={provider === "gemini"}>
          <input type="radio" value="gemini" bind:group={provider} />
          <span class="label">
            <span class="name">Gemini (cloud)</span>
            <span class="desc">
              Calls Google's API. Faster and smarter, but prompts leave your
              machine.
            </span>
          </span>
        </label>
      </fieldset>

      {#if provider === "gemini"}
        <div class="key-field">
          <label for="gemini-key">Gemini API key</label>
          <input
            id="gemini-key"
            type="password"
            autocomplete="off"
            spellcheck="false"
            bind:value={geminiKey}
            placeholder="AIza…"
          />
          <p class="hint">
            Get a free key at
            <a href="https://aistudio.google.com" target="_blank" rel="noreferrer">
              aistudio.google.com</a>.
            Stored in your system keychain (macOS/Linux/Windows) when available.
          </p>
        </div>
      {/if}

      <div class="actions">
        <p class="hint">
          June will send one short verification prompt to confirm the provider works.
        </p>
        <button
          type="button"
          onclick={handleApply}
          disabled={step === "verifying" ||
            (provider === "gemini" && !geminiKey.trim())}
          class="primary"
        >
          {step === "verifying" ? "Verifying…" : "Verify and continue"}
        </button>
      </div>
    </section>

    {#if step === "error" && result}
      <div class="notice danger">
        <p class="notice-title">{result.message || "Verification failed."}</p>
        {#if result.hint}<p>{result.hint}</p>{/if}
      </div>
    {/if}

    {#if step === "done"}
      <div class="notice success">
        <p class="notice-title">Ready to chat.</p>
        <p>June is set up with {result?.model || status?.model || "your chosen model"}.</p>
        <button type="button" onclick={continueToChat} class="primary">
          Continue to chat
        </button>
      </div>
    {/if}
  {/if}
</main>

<style>
  .page {
    max-width: 640px;
    margin: 0 auto;
    padding: var(--space-8) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }

  header {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .eyebrow {
    margin: 0;
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--color-fg-subtle);
  }
  h1 {
    margin: 0;
    font-size: calc(var(--size-xl) * 1.25);
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .lede {
    margin: 0;
    color: var(--color-fg-muted);
    line-height: var(--leading-relaxed);
  }

  .card {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }

  fieldset {
    border: 0;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  legend {
    padding: 0;
    font-size: var(--size-sm);
    font-weight: 600;
    color: var(--color-fg-primary);
    margin-bottom: var(--space-1);
  }

  .option {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-4);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .option:hover {
    border-color: var(--color-border-strong);
  }
  .option.selected {
    border-color: var(--color-accent);
    background: var(--color-bg-sunken);
  }
  .option input {
    margin-top: 0.2rem;
  }
  .label {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .name {
    font-weight: 500;
    color: var(--color-fg-primary);
  }
  .desc {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .status-line {
    font-size: var(--size-xs);
    margin-top: var(--space-1);
  }
  .status-line .ok {
    color: var(--color-success);
  }
  .status-line .warn {
    color: var(--color-accent);
  }
  .status-line a {
    color: var(--color-accent);
    text-decoration: underline;
  }

  .key-field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .key-field label {
    font-size: var(--size-sm);
    font-weight: 500;
  }
  .key-field input {
    width: 100%;
    padding: var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    background: var(--color-bg-sunken);
    color: var(--color-fg-primary);
    font-family: var(--font-mono);
    font-size: var(--size-sm);
  }
  .key-field input:focus {
    outline: none;
    border-color: var(--color-accent);
  }
  .key-field a {
    color: var(--color-accent);
  }

  .hint {
    margin: 0;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    line-height: var(--leading-relaxed);
  }

  .actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    flex-wrap: wrap;
  }
  .actions .hint {
    flex: 1;
    min-width: 0;
  }

  button.primary {
    background: var(--color-accent);
    color: var(--color-bg-base);
    border: 0;
    font-weight: 500;
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: var(--size-sm);
  }
  button.primary:hover:not(:disabled) {
    background: var(--color-accent-muted);
  }
  button.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .notice {
    padding: var(--space-4);
    border-radius: var(--radius-md);
    border: 1px solid;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .notice-title {
    margin: 0;
    font-weight: 500;
  }
  .notice p {
    margin: 0;
  }
  .notice.danger {
    border-color: var(--color-danger);
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 12%, transparent);
  }
  .notice.success {
    border-color: var(--color-success);
    background: color-mix(in srgb, var(--color-success) 10%, transparent);
  }
  .notice.success p:first-of-type {
    color: var(--color-success);
  }
</style>
