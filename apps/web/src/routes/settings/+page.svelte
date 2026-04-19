<script lang="ts">
  import { onMount } from "svelte";
  import {
    createJuneClient,
    type SettingsView,
    type SetupApplyResponse,
  } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";
  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl });

  type Provider = "gemma" | "gemini";

  let settings: SettingsView | null = $state(null);
  let provider: Provider = $state("gemma");
  let gemmaModel = $state("");
  let geminiModel = $state("");
  let newKey = $state("");
  let busy = $state(false);
  let result: SetupApplyResponse | null = $state(null);
  let note: string | null = $state(null);

  onMount(refresh);

  async function refresh() {
    try {
      settings = await client.getSettings();
      provider = ((settings.provider as Provider) || "gemma") as Provider;
      gemmaModel = settings.gemma_model || "";
      geminiModel = settings.gemini_model || "";
    } catch (err) {
      console.warn("June: /settings unreachable", err);
    }
  }

  async function handleApply() {
    if (provider === "gemini" && !newKey.trim() && !settings?.api_key_present) return;
    busy = true;
    note = null;
    result = null;
    try {
      result = await client.applySetup({
        provider,
        gemini_api_key:
          provider === "gemini" && newKey.trim() ? newKey.trim() : null,
        gemma_model: gemmaModel.trim() || null,
        gemini_model: geminiModel.trim() || null,
      });
      if (result.ok) {
        newKey = "";
        note = "Saved.";
      }
    } catch (err) {
      result = {
        ok: false,
        provider,
        model: "",
        verified: false,
        message: `Couldn't reach the API at ${apiUrl}.`,
        hint: "Start it with ./tools/dev.sh and reload this page.",
      };
    }
    await refresh();
    busy = false;
  }

  async function handleForgetKey() {
    if (!confirm("Remove the saved Gemini key from this machine?")) return;
    busy = true;
    note = null;
    try {
      const res = await client.forgetGeminiKey();
      note =
        res.cleared_from === "none"
          ? "No key was stored."
          : `Removed from ${res.cleared_from === "keyring" ? "system keychain" : "local file"}.`;
    } catch (err) {
      note = `Couldn't clear the key: ${String(err)}`;
    }
    await refresh();
    busy = false;
  }
</script>

<svelte:head>
  <title>Settings — June</title>
</svelte:head>

<main class="page">
  <header>
    <a class="back" href="/">← Chat</a>
    <h1>Settings</h1>
  </header>

  {#if !settings}
    <p class="hint">Loading…</p>
  {:else}
    <section class="card">
      <h2>Model provider</h2>
      <p class="hint">
        Switch between local Gemma 4 and cloud Gemini. Your conversations and
        memories stay on this machine either way.
      </p>

      <fieldset class="providers">
        <label class="option" class:selected={provider === "gemma"}>
          <input type="radio" value="gemma" bind:group={provider} />
          <span class="label">
            <span class="name">Gemma 4 (local)</span>
            <span class="desc">
              {#if provider === "gemma"}
                {#if settings.ollama_reachable && settings.ollama_has_model}
                  <span class="ok">Ollama ready with {settings.model || "gemma4:e4b"}.</span>
                {:else if settings.ollama_reachable}
                  <span class="warn">Model not pulled. Run <code>ollama pull gemma4:e4b</code>.</span>
                {:else}
                  <span class="warn">Ollama isn't reachable. Run <code>ollama serve</code>.</span>
                {/if}
              {:else}
                <span>Local inference via Ollama.</span>
              {/if}
            </span>
          </span>
        </label>

        <label class="option" class:selected={provider === "gemini"}>
          <input type="radio" value="gemini" bind:group={provider} />
          <span class="label">
            <span class="name">Gemini (cloud)</span>
            <span class="desc">
              {#if settings.api_key_present}
                <span class="ok">Key set · {settings.key_storage_label}.</span>
              {:else}
                <span class="warn">No key set.</span>
              {/if}
            </span>
          </span>
        </label>
      </fieldset>

      <div class="grid">
        <div class="field">
          <label for="gemma-model">Gemma model tag</label>
          <input
            id="gemma-model"
            type="text"
            bind:value={gemmaModel}
            placeholder="gemma4:e4b"
          />
        </div>
        <div class="field">
          <label for="gemini-model">Gemini model</label>
          <input
            id="gemini-model"
            type="text"
            bind:value={geminiModel}
            placeholder="gemini-2.0-flash"
          />
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Gemini API key</h2>
      <p class="hint">
        Stored in your system keychain when available, otherwise a mode-0600 file.
        The key is never echoed back to this screen once saved.
      </p>

      <p class="storage">
        {#if settings.api_key_present}
          <span class="dot ok"></span>
          <span>Key is set · {settings.key_storage_label}</span>
        {:else}
          <span class="dot empty"></span>
          <span>No key stored.</span>
        {/if}
      </p>

      <div class="field">
        <label for="new-key">
          {settings.api_key_present ? "Replace key" : "Paste key"}
        </label>
        <input
          id="new-key"
          type="password"
          autocomplete="off"
          spellcheck="false"
          bind:value={newKey}
          placeholder="AIza…"
        />
        <p class="hint">
          Get a free key at
          <a href="https://aistudio.google.com" target="_blank" rel="noreferrer">
            aistudio.google.com</a>.
        </p>
      </div>

      <div class="row">
        <button
          type="button"
          class="ghost"
          onclick={handleForgetKey}
          disabled={busy || !settings.api_key_present}
        >
          Forget key
        </button>
      </div>
    </section>

    <div class="actions">
      {#if note}<p class="note">{note}</p>{/if}
      <button
        type="button"
        class="primary"
        onclick={handleApply}
        disabled={busy ||
          (provider === "gemini" && !newKey.trim() && !settings.api_key_present)}
      >
        {busy ? "Saving…" : "Save and verify"}
      </button>
    </div>

    {#if result && !result.ok}
      <div class="notice danger">
        <p class="notice-title">{result.message || "Verification failed."}</p>
        {#if result.hint}<p>{result.hint}</p>{/if}
      </div>
    {/if}
  {/if}
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

  .card {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  h2 {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
  }
  .hint {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    line-height: var(--leading-relaxed);
  }
  .hint a {
    color: var(--color-accent);
  }

  fieldset {
    border: 0;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
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
  .label {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .name {
    font-weight: 500;
  }
  .desc {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .ok {
    color: var(--color-success);
  }
  .warn {
    color: var(--color-accent);
  }
  code {
    font-family: var(--font-mono);
    font-size: 0.92em;
    padding: 0.1em 0.35em;
    background: var(--color-bg-sunken);
    border-radius: var(--radius-sm);
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .field label {
    font-size: var(--size-sm);
    font-weight: 500;
  }
  .field input {
    padding: var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    background: var(--color-bg-sunken);
    color: var(--color-fg-primary);
    font-family: var(--font-mono);
    font-size: var(--size-sm);
  }
  .field input:focus {
    outline: none;
    border-color: var(--color-accent);
  }

  .storage {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    margin: 0;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    display: inline-block;
  }
  .dot.ok {
    background: var(--color-success);
  }
  .dot.empty {
    background: var(--color-fg-subtle);
  }

  .row {
    display: flex;
    gap: var(--space-3);
    flex-wrap: wrap;
  }

  .actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-4);
  }
  .note {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }

  button.primary,
  button.ghost {
    border: 0;
    font-weight: 500;
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: var(--size-sm);
  }
  button.primary {
    background: var(--color-accent);
    color: var(--color-bg-base);
  }
  button.primary:hover:not(:disabled) {
    background: var(--color-accent-muted);
  }
  button.ghost {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
  }
  button.ghost:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
  }
  button.primary:disabled,
  button.ghost:disabled {
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
  .notice.danger {
    border-color: var(--color-danger);
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 12%, transparent);
  }
  .notice-title {
    margin: 0;
    font-weight: 500;
  }
  .notice p {
    margin: 0;
  }

  @media (max-width: 520px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>
