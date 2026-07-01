<script lang="ts">
  import { onMount } from "svelte";
  import { ConfirmDialog, OfflineNotice, type PrivacyDial, type SettingsView, type SetupApplyResponse } from "@june/ui";
  import { client, apiUrl } from "$lib/api.js";
  import { getPlatform, UnsupportedError } from "@june/ui/platform";
  import { profileName, setProfileName } from "$lib/stores/user.svelte.js";
  const platform = getPlatform();

  type Provider = "gemma" | "gemini";

  const DEFAULT_MODELS: Record<Provider, string> = {
    gemma: "gemma4:e2b",
    gemini: "gemini-2.0-flash",
  };
  const MODEL_ENV_HINTS: Record<Provider, string> = {
    gemma: "GEMMA_MODEL or MODEL_NAME",
    gemini: "GEMINI_MODEL or MODEL_NAME",
  };

  let settings: SettingsView | null = $state(null);
  let loaded = $state(false);
  let unreachable = $state(false);
  let retrying = $state(false);
  let provider: Provider = $state("gemma");
  let gemmaModel = $state("");
  let geminiModel = $state("");
  let newKey = $state("");
  let busy = $state(false);
  let dialBusy = $state(false);
  let dialNote: string | null = $state(null);
  let result: SetupApplyResponse | null = $state(null);
  let note: string | null = $state(null);
  let notifyState: "idle" | "ok" | "denied" | "unsupported" = $state("idle");
  let profileInput = $state(profileName.value);

  let confirmForgetOpen = $state(false);

  function asProvider(value: string | undefined): Provider | null {
    return value === "gemma" || value === "gemini" ? value : null;
  }

  function providerName(value: Provider | string | null): string {
    if (value === "gemma") return "Gemma 4 (local)";
    if (value === "gemini") return "Gemini (cloud)";
    return "No provider";
  }

  function savedModelFor(value: Provider | null): string {
    if (!settings || !value) return "";
    return value === "gemma" ? settings.gemma_model || "" : settings.gemini_model || "";
  }

  const activeProvider = $derived.by(() => asProvider(settings?.provider));
  const activeSavedModel = $derived(
    activeProvider ? savedModelFor(activeProvider) : "",
  );
  const activeDefaultModel = $derived(
    activeProvider ? DEFAULT_MODELS[activeProvider] : "",
  );
  const activeEnvHint = $derived(
    activeProvider ? MODEL_ENV_HINTS[activeProvider] : "model environment variables",
  );
  const activeModelNote = $derived.by(() => {
    if (!settings || !activeProvider) return "";
    const activeModel = settings.model || "";
    if (!activeModel) return "No active model is resolved for this provider yet.";
    if (activeSavedModel && activeModel !== activeSavedModel) {
      return `Saved ${providerName(activeProvider)} model is ${activeSavedModel}; the API is currently using ${activeModel}. Environment variables (${activeEnvHint}) can override saved Settings at startup.`;
    }
    if (!activeSavedModel && activeModel !== activeDefaultModel) {
      return `No saved model override is set; the API is currently using ${activeModel}. This usually means ${activeEnvHint} is set in the API environment.`;
    }
    if (activeSavedModel) return `Matches the saved ${providerName(activeProvider)} model override.`;
    return `Using the ${providerName(activeProvider)} preset default.`;
  });

  const isDesktop = platform.runtime === "tauri";
  let autostart = $state(false);
  let autostartBusy = $state(false);
  let autostartError: string | null = $state(null);

  onMount(async () => {
    await refresh();
    if (isDesktop) await loadAutostart();
  });

  async function loadAutostart() {
    try {
      autostart = await platform.getAutostart();
    } catch (err) {
      autostartError = String((err as Error)?.message ?? err);
    }
  }

  async function toggleAutostart(next: boolean) {
    autostartBusy = true;
    autostartError = null;
    try {
      await platform.setAutostart(next);
      autostart = next;
    } catch (err) {
      autostartError = String((err as Error)?.message ?? err);
    }
    autostartBusy = false;
  }

  async function refresh() {
    retrying = true;
    try {
      settings = await client.getSettings();
      provider = ((settings.provider as Provider) || "gemma") as Provider;
      gemmaModel = settings.gemma_model || "";
      geminiModel = settings.gemini_model || "";
      unreachable = false;
    } catch (err) {
      console.warn("June: /settings unreachable", err);
      unreachable = true;
    } finally {
      loaded = true;
      retrying = false;
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

  async function handleTestNotification() {
    notifyState = "idle";
    try {
      await platform.notify({
        title: "June",
        body: "Notifications are wired up on this shell.",
        tag: "june-test",
      });
      notifyState = "ok";
    } catch (err) {
      if (err instanceof UnsupportedError) notifyState = "unsupported";
      else notifyState = "denied";
    }
  }

  async function handleDialChange(next: PrivacyDial) {
    if (!settings || settings.privacy_dial === next || dialBusy) return;
    dialBusy = true;
    dialNote = null;
    const previous = settings.privacy_dial;
    settings = { ...settings, privacy_dial: next };
    try {
      await client.updatePrivacyDial(next);
      dialNote = "Saved.";
    } catch (err) {
      settings = { ...settings, privacy_dial: previous };
      dialNote = `Couldn't update: ${String(err)}`;
    } finally {
      dialBusy = false;
    }
  }

  function handleForgetKey() {
    confirmForgetOpen = true;
  }

  async function doForgetKey() {
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

<main class="page" id="main-content">
  <header>
    <h1>Settings</h1>
  </header>

  {#if !loaded}
    <p class="hint">Loading…</p>
  {:else if unreachable || !settings}
    <OfflineNotice kind="settings" onRetry={refresh} {retrying} />
  {:else}
    <section class="card">
      <h2>Model provider</h2>
      <p class="hint">
        Switch between local Gemma 4 and cloud Gemini. Memory is stored on this
        machine; Gemini mode sends the current prompt and relevant recalled
        context to Google's API for inference.
      </p>

      <fieldset class="providers">
        <label class="option" class:selected={provider === "gemma"}>
          <input type="radio" value="gemma" bind:group={provider} />
          <span class="label">
            <span class="name">Gemma 4 (local)</span>
            <span class="desc">
              {#if provider === "gemma"}
                {#if activeProvider === "gemma"}
                  {#if settings.ollama_reachable && settings.ollama_has_model}
                    <span class="ok">Ollama ready with active model {settings.model || DEFAULT_MODELS.gemma}.</span>
                  {:else if settings.ollama_reachable}
                    <span class="warn">
                      Active model not pulled. <a href="/help/ollama">Fix it</a>.
                    </span>
                  {:else}
                    <span class="warn">
                      Ollama isn't reachable. <a href="/help/ollama">Fix it</a>.
                    </span>
                  {/if}
                {:else}
                  <span>Save and verify to switch to local inference via Ollama.</span>
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

      <div class="runtime-summary">
        <span class="summary-label">Active runtime</span>
        <div class="runtime-line">
          <span>{providerName(activeProvider)}</span>
          <span aria-hidden="true">·</span>
          <code>{settings.model || "not resolved"}</code>
        </div>
        {#if activeModelNote}
          <p>{activeModelNote}</p>
        {/if}
      </div>

      <div class="grid">
        <div class="field">
          <label for="gemma-model">Saved Gemma model tag</label>
          <input
            id="gemma-model"
            type="text"
            bind:value={gemmaModel}
            placeholder="gemma4:e2b"
          />
        </div>
        <div class="field">
          <label for="gemini-model">Saved Gemini model</label>
          <input
            id="gemini-model"
            type="text"
            bind:value={geminiModel}
            placeholder="gemini-2.0-flash"
          />
        </div>
      </div>
      <p class="hint">
        Saved model fields apply when you Save and verify. The active runtime above is
        what the API process is using right now.
      </p>
    </section>

    <section class="card">
      <h2>Privacy dial</h2>
      <p class="hint">
        Controls when June can call the cloud. Chat and recall always run on the model
        you picked above; this dial decides whether agentic skills (planning,
        long-context, browser, computer use) are allowed to reach Gemini.
      </p>

      <fieldset class="providers">
        <label class="option" class:selected={settings.privacy_dial === "local_only"}>
          <input
            type="radio"
            value="local_only"
            checked={settings.privacy_dial === "local_only"}
            onchange={() => handleDialChange("local_only")}
            disabled={dialBusy}
          />
          <span class="label">
            <span class="name">Local-only</span>
            <span class="desc">
              Never call the cloud. Skills that need cloud are disabled with a visible
              explanation.
            </span>
          </span>
        </label>

        <label class="option" class:selected={settings.privacy_dial === "private_by_default"}>
          <input
            type="radio"
            value="private_by_default"
            checked={settings.privacy_dial === "private_by_default"}
            onchange={() => handleDialChange("private_by_default")}
            disabled={dialBusy}
          />
          <span class="label">
            <span class="name">Private by default <span class="badge">recommended</span></span>
            <span class="desc">
              Chat and recall stay local. Agentic skills may call Gemini, with
              confirmation on the first call of each kind per session.
            </span>
          </span>
        </label>

        <label class="option" class:selected={settings.privacy_dial === "cloud_first"}>
          <input
            type="radio"
            value="cloud_first"
            checked={settings.privacy_dial === "cloud_first"}
            onchange={() => handleDialChange("cloud_first")}
            disabled={dialBusy}
          />
          <span class="label">
            <span class="name">Cloud-first</span>
            <span class="desc">
              Prefer Gemini for capability; fall back to Gemma only when the cloud is
              offline.
            </span>
          </span>
        </label>
      </fieldset>

      {#if dialNote}
        <p class="hint" aria-live="polite">{dialNote}</p>
      {/if}
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

    {#if isDesktop}
      <section class="card">
        <h2>Native shell</h2>
        <p class="hint">
          June can open with your Mac, sit in the menu bar, and pop open from
          anywhere with a global hotkey. None of these are on by default.
        </p>

        <div class="row toggle-row">
          <div class="toggle-text">
            <div class="name">Launch June at login</div>
            <p class="hint">
              Adds June to your login items via the system autostart manager.
            </p>
          </div>
          <label class="switch">
            <input
              type="checkbox"
              checked={autostart}
              disabled={autostartBusy}
              onchange={(e) => toggleAutostart((e.target as HTMLInputElement).checked)}
            />
            <span class="slider"></span>
          </label>
        </div>
        {#if autostartError}
          <p class="warn">{autostartError}</p>
        {/if}

        <div class="row toggle-row">
          <div class="toggle-text">
            <div class="name">Global hotkey</div>
            <p class="hint">
              <code>⌘⇧J</code> on macOS, <code>Ctrl+Shift+J</code> on Windows
              and Linux. Toggles the June window from anywhere. Hard-coded for
              now; we'll make it configurable once usage tells us a different
              default fits better.
            </p>
          </div>
        </div>

        <div class="row toggle-row">
          <div class="toggle-text">
            <div class="name">Tray icon</div>
            <p class="hint">
              June lives in your menu bar. Click to open or hide; right-click
              for the Open / Quit menu. Always on while the app is running.
            </p>
          </div>
        </div>
      </section>
    {/if}

    <section class="card">
      <h2>User profile</h2>
      <p class="hint">
        Each profile has its own memory store. Switch to keep work and personal
        conversations separate, or create a demo profile to explore features.
      </p>
      <div class="field">
        <label for="profile-name">Profile name</label>
        <div class="profile-row">
          <input
            id="profile-name"
            type="text"
            bind:value={profileInput}
            placeholder="local"
            onkeydown={(e) => {
              if (e.key === "Enter") {
                setProfileName(profileInput);
                profileInput = profileName.value;
              }
            }}
          />
          <button
            type="button"
            class="ghost"
            onclick={() => { setProfileName(profileInput); profileInput = profileName.value; }}
            disabled={profileInput === profileName.value || !profileInput.trim()}
          >Switch</button>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Notifications</h2>
      <p class="hint">
        Send a test notification through the active shell. On the desktop app
        this uses the OS notification center; in a browser it uses the Web
        Notifications API. Running in: <code>{platform.runtime}</code>.
      </p>
      <div class="row">
        <button type="button" class="ghost" onclick={handleTestNotification}>
          Send test notification
        </button>
        {#if notifyState === "ok"}
          <span class="ok">Sent.</span>
        {:else if notifyState === "denied"}
          <span class="warn">Permission denied or unavailable.</span>
        {:else if notifyState === "unsupported"}
          <span class="warn">Not supported on this shell.</span>
        {/if}
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

<ConfirmDialog
  bind:open={confirmForgetOpen}
  title="Forget Gemini key"
  message="Remove the saved Gemini key from this machine?"
  confirmLabel="Remove key"
  danger={true}
  onConfirm={doForgetKey}
/>

<style>
  .page {
    max-width: 640px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    min-width: 0;
  }

  header {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
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
    min-width: 0;
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
    min-width: 0;
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
    min-width: 0;
  }
  .name {
    font-weight: 500;
    overflow-wrap: anywhere;
  }
  .desc {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    overflow-wrap: anywhere;
  }
  .ok {
    color: var(--color-success);
  }
  .warn {
    color: var(--color-accent);
  }
  .warn a {
    color: var(--color-accent);
    text-decoration: underline;
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
    min-width: 0;
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
    flex-wrap: wrap;
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
    flex-wrap: wrap;
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

  .toggle-row {
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: nowrap;
    gap: var(--space-4);
  }
  .toggle-text {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    flex: 1;
    min-width: 0;
  }
  .toggle-text .name {
    font-weight: 500;
  }

  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
    flex-shrink: 0;
  }
  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  .slider {
    position: absolute;
    inset: 0;
    background: var(--color-bg-sunken);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-pill);
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease;
  }
  .slider::before {
    content: "";
    position: absolute;
    width: 18px;
    height: 18px;
    top: 2px;
    left: 2px;
    background: var(--color-fg-muted);
    border-radius: 50%;
    transition: transform 120ms ease, background 120ms ease;
  }
  .switch input:checked + .slider {
    background: var(--color-accent);
    border-color: var(--color-accent);
  }
  .switch input:checked + .slider::before {
    transform: translateX(20px);
    background: var(--color-bg-base);
  }
  .switch input:disabled + .slider {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .profile-row {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .profile-row input {
    flex: 1;
    min-width: 0;
  }
  .profile-row button {
    flex-shrink: 0;
    white-space: nowrap;
  }

  @media (max-width: 520px) {
    .page {
      padding-inline: var(--space-3);
    }
    .grid {
      grid-template-columns: 1fr;
    }
    .option {
      align-items: flex-start;
    }
    .profile-row {
      align-items: stretch;
      flex-direction: column;
    }
    .actions {
      align-items: stretch;
    }
    .actions button {
      width: 100%;
    }
  }

  .runtime-summary {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-bg-sunken);
    min-width: 0;
  }
  .summary-label {
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .runtime-line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    flex-wrap: wrap;
    color: var(--color-fg-primary);
    font-size: var(--size-sm);
  }
  .runtime-summary p {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
    overflow-wrap: anywhere;
  }
  code {
    font-family: var(--font-mono);
    overflow-wrap: anywhere;
  }
</style>
