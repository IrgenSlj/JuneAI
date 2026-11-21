<script lang="ts">
  import { onMount } from "svelte";
  import { ApiError, type ProviderInfo, type Transparency } from "@june/ui/api";
  import { routeGroup } from "#lib/chat.svelte.js";
  import { session } from "#lib/session.svelte.js";

  // Keys go straight to the daemon, which puts them in the macOS keychain.
  // This page never receives a key back; it only learns whether one is stored.
  let routes: ProviderInfo[] = $state([]);
  let drafts: Record<string, string> = $state({});
  let busy: string | undefined = $state();
  let notice: { id: string; text: string; error?: boolean } | undefined = $state();

  const groups = $derived(
    ["On this Mac", "Your keys", "Free"]
      .map((name) => ({ name, routes: routes.filter((p) => routeGroup(p.route) === name) }))
      .filter((g) => g.routes.length > 0),
  );

  let level: Transparency | undefined = $state();
  const LEVELS: { value: Transparency; name: string; text: string }[] = [
    { value: "quiet", name: "Quiet", text: "Show nothing extra. Receipts are still kept on the Receipts page." },
    { value: "glance", name: "Glance", text: "One line under the message box: model, route, and whether it leaves this Mac." },
    { value: "full", name: "Full", text: "Before sending to a new route, or with new kinds of context, show exactly what leaves and ask. Each reply shows its receipt." },
  ];

  onMount(async () => {
    const [providers, settings] = await Promise.all([session.client.providers(), session.client.settings()]);
    routes = providers;
    level = settings.transparency;
  });

  async function setLevel(next: Transparency): Promise<void> {
    level = (await session.client.putSettings({ transparency: next })).transparency;
  }

  async function save(id: string, apiKey: string | null): Promise<void> {
    busy = id;
    notice = undefined;
    try {
      routes = await session.client.putProvider(id, { apiKey: apiKey === null ? null : apiKey.trim() });
      drafts[id] = "";
      notice = { id, text: apiKey === null ? "Key removed from the keychain." : "Key saved in the macOS keychain." };
    } catch (error) {
      notice = { id, text: error instanceof ApiError ? error.reason : String(error), error: true };
    } finally {
      busy = undefined;
    }
  }
</script>

<main class="settings">
  <h1>What June shows</h1>
  <p class="intro">Everything sent is recorded the same way at every level; this only changes what you see.</p>
  <fieldset class="levels" aria-label="Transparency">
    {#each LEVELS as option (option.value)}
      <label class:chosen={level === option.value}>
        <input type="radio" name="transparency" value={option.value} checked={level === option.value} onchange={() => setLevel(option.value)} />
        <span><strong>{option.name}</strong> {option.text}</span>
      </label>
    {/each}
  </fieldset>
  <p class="note"><a href="/receipts">Receipts</a>: every send, import and export, with a check that none was removed or changed.</p>

  <h1 class="models">Models</h1>
  <p class="intro">June talks to the models you bring. Keys are kept in the macOS keychain, never in your vault or a file.</p>

  {#each groups as group (group.name)}
    <section>
      <h2>{group.name}</h2>
      {#each group.routes as p (p.route.id)}
        <article class="route" aria-label={p.route.name}>
          <header>
            <strong>{p.route.name}</strong>
            <span class="where">{p.route.leavesDevice ? `sends to ${p.route.host}` : "stays on this Mac"}</span>
            <span class="state" class:ok={p.ready}>{p.needsKey ? (p.ready ? "Key stored" : "No key") : p.running ? "Running" : "Not running"}</span>
          </header>
          <p class="note">Data practice: {p.route.dataNote === "unknown" ? "not yet checked by June" : p.route.dataNote}</p>
          {#if p.route.kind === "local" && !p.running}
            <p class="note">Start {p.route.name.replace(" (this Mac)", "")} on this Mac, then reload this page.</p>
          {/if}
          {#if p.needsKey}
            <form onsubmit={(e) => { e.preventDefault(); void save(p.route.id, drafts[p.route.id] ?? ""); }}>
              <input
                type="password"
                autocomplete="off"
                spellcheck="false"
                aria-label={`${p.route.name} API key`}
                placeholder={p.ready ? "Replace the stored key" : "Paste an API key"}
                bind:value={drafts[p.route.id]}
              />
              <button type="submit" class="primary" disabled={busy !== undefined || (drafts[p.route.id] ?? "").trim() === ""}>Save</button>
              {#if p.ready}
                <button type="button" disabled={busy !== undefined} onclick={() => save(p.route.id, null)}>Remove</button>
              {/if}
            </form>
          {/if}
          {#if notice?.id === p.route.id}
            <p class="notice" class:error={notice.error} role={notice.error ? "alert" : "status"}>{notice.text}</p>
          {/if}
        </article>
      {/each}
    </section>
  {/each}
</main>

<style>
  .settings {
    max-width: 40rem;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 4rem;
    overflow-y: auto;
  }
  .levels {
    border: none;
    padding: 0;
    margin: 0.5rem 0;
    display: grid;
    gap: 0.5rem;
  }
  .levels label {
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: 0.9rem;
  }
  .levels label.chosen {
    border-color: var(--color-accent);
    background: var(--color-bg-raised);
  }
  .levels span {
    color: var(--color-fg-secondary);
  }
  .levels strong {
    color: var(--color-fg-primary);
    margin-right: 0.25rem;
  }
  .models {
    margin-top: 2.5rem;
  }
  .intro,
  .note {
    opacity: 0.7;
  }
  .note {
    font-size: 0.85rem;
    margin: 0.25rem 0 0.5rem;
  }
  h2 {
    font-size: 1rem;
    margin-top: 2rem;
  }
  .route {
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--color-border, rgba(127, 127, 127, 0.2));
  }
  .route header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: baseline;
  }
  .where {
    font-size: 0.85rem;
    opacity: 0.7;
  }
  .state {
    margin-left: auto;
    font-size: 0.85rem;
    opacity: 0.7;
  }
  .state.ok {
    opacity: 1;
  }
  form {
    display: flex;
    gap: 0.5rem;
  }
  form input {
    flex: 1;
    min-width: 0;
  }
  .notice {
    font-size: 0.85rem;
    margin: 0.5rem 0 0;
  }
  .notice.error {
    border-left: 3px solid var(--color-danger, #b3261e);
    padding-left: 0.5rem;
  }
</style>
