<script lang="ts">
  import { goto } from "$app/navigation";
  import { ApiError } from "@june/ui/api";
  import { getPlatform } from "@june/ui/platform";
  import { session } from "#lib/session.svelte.js";

  const platform = getPlatform();
  let path = $state(session.vault?.defaultPath ?? "~/June");
  let busy = $state(false);
  let error: string | undefined = $state();

  async function open(target: string): Promise<void> {
    busy = true;
    error = undefined;
    try {
      session.vault = await session.client.openVault({ path: target });
      await goto("/chat", { replaceState: true });
    } catch (e) {
      error = e instanceof ApiError ? e.reason : String(e);
    } finally {
      busy = false;
    }
  }

  async function browse(): Promise<void> {
    const picked = await platform.pickFolder({ title: "Choose your June vault", defaultPath: path });
    if (picked !== null) path = picked;
  }
</script>

<main class="setup">
  <h1>Where should June keep your conversations?</h1>
  <p>
    June saves everything as plain files in one folder, your vault. It stays readable without June, and you
    can move or back it up like any other folder.
  </p>
  <form onsubmit={(e) => { e.preventDefault(); void open(path); }}>
    <label for="vault-path">Vault folder</label>
    <div class="row">
      <input id="vault-path" bind:value={path} spellcheck="false" autocomplete="off" />
      {#if platform.runtime === "tauri"}
        <button type="button" onclick={browse} disabled={busy}>Choose…</button>
      {/if}
    </div>
    <button type="submit" class="primary" disabled={busy || path.trim() === ""}>Use this folder</button>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  </form>
</main>

<style>
  .setup {
    max-width: 36rem;
    margin: 12vh auto 0;
    padding: 0 1.5rem;
  }
  form {
    display: grid;
    gap: 0.75rem;
    margin-top: 1.5rem;
  }
  .row {
    display: flex;
    gap: 0.5rem;
  }
  input {
    flex: 1;
    font-family: var(--font-mono, monospace);
    padding: 0.5rem 0.75rem;
  }
  .error {
    color: var(--color-danger, #c0392b);
  }
</style>
