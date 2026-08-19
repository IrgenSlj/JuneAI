<script lang="ts">
  import type { RegistryEntry } from "../api/client.js";

  /**
   * The MCP registry section of the Skills page.
   *
   * Extracted from `routes/skills/+page.svelte` (D.8), which had grown to 1335
   * lines — 692 of them CSS — while chat was decomposed into this package long
   * ago. Presentational only: it owns no state and makes no client calls, so
   * the page keeps the loading, error and pending bookkeeping and this renders
   * it.
   *
   * The styles include a few rules the page also uses for its own skills list
   * (`.error`, `.muted`, the skeleton shimmer). They are duplicated rather than
   * shared because Svelte scopes styles per component and the two sections are
   * now separate; the alternative is a global stylesheet, which trades a little
   * duplication for action at a distance.
   */
  let {
    entries = [],
    loading = false,
    error = null,
    pendingKey = null,
    policyLabel,
    onRefresh,
    onInstall,
    onUninstall,
  }: {
    entries?: RegistryEntry[];
    loading?: boolean;
    error?: string | null;
    pendingKey?: string | null;
    policyLabel: (policy: string) => string;
    onRefresh: () => void;
    onInstall: (entry: RegistryEntry) => void;
    onUninstall: (entry: RegistryEntry) => void;
  } = $props();

</script>

<section class="registry-section" aria-label="MCP registry">
  <header class="registry-head">
    <h2>Browse the MCP registry</h2>
    <button
      type="button"
      class="registry-refresh"
      onclick={onRefresh}
      disabled={loading}
    >
      {loading ? "Loading…" : "Refresh"}
    </button>
  </header>
  <p class="registry-lead">
    Any third-party MCP server can run as a June skill. These are curated and shipped
    with the app. Installed skills appear in the list above and become callable by the
    agent on the next reload.
  </p>

  {#if error}
    <div class="error" role="alert">Registry failed to load: {error}</div>
  {/if}

  {#if loading && entries.length === 0}
    <div class="skeleton-list" aria-label="Loading registry…">
      {#each [1, 2, 3] as _ (_)}
        <div class="skeleton skeleton-card"></div>
      {/each}
    </div>
  {:else if entries.length === 0 && !error}
    <p class="muted">No registry entries available.</p>
  {:else}
    <ul class="registry">
      {#each entries as entry (entry.key)}
        <li class="registry-entry">
          <div class="registry-entry-head">
            <div class="registry-ident">
              <div class="registry-name">
                {entry.name}
                {#if entry.verified}<span class="badge verified">verified</span>{/if}
                <span class="badge policy">{policyLabel(entry.model_policy)}</span>
              </div>
              {#if entry.description}
                <div class="registry-desc">{entry.description}</div>
              {/if}
              {#if entry.homepage}
                <a class="registry-link" href={entry.homepage} target="_blank" rel="noopener">
                  {entry.publisher} · source
                </a>
              {/if}
            </div>
            <div class="registry-actions">
              {#if entry.installed}
                <button
                  type="button"
                  class="registry-btn"
                  onclick={() => onUninstall(entry)}
                  disabled={pendingKey !== null}
                >
                  {pendingKey === entry.key ? "…" : "Uninstall"}
                </button>
              {:else}
                <button
                  type="button"
                  class="registry-btn primary"
                  onclick={() => onInstall(entry)}
                  disabled={pendingKey !== null}
                >
                  {#if pendingKey === entry.key}
                    <span class="spinner" aria-hidden="true"></span>
                    Installing…
                  {:else}
                    Install
                  {/if}
                </button>
              {/if}
            </div>
          </div>

          {#if entry.install.env_required && entry.install.env_required.length > 0}
            <div class="registry-env">
              Requires:
              {#each entry.install.env_required as envName (envName)}
                <code>{envName}</code>
              {/each}
            </div>
          {/if}

          {#if entry.tools_preview && entry.tools_preview.length > 0}
            <div class="registry-tools">
              Tools:
              {#each entry.tools_preview as toolName (toolName)}
                <code>{toolName}</code>
              {/each}
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .error {
    background: color-mix(in srgb, var(--color-danger) 15%, transparent);
    color: var(--color-danger);
    border: 1px solid color-mix(in srgb, var(--color-danger) 40%, transparent);
    border-radius: var(--radius-md);
    padding: var(--space-3);
    font-size: var(--size-sm);
  }
  @keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
  }
  .skeleton-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .skeleton {
    background: linear-gradient(90deg, var(--color-bg-raised) 25%, var(--color-border) 50%, var(--color-bg-raised) 75%);
    background-size: 800px 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: var(--radius-sm);
  }
  .skeleton-card {
    height: 100px;
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
  }

  .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    margin: 0;
  }
  .registry-section {
    margin-top: var(--space-5);
    padding-top: var(--space-4);
    border-top: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .registry-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .registry-head h2 {
    margin: 0;
    font-size: var(--size-lg);
    font-weight: 600;
  }
  .registry-refresh {
    background: var(--color-bg-raised);
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .registry-refresh:hover:not(:disabled) {
    color: var(--color-accent);
    border-color: var(--color-accent);
  }
  .registry-lead {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    margin: 0;
  }

  .registry {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .registry-entry {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .registry-entry-head {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
    align-items: flex-start;
  }

  .registry-ident {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
    flex: 1;
  }

  .registry-name {
    font-weight: 600;
    color: var(--color-fg-primary);
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .registry-desc {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }

  .registry-link {
    color: var(--color-fg-subtle);
    font-size: var(--size-xs);
    text-decoration: none;
    font-family: var(--font-mono);
  }
  .registry-link:hover {
    color: var(--color-accent);
    text-decoration: underline;
  }

  .registry-actions {
    flex-shrink: 0;
  }

  .registry-btn {
    background: transparent;
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .registry-btn:hover:not(:disabled) {
    border-color: var(--color-accent);
    color: var(--color-accent);
  }
  .registry-btn.primary {
    background: var(--color-accent);
    color: var(--color-bg);
    border-color: var(--color-accent);
  }
  .registry-btn.primary:hover:not(:disabled) {
    color: var(--color-bg);
  }
  .registry-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .badge {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    padding: 1px var(--space-2);
    border-radius: var(--radius-sm);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge.verified {
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .badge.policy {
    color: var(--color-fg-subtle);
    border: 1px dashed var(--color-border);
  }
  .registry-env,
  .registry-tools {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .registry-env code,
  .registry-tools code {
    font-family: var(--font-mono);
    color: var(--color-fg-muted);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 1px var(--space-2);
  }

  @keyframes registry-spin {
    to { transform: rotate(360deg); }
  }
  .spinner {
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    margin-right: 0.4em;
    vertical-align: -0.1em;
    border: 1.5px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: registry-spin 0.8s linear infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation: none; }
  }
</style>
