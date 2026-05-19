<script lang="ts">
  import { onMount } from "svelte";
  import {
    OfflineNotice,
    type RegistryEntry,
    type SkillInfo,
    type SkillsResponse,
    type SkillWriteRecord,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let skills: SkillInfo[] = $state([]);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let actionError: string | null = $state(null);
  let pendingToggle: string | null = $state(null);
  let pendingToolToggle: string | null = $state(null);

  let registry: RegistryEntry[] = $state([]);
  let registryLoading = $state(false);
  let registryError: string | null = $state(null);
  let pendingRegistryAction: string | null = $state(null);

  type WritesState = {
    loading: boolean;
    error: string | null;
    items: SkillWriteRecord[] | null;
  };
  let writesByKey: Record<string, WritesState> = $state({});

  async function loadWrites(skillKey: string) {
    const current = writesByKey[skillKey];
    if (current?.loading) return;
    writesByKey = {
      ...writesByKey,
      [skillKey]: { loading: true, error: null, items: current?.items ?? null },
    };
    try {
      const res = await client.getSkillWrites(skillKey, profileName.value, 30);
      writesByKey = {
        ...writesByKey,
        [skillKey]: { loading: false, error: null, items: res.writes ?? [] },
      };
    } catch (err) {
      writesByKey = {
        ...writesByKey,
        [skillKey]: {
          loading: false,
          error: err instanceof Error ? err.message : String(err),
          items: current?.items ?? null,
        },
      };
    }
  }

  async function refresh() {
    loading = true;
    loadError = null;
    actionError = null;
    try {
      const response: SkillsResponse = await client.getSkills();
      skills = response.skills ?? [];
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function refreshRegistry() {
    registryLoading = true;
    registryError = null;
    try {
      const response = await client.getSkillRegistry();
      registry = response.entries ?? [];
    } catch (err) {
      registryError = err instanceof Error ? err.message : String(err);
    } finally {
      registryLoading = false;
    }
  }

  async function installRegistry(entry: RegistryEntry) {
    if (pendingRegistryAction) return;
    pendingRegistryAction = entry.key;
    actionError = null;
    try {
      await client.installRegistryEntry(entry.key);
      await Promise.all([refresh(), refreshRegistry()]);
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingRegistryAction = null;
    }
  }

  async function uninstallRegistry(entry: RegistryEntry) {
    if (pendingRegistryAction) return;
    if (!confirm(`Remove ${entry.name} from your installed skills?`)) return;
    pendingRegistryAction = entry.key;
    actionError = null;
    try {
      await client.uninstallRegistryEntry(entry.key);
      await Promise.all([refresh(), refreshRegistry()]);
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingRegistryAction = null;
    }
  }

  function policyLabel(policy: string): string {
    switch (policy) {
      case "local_only":
        return "local-only";
      case "cloud_required":
        return "cloud-required";
      default:
        return "cloud-allowed";
    }
  }

  onMount(async () => {
    await refresh();
    void refreshRegistry();
  });

  async function toggle(skill: SkillInfo) {
    if (pendingToggle) return;
    pendingToggle = skill.key;
    actionError = null;
    try {
      await client.toggleSkill(skill.key, !skill.enabled);
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingToggle = null;
    }
  }

  async function toggleTool(skill: SkillInfo, toolName: string, enabled: boolean) {
    const key = `${skill.key}:${toolName}`;
    if (pendingToolToggle) return;
    pendingToolToggle = key;
    actionError = null;
    try {
      await client.toggleSkillTool(skill.key, toolName, enabled);
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingToolToggle = null;
    }
  }

  function statusLabel(skill: SkillInfo): string {
    if (!skill.enabled) return "disabled";
    return skill.status || "stopped";
  }

  function statusKind(skill: SkillInfo): "ok" | "warn" | "bad" | "muted" {
    if (!skill.enabled) return "muted";
    switch (skill.status) {
      case "running":
        return "ok";
      case "starting":
        return "warn";
      case "crashed":
        return "bad";
      default:
        return "muted";
    }
  }

  function statusExplain(skill: SkillInfo): string {
    if (!skill.enabled) {
      return "Skill is turned off. Its tools aren't available to June.";
    }
    switch (skill.status) {
      case "running":
        return "Subprocess is up. Tools are ready to call.";
      case "starting":
        return "Subprocess is launching. Tools become available once the handshake completes.";
      case "crashed":
        return "Subprocess exited unexpectedly. Check the error below and toggle off/on to retry.";
      case "stopped":
        return "Subprocess isn't running. It'll start the first time a tool is needed.";
      default:
        return `Status: ${skill.status}`;
    }
  }
</script>

<svelte:head>
  <title>Skills — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <div class="heading">
      <h1>Skills</h1>
    </div>
    <div class="controls">
      <button type="button" onclick={refresh} disabled={loading}>
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  </header>

  {#if loadError && skills.length === 0}
    <OfflineNotice
      kind="skills"
      detail={loadError}
      onRetry={refresh}
      retrying={loading}
    />
  {:else if actionError}
    <div class="error">Couldn't update that skill: {actionError}</div>
  {/if}

  {#if loading && skills.length === 0}
    <div class="skeleton-list" aria-label="Loading skills&hellip;">
      {#each [1, 2, 3] as _ (_)}
        <div class="skeleton skeleton-card"></div>
      {/each}
    </div>
  {:else if loadError && skills.length === 0}
    <!-- OfflineNotice above already covers this state. -->
  {:else if skills.length === 0}
    <div class="empty">No skills installed.</div>
  {:else}
    <p class="summary">
      {skills.filter((s) => s.enabled).length} of {skills.length} enabled
    </p>

    <ul class="skills">
      {#each skills as skill (skill.key)}
        <li class="skill">
          <div class="skill-head">
            <div class="skill-ident">
              <div class="skill-name">{skill.key}</div>
              {#if skill.description}
                <div class="skill-desc">{skill.description}</div>
              {/if}
            </div>
            <div class="skill-actions">
              <span
                class="status status-{statusKind(skill)}"
                title={statusExplain(skill)}
                aria-label="Status: {statusLabel(skill)}. {statusExplain(skill)}"
              >
                {statusLabel(skill)}
              </span>
              <button
                type="button"
                class="toggle"
                onclick={() => toggle(skill)}
                disabled={pendingToggle === skill.key}
                aria-label={skill.enabled ? "Disable skill" : "Enable skill"}
              >
                {pendingToggle === skill.key
                  ? "…"
                  : skill.enabled
                    ? "Disable"
                    : "Enable"}
              </button>
            </div>
          </div>

          {#if skill.error}
            <div class="skill-error" role="alert">{skill.error}</div>
          {/if}

          {#if skill.tools?.length}
            <details class="tools-wrap" open={skill.tools.length <= 5}>
              <summary>
                {skill.tools.length} tool{skill.tools.length === 1 ? "" : "s"}
              </summary>
            <ul class="tools">
              {#each skill.tools as tool (tool.name)}
                <li class="tool">
                  <div class="tool-main">
                    <div class="tool-line">
                      <code>{tool.name}</code>
                      <span class="tool-state" class:off={!tool.enabled}>
                        {tool.enabled ? "enabled" : "disabled"}
                      </span>
                    </div>
                    {#if tool.description}
                      <span class="tool-desc">{tool.description}</span>
                    {/if}
                  </div>
                  <button
                    type="button"
                    class="tool-toggle"
                    onclick={() => toggleTool(skill, tool.name, !tool.enabled)}
                    disabled={pendingToolToggle === `${skill.key}:${tool.name}` || !skill.enabled}
                    aria-label={tool.enabled ? "Disable tool" : "Enable tool"}
                  >
                    {pendingToolToggle === `${skill.key}:${tool.name}`
                      ? "…"
                      : tool.enabled
                        ? "Disable"
                        : "Enable"}
                  </button>
                </li>
              {/each}
            </ul>
            </details>
          {:else if skill.enabled && skill.status === "running"}
            <p class="muted">No tools discovered yet.</p>
          {:else if skill.enabled}
            <p class="muted">Tools load once the skill starts.</p>
          {/if}

          <details
            class="writes-wrap"
            ontoggle={(e) => {
              if (
                (e.currentTarget as HTMLDetailsElement).open &&
                !writesByKey[skill.key]?.items
              )
                loadWrites(skill.key);
            }}
          >
            <summary>
              What this skill remembered
              {#if writesByKey[skill.key]?.items}
                <span class="count">{writesByKey[skill.key].items!.length}</span>
              {/if}
            </summary>
            {#if writesByKey[skill.key]?.loading}
              <p class="muted">Loading…</p>
            {:else if writesByKey[skill.key]?.error}
              <p class="error-inline">{writesByKey[skill.key].error}</p>
            {:else if writesByKey[skill.key]?.items && writesByKey[skill.key].items!.length === 0}
              <p class="muted">
                Nothing yet. Once this skill writes via <code>MemoryManager.write()</code>,
                each paraphrase shows up here and feeds recall.
              </p>
            {:else if writesByKey[skill.key]?.items}
              <ul class="writes">
                {#each writesByKey[skill.key].items! as w (w.ref)}
                  <li class="write">
                    <div class="write-head">
                      {#if w.tool}
                        <code class="write-tool">{w.tool}</code>
                      {/if}
                      {#if w.created_at}
                        <time class="write-time" datetime={w.created_at}>
                          {formatRelative(w.created_at)}
                        </time>
                      {/if}
                    </div>
                    <div class="write-text">{w.text}</div>
                  </li>
                {/each}
              </ul>
            {/if}
          </details>
        </li>
      {/each}
    </ul>
  {/if}

  <section class="registry-section" aria-label="MCP registry">
    <header class="registry-head">
      <h2>Browse the MCP registry</h2>
      <button
        type="button"
        class="registry-refresh"
        onclick={refreshRegistry}
        disabled={registryLoading}
      >
        {registryLoading ? "Loading…" : "Refresh"}
      </button>
    </header>
    <p class="registry-lead">
      Any third-party MCP server can run as a June skill. These are curated and shipped
      with the app. Installed skills appear in the list above and become callable by the
      agent on the next reload.
    </p>

    {#if registryError}
      <div class="error" role="alert">Registry failed to load: {registryError}</div>
    {/if}

    {#if registryLoading && registry.length === 0}
      <div class="skeleton-list" aria-label="Loading registry…">
        {#each [1, 2, 3] as _ (_)}
          <div class="skeleton skeleton-card"></div>
        {/each}
      </div>
    {:else if registry.length === 0 && !registryError}
      <p class="muted">No registry entries available.</p>
    {:else}
      <ul class="registry">
        {#each registry as entry (entry.key)}
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
                    onclick={() => uninstallRegistry(entry)}
                    disabled={pendingRegistryAction !== null}
                  >
                    {pendingRegistryAction === entry.key ? "…" : "Uninstall"}
                  </button>
                {:else}
                  <button
                    type="button"
                    class="registry-btn primary"
                    onclick={() => installRegistry(entry)}
                    disabled={pendingRegistryAction !== null}
                  >
                    {pendingRegistryAction === entry.key ? "Installing…" : "Install"}
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
</main>

<style>
  .page {
    max-width: 760px;
    margin: 0 auto;
    padding: var(--space-5) var(--space-4) var(--space-7);
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
  }

  .top {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--color-border);
  }

  .heading {
    display: flex;
    align-items: baseline;
    gap: var(--space-4);
  }

  h1 {
    margin: 0;
    font-size: var(--size-xl);
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  .controls {
    display: flex;
    gap: var(--space-2);
    justify-content: flex-end;
  }

  .controls button {
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    cursor: pointer;
  }
  .controls button:hover:not(:disabled) {
    border-color: var(--color-accent);
  }
  .controls button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .summary {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    margin: 0;
  }

  .error {
    background: color-mix(in srgb, var(--color-danger) 15%, transparent);
    color: var(--color-danger);
    border: 1px solid color-mix(in srgb, var(--color-danger) 40%, transparent);
    border-radius: var(--radius-md);
    padding: var(--space-3);
    font-size: var(--size-sm);
  }

  .empty {
    color: var(--color-fg-muted);
    text-align: center;
    padding: var(--space-7) 0;
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

  .skills {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .skill {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .skill-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
  }

  .skill-ident {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }

  .skill-name {
    font-weight: 600;
    color: var(--color-fg-primary);
    text-transform: capitalize;
  }

  .skill-desc {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }

  .skill-actions {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }

  .status {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
  }
  .status-ok {
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .status-warn {
    color: var(--color-fg-primary);
    background: color-mix(in srgb, var(--color-fg-muted) 20%, transparent);
  }
  .status-bad {
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 15%, transparent);
  }
  .status-muted {
    color: var(--color-fg-subtle);
    background: transparent;
    border: 1px dashed var(--color-border);
  }

  .toggle {
    background: transparent;
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .toggle:hover:not(:disabled) {
    border-color: var(--color-accent);
  }
  .toggle:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .skill-error {
    color: var(--color-danger);
    font-size: var(--size-sm);
    font-family: var(--font-mono);
  }

  .tools-wrap {
    border-top: 1px solid var(--color-border);
    padding-top: var(--space-2);
  }
  .tools-wrap summary {
    cursor: pointer;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    padding: var(--space-1) 0;
    list-style: none;
  }
  .tools-wrap summary::-webkit-details-marker {
    display: none;
  }
  .tools-wrap summary::before {
    content: "▸ ";
    color: var(--color-fg-subtle);
    display: inline-block;
    transition: transform 120ms ease;
  }
  .tools-wrap[open] summary::before {
    transform: rotate(90deg);
  }
  .tools-wrap[open] .tools {
    margin-top: var(--space-2);
  }

  .tools {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .tool {
    display: flex;
    gap: var(--space-2);
    align-items: flex-start;
    justify-content: space-between;
    font-size: var(--size-sm);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2);
    background: var(--color-bg);
  }

  .tool-main {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .tool-line {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .tool code {
    font-family: var(--font-mono);
    color: var(--color-fg-primary);
  }

  .tool-state {
    color: var(--color-accent);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }
  .tool-state.off {
    color: var(--color-fg-subtle);
  }

  .tool-desc {
    color: var(--color-fg-muted);
  }

  .tool-toggle {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-size: var(--size-xs);
    cursor: pointer;
    flex-shrink: 0;
  }
  .tool-toggle:hover:not(:disabled) {
    color: var(--color-accent);
    border-color: var(--color-accent);
  }
  .tool-toggle:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    margin: 0;
  }

  .writes-wrap {
    margin-top: var(--space-2);
    border-top: 1px dashed var(--color-border);
    padding-top: var(--space-2);
  }
  .writes-wrap summary {
    cursor: pointer;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    list-style: none;
    padding: var(--space-1) 0;
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .writes-wrap summary::-webkit-details-marker {
    display: none;
  }
  .writes-wrap summary::before {
    content: "▸ ";
    font-size: 0.85em;
    color: var(--color-fg-subtle);
  }
  .writes-wrap[open] summary::before {
    content: "▾ ";
  }
  .writes-wrap summary:hover {
    color: var(--color-fg-primary);
  }
  .writes-wrap .count {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }

  .writes {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .write {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .write-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  .write-tool {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  .write-time {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    flex-shrink: 0;
  }
  .write-text {
    color: var(--color-fg-primary);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
  }

  .error-inline {
    color: var(--color-danger);
    font-size: var(--size-sm);
    margin: var(--space-2) 0 0;
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
</style>
