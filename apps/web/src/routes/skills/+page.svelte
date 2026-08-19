<script lang="ts">
  import { onMount } from "svelte";
  import {
    OfflineNotice,
    ConfirmDialog,
    SkillRegistry,
    type RegistryEntry,
    type SkillInfo,
    type SkillsResponse,
    type SkillToolInfo,
    type SkillToolInvokeResponse,
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

  let confirmUninstallOpen = $state(false);
  let confirmEntry: RegistryEntry | null = $state(null);

  // Playground state per (skill_key + tool_name).
  let playgroundArgs: Record<string, Record<string, string>> = $state({});
  let playgroundResults: Record<string, SkillToolInvokeResponse> = $state({});
  let playgroundPending: string | null = $state(null);

  function playgroundKey(skillKey: string, toolName: string): string {
    return `${skillKey}::${toolName}`;
  }

  function schemaProperties(tool: SkillToolInfo): Array<[string, { type?: string; description?: string; default?: unknown }]> {
    const schema = (tool.input_schema ?? {}) as { properties?: Record<string, { type?: string; description?: string; default?: unknown }> };
    const props = schema.properties ?? {};
    return Object.entries(props);
  }

  function requiredFields(tool: SkillToolInfo): string[] {
    const schema = (tool.input_schema ?? {}) as { required?: string[] };
    return Array.isArray(schema.required) ? schema.required : [];
  }

  function coerceArg(raw: string, type: string | undefined): unknown {
    const text = raw ?? "";
    switch (type) {
      case "number":
      case "integer":
        if (text.trim() === "") return undefined;
        return type === "integer" ? parseInt(text, 10) : parseFloat(text);
      case "boolean":
        return text === "true";
      default:
        return text;
    }
  }

  async function runTool(skill: SkillInfo, tool: SkillToolInfo) {
    const key = playgroundKey(skill.key, tool.name);
    if (playgroundPending) return;
    playgroundPending = key;
    const inputs = playgroundArgs[key] ?? {};
    const args: Record<string, unknown> = {};
    for (const [name, prop] of schemaProperties(tool)) {
      const value = coerceArg(inputs[name] ?? "", prop.type);
      if (value !== undefined && value !== "") args[name] = value;
    }
    try {
      const response = await client.invokeSkillTool(skill.key, tool.name, args);
      playgroundResults = { ...playgroundResults, [key]: response };
    } catch (err) {
      playgroundResults = {
        ...playgroundResults,
        [key]: {
          skill: skill.key,
          tool: tool.name,
          ok: false,
          result: "",
          error: err instanceof Error ? err.message : String(err),
          latency_ms: 0,
        },
      };
    } finally {
      playgroundPending = null;
    }
  }

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

  function uninstallRegistry(entry: RegistryEntry) {
    if (pendingRegistryAction) return;
    confirmEntry = entry;
    confirmUninstallOpen = true;
  }

  async function doUninstallRegistry() {
    if (!confirmEntry) return;
    const entry = confirmEntry;
    pendingRegistryAction = entry.key;
    actionError = null;
    try {
      await client.uninstallRegistryEntry(entry.key);
      await Promise.all([refresh(), refreshRegistry()]);
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingRegistryAction = null;
      confirmEntry = null;
    }
  }

  function policyLabel(policy: string): string {
    switch (policy) {
      case "local_only":
        return "local only";
      case "cloud_required":
        return "needs cloud";
      default:
        return "cloud allowed";
    }
  }

  function policyKind(policy: string): "local" | "cloud" {
    return policy === "local_only" ? "local" : "cloud";
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
        <li class="skill" id="skill-{skill.key}">
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
              <span
                class="badge skill-policy skill-policy-{policyKind(skill.model_policy)}"
                title="Declared model policy: {skill.model_policy}"
                aria-label="Declared model policy: {policyLabel(skill.model_policy)}"
              >
                {policyLabel(skill.model_policy)}
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

          {#if skill.scopes?.length}
            <div class="skill-scopes" aria-label="What this skill can do">
              {#each skill.scopes as scope (scope)}
                <span
                  class="scope-badge"
                  class:scope-sensitive={scope === "sends data off device" || scope === "runs code"}
                >
                  {scope}
                </span>
              {/each}
            </div>
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
                  <details
                    class="playground-wrap"
                    aria-label="Try this tool"
                  >
                    <summary>Try</summary>
                    {#each schemaProperties(tool) as [name, prop] (name)}
                      {@const isRequired = requiredFields(tool).includes(name)}
                      {@const pkey = playgroundKey(skill.key, tool.name)}
                      <label class="play-field">
                        <span class="play-label">
                          {name}{isRequired ? " *" : ""}
                          {#if prop.type}<span class="play-type">{prop.type}</span>{/if}
                        </span>
                        {#if prop.type === "boolean"}
                          <select
                            value={(playgroundArgs[pkey] ?? {})[name] ?? ""}
                            onchange={(e) => {
                              const next = { ...(playgroundArgs[pkey] ?? {}) };
                              next[name] = (e.currentTarget as HTMLSelectElement).value;
                              playgroundArgs = { ...playgroundArgs, [pkey]: next };
                            }}
                          >
                            <option value="">—</option>
                            <option value="true">true</option>
                            <option value="false">false</option>
                          </select>
                        {:else}
                          <input
                            type={prop.type === "number" || prop.type === "integer" ? "number" : "text"}
                            placeholder={prop.description ?? ""}
                            value={(playgroundArgs[pkey] ?? {})[name] ?? ""}
                            oninput={(e) => {
                              const next = { ...(playgroundArgs[pkey] ?? {}) };
                              next[name] = (e.currentTarget as HTMLInputElement).value;
                              playgroundArgs = { ...playgroundArgs, [pkey]: next };
                            }}
                          />
                        {/if}
                      </label>
                    {/each}
                    <div class="play-actions">
                      <button
                        type="button"
                        class="play-run"
                        onclick={() => runTool(skill, tool)}
                        disabled={playgroundPending === playgroundKey(skill.key, tool.name) || !skill.enabled}
                      >
                        {playgroundPending === playgroundKey(skill.key, tool.name) ? "Running…" : "Run"}
                      </button>
                      {#if playgroundResults[playgroundKey(skill.key, tool.name)]}
                        {@const res = playgroundResults[playgroundKey(skill.key, tool.name)]}
                        <span class="play-meta" class:bad={!res.ok}>
                          {res.ok ? "ok" : "fail"} · {res.latency_ms}ms
                        </span>
                      {/if}
                    </div>
                    {#if playgroundResults[playgroundKey(skill.key, tool.name)]}
                      {@const res = playgroundResults[playgroundKey(skill.key, tool.name)]}
                      <pre class="play-result" class:bad={!res.ok}>{res.ok ? res.result : res.error}</pre>
                    {/if}
                  </details>
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

  <SkillRegistry
    entries={registry}
    loading={registryLoading}
    error={registryError}
    pendingKey={pendingRegistryAction}
    {policyLabel}
    onRefresh={refreshRegistry}
    onInstall={installRegistry}
    onUninstall={uninstallRegistry}
  />
</main>

<ConfirmDialog
  bind:open={confirmUninstallOpen}
  title="Uninstall skill"
  message={confirmEntry ? `Remove ${confirmEntry.name} from your installed skills?` : "Remove this skill?"}
  confirmLabel="Uninstall"
  danger={true}
  onConfirm={doUninstallRegistry}
  onCancel={() => { confirmEntry = null; }}
/>

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

  .skill-scopes {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .scope-badge {
    font-size: var(--size-xs);
    padding: 1px var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    color: var(--color-fg-muted);
    background: var(--color-bg);
  }
  /* Network egress and code execution are the scopes worth scrutinizing. */
  .scope-sensitive {
    color: var(--color-warn);
    border-color: var(--color-warn);
    background: color-mix(in srgb, var(--color-warn) 10%, transparent);
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
    flex-wrap: wrap;
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

  .skill-policy {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    padding: 1px var(--space-2);
    border-radius: var(--radius-sm);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .skill-policy-local {
    color: var(--color-fg-subtle);
    border: 1px dashed var(--color-border);
  }
  .skill-policy-cloud {
    color: var(--color-fg-primary);
    background: color-mix(in srgb, var(--color-fg-muted) 15%, transparent);
    border: 1px solid color-mix(in srgb, var(--color-fg-muted) 30%, transparent);
  }

  .playground-wrap {
    order: 3;
    flex-basis: 100%;
    margin-top: var(--space-1);
    border-top: 1px dashed var(--color-border);
    padding-top: var(--space-2);
  }
  .playground-wrap summary {
    cursor: pointer;
    color: var(--color-fg-subtle);
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    list-style: none;
    padding: var(--space-1) 0;
  }
  .playground-wrap summary::-webkit-details-marker { display: none; }
  .playground-wrap summary::before { content: "▸ "; }
  .playground-wrap[open] summary::before { content: "▾ "; }
  .playground-wrap summary:hover { color: var(--color-accent); }

  .play-field {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: var(--space-2);
  }
  .play-label {
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
  }
  .play-type {
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }
  .play-field input,
  .play-field select {
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-size: var(--size-sm);
  }

  .play-actions {
    display: flex;
    gap: var(--space-2);
    align-items: center;
    margin-top: var(--space-2);
  }
  .play-run {
    background: var(--color-accent);
    color: var(--color-bg);
    border: 1px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .play-run:disabled { opacity: 0.5; cursor: default; }
  .play-meta {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-accent);
  }
  .play-meta.bad { color: var(--color-danger); }

  .play-result {
    margin-top: var(--space-2);
    padding: var(--space-2);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    color: var(--color-fg-muted);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 280px;
    overflow-y: auto;
  }
  .play-result.bad { color: var(--color-danger); }

</style>
