<script lang="ts">
  import { onMount } from "svelte";
  import {
    createJuneClient,
    type SkillInfo,
    type SkillsResponse,
  } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";

  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl });

  let skills: SkillInfo[] = $state([]);
  let loading = $state(true);
  let error: string | null = $state(null);
  let pendingToggle: string | null = $state(null);

  async function refresh() {
    loading = true;
    error = null;
    try {
      const response: SkillsResponse = await client.getSkills();
      skills = response.skills ?? [];
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(refresh);

  async function toggle(skill: SkillInfo) {
    if (pendingToggle) return;
    pendingToggle = skill.key;
    try {
      await client.toggleSkill(skill.key, !skill.enabled);
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      pendingToggle = null;
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
</script>

<svelte:head>
  <title>Skills — June</title>
</svelte:head>

<main class="page">
  <header class="top">
    <div class="heading">
      <a class="back" href="/">← Chat</a>
      <h1>Skills</h1>
    </div>
    <div class="controls">
      <button type="button" onclick={refresh} disabled={loading}>
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  </header>

  {#if error}
    <div class="error">Couldn't load skills: {error}</div>
  {/if}

  {#if loading && skills.length === 0}
    <div class="empty">Loading June's skills…</div>
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
              <span class="status status-{statusKind(skill)}">
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
            <div class="skill-error">{skill.error}</div>
          {/if}

          {#if skill.tools?.length}
            <ul class="tools">
              {#each skill.tools as tool (tool.name)}
                <li class="tool">
                  <code>{tool.name}</code>
                  {#if tool.description}
                    <span class="tool-desc">{tool.description}</span>
                  {/if}
                </li>
              {/each}
            </ul>
          {:else if skill.enabled}
            <p class="muted">No tools discovered yet.</p>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
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

  .back {
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    text-decoration: none;
  }
  .back:hover {
    color: var(--color-accent);
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
    align-items: baseline;
    font-size: var(--size-sm);
  }

  .tool code {
    font-family: var(--font-mono);
    color: var(--color-fg-primary);
  }

  .tool-desc {
    color: var(--color-fg-muted);
  }

  .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    margin: 0;
  }
</style>
