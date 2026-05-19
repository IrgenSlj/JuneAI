<script lang="ts">
  import { onMount } from "svelte";
  import { OfflineNotice, ConfirmDialog, type TaskView } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let tasks: TaskView[] = $state([]);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let actionError: string | null = $state(null);
  let pendingAction: string | null = $state(null);

  let newGoal = $state("");
  let creating = $state(false);

  let expanded: Record<string, boolean> = $state({});

  let confirmOpen = $state(false);
  let confirmTask: TaskView | null = $state(null);

  async function refresh() {
    loading = true;
    loadError = null;
    actionError = null;
    try {
      const response = await client.getTasks(profileName.value);
      tasks = response.tasks ?? [];
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(refresh);

  async function createTask(event: SubmitEvent) {
    event.preventDefault();
    const goal = newGoal.trim();
    if (!goal || creating) return;
    creating = true;
    actionError = null;
    try {
      await client.createTask(profileName.value, { goal });
      newGoal = "";
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      creating = false;
    }
  }

  async function patchStatus(task: TaskView, status: string) {
    const key = `${task.id}:${status}`;
    if (pendingAction) return;
    pendingAction = key;
    actionError = null;
    try {
      await client.patchTask(profileName.value, task.id, { status });
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingAction = null;
    }
  }

  async function removeTask(task: TaskView) {
    if (pendingAction) return;
    confirmTask = task;
    confirmOpen = true;
  }

  async function doRemoveTask() {
    if (!confirmTask) return;
    const task = confirmTask;
    pendingAction = `${task.id}:delete`;
    actionError = null;
    try {
      await client.deleteTask(profileName.value, task.id);
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingAction = null;
      confirmTask = null;
    }
  }

  function statusKind(status: string): "ok" | "warn" | "bad" | "muted" {
    switch (status) {
      case "running":
        return "ok";
      case "planning":
      case "awaiting_user":
      case "paused":
        return "warn";
      case "failed":
        return "bad";
      case "completed":
        return "muted";
      case "cancelled":
        return "muted";
      default:
        return "muted";
    }
  }

  function statusLabel(status: string): string {
    return status.replace(/_/g, " ");
  }

  function isActive(status: string): boolean {
    return ["planning", "running", "paused", "awaiting_user"].includes(status);
  }

  const activeTasks = $derived(tasks.filter((t) => isActive(t.status)));
  const finishedTasks = $derived(tasks.filter((t) => !isActive(t.status)));
</script>

<svelte:head>
  <title>Tasks — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <div class="heading">
      <h1>Tasks</h1>
    </div>
    <p class="lead">
      Long-running work June is doing for you. A task survives the conversation that spawned
      it and reports back when it is done.
    </p>
    <div class="controls">
      <button type="button" onclick={refresh} disabled={loading}>
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  </header>

  <section class="composer-section" aria-label="Create task">
    <form class="composer" onsubmit={createTask}>
      <label for="task-goal" class="sr-only">Task goal</label>
      <input
        id="task-goal"
        type="text"
        bind:value={newGoal}
        placeholder="Tell June what to do — e.g. 'find every PDF from this week about taxes'"
        disabled={creating}
        autocomplete="off"
      />
      <button type="submit" disabled={creating || newGoal.trim().length === 0}>
        {creating ? "Creating…" : "Create task"}
      </button>
    </form>
  </section>

  {#if loadError && tasks.length === 0}
    <OfflineNotice kind="memory" detail={loadError} onRetry={refresh} retrying={loading} />
  {:else if actionError}
    <div class="error" role="alert">Couldn't update that task: {actionError}</div>
  {/if}

  {#if loading && tasks.length === 0}
    <div class="skeleton-list" aria-label="Loading tasks…">
      {#each [1, 2, 3] as _ (_)}
        <div class="skeleton skeleton-card"></div>
      {/each}
    </div>
  {:else if tasks.length === 0 && !loadError}
    <div class="empty">
      <p>No tasks yet.</p>
      <p class="muted">Use the box above to give June her first one.</p>
    </div>
  {:else}
    {#if activeTasks.length > 0}
      <section aria-label="Active tasks">
        <h2 class="group-label">Active</h2>
        <ul class="tasks">
          {#each activeTasks as task (task.id)}
            {@render taskCard(task)}
          {/each}
        </ul>
      </section>
    {/if}

    {#if finishedTasks.length > 0}
      <section aria-label="Finished tasks">
        <h2 class="group-label">Recent</h2>
        <ul class="tasks">
          {#each finishedTasks as task (task.id)}
            {@render taskCard(task)}
          {/each}
        </ul>
      </section>
    {/if}
  {/if}
</main>

<ConfirmDialog
  bind:open={confirmOpen}
  title="Delete task"
  message={confirmTask ? `Delete this task?\n\n"${confirmTask.goal}"` : "Delete this task?"}
  confirmLabel="Delete"
  danger={true}
  onConfirm={doRemoveTask}
  onCancel={() => { confirmTask = null; }}
/>

{#snippet taskCard(task: TaskView)}
  <li class="task">
    <div class="task-head">
      <div class="task-ident">
        <div class="task-goal">{task.goal}</div>
        <div class="task-meta">
          <span class="status status-{statusKind(task.status)}">
            {statusLabel(task.status)}
          </span>
          {#if task.owner_skill}
            <span class="meta-chip">via {task.owner_skill}</span>
          {/if}
          <time class="meta-time" datetime={task.updated_at}>
            {formatRelative(task.updated_at)}
          </time>
        </div>
      </div>
      <div class="task-actions">
        {#if task.status === "planning"}
          <button
            type="button"
            class="task-btn primary"
            onclick={() => patchStatus(task, "running")}
            disabled={pendingAction !== null}
          >
            {pendingAction === `${task.id}:running` ? "…" : "Start"}
          </button>
        {/if}
        {#if task.status === "running"}
          <button
            type="button"
            class="task-btn"
            onclick={() => patchStatus(task, "paused")}
            disabled={pendingAction !== null}
          >
            Pause
          </button>
          <button
            type="button"
            class="task-btn"
            onclick={() => patchStatus(task, "completed")}
            disabled={pendingAction !== null}
          >
            Done
          </button>
        {/if}
        {#if task.status === "paused"}
          <button
            type="button"
            class="task-btn primary"
            onclick={() => patchStatus(task, "running")}
            disabled={pendingAction !== null}
          >
            Resume
          </button>
        {/if}
        {#if isActive(task.status)}
          <button
            type="button"
            class="task-btn"
            onclick={() => patchStatus(task, "cancelled")}
            disabled={pendingAction !== null}
          >
            Cancel
          </button>
        {/if}
        <button
          type="button"
          class="task-btn danger"
          onclick={() => removeTask(task)}
          disabled={pendingAction !== null}
          aria-label="Delete task"
        >
          {pendingAction === `${task.id}:delete` ? "…" : "Delete"}
        </button>
      </div>
    </div>

    {#if task.error}
      <div class="task-error" role="alert">{task.error}</div>
    {/if}

    {#if task.plan && task.plan.length > 0}
      <details
        class="trace-wrap"
        open={expanded[task.id]}
        ontoggle={(e) => {
          expanded = {
            ...expanded,
            [task.id]: (e.currentTarget as HTMLDetailsElement).open,
          };
        }}
      >
        <summary>
          {task.plan.length} step{task.plan.length === 1 ? "" : "s"}
        </summary>
        <ol class="trace">
          {#each task.plan as step (step.id)}
            <li class="step">
              <div class="step-line">
                <span class="step-index">{step.index + 1}.</span>
                <span class="step-desc">{step.description || step.tool_name || "step"}</span>
                <span class="step-status step-status-{statusKind(step.status)}">
                  {statusLabel(step.status)}
                </span>
              </div>
              {#if step.tool_name}
                <div class="step-tool">
                  <code>{step.tool_name}</code>
                  {#if step.model_provenance}
                    {@const prov = step.model_provenance as {
                      provider?: string;
                      model?: string;
                      tier?: string;
                    }}
                    <span class="prov">
                      {prov.tier ?? ""} · {prov.provider ?? ""}{prov.model
                        ? ` · ${prov.model}`
                        : ""}
                    </span>
                  {/if}
                </div>
              {/if}
              {#if step.error}
                <div class="step-error">{step.error}</div>
              {/if}
            </li>
          {/each}
        </ol>
      </details>
    {/if}
  </li>
{/snippet}

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
    gap: var(--space-2);
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

  .lead {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }

  .controls {
    display: flex;
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

  .composer-section {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-3);
  }

  .composer {
    display: flex;
    gap: var(--space-2);
  }
  .composer input {
    flex: 1;
    background: var(--color-bg);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-3);
    font: inherit;
  }
  .composer input:focus {
    outline: 2px solid var(--color-accent);
    outline-offset: -1px;
  }
  .composer button {
    background: var(--color-accent);
    color: var(--color-bg);
    border: 1px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }
  .composer button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
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
    text-align: center;
    padding: var(--space-7) 0;
    color: var(--color-fg-muted);
  }
  .empty .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    margin-top: var(--space-2);
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

  .group-label {
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--color-fg-subtle);
    margin: 0 0 var(--space-2);
  }

  .tasks {
    list-style: none;
    padding: 0;
    margin: 0 0 var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .task {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .task-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
  }

  .task-ident {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    flex: 1;
  }

  .task-goal {
    font-weight: 600;
    color: var(--color-fg-primary);
    word-break: break-word;
  }

  .task-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }

  .meta-chip {
    font-family: var(--font-mono);
  }
  .meta-time {
    font-family: var(--font-mono);
  }

  .status {
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

  .task-actions {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .task-btn {
    background: transparent;
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .task-btn:hover:not(:disabled) {
    border-color: var(--color-accent);
    color: var(--color-accent);
  }
  .task-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .task-btn.primary {
    background: var(--color-accent);
    color: var(--color-bg);
    border-color: var(--color-accent);
  }
  .task-btn.primary:hover:not(:disabled) {
    color: var(--color-bg);
  }
  .task-btn.danger {
    color: var(--color-fg-muted);
  }
  .task-btn.danger:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
  }

  .task-error {
    color: var(--color-danger);
    font-size: var(--size-sm);
    font-family: var(--font-mono);
  }

  .trace-wrap {
    border-top: 1px solid var(--color-border);
    padding-top: var(--space-2);
  }
  .trace-wrap summary {
    cursor: pointer;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    padding: var(--space-1) 0;
    list-style: none;
  }
  .trace-wrap summary::-webkit-details-marker {
    display: none;
  }
  .trace-wrap summary::before {
    content: "▸ ";
    color: var(--color-fg-subtle);
  }
  .trace-wrap[open] summary::before {
    content: "▾ ";
  }

  .trace {
    list-style: none;
    padding: 0;
    margin: var(--space-2) 0 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    /* A runaway plan can have dozens of steps; keep the card a sane height. */
    max-height: 60vh;
    overflow-y: auto;
    /* Reserve scrollbar gutter so steps don't reflow when the list grows. */
    scrollbar-gutter: stable;
  }

  .step {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .step-line {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .step-index {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }

  .step-desc {
    flex: 1;
    color: var(--color-fg-primary);
    font-size: var(--size-sm);
    min-width: 0;
  }

  .step-status {
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: var(--size-xs);
    padding: 2px var(--space-2);
    border-radius: var(--radius-sm);
  }

  .step-tool {
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  .step-tool code {
    font-family: var(--font-mono);
    color: var(--color-fg-muted);
  }
  .prov {
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
  }

  .step-error {
    color: var(--color-danger);
    font-size: var(--size-xs);
    font-family: var(--font-mono);
  }
</style>
