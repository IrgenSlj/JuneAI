<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { OfflineNotice, ConfirmDialog, type TaskView, type TaskStepView } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative, dueState } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let tasks: TaskView[] = $state([]);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let actionError: string | null = $state(null);
  let pendingAction: string | null = $state(null);

  let newGoal = $state("");
  let newDue = $state("");
  let creating = $state(false);
  let pendingDue: string | null = $state(null);

  const DUE_LABEL: Record<string, string> = {
    overdue: "overdue",
    due_soon: "due soon",
    scheduled: "due",
  };

  let expanded: Record<string, boolean> = $state({});

  let confirmOpen = $state(false);
  let confirmTask: TaskView | null = $state(null);

  // Live event stream tracking.
  // Maps task_id -> AbortController for the open SSE stream.
  let liveStreams: Map<string, AbortController> = new Map();
  // Set of task IDs for which a live stream is open (used for pulsing dot).
  let streamingTaskIds: Set<string> = $state(new Set());

  function openStream(task: TaskView): void {
    if (liveStreams.has(task.id)) return;
    const ac = new AbortController();
    liveStreams.set(task.id, ac);
    streamingTaskIds = new Set([...streamingTaskIds, task.id]);

    (async () => {
      try {
        for await (const frame of client.streamTaskEvents(
          profileName.value,
          task.id,
          ac.signal,
        )) {
          if (frame.type === "step" && frame.step) {
            const incoming = frame.step as TaskStepView;
            tasks = tasks.map((t) => {
              if (t.id !== task.id) return t;
              const plan = t.plan ?? [];
              const already = plan.some((s) => s.id === incoming.id);
              if (already) return t;
              return { ...t, plan: [...plan, incoming] };
            });
          } else if (frame.type === "status" && frame.status) {
            tasks = tasks.map((t) =>
              t.id === task.id ? { ...t, status: frame.status! } : t,
            );
          } else if (frame.type === "done") {
            break;
          }
        }
      } catch {
        // AbortError on navigate-away or explicit close — suppress.
      } finally {
        liveStreams.delete(task.id);
        streamingTaskIds = new Set(
          [...streamingTaskIds].filter((id) => id !== task.id),
        );
      }
    })();
  }

  function closeStream(taskId: string): void {
    const ac = liveStreams.get(taskId);
    if (ac) {
      ac.abort();
      liveStreams.delete(taskId);
      streamingTaskIds = new Set(
        [...streamingTaskIds].filter((id) => id !== taskId),
      );
    }
  }

  async function refresh() {
    loading = true;
    loadError = null;
    actionError = null;
    try {
      const response = await client.getTasks(profileName.value);
      tasks = response.tasks ?? [];
      // Re-open streams for tasks that are still running after a refresh.
      for (const t of tasks) {
        if (t.status === "running" && !liveStreams.has(t.id)) {
          openStream(t);
        }
      }
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(refresh);

  onDestroy(() => {
    for (const ac of liveStreams.values()) ac.abort();
    liveStreams.clear();
  });

  async function createTask(event: SubmitEvent) {
    event.preventDefault();
    const goal = newGoal.trim();
    if (!goal || creating) return;
    creating = true;
    actionError = null;
    try {
      // <input type="datetime-local"> yields local time without a zone; send as-is.
      const due_at = newDue ? new Date(newDue).toISOString() : undefined;
      await client.createTask(profileName.value, { goal, due_at });
      newGoal = "";
      newDue = "";
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      creating = false;
    }
  }

  async function setDue(task: TaskView, value: string) {
    if (pendingDue) return;
    pendingDue = task.id;
    actionError = null;
    try {
      // Empty clears the deadline; a value sets it (ISO 8601).
      const due_at = value ? new Date(value).toISOString() : "";
      const updated = await client.patchTask(profileName.value, task.id, { due_at });
      tasks = tasks.map((t) => (t.id === updated.id ? updated : t));
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingDue = null;
    }
  }

  /** ISO -> value for <input type="datetime-local"> (local, no seconds/zone). */
  function toLocalInput(iso: string | null | undefined): string {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "";
    const d = new Date(t);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  async function patchStatus(task: TaskView, status: string) {
    const key = `${task.id}:${status}`;
    if (pendingAction) return;
    pendingAction = key;
    actionError = null;
    try {
      const updated = await client.patchTask(profileName.value, task.id, { status });
      tasks = tasks.map((t) => (t.id === updated.id ? updated : t));
      if (status === "running") {
        openStream(updated);
      } else {
        closeStream(task.id);
        await refresh();
      }
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingAction = null;
    }
  }

  async function approveAndRetry(task: TaskView, tool: string) {
    const key = `${task.id}:approve`;
    if (pendingAction || !tool) return;
    pendingAction = key;
    actionError = null;
    try {
      const updated = await client.approveTaskTool(profileName.value, task.id, tool);
      tasks = tasks.map((t) => (t.id === updated.id ? updated : t));
      openStream(updated);
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

  function waitingStep(task: TaskView): TaskStepView | null {
    if (task.status !== "awaiting_user") return null;
    const pending = (task.plan ?? [])
      .slice()
      .reverse()
      .find((step) => step.status === "pending");
    return pending ?? null;
  }

  function stepDetail(step: TaskStepView): string {
    if (step.tool_result === null || step.tool_result === undefined) return "";
    if (typeof step.tool_result === "string") return step.tool_result;
    return JSON.stringify(step.tool_result, null, 2);
  }

  function isActive(status: string): boolean {
    return ["planning", "running", "paused", "awaiting_user"].includes(status);
  }

  const MAX_TASK_ATTEMPTS = 5;

  function isExhausted(task: TaskView): boolean {
    return (
      task.status === "failed" &&
      task.attempts >= MAX_TASK_ATTEMPTS &&
      (task.error ?? "").startsWith("Stopped after ")
    );
  }

  const activeTasks = $derived(tasks.filter((t) => isActive(t.status)));
  const finishedTasks = $derived(tasks.filter((t) => !isActive(t.status)));
</script>

<svelte:head>
  <title>Promises — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <div class="heading">
      <h1>Promises</h1>
    </div>
    <p class="lead">
      Standing work June is holding for you. A promise can run, wait, ask for
      approval, or finish with a visible trace.
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
      <label for="task-due" class="sr-only">Optional deadline</label>
      <input
        id="task-due"
        type="datetime-local"
        class="due-input"
        bind:value={newDue}
        disabled={creating}
        title="Optional deadline"
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
          {#if streamingTaskIds.has(task.id)}
            <span class="live-dot" title="Live"></span>
          {/if}
          {#if task.attempts > 1 && isActive(task.status)}
            <span class="attempt-chip" aria-label="Attempt {task.attempts} of {MAX_TASK_ATTEMPTS}">
              attempt {task.attempts} of {MAX_TASK_ATTEMPTS}
            </span>
          {/if}
          {#if task.owner_skill}
            <a class="meta-chip skill-link" href="/skills#skill-{task.owner_skill}">
              via {task.owner_skill}
            </a>
          {/if}
          <time class="meta-time" datetime={task.updated_at}>
            {formatRelative(task.updated_at)}
          </time>
          {#if task.due_at && dueState(task.due_at)}
            {@const ds = dueState(task.due_at)}
            <span class="due-badge due-{ds}" title={`Due ${formatRelative(task.due_at)}`}>
              {DUE_LABEL[ds ?? "scheduled"]} {formatRelative(task.due_at)}
            </span>
          {/if}
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
        {#if task.status === "awaiting_user"}
          <button
            type="button"
            class="task-btn primary"
            onclick={() => patchStatus(task, "running")}
            disabled={pendingAction !== null}
          >
            {pendingAction === `${task.id}:running` ? "…" : "Retry"}
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

    {#if isActive(task.status)}
      <div class="task-due-editor">
        <label for="due-{task.id}">Deadline</label>
        <input
          id="due-{task.id}"
          type="datetime-local"
          class="due-input"
          value={toLocalInput(task.due_at)}
          disabled={pendingDue === task.id}
          onchange={(e) => setDue(task, e.currentTarget.value)}
        />
        {#if task.due_at}
          <button
            type="button"
            class="due-clear"
            onclick={() => setDue(task, "")}
            disabled={pendingDue === task.id}
          >
            Clear
          </button>
        {/if}
      </div>
    {/if}

    {#if isExhausted(task)}
      <div class="task-exhausted" role="status" aria-label="Promise exhausted after {task.attempts} attempts">
        <div class="exhausted-title">Needs a new approach</div>
        <p>{task.error}</p>
        {#if task.next_action}
          <p class="exhausted-action">{task.next_action}</p>
        {/if}
        {#if task.blocked_reason}
          <p class="exhausted-reason">{task.blocked_reason}</p>
        {/if}
      </div>
    {:else if task.error}
      <div class="task-error" role="alert">{task.error}</div>
    {/if}

    {#if waitingStep(task)}
      {@const step = waitingStep(task)}
      <div class="task-waiting" role="status">
        <div class="waiting-title">Waiting on you</div>
        <p>
          {task.blocked_reason ||
            step?.description ||
            "June needs your approval before this promise can continue."}
        </p>
        {#if task.next_action}
          <p>{task.next_action}</p>
        {/if}
        {#if step?.tool_name}
          <p class="waiting-meta">
            Tool: <code>{step.tool_name}</code>
          </p>
        {/if}
        {#if task.blocked_kind === "approval" && step?.tool_name}
          <button
            type="button"
            class="waiting-approve"
            onclick={() => approveAndRetry(task, step!.tool_name!)}
            disabled={pendingAction !== null}
          >
            {pendingAction === `${task.id}:approve` ? "…" : "Approve & retry"}
          </button>
        {/if}
      </div>
    {/if}

    {#if task.status === "completed" && task.final_deliverable}
      <div class="task-deliverable">
        <div class="deliverable-title">Deliverable</div>
        <p>{task.final_deliverable}</p>
      </div>
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
              {#if step.status === "pending" && stepDetail(step)}
                <pre class="step-detail">{stepDetail(step)}</pre>
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
  .composer input.due-input {
    flex: 0 0 auto;
    color-scheme: light dark;
  }

  /* Deadline badge on a card — semantic color by urgency. */
  .due-badge {
    font-size: var(--size-xs);
    padding: 1px var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    white-space: nowrap;
  }
  .due-overdue {
    color: var(--color-danger);
    border-color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }
  .due-due_soon {
    color: var(--color-warn);
    border-color: var(--color-warn);
    background: color-mix(in srgb, var(--color-warn) 10%, transparent);
  }
  .due-scheduled {
    color: var(--color-fg-muted);
  }

  .task-due-editor {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-2);
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
  }
  .task-due-editor .due-input {
    background: var(--color-bg);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    color-scheme: light dark;
  }
  .due-clear {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    cursor: pointer;
    font-size: var(--size-xs);
  }
  .due-clear:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
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
  .meta-chip.skill-link {
    color: var(--color-fg-subtle);
    text-decoration: none;
    border-bottom: 1px dotted var(--color-border);
  }
  .meta-chip.skill-link:hover {
    color: var(--color-accent);
    border-bottom-color: var(--color-accent);
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

  .attempt-chip {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
    background: color-mix(in srgb, var(--color-fg-muted) 10%, transparent);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 1px var(--space-2);
  }

  .task-exhausted {
    background: color-mix(in srgb, var(--color-warn) 8%, var(--color-bg));
    border: 1px solid color-mix(in srgb, var(--color-warn) 40%, var(--color-border));
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .task-exhausted p {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    line-height: var(--leading-normal);
  }
  .exhausted-title {
    font-size: var(--size-xs);
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    color: var(--color-warn);
  }
  .exhausted-action {
    font-style: italic;
  }

  .task-waiting {
    background: color-mix(in srgb, var(--color-accent) 10%, var(--color-bg));
    border: 1px solid color-mix(in srgb, var(--color-accent) 35%, var(--color-border));
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    color: var(--color-fg-primary);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .task-waiting p {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .waiting-title {
    font-size: var(--size-xs);
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    color: var(--color-accent);
  }
  .waiting-meta {
    font-family: var(--font-mono);
  }
  .waiting-meta code {
    color: var(--color-fg-primary);
  }
  .waiting-approve {
    align-self: flex-start;
    margin-top: var(--space-2);
    background: var(--color-accent);
    color: var(--color-accent-ink);
    border: none;
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-3);
    font-size: var(--size-sm);
    font-weight: 500;
    cursor: pointer;
  }
  .waiting-approve:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .task-deliverable {
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .task-deliverable p {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
    display: -webkit-box;
    line-clamp: 4;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .deliverable-title {
    font-size: var(--size-xs);
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    color: var(--color-fg-subtle);
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

  .step-detail {
    margin: var(--space-1) 0 0;
    max-height: 140px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--color-bg-raised);
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.75); }
  }
  .live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--color-accent);
    animation: pulse-dot 1.2s ease-in-out infinite;
    flex-shrink: 0;
  }
</style>
