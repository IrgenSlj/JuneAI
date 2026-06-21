<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    OfflineNotice,
    ConfirmDialog,
    TraceEventList,
    type ActivityEntryView,
    type MemorySnapshot,
    type SkillInfo,
    type SkillsResponse,
    type SystemStatus,
    type TraceSummary,
    type TraceEventView,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  const REFRESH_MS = 5000;

  let memory: MemorySnapshot | null = $state(null);
  let skills: SkillInfo[] = $state([]);
  let system: SystemStatus | null = $state(null);
  let activity: ActivityEntryView[] = $state([]);
  let loading = $state(true);
  let clearingActivity = $state(false);
  let confirmClearOpen = $state(false);
  let lastRefresh: Date | null = $state(null);
  let loadError: string | null = $state(null);
  let timer: ReturnType<typeof setInterval> | null = null;

  // Turn traces state
  let traces: TraceSummary[] = $state([]);
  let tracesError: string | null = $state(null);
  let selectedTraceId: string | null = $state(null);
  let selectedTraceEvents: TraceEventView[] = $state([]);
  let traceDetailLoading = $state(false);
  let traceDetailError: string | null = $state(null);

  async function refresh() {
    loading = true;
    loadError = null;
    tracesError = null;
    try {
      const [snap, sk, sys, act, trl] = await Promise.all([
        client.getMemory(profileName.value),
        client.getSkills(),
        client.getSystem(),
        client.getActivity(50).catch(() => ({ entries: [], count: 0 })),
        client.getTraces(50).catch((e: unknown) => {
          tracesError = e instanceof Error ? e.message : String(e);
          return { traces: [], count: 0 };
        }),
      ]);
      memory = snap;
      skills = (sk as SkillsResponse).skills ?? [];
      system = sys;
      activity = act.entries ?? [];
      traces = trl.traces ?? [];
      // Keep an open trace's detail in sync with its (possibly updated) summary,
      // so the event-count badge can never disagree with the expanded panel.
      if (selectedTraceId) {
        try {
          const view = await client.getTrace(selectedTraceId);
          selectedTraceEvents = view.events ?? [];
        } catch {
          // Keep the previously-loaded events on a transient failure.
        }
      }
      lastRefresh = new Date();
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function selectTrace(turnId: string): Promise<void> {
    if (selectedTraceId === turnId) {
      // Toggle off
      selectedTraceId = null;
      selectedTraceEvents = [];
      return;
    }
    selectedTraceId = turnId;
    selectedTraceEvents = [];
    traceDetailLoading = true;
    traceDetailError = null;
    try {
      const view = await client.getTrace(turnId);
      selectedTraceEvents = view.events ?? [];
    } catch (err) {
      traceDetailError = err instanceof Error ? err.message : String(err);
    } finally {
      traceDetailLoading = false;
    }
  }

  function formatEpoch(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleString();
  }

  function relativeEpoch(epochSeconds: number): string {
    const diffMs = Date.now() - epochSeconds * 1000;
    const diffS = Math.floor(diffMs / 1000);
    if (diffS < 60) return `${diffS}s ago`;
    const diffM = Math.floor(diffS / 60);
    if (diffM < 60) return `${diffM}m ago`;
    const diffH = Math.floor(diffM / 60);
    if (diffH < 24) return `${diffH}h ago`;
    return formatEpoch(epochSeconds);
  }

  onMount(() => {
    refresh();
    timer = setInterval(refresh, REFRESH_MS);
  });

  onDestroy(() => {
    if (timer) clearInterval(timer);
  });

  const memoryCounts = $derived.by(() => {
    if (!memory) return null;
    return {
      goals: memory.goals?.length ?? 0,
      open_loops: memory.open_loops?.length ?? 0,
      calendar: memory.calendar?.length ?? 0,
      journal: memory.journal?.length ?? 0,
      body_metrics: memory.body_metrics?.length ?? 0,
      semantic: memory.semantic_facts?.length ?? 0,
      entities: memory.entities?.length ?? 0,
      messages: memory.recent_messages ?? 0,
    };
  });

  const totalSqliteRows = $derived(
    memoryCounts
      ? memoryCounts.goals +
          memoryCounts.open_loops +
          memoryCounts.calendar +
          memoryCounts.journal +
          memoryCounts.body_metrics
      : 0,
  );

  function clearActivity() {
    if (clearingActivity) return;
    confirmClearOpen = true;
  }

  async function doClearActivity() {
    clearingActivity = true;
    try {
      await client.clearActivity();
      activity = [];
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      clearingActivity = false;
    }
  }

  function statusKind(status: number | null | undefined): "ok" | "warn" | "bad" | "muted" {
    if (!status) return "muted";
    if (status >= 500) return "bad";
    if (status >= 400) return "warn";
    if (status >= 200) return "ok";
    return "muted";
  }

  const enabledSkillCount = $derived(skills.filter((s) => s.enabled).length);
  const runningSkillCount = $derived(
    skills.filter((s) => s.status === "running").length,
  );
  const skillToolCounts = $derived.by(() => {
    const tools = skills.flatMap((skill) =>
      (skill.tools ?? []).map((tool) => ({ ...tool, skill: skill.key })),
    );
    return {
      total: tools.length,
      enabled: tools.filter((tool) => tool.enabled).length,
      disabled: tools.filter((tool) => !tool.enabled),
    };
  });

</script>

<svelte:head>
  <title>System — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <div class="heading">
      <h1>System</h1>
      <p class="lede">
        How June is wired together right now. Counts and status pull from the running stack
        and refresh every {REFRESH_MS / 1000} seconds.
      </p>
    </div>
    <div class="controls">
      {#if lastRefresh}
        <span class="hint">updated {formatRelative(lastRefresh.toISOString())}</span>
      {/if}
      <button type="button" onclick={refresh} disabled={loading}>
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  </header>

  {#if loadError && !memory}
    <OfflineNotice kind="memory" detail={loadError} onRetry={refresh} retrying={loading} />
  {/if}

  <!-- Layer 1 — Shells -->
  <section class="layer">
    <div class="layer-label">Shells</div>
    <div class="row">
      <div class="card">
        <div class="card-head">
          <h2>Web PWA</h2>
          <span class="badge ok">active</span>
        </div>
        <p class="card-body">
          SvelteKit, served from the same build the desktop shell wraps.
          Installable, offline-capable, talks to the API over HTTP + SSE.
        </p>
      </div>
      <div class="card">
        <div class="card-head">
          <h2>Desktop Shell</h2>
          <span class="badge muted">scaffolded — not built</span>
        </div>
        <p class="card-body">
          Tauri 2.x at <code>apps/desktop/</code>. Phases 1-4 committed; the
          Rust has not been compiled yet on this machine. <code>pnpm desktop:dev</code>
          requires <code>rustup</code>.
        </p>
      </div>
    </div>
  </section>

  <div class="connector" aria-hidden="true">↓ HTTP / Server-Sent Events</div>

  <!-- Layer 2 — API -->
  <section class="layer">
    <div class="layer-label">API</div>
    <div class="row">
      <div class="card wide">
        <div class="card-head">
          <h2>FastAPI · :8000</h2>
          {#if system}
            <span class="badge ok">healthy</span>
          {:else}
            <span class="badge warn">unreachable</span>
          {/if}
        </div>
        <ul class="endpoints">
          <li><code>POST /chat</code> — SSE stream of <code>token</code> · <code>tool_call</code> · <code>recall</code> · <code>tool_result</code> · <code>done</code></li>
          <li><code>GET /memory/{"{user}"}</code>, <code>POST /memory/{"{user}"}/fact</code>, <code>PATCH /memory/{"{user}"}/fact/{"{ref}"}</code>, <code>DELETE /memory/{"{user}"}/fact/{"{ref}"}</code></li>
          <li><code>POST /memory/{"{user}"}/feedback</code> — thumbs up / down on recalled memories</li>
          <li><code>GET /skills</code>, <code>POST /skills/{"{key}"}/toggle</code>, <code>POST /skills/{"{key}"}/tools/{"{tool}"}/toggle</code></li>
          <li><code>GET /system</code>, <code>GET /setup</code>, <code>POST /setup</code>, <code>GET /settings</code></li>
        </ul>
      </div>
    </div>
  </section>

  <div class="connector" aria-hidden="true">↓ in-process Python</div>

  <!-- Layer 3 — Brain -->
  <section class="layer">
    <div class="layer-label">Brain</div>
    <div class="row">
      <div class="card wide">
        <div class="card-head">
          <h2>Handwritten loop</h2>
          <span class="badge ok">running</span>
        </div>
        <div class="flow">
          <span class="flow-step">recall</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">generate</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">extract</span>
        </div>
        <p class="card-body">
          Every turn fans out to all three memory stores for recall, runs the LLM,
          then writes durable facts/entities/relations back through
          <code>MemoryManager.write()</code> — the single write entry point shared
          by extract and skills.
        </p>
      </div>
    </div>
  </section>

  <div class="connector branch" aria-hidden="true">
    <span>↓ Memory</span>
    <span>↓ Skills</span>
    <span>↓ Provider</span>
  </div>

  <!-- Layer 4 — split: Memory · Skills · Provider -->
  <section class="layer triple">
    <div class="row triple-row">
      <!-- Memory -->
      <div class="card">
        <div class="card-head">
          <h2>Memory</h2>
          {#if memoryCounts}
            <span class="badge ok"
              >{totalSqliteRows + memoryCounts.semantic + memoryCounts.entities} items</span
            >
          {/if}
        </div>
        <div class="store">
          <h3>SQLite — structured</h3>
          {#if memoryCounts}
            <ul class="counts">
              <li><span>goals</span><b>{memoryCounts.goals}</b></li>
              <li><span>open loops</span><b>{memoryCounts.open_loops}</b></li>
              <li><span>calendar</span><b>{memoryCounts.calendar}</b></li>
              <li><span>journal</span><b>{memoryCounts.journal}</b></li>
              <li><span>body metrics</span><b>{memoryCounts.body_metrics}</b></li>
              <li><span>chat history</span><b>{memoryCounts.messages}</b></li>
            </ul>
          {/if}
        </div>
        <div class="store">
          <h3>ChromaDB — semantic</h3>
          {#if memoryCounts}
            <ul class="counts">
              <li><span>facts</span><b>{memoryCounts.semantic}</b></li>
            </ul>
          {/if}
        </div>
        <div class="store">
          <h3>Knowledge graph</h3>
          {#if memoryCounts}
            <ul class="counts">
              <li><span>entities</span><b>{memoryCounts.entities}</b></li>
            </ul>
          {/if}
        </div>
      </div>

      <!-- Skills -->
      <div class="card">
        <div class="card-head">
          <h2>Skills</h2>
          <span class="badge ok">{runningSkillCount} of {enabledSkillCount} running</span>
        </div>
        <div class="store">
          <h3>Tool surface</h3>
          <ul class="counts">
            <li><span>tools enabled</span><b>{skillToolCounts.enabled}</b></li>
            <li><span>tools total</span><b>{skillToolCounts.total}</b></li>
          </ul>
        </div>
        <ul class="skill-list">
          {#each skills as skill (skill.key)}
            <li class="skill-row">
              <div class="skill-line">
                <span class="skill-name">{skill.key}</span>
                <span class="skill-tag" data-status={skill.status}>{skill.status}</span>
              </div>
              <div class="skill-tools">
                {(skill.tools ?? []).filter((tool) => tool.enabled).length} of
                {skill.tools?.length ?? 0}
                {(skill.tools?.length ?? 0) === 1 ? "tool" : "tools"} enabled
              </div>
            </li>
          {/each}
        </ul>
        {#if skillToolCounts.disabled.length}
          <div class="disabled-tools">
            <h3>Disabled tools</h3>
            {#each skillToolCounts.disabled as tool (`${tool.skill}:${tool.name}`)}
              <code>{tool.skill}.{tool.name}</code>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Provider -->
      <div class="card">
        <div class="card-head">
          <h2>Provider</h2>
          {#if system}
            <span class="badge" class:ok={system.mode === "local"} class:accent={system.mode === "api"}>
              {system.privacy_label}
            </span>
          {/if}
        </div>
        {#if system}
          <dl class="kv">
            <div class="kv-row"><dt>label</dt><dd>{system.label}</dd></div>
            <div class="kv-row"><dt>model</dt><dd><code>{system.model || "—"}</code></dd></div>
            <div class="kv-row"><dt>mode</dt><dd>{system.mode}</dd></div>
            {#if system.base_url}
              <div class="kv-row"><dt>endpoint</dt><dd><code>{system.base_url}</code></dd></div>
            {/if}
            {#if system.provider === "gemma"}
              <div class="kv-row">
                <dt>ollama</dt>
                <dd>
                  {system.ollama_reachable ? "reachable" : "offline"}
                  {#if system.ollama_reachable}
                    · {system.ollama_has_model ? "model pulled" : "model missing"}
                  {/if}
                </dd>
              </div>
            {:else}
              <div class="kv-row">
                <dt>api key</dt>
                <dd>{system.api_key_present ? "set" : "missing"}</dd>
              </div>
            {/if}
          </dl>
        {/if}
      </div>
    </div>
  </section>

  <section class="layer">
    <div class="layer-label">Activity</div>
    <div class="card activity-card">
      <div class="card-head">
        <h2>Recent activity</h2>
        <div class="activity-actions">
          <span class="badge muted">{activity.length} entr{activity.length === 1 ? "y" : "ies"}</span>
          <button
            type="button"
            class="activity-clear"
            onclick={clearActivity}
            disabled={clearingActivity || activity.length === 0}
          >
            {clearingActivity ? "Clearing…" : "Clear log"}
          </button>
        </div>
      </div>
      <p class="card-body">
        Every API request June handled this session. The trust primitive — if June
        did something, it's here.
      </p>
      {#if activity.length === 0}
        <p class="hint">No activity recorded yet. Open the chat or click around to populate.</p>
      {:else}
        <ul class="activity-list">
          {#each activity as entry (entry.id)}
            <li class="activity-row">
              <span class="activity-kind">{entry.kind}</span>
              <span class="activity-label">{entry.label}</span>
              {#if entry.status !== null && entry.status !== undefined}
                <span class="activity-status activity-status-{statusKind(entry.status)}">
                  {entry.status}
                </span>
              {/if}
              {#if entry.latency_ms !== null && entry.latency_ms !== undefined}
                <span class="activity-latency">{entry.latency_ms}ms</span>
              {/if}
              <time class="activity-time" datetime={entry.timestamp}>
                {formatRelative(entry.timestamp)}
              </time>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>

  <!-- Turn traces — glass-box view of persisted per-turn harness operations -->
  <section class="layer">
    <div class="layer-label">Turn traces</div>
    <div class="card activity-card">
      <div class="card-head">
        <h2>Turn traces</h2>
        <span class="badge muted">{traces.length} turn{traces.length === 1 ? "" : "s"}</span>
      </div>
      <p class="card-body">
        The complete glass-box record for every turn: the assembled prompt, each LLM iteration's
        raw output, the model's reasoning and thinking, every tool call and result, and the
        cloud-boundary provenance line. Select a turn to expand its full trace.
      </p>
      {#if tracesError}
        <p class="hint trace-error">{tracesError}</p>
      {:else if traces.length === 0}
        <p class="hint">No turns recorded yet. Send a message to generate a trace.</p>
      {:else}
        <ul class="trace-summary-list">
          {#each traces as trace (trace.turn_id)}
            {@const isSelected = selectedTraceId === trace.turn_id}
            <li class="trace-summary-row" class:selected={isSelected}>
              <button
                type="button"
                class="trace-summary-btn"
                onclick={() => selectTrace(trace.turn_id)}
                aria-expanded={isSelected && !traceDetailError}
                aria-controls={isSelected ? `trace-panel-${trace.turn_id}` : undefined}
              >
                <span class="trace-id">{trace.turn_id}</span>
                <span class="trace-count">{trace.event_count} event{trace.event_count === 1 ? "" : "s"}</span>
                <time class="trace-time" title={formatEpoch(trace.started_at)}>
                  {relativeEpoch(trace.started_at)}
                </time>
                <span class="trace-chevron" aria-hidden="true">{isSelected ? "▲" : "▼"}</span>
              </button>
              {#if isSelected}
                <div class="trace-detail-panel" id={`trace-panel-${trace.turn_id}`}>
                  {#if traceDetailLoading}
                    <p class="hint">Loading trace…</p>
                  {:else if traceDetailError}
                    <p class="hint trace-error">{traceDetailError}</p>
                  {:else}
                    <TraceEventList events={selectedTraceEvents} />
                  {/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>
</main>

<ConfirmDialog
  bind:open={confirmClearOpen}
  title="Clear activity log"
  message="Clear the entire activity log? This cannot be undone."
  confirmLabel="Clear log"
  danger={true}
  onConfirm={doClearActivity}
/>

<style>
  .page {
    max-width: 980px;
    margin: 0 auto;
    padding: var(--space-5) var(--space-4) var(--space-7);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }

  .top {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--color-border);
  }

  .heading h1 {
    margin: 0 0 var(--space-1);
    font-size: var(--size-xl);
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .lede {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    max-width: 60ch;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .hint {
    color: var(--color-fg-subtle);
    font-size: var(--size-xs);
    font-family: var(--font-mono);
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
  .controls button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .layer {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .layer-label {
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }

  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }
  .row.triple-row {
    grid-template-columns: 1.2fr 1fr 1fr;
  }
  @media (max-width: 860px) {
    .row,
    .row.triple-row {
      grid-template-columns: 1fr;
    }
  }

  .card {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .card.wide {
    grid-column: 1 / -1;
  }

  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .card h2 {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
  }
  .card h3 {
    margin: var(--space-2) 0 var(--space-1);
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--color-fg-subtle);
    font-weight: 500;
  }

  .card-body {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
  }

  .badge {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--color-border);
    color: var(--color-fg-muted);
    background: var(--color-bg);
    flex-shrink: 0;
  }
  .badge.ok {
    color: var(--color-success);
    border-color: color-mix(in srgb, var(--color-success) 40%, transparent);
    background: color-mix(in srgb, var(--color-success) 10%, transparent);
  }
  .badge.warn {
    color: var(--color-danger);
    border-color: color-mix(in srgb, var(--color-danger) 40%, transparent);
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }
  .badge.accent {
    color: var(--color-accent);
    border-color: color-mix(in srgb, var(--color-accent) 40%, transparent);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
  }
  .badge.muted {
    color: var(--color-fg-subtle);
  }

  .connector {
    align-self: center;
    color: var(--color-fg-subtle);
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    padding: var(--space-1) 0;
    text-align: center;
  }
  .connector.branch {
    display: flex;
    gap: var(--space-6);
    justify-content: center;
  }

  .endpoints {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .endpoints code {
    font-size: var(--size-xs);
  }

  .flow {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .flow-step {
    padding: 4px 12px;
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-pill);
    font-size: var(--size-sm);
    font-family: var(--font-mono);
  }
  .flow-arrow {
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }

  .store {
    border-top: 1px dashed var(--color-border);
    padding-top: var(--space-2);
  }
  .store:first-of-type {
    border-top: none;
    padding-top: 0;
  }

  .counts {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-1) var(--space-3);
    font-size: var(--size-sm);
  }
  .counts li {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
    color: var(--color-fg-muted);
  }
  .counts b {
    font-family: var(--font-mono);
    color: var(--color-fg-primary);
  }

  .skill-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .skill-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .skill-line {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-3);
  }
  .skill-name {
    font-weight: 500;
  }
  .skill-tag {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
  }
  .skill-tag[data-status="running"] {
    color: var(--color-success);
  }
  .skill-tag[data-status="crashed"] {
    color: var(--color-danger);
  }
  .skill-tools {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }

  .disabled-tools {
    border-top: 1px dashed var(--color-border);
    padding-top: var(--space-2);
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .disabled-tools h3 {
    flex-basis: 100%;
  }
  .disabled-tools code {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 2px 6px;
  }

  .kv {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    font-size: var(--size-sm);
  }
  .kv-row {
    display: flex;
    gap: var(--space-3);
  }
  .kv dt {
    flex-shrink: 0;
    width: 6em;
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    align-self: center;
  }
  .kv dd {
    margin: 0;
    color: var(--color-fg-muted);
    flex: 1;
  }

  code {
    font-family: var(--font-mono);
  }

  .activity-card {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .activity-actions {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .activity-clear {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }
  .activity-clear:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
  }
  .activity-clear:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .activity-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    max-height: 400px;
    overflow-y: auto;
  }
  .activity-row {
    display: grid;
    grid-template-columns: 70px 1fr auto auto auto;
    gap: var(--space-2);
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg);
    align-items: center;
    font-size: var(--size-sm);
  }
  .activity-kind {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .activity-label {
    font-family: var(--font-mono);
    color: var(--color-fg-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .activity-status {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    padding: 1px var(--space-2);
    border-radius: var(--radius-sm);
  }
  .activity-status-ok {
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .activity-status-warn {
    color: var(--color-fg-primary);
    background: color-mix(in srgb, var(--color-fg-muted) 20%, transparent);
  }
  .activity-status-bad {
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 15%, transparent);
  }
  .activity-status-muted {
    color: var(--color-fg-subtle);
    border: 1px dashed var(--color-border);
  }
  .activity-latency {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  .activity-time {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    text-align: right;
  }

  /* Turn traces */
  .trace-summary-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
  .trace-summary-row {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg);
    overflow: hidden;
  }
  .trace-summary-row.selected {
    border-color: color-mix(in srgb, var(--color-accent) 40%, transparent);
  }
  .trace-summary-btn {
    display: grid;
    grid-template-columns: 1fr auto auto auto;
    gap: var(--space-3);
    width: 100%;
    padding: var(--space-2) var(--space-3);
    border: none;
    background: transparent;
    text-align: left;
    font: inherit;
    cursor: pointer;
    align-items: center;
  }
  .trace-summary-btn:hover {
    background: color-mix(in srgb, var(--color-fg-muted) 6%, transparent);
  }
  .trace-id {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .trace-count {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    flex-shrink: 0;
  }
  .trace-time {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    flex-shrink: 0;
  }
  .trace-chevron {
    font-size: 9px;
    color: var(--color-fg-subtle);
    flex-shrink: 0;
  }
  .trace-detail-panel {
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--color-border);
    background: var(--color-term-bg, var(--color-bg));
  }
  .trace-error {
    color: var(--color-danger);
  }
</style>
