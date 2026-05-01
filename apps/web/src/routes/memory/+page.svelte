<script lang="ts">
  import { onMount } from "svelte";
  import {
    createJuneClient,
    OfflineNotice,
    type MemoryFact,
    type MemorySnapshot,
  } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";
  const USER_ID = "local";

  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl });

  let snapshot: MemorySnapshot | null = $state(null);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let actionError: string | null = $state(null);
  let query = $state("");
  let pendingDelete: string | null = $state(null);

  type Section = {
    id: string;
    title: string;
    empty: string;
    facts: MemoryFact[];
    deletable: boolean;
  };

  const sections = $derived.by<Section[]>(() => {
    if (!snapshot) return [];
    return [
      {
        id: "semantic",
        title: "Facts",
        empty:
          "No durable facts yet. As June learns that you prefer tea or live in Berlin, those land here.",
        facts: snapshot.semantic_facts ?? [],
        deletable: true,
      },
      {
        id: "entities",
        title: "People, places, and things",
        empty:
          "No entities mapped yet. People and projects you mention become nodes here.",
        facts: snapshot.entities ?? [],
        deletable: true,
      },
      {
        id: "goals",
        title: "Goals",
        empty: "No goals tracked. Ask June to remember a goal and it shows up here.",
        facts: snapshot.goals ?? [],
        deletable: true,
      },
      {
        id: "open_loops",
        title: "Open loops",
        empty:
          "No open threads. Questions left dangling or follow-ups June is holding will appear here.",
        facts: snapshot.open_loops ?? [],
        deletable: true,
      },
      {
        id: "calendar",
        title: "Calendar",
        empty: "Nothing on the calendar. Ask June to save events or reminders.",
        facts: snapshot.calendar ?? [],
        deletable: true,
      },
    ];
  });

  const TIMESTAMP_KEYS = [
    "timestamp",
    "created_at",
    "updated_at",
    "last_seen",
    "when",
    "at",
  ];

  function pickTimestamp(fact: MemoryFact): string | null {
    const meta = fact.metadata ?? {};
    for (const key of TIMESTAMP_KEYS) {
      const raw = meta[key];
      if (typeof raw === "string" && raw.length > 0) return raw;
    }
    return null;
  }

  function formatRelative(iso: string): string {
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return iso;
    const diffMs = Date.now() - then;
    const abs = Math.abs(diffMs);
    const minute = 60_000;
    const hour = 3_600_000;
    const day = 86_400_000;
    if (abs < minute) return "just now";
    if (abs < hour) {
      const m = Math.round(diffMs / minute);
      return m > 0 ? `${m}m ago` : `in ${-m}m`;
    }
    if (abs < day) {
      const h = Math.round(diffMs / hour);
      return h > 0 ? `${h}h ago` : `in ${-h}h`;
    }
    if (abs < day * 30) {
      const d = Math.round(diffMs / day);
      return d > 0 ? `${d}d ago` : `in ${-d}d`;
    }
    return new Date(then).toLocaleDateString();
  }

  const filteredSections = $derived.by<Section[]>(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sections;
    return sections.map((section) => ({
      ...section,
      facts: section.facts.filter((f) =>
        `${f.title} ${f.body} ${Object.values(f.metadata ?? {}).join(" ")}`
          .toLowerCase()
          .includes(q),
      ),
    }));
  });

  const totalShown = $derived(
    filteredSections.reduce((n, s) => n + s.facts.length, 0),
  );

  async function refresh() {
    loading = true;
    loadError = null;
    actionError = null;
    try {
      snapshot = await client.getMemory(USER_ID);
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(refresh);

  async function handleDelete(fact: MemoryFact) {
    if (!fact.ref || pendingDelete) return;
    pendingDelete = fact.ref;
    actionError = null;
    try {
      await client.deleteMemoryFact(USER_ID, fact.ref);
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingDelete = null;
    }
  }

  function metadataEntries(fact: MemoryFact): [string, string][] {
    const meta = fact.metadata ?? {};
    return Object.entries(meta)
      .filter(([k]) => !TIMESTAMP_KEYS.includes(k))
      .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
      .map(([k, v]) => [k, String(v)]);
  }
</script>

<svelte:head>
  <title>Memory — June</title>
</svelte:head>

<main class="page" id="main-content">
  <header class="top">
    <div class="heading">
      <h1>Memory</h1>
    </div>
    <div class="controls">
      <input
        type="search"
        placeholder="Search memories…"
        bind:value={query}
        aria-label="Filter memories"
      />
      <button type="button" onclick={refresh} disabled={loading}>
        {loading ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  </header>

  {#if loadError && !snapshot}
    <OfflineNotice
      kind="memory"
      detail={loadError}
      onRetry={refresh}
      retrying={loading}
    />
  {:else if actionError}
    <div class="error">
      Couldn't save that change: {actionError}
    </div>
  {/if}

  {#if !snapshot && loading}
    <div class="empty">Loading what June remembers…</div>
  {:else if snapshot}
    <p class="summary">
      {#if query.trim()}
        {totalShown} match{totalShown === 1 ? "" : "es"} for "{query}"
      {:else}
        {snapshot.semantic_facts?.length ?? 0} fact{(snapshot.semantic_facts?.length ?? 0) === 1 ? "" : "s"},
        {snapshot.entities?.length ?? 0} entit{(snapshot.entities?.length ?? 0) === 1 ? "y" : "ies"},
        {snapshot.recent_messages ?? 0} message{(snapshot.recent_messages ?? 0) === 1 ? "" : "s"} in history
      {/if}
    </p>

    {#each filteredSections as section (section.id)}
      {#if !query.trim() || section.facts.length > 0}
        <section class="group">
          <div class="group-head">
            <h2>{section.title}</h2>
            <span class="count">{section.facts.length}</span>
          </div>
          {#if section.facts.length === 0}
            <p class="muted">{section.empty}</p>
          {:else}
            <ul class="facts">
              {#each section.facts as fact (fact.ref || `${section.id}-${fact.title}`)}
                {@const ts = pickTimestamp(fact)}
                <li class="fact">
                  <div class="fact-main">
                    <div class="fact-head">
                      {#if fact.title}
                        <div class="fact-title">{fact.title}</div>
                      {/if}
                      {#if ts}
                        <time class="fact-time" datetime={ts}>{formatRelative(ts)}</time>
                      {/if}
                    </div>
                    {#if fact.body && fact.body !== fact.title}
                      <div class="fact-body">{fact.body}</div>
                    {/if}
                    {#if metadataEntries(fact).length}
                      <dl class="meta">
                        {#each metadataEntries(fact) as [key, value] (key)}
                          <div class="meta-row">
                            <dt>{key}</dt>
                            <dd>{value}</dd>
                          </div>
                        {/each}
                      </dl>
                    {/if}
                  </div>
                  {#if section.deletable && fact.ref}
                    <button
                      type="button"
                      class="delete"
                      onclick={() => handleDelete(fact)}
                      disabled={pendingDelete === fact.ref}
                      aria-label="Forget this"
                    >
                      {pendingDelete === fact.ref ? "…" : "Forget"}
                    </button>
                  {/if}
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      {/if}
    {/each}
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

  .controls {
    display: flex;
    gap: var(--space-2);
  }

  .controls input {
    flex: 1;
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-3);
    font: inherit;
  }
  .controls input:focus {
    outline: none;
    border-color: var(--color-accent);
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

  .group {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .group-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .group h2 {
    margin: 0;
    font-size: var(--size-sm);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--color-fg-muted);
  }
  .count {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }

  .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
    margin: 0;
  }

  .facts {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .fact {
    display: flex;
    gap: var(--space-3);
    align-items: flex-start;
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
  }

  .fact-main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .fact-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    justify-content: space-between;
  }
  .fact-title {
    font-weight: 500;
    color: var(--color-fg-primary);
    min-width: 0;
    flex: 1;
  }
  .fact-time {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
    flex-shrink: 0;
  }

  .fact-body {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    line-height: var(--leading-normal);
  }

  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    margin: var(--space-1) 0 0;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
  }
  .meta-row {
    display: inline-flex;
    gap: var(--space-1);
  }
  .meta dt {
    color: var(--color-fg-subtle);
  }
  .meta dt::after {
    content: ":";
  }
  .meta dd {
    margin: 0;
    color: var(--color-fg-muted);
  }

  .delete {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font-size: var(--size-xs);
    cursor: pointer;
  }
  .delete:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
  }
  .delete:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
