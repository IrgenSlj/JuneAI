<script lang="ts">
  import { onMount } from "svelte";
  import {
    OfflineNotice,
    type MemoryFact,
    type MemorySnapshot,
    type MemoryStats,
  } from "@june/ui";
  import { client } from "$lib/api.js";
  import { formatRelative } from "$lib/dates.js";
  import { profileName } from "$lib/stores/user.svelte.js";

  let snapshot: MemorySnapshot | null = $state(null);
  let stats: MemoryStats | null = $state(null);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let actionError: string | null = $state(null);
  let query = $state("");
  let pendingDelete: string | null = $state(null);
  let editingRef: string | null = $state(null);
  let editFields: Record<string, string> = $state({});
  let pendingSave = $state(false);

  type SectionKind =
    | "semantic"
    | "entities"
    | "goals"
    | "open_loops"
    | "calendar"
    | "journal"
    | "body_metrics";

  type Section = {
    id: SectionKind;
    title: string;
    caption: string;
    empty: string;
    facts: MemoryFact[];
    deletable: boolean;
    editable: boolean;
  };

  const sections = $derived.by<Section[]>(() => {
    if (!snapshot) return [];
    return [
      {
        id: "semantic",
        title: "Facts",
        caption: "What June has learned and kept.",
        empty:
          "No durable facts yet. As June learns that you prefer tea or live in Berlin, those land here.",
        facts: snapshot.semantic_facts ?? [],
        deletable: true,
        editable: false,
      },
      {
        id: "entities",
        title: "People, places, and things",
        caption: "Entities and how they connect.",
        empty:
          "No entities mapped yet. People and projects you mention become nodes here.",
        facts: snapshot.entities ?? [],
        deletable: true,
        editable: false,
      },
      {
        id: "goals",
        title: "Goals",
        caption: "What you're working toward.",
        empty: "No goals tracked. Ask June to remember a goal and it shows up here.",
        facts: snapshot.goals ?? [],
        deletable: true,
        editable: true,
      },
      {
        id: "open_loops",
        title: "Open loops",
        caption: "Threads left open, waiting to close.",
        empty:
          "No open threads. Questions left dangling or follow-ups June is holding will appear here.",
        facts: snapshot.open_loops ?? [],
        deletable: true,
        editable: true,
      },
      {
        id: "calendar",
        title: "Calendar",
        caption: "Events and reminders you've asked June to hold.",
        empty: "Nothing on the calendar. Ask June to save events or reminders.",
        facts: snapshot.calendar ?? [],
        deletable: true,
        editable: true,
      },
      {
        id: "journal",
        title: "Journal",
        caption: "Reflections and daily entries, written when you speak.",
        empty:
          "No journal entries yet. The daily skill writes here when you reflect; nothing to read until then.",
        facts: snapshot.journal ?? [],
        deletable: true,
        editable: false,
      },
      {
        id: "body_metrics",
        title: "Body metrics",
        caption: "Health data logged by the health skill, one row per day.",
        empty:
          "No body metrics logged. The health skill records weight, sleep, and energy; one row per day.",
        facts: snapshot.body_metrics ?? [],
        deletable: true,
        editable: false,
      },
    ];
  });

  type EditField = { key: string; label: string };

  const EDIT_FIELDS: Partial<Record<SectionKind, EditField[]>> = {
    goals: [
      { key: "title", label: "Title" },
      { key: "category", label: "Category" },
      { key: "next_step", label: "Next step" },
      { key: "target_date", label: "Target date" },
      { key: "status", label: "Status" },
    ],
    open_loops: [
      { key: "topic", label: "Topic" },
      { key: "next_step", label: "Next step" },
      { key: "due_date", label: "Due date" },
      { key: "status", label: "Status" },
    ],
    calendar: [
      { key: "title", label: "Title" },
      { key: "date", label: "Date" },
      { key: "time", label: "Time" },
      { key: "details", label: "Details" },
      { key: "status", label: "Status" },
    ],
  };

  function fieldsFromFact(section: SectionKind, fact: MemoryFact): Record<string, string> {
    const meta = fact.metadata ?? {};
    if (section === "goals") {
      return {
        title: fact.title ?? "",
        category: String(meta.category ?? ""),
        next_step: fact.body ?? "",
        target_date: String(meta.target_date ?? ""),
        status: String(meta.status ?? ""),
      };
    }
    if (section === "open_loops") {
      return {
        topic: fact.title ?? "",
        next_step: fact.body ?? "",
        due_date: String(meta.due_date ?? ""),
        status: String(meta.status ?? ""),
      };
    }
    if (section === "calendar") {
      return {
        title: fact.title ?? "",
        date: String(meta.date ?? ""),
        time: String(meta.time ?? ""),
        details: fact.body ?? "",
        status: String(meta.status ?? ""),
      };
    }
    return {};
  }

  function startEdit(section: SectionKind, fact: MemoryFact) {
    if (!fact.ref) return;
    editingRef = fact.ref;
    editFields = fieldsFromFact(section, fact);
    actionError = null;
  }

  function cancelEdit() {
    editingRef = null;
    editFields = {};
  }

  async function saveEdit() {
    if (!editingRef || pendingSave) return;
    pendingSave = true;
    actionError = null;
    try {
      await client.patchMemoryFact(profileName.value, editingRef, editFields);
      editingRef = null;
      editFields = {};
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingSave = false;
    }
  }

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
      const [snap, stat] = await Promise.all([
        client.getMemory(profileName.value),
        client.getMemoryStats(profileName.value).catch(() => null),
      ]);
      snapshot = snap;
      stats = stat;
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
      await client.deleteMemoryFact(profileName.value, fact.ref);
      await refresh();
    } catch (err) {
      actionError = err instanceof Error ? err.message : String(err);
    } finally {
      pendingDelete = null;
    }
  }

  let copiedRef: string | null = $state(null);

  async function handleCopy(fact: MemoryFact) {
    if (!fact.ref) return;
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    // Prefer body; fall back to title when body is empty or identical.
    const text = fact.body && fact.body !== fact.title ? fact.body : (fact.title ?? "");
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copiedRef = fact.ref;
      setTimeout(() => {
        if (copiedRef === fact.ref) copiedRef = null;
      }, 1500);
    } catch {
      // Clipboard write fails silently in insecure contexts; stay quiet.
    }
  }

  function metadataEntries(fact: MemoryFact): [string, string][] {
    const meta = fact.metadata ?? {};
    return Object.entries(meta)
      .filter(([k]) => !TIMESTAMP_KEYS.includes(k) && k !== "source")
      .filter(([, v]) => v !== null && v !== undefined && String(v).trim() !== "")
      .map(([k, v]) => [k, String(v)]);
  }

  function factSource(fact: MemoryFact): string | null {
    const raw = (fact.metadata ?? {})["source"];
    if (typeof raw === "string" && raw.trim().length > 0) return raw.trim();
    return null;
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

  {#if stats}
    <section class="stats-card" aria-label="Memory at a glance">
      <div class="stats-headline">
        <span class="stats-total">{stats.total}</span>
        <span class="stats-total-label">
          {stats.total === 1 ? "thing" : "things"} June remembers
        </span>
        {#if stats.last_write}
          <span class="stats-last">
            last write {formatRelative(stats.last_write)}
          </span>
        {/if}
      </div>
      <ul class="stats-buckets">
        {#each stats.buckets as bucket (bucket.kind)}
          <li class="stats-bucket" data-store={bucket.store}>
            <span class="stats-bucket-count">{bucket.count}</span>
            <span class="stats-bucket-kind">{bucket.kind.replace(/_/g, " ")}</span>
            <span class="stats-bucket-store">{bucket.store}</span>
          </li>
        {/each}
      </ul>
      {#if stats.recent_facts && stats.recent_facts.length > 0}
        <details class="stats-recent">
          <summary>Recently learned</summary>
          <ul>
            {#each stats.recent_facts as fact (fact.ref)}
              <li>{fact.body || fact.title || fact.ref}</li>
            {/each}
          </ul>
        </details>
      {/if}
    </section>
  {/if}

  {#if !snapshot && loading}
    <div class="skeleton-list" aria-label="Loading memories&hellip;">
      {#each [1, 2, 3, 4] as _ (_)}
        <div class="skeleton-group">
          <div class="skeleton skeleton-heading"></div>
          <div class="skeleton skeleton-row"></div>
          <div class="skeleton skeleton-row"></div>
          <div class="skeleton skeleton-row skeleton-short"></div>
        </div>
      {/each}
    </div>
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
          <p class="section-caption">{section.caption}</p>
          {#if section.facts.length === 0}
            <p class="muted">{section.empty}</p>
          {:else}
            <ul class="facts">
              {#each section.facts as fact (fact.ref || `${section.id}-${fact.title}`)}
                {@const ts = pickTimestamp(fact)}
                {@const isEditing = editingRef !== null && editingRef === fact.ref}
                <li class="fact">
                  {#if isEditing && section.editable}
                    <form
                      class="fact-editor"
                      onsubmit={(e) => {
                        e.preventDefault();
                        saveEdit();
                      }}
                    >
                      {#each EDIT_FIELDS[section.id] ?? [] as field (field.key)}
                        <label class="field">
                          <span class="field-label">{field.label}</span>
                          <input
                            type="text"
                            value={editFields[field.key] ?? ""}
                            oninput={(e) =>
                              (editFields[field.key] = e.currentTarget.value)}
                          />
                        </label>
                      {/each}
                      <div class="editor-actions">
                        <button type="submit" class="primary" disabled={pendingSave}>
                          {pendingSave ? "Saving…" : "Save"}
                        </button>
                        <button type="button" onclick={cancelEdit} disabled={pendingSave}>
                          Cancel
                        </button>
                      </div>
                    </form>
                  {:else}
                    <div class="fact-main">
                      <div class="fact-head">
                        {#if fact.title}
                          <div class="fact-title">{fact.title}</div>
                        {/if}
                        <div class="fact-meta-right">
                          {#if ts}
                            <time class="fact-time" datetime={ts}>{formatRelative(ts)}</time>
                          {/if}
                          {#if factSource(fact)}
                            <span class="fact-source">via {factSource(fact)}</span>
                          {/if}
                        </div>
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
                    <div class="fact-actions">
                      {#if fact.ref && (fact.body || fact.title)}
                        <button
                          type="button"
                          class="copy"
                          onclick={() => handleCopy(fact)}
                          disabled={editingRef !== null}
                          aria-label={copiedRef === fact.ref ? "Copied" : "Copy text"}
                          title={copiedRef === fact.ref ? "Copied" : "Copy"}
                        >
                          {copiedRef === fact.ref ? "Copied" : "Copy"}
                        </button>
                      {/if}
                      {#if section.editable && fact.ref}
                        <button
                          type="button"
                          class="edit"
                          onclick={() => startEdit(section.id, fact)}
                          disabled={editingRef !== null}
                          aria-label="Edit this"
                        >
                          Edit
                        </button>
                      {/if}
                      {#if section.deletable && fact.ref}
                        <button
                          type="button"
                          class="delete"
                          onclick={() => handleDelete(fact)}
                          disabled={pendingDelete === fact.ref || editingRef !== null}
                          aria-label="Forget this"
                        >
                          {pendingDelete === fact.ref ? "…" : "Forget"}
                        </button>
                      {/if}
                    </div>
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
    gap: var(--space-6);
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

  @keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
  }
  .skeleton-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-6);
  }
  .skeleton-group {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .skeleton {
    background: linear-gradient(90deg, var(--color-bg-raised) 25%, var(--color-border) 50%, var(--color-bg-raised) 75%);
    background-size: 800px 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: var(--radius-sm);
    height: 16px;
  }
  .skeleton-heading {
    width: 120px;
    height: 12px;
  }
  .skeleton-row {
    height: 48px;
    border-radius: var(--radius-md);
  }
  .skeleton-short {
    width: 60%;
  }

  .group {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding-top: var(--space-2);
  }

  .group-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .group h2 {
    margin: 0;
    font-size: var(--size-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--color-fg-subtle);
  }
  .count {
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    opacity: 0.7;
  }

  .section-caption {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    line-height: var(--leading-normal);
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
    gap: var(--space-3);
  }

  .fact {
    display: flex;
    gap: var(--space-3);
    align-items: flex-start;
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4) var(--space-4);
  }

  .fact-main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .fact-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    justify-content: space-between;
  }
  .fact-title {
    font-size: var(--size-sm);
    font-weight: 500;
    color: var(--color-fg-primary);
    line-height: var(--leading-normal);
    min-width: 0;
    flex: 1;
  }
  .fact-meta-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    flex-shrink: 0;
  }
  .fact-time {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
  }
  .fact-source {
    font-size: var(--size-xs);
    font-family: var(--font-mono);
    color: var(--color-fg-subtle);
    opacity: 0.7;
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

  .fact-actions {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    flex-shrink: 0;
  }

  .edit,
  .delete,
  .copy {
    background: transparent;
    color: var(--color-fg-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font-size: var(--size-xs);
    cursor: pointer;
  }
  .edit:hover:not(:disabled),
  .copy:hover:not(:disabled) {
    color: var(--color-accent);
    border-color: var(--color-accent);
  }
  .delete:hover:not(:disabled) {
    color: var(--color-danger);
    border-color: var(--color-danger);
  }
  .edit:disabled,
  .delete:disabled,
  .copy:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .fact-editor {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  .field-label {
    font-size: var(--size-xs);
    font-weight: 500;
    color: var(--color-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .fact-editor input {
    background: var(--color-bg);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
  }

  .fact-editor input:focus {
    outline: none;
    border-color: var(--color-accent);
  }

  .editor-actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-1);
  }

  .editor-actions button {
    background: var(--color-bg);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
  }

  .editor-actions button.primary {
    background: var(--color-accent);
    color: var(--color-bg);
    border-color: var(--color-accent);
  }

  .editor-actions button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .stats-card {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .stats-headline {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    flex-wrap: wrap;
  }

  .stats-total {
    font-size: var(--size-xl);
    font-weight: 600;
    color: var(--color-fg-primary);
    font-family: var(--font-mono);
  }

  .stats-total-label {
    color: var(--color-fg-muted);
  }

  .stats-last {
    margin-left: auto;
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }

  .stats-buckets {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: var(--space-2);
  }

  .stats-bucket {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg);
  }

  .stats-bucket-count {
    font-family: var(--font-mono);
    font-size: var(--size-lg);
    color: var(--color-fg-primary);
    font-weight: 600;
  }

  .stats-bucket-kind {
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }

  .stats-bucket-store {
    color: var(--color-fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .stats-recent {
    border-top: 1px dashed var(--color-border);
    padding-top: var(--space-2);
  }
  .stats-recent summary {
    cursor: pointer;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
    padding: var(--space-1) 0;
    list-style: none;
  }
  .stats-recent summary::-webkit-details-marker {
    display: none;
  }
  .stats-recent summary::before {
    content: "▸ ";
    color: var(--color-fg-subtle);
  }
  .stats-recent[open] summary::before {
    content: "▾ ";
  }
  .stats-recent ul {
    list-style: none;
    padding: 0;
    margin: var(--space-2) 0 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .stats-recent li {
    padding: var(--space-1) var(--space-2);
    background: var(--color-bg);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
  }
</style>
