<script lang="ts">
  import { Composer, MessageList, groupByDay, snippetParts, sourceName } from "@june/ui";
  import { getPlatform } from "@june/ui/platform";
  import { onMount } from "svelte";
  import LabelSheet from "#lib/LabelSheet.svelte";
  import { chat, routeGroup } from "#lib/chat.svelte.js";

  onMount(() => void chat.load());

  const groups = $derived(
    ["On this Mac", "Your keys", "Your subscriptions", "Free"]
      .map((name) => ({ name, routes: chat.ready.filter((p) => routeGroup(p.route) === name) }))
      .filter((g) => g.routes.length > 0),
  );
  const local = $derived(chat.routes.filter((p) => p.route.kind === "local"));
  const DOWNLOADS: Record<string, string> = { ollama: "https://ollama.com/download", "lm-studio": "https://lmstudio.ai" };
  function download(id: string): void {
    const url = DOWNLOADS[id];
    if (url !== undefined) void getPlatform().openExternal(url);
  }
  const placeholder = $derived(chat.route === undefined ? "Connect a model to start" : "Message");

  const listed = $derived(groupByDay(chat.threads, new Date()));
  let draft = $state("");
  function cancelLabel(): void {
    draft = chat.cancelPending() ?? draft;
  }
  let searchTimer: ReturnType<typeof setTimeout> | undefined;
  function onSearch(value: string): void {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => void chat.search(value), 120);
  }
  let notice: string | undefined = $state();
  let noticeTimer: ReturnType<typeof setTimeout> | undefined;
  function tell(text: string): void {
    notice = text;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => (notice = undefined), 4000);
  }
  async function exportAs(format: "md" | "json"): Promise<void> {
    const where = await chat.exportThread(format);
    if (where !== undefined) tell(`Exported to ${where.split(/[\\/]/).at(-1)}`);
  }
  async function copy(): Promise<void> {
    if (await chat.copyThread()) tell("Copied as Markdown");
  }
</script>

<div class="layout">
  <nav class="threads" aria-label="Conversations">
    <button type="button" class="new primary" onclick={() => chat.startNew()} disabled={chat.streaming}>New conversation</button>
    <input class="search" type="search" placeholder="Search conversations" aria-label="Search conversations" value={chat.query} oninput={(e) => onSearch(e.currentTarget.value)} />
    {#if chat.results !== undefined}
      <p class="group">{chat.results.length === 0 ? "No matches" : chat.results.length === 1 ? "1 conversation" : `${chat.results.length} conversations`}</p>
      <ul aria-label="Search results">
        {#each chat.results as hit (hit.id)}
          <li>
            <button type="button" class="hit" class:active={chat.thread?.id === hit.id} onclick={() => chat.open(hit.id)} disabled={chat.streaming}>
              <span class="title">{hit.title}{#if sourceName(hit.source)}<span class="badge">{sourceName(hit.source)}</span>{/if}</span>
              {#if hit.match && hit.match.turn !== "title"}
                <span class="snippet">{#each snippetParts(hit.match.snippet) as part, i (i)}{#if part.match}<mark>{part.text}</mark>{:else}{part.text}{/if}{/each}</span>
              {/if}
            </button>
          </li>
        {/each}
      </ul>
    {:else}
      {#each listed as group (group.name)}
        <p class="group">{group.name}</p>
        <ul>
          {#each group.items as thread (thread.id)}
            <li>
              <button type="button" class:active={chat.thread?.id === thread.id} onclick={() => chat.open(thread.id)} disabled={chat.streaming}>
                <span class="title">{thread.title}{#if sourceName(thread.source)}<span class="badge">{sourceName(thread.source)}</span>{/if}</span>
              </button>
            </li>
          {/each}
        </ul>
      {/each}
      {#if chat.moreThreads}
        <button type="button" class="quiet more" onclick={() => chat.loadMore()}>Show older</button>
      {/if}
      {#if chat.loaded && chat.threads.length === 0}
        <p class="hint">Your conversations appear here. <a href="/import">Import your Claude history</a> to bring the old ones.</p>
      {/if}
    {/if}
  </nav>

  <main class="chat">
    {#if chat.thread}
      <div class="thread-head">
        <h1>{chat.thread.title}</h1>
        {#if sourceName(chat.thread.source)}<span class="badge">Imported from {sourceName(chat.thread.source)}</span>{/if}
        <span class="actions">
          {#if notice}<span class="notice" role="status">{notice}</span>{/if}
          <button type="button" class="quiet" onclick={copy} disabled={chat.streaming}>Copy</button>
          <button type="button" class="quiet" onclick={() => exportAs("md")} disabled={chat.streaming} title="Save this conversation as a Markdown file">Export .md</button>
          <button type="button" class="quiet" onclick={() => exportAs("json")} disabled={chat.streaming} title="Save this conversation as JSON">.json</button>
        </span>
      </div>
    {/if}
    {#if chat.thread?.parent}
      {@const parentId = chat.thread.parent.split("#")[0] ?? ""}
      <p class="branch-note">
        Branched from
        <button type="button" class="link" onclick={() => chat.open(parentId)} disabled={chat.streaming}>
          {chat.threads.find((t) => t.id === parentId)?.title ?? "another conversation"}
        </button>
      </p>
    {/if}
    <section class="messages" aria-label="Conversation">
      {#if chat.messages.length === 0 && chat.loaded && chat.ready.length === 0}
        <div class="doors">
          <h1>Connect a model to start</h1>
          <p class="lede">June talks to the models you bring. Pick one way in; you can add the other later.</p>
          <section class="door" aria-label="Run a model on this Mac">
            <h2>Run a model on this Mac</h2>
            <p>Free, private, nothing leaves this Mac. Install one of these, start it, then check again.</p>
            <ul>
              {#each local as p (p.route.id)}
                <li>
                  <span>{p.route.name.replace(" (this Mac)", "")}</span>
                  <span class="state">{p.running ? "running" : "not running"}</span>
                  <button type="button" class="quiet" onclick={() => download(p.route.id)}>Download</button>
                </li>
              {/each}
            </ul>
            <button type="button" onclick={() => chat.recheck()}>Check again</button>
          </section>
          <section class="door" aria-label="Use a key">
            <h2>Use a key</h2>
            <p>OpenAI, Anthropic, Mistral, OpenRouter or OpenCode. The key goes to the macOS keychain; replies still save to your vault.</p>
            <a class="button primary" href="/settings">Add a key</a>
          </section>
        </div>
      {:else if chat.messages.length === 0 && chat.loaded}
        <p class="empty">Start a conversation. It is saved as a Markdown file in your vault.</p>
      {:else}
        <MessageList messages={chat.messages} streaming={chat.streaming} onBranch={(id) => chat.branchAt(id)} />
      {/if}
    </section>

    {#if chat.error}
      <p class="error" role="alert">{chat.error}</p>
    {/if}

    {#if chat.readOnly}
      <div class="continue">
        <p>Imported conversations stay exactly as the export had them. Continue this one on any model as a new thread; this copy stays untouched.</p>
        <button type="button" class="primary" onclick={() => chat.continueThread()} disabled={chat.route === undefined}>Continue in a new thread</button>
      </div>
    {:else}
      <Composer bind:value={draft} streaming={chat.streaming} disabled={chat.route === undefined || chat.model === "" || chat.pending !== undefined} {placeholder} onSubmit={(text) => chat.submit(text)} onCancel={() => chat.stop()} />
    {/if}

    <div class="glance" aria-label="Where this message goes">
      <select class="pick" value={chat.routeId} onchange={(e) => chat.pickRoute(e.currentTarget.value)} disabled={chat.streaming || groups.length === 0} aria-label="Route">
        {#if groups.length === 0}<option value={undefined}>No model connected</option>{/if}
        {#each groups as group (group.name)}
          <optgroup label={group.name}>
            {#each group.routes as p (p.route.id)}
              <option value={p.route.id}>{p.route.name}</option>
            {/each}
          </optgroup>
        {/each}
      </select>
      {#if chat.models.length > 0}
        <select class="pick" bind:value={chat.model} disabled={chat.streaming} aria-label="Model">
          {#each chat.models as model (model)}
            <option value={model}>{model}</option>
          {/each}
        </select>
      {:else if chat.route}
        <input class="pick" bind:value={chat.model} placeholder="model name" aria-label="Model" disabled={chat.streaming} />
      {/if}
      {#if chat.route && chat.transparency !== "quiet"}
        <span aria-hidden="true">·</span>
        {#if chat.route.leavesDevice}
          <span class="leaves">leaves this Mac for {chat.route.host}</span>
        {:else}
          <span class="stays">stays on this Mac</span>
        {/if}
      {:else if !chat.route}
        <a href="/settings">Connect a model</a>
      {/if}
      {#if chat.savedTurn && !chat.streaming}
        <span aria-hidden="true">·</span>
        <span class="saved">Saved to your vault</span>
      {/if}
    </div>
  </main>
</div>

{#if chat.pending}
  <LabelSheet label={chat.pending.label} onSend={() => chat.approvePending()} onCancel={cancelLabel} />
{/if}

<style>
  .layout {
    display: grid;
    grid-template-columns: 15rem 1fr;
    min-height: 0;
  }
  .threads {
    border-right: 1px solid var(--color-border, rgba(127, 127, 127, 0.2));
    padding: 0.75rem;
    overflow-y: auto;
  }
  .threads ul {
    list-style: none;
    margin: 0.75rem 0 0;
    padding: 0;
  }
  .threads li button {
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    color: var(--color-fg-secondary);
    padding: 0.45rem 0.6rem;
    border-radius: var(--radius-md);
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .threads li button:hover:not(:disabled) {
    background: var(--color-bg-sunken);
  }
  .threads li button.active {
    background: var(--color-bg-sunken);
    color: var(--color-fg-primary);
  }
  .new {
    width: 100%;
  }
  .search {
    width: 100%;
    margin-top: var(--space-2);
  }
  .group {
    margin: var(--space-3) 0 0 0.6rem;
    font-size: var(--size-xs);
    color: var(--color-fg-subtle);
  }
  .threads .group + ul {
    margin-top: var(--space-1);
  }
  .threads .title {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .badge {
    flex: none;
    padding: 0 var(--space-1);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
  }
  .threads .hit {
    white-space: normal;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .hit .title {
    white-space: nowrap;
  }
  .snippet {
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  mark {
    background: var(--color-accent-soft, rgba(255, 200, 0, 0.3));
    color: inherit;
    border-radius: 2px;
  }
  .more {
    margin: var(--space-2) 0 0 0.6rem;
  }
  .hint {
    margin: var(--space-4) 0.6rem;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }
  .thread-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    padding-bottom: var(--space-2);
    border-bottom: 1px solid var(--color-border);
  }
  .thread-head h1 {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .actions {
    margin-left: auto;
    display: flex;
    align-items: baseline;
    gap: var(--space-1);
    flex: none;
  }
  .actions button {
    font-size: var(--size-xs);
    padding: 2px var(--space-2);
  }
  .notice {
    font-size: var(--size-xs);
    color: var(--color-success);
  }
  .continue {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
  }
  .continue p {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }
  .continue button {
    flex: none;
  }
  .chat {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-height: 0;
    max-width: 48rem;
    width: 100%;
    margin: 0 auto;
    padding: 1rem 1.25rem;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .empty {
    margin: auto 0;
    text-align: center;
    color: var(--color-fg-muted);
  }
  .doors {
    margin: auto 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }
  .doors h1 {
    grid-column: 1 / -1;
    margin: 0;
    font-size: var(--size-xl);
  }
  .lede {
    grid-column: 1 / -1;
    margin: 0 0 var(--space-2);
    color: var(--color-fg-muted);
  }
  .door {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-5);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
  }
  .door h2 {
    margin: 0;
    font-size: var(--size-md);
  }
  .door p {
    margin: 0;
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .door ul {
    list-style: none;
    margin: 0;
    padding: 0;
    width: 100%;
  }
  .door li {
    display: flex;
    gap: var(--space-3);
    align-items: baseline;
    padding: var(--space-1) 0;
  }
  .door .state {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
  }
  .door li .quiet {
    margin-left: auto;
  }
  .door .button {
    margin-top: auto;
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-md);
    background: var(--color-accent);
    color: var(--color-accent-ink);
    font-weight: 600;
    text-decoration: none;
  }
  @media (max-width: 52rem) {
    .doors {
      grid-template-columns: 1fr;
    }
  }
  .error {
    margin: 0;
    padding: 0.5rem 0.75rem;
    border-left: 3px solid var(--color-danger, #b3261e);
    font-size: 0.9rem;
  }
  .branch-note {
    margin: 0;
    font-size: 0.85rem;
    opacity: 0.7;
  }
  .link {
    background: none;
    border: none;
    padding: 0;
    color: var(--color-accent, inherit);
    cursor: pointer;
    font: inherit;
  }
  .glance {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
  }
  .pick {
    padding: 2px 4px;
    font-size: var(--size-xs);
    color: var(--color-fg-secondary);
    background: transparent;
    border-color: transparent;
    border-radius: var(--radius-sm);
    max-width: 16rem;
  }
  .pick:hover:not(:disabled) {
    border-color: var(--color-border-strong);
  }
  input.pick {
    width: 9rem;
  }
  .stays {
    color: var(--color-success);
  }
  .saved {
    color: var(--color-fg-secondary);
  }
  @media (max-width: 40rem) {
    .layout {
      grid-template-columns: 1fr;
    }
    .threads {
      display: none;
    }
  }
</style>
