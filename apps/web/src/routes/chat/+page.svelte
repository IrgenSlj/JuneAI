<script lang="ts">
  import { Composer, MessageList } from "@june/ui";
  import { getPlatform } from "@june/ui/platform";
  import { onMount } from "svelte";
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
</script>

<div class="layout">
  <nav class="threads" aria-label="Conversations">
    <button type="button" class="new primary" onclick={() => chat.startNew()} disabled={chat.streaming}>New conversation</button>
    <ul>
      {#each chat.threads as thread (thread.id)}
        <li>
          <button type="button" class:active={chat.thread?.id === thread.id} onclick={() => chat.open(thread.id)} disabled={chat.streaming}>
            {thread.title}
          </button>
        </li>
      {/each}
    </ul>
  </nav>

  <main class="chat">
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

    <Composer streaming={chat.streaming} disabled={chat.route === undefined || chat.model === ""} {placeholder} onSubmit={(text) => chat.send(text)} onCancel={() => chat.stop()} />

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
      {#if chat.route}
        <span aria-hidden="true">·</span>
        {#if chat.route.leavesDevice}
          <span class="leaves">leaves this Mac for {chat.route.host}</span>
        {:else}
          <span class="stays">stays on this Mac</span>
        {/if}
      {:else}
        <a href="/settings">Connect a model</a>
      {/if}
      {#if chat.savedTurn && !chat.streaming}
        <span aria-hidden="true">·</span>
        <span class="saved">Saved to your vault</span>
      {/if}
    </div>
  </main>
</div>

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
