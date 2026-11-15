<script lang="ts">
  import { Composer, MessageList } from "@june/ui";
  import { onMount } from "svelte";
  import { chat, routeGroup } from "#lib/chat.svelte.js";

  onMount(() => void chat.load());

  const groups = $derived(
    ["On this Mac", "Your keys", "Your subscriptions", "Free"]
      .map((name) => ({ name, routes: chat.ready.filter((p) => routeGroup(p.route) === name) }))
      .filter((g) => g.routes.length > 0),
  );
  const placeholder = $derived(chat.route === undefined ? "Connect a model to start" : "Message");
</script>

<div class="layout">
  <nav class="threads" aria-label="Conversations">
    <button type="button" class="new" onclick={() => chat.startNew()} disabled={chat.streaming}>New conversation</button>
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
    <section class="messages" aria-label="Conversation">
      {#if chat.messages.length === 0}
        <p class="empty">
          {#if chat.routes.length > 0 && chat.ready.length === 0}
            No model is connected. Start Ollama or LM Studio on this Mac, or add an API key in Settings.
          {:else}
            No conversation yet.
          {/if}
        </p>
      {:else}
        <MessageList messages={chat.messages} streaming={chat.streaming} />
      {/if}
    </section>

    {#if chat.error}
      <p class="error" role="alert">{chat.error}</p>
    {/if}

    <div class="controls">
      <select value={chat.routeId} onchange={(e) => chat.pickRoute(e.currentTarget.value)} disabled={chat.streaming || groups.length === 0} aria-label="Route">
          {#each groups as group (group.name)}
            <optgroup label={group.name}>
              {#each group.routes as p (p.route.id)}
                <option value={p.route.id}>{p.route.name}</option>
              {/each}
            </optgroup>
          {/each}
      </select>
      {#if chat.models.length > 0}
        <select bind:value={chat.model} disabled={chat.streaming} aria-label="Model">
          {#each chat.models as model (model)}
            <option value={model}>{model}</option>
          {/each}
        </select>
      {:else if chat.route}
        <input bind:value={chat.model} placeholder="Model name" aria-label="Model" disabled={chat.streaming} />
      {/if}
    </div>

    <Composer streaming={chat.streaming} disabled={chat.route === undefined || chat.model === ""} {placeholder} onSubmit={(text) => chat.send(text)} onCancel={() => chat.stop()} />

    {#if chat.route}
      <p class="glance" aria-label="Where this message goes">
        <span>{chat.model || "no model"}</span>
        <span aria-hidden="true">·</span>
        <span>{chat.route.name}</span>
        <span aria-hidden="true">·</span>
        {#if chat.route.leavesDevice}
          <span class="leaves">leaves this Mac for {chat.route.host}</span>
        {:else}
          <span class="stays">stays on this Mac</span>
        {/if}
        {#if chat.savedTurn && !chat.streaming}
          <span aria-hidden="true">·</span>
          <span class="saved">Saved to your vault</span>
        {/if}
      </p>
    {/if}
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
    color: inherit;
    padding: 0.4rem 0.5rem;
    border-radius: 6px;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .threads li button.active {
    background: var(--color-surface-2, rgba(127, 127, 127, 0.12));
  }
  .new {
    width: 100%;
  }
  .chat {
    display: grid;
    grid-template-rows: 1fr auto auto auto auto;
    gap: 0.5rem;
    min-height: 0;
    max-width: 48rem;
    width: 100%;
    margin: 0 auto;
    padding: 1rem 1.25rem;
  }
  .messages {
    overflow-y: auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .empty {
    opacity: 0.6;
    margin-top: 30vh;
    text-align: center;
  }
  .error {
    margin: 0;
    padding: 0.5rem 0.75rem;
    border-left: 3px solid var(--color-danger, #b3261e);
    font-size: 0.9rem;
  }
  .controls {
    display: flex;
    gap: 0.5rem;
  }
  .glance {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    font-size: 0.8rem;
    opacity: 0.7;
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
