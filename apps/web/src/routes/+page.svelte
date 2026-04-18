<script lang="ts">
  import { onMount } from "svelte";
  import {
    ChatBubble,
    Composer,
    MessageList,
    createJuneClient,
    type ChatMessage,
    type SystemStatus,
  } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";
  const USER_ID = "local";

  // PUBLIC_JUNE_API_URL is picked up at build time by SvelteKit. Any
  // env var prefixed with PUBLIC_ is bundled into the client, which is
  // exactly what we want for a PWA that may run against a different
  // API host (localhost in dev, a Fly/Render-hosted instance in prod).
  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;

  const client = createJuneClient({ baseUrl: apiUrl });

  let messages: ChatMessage[] = $state([]);
  let streaming = $state(false);
  let system: SystemStatus | null = $state(null);
  let abortController: AbortController | null = $state(null);

  onMount(async () => {
    try {
      system = await client.getSystem();
    } catch (err) {
      console.warn("June: /system unreachable", err);
    }
  });

  function pushMessage(msg: ChatMessage) {
    messages = [...messages, msg];
  }

  function appendToken(id: string, token: string) {
    messages = messages.map((m) =>
      m.id === id ? { ...m, content: m.content + token } : m,
    );
  }

  async function handleSubmit(text: string) {
    if (streaming) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
    };
    const assistantId = `a-${Date.now()}`;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
    };

    messages = [...messages, userMsg, assistantMsg];
    streaming = true;
    abortController = new AbortController();

    try {
      for await (const event of client.streamChat({
        user_id: USER_ID,
        message: text,
        skill: "assistant",
        signal: abortController.signal,
      })) {
        switch (event.type) {
          case "token":
            appendToken(assistantId, event.content);
            break;
          case "tool_call":
            pushMessage({
              id: `t-${Date.now()}-${Math.random()}`,
              role: "tool",
              content: formatToolCall(event.tool_name, event.tool_args),
              toolName: event.tool_name,
            });
            break;
          case "tool_result":
            // Tool results attach onto the most recent tool bubble with
            // the same name, preserving transcript order.
            messages = messages.map((m) =>
              m.role === "tool" && m.toolName === event.tool_name && !m.content.includes("→")
                ? { ...m, content: `${m.content}\n  → ${event.tool_result}` }
                : m,
            );
            break;
          case "error":
            appendToken(assistantId, `\n\n[error: ${event.content}]`);
            break;
          case "done":
            break;
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        appendToken(assistantId, " [stopped]");
      } else {
        appendToken(assistantId, `\n\n[stream failed: ${String(err)}]`);
      }
    } finally {
      streaming = false;
      abortController = null;
    }
  }

  function handleCancel() {
    abortController?.abort();
  }

  function formatToolCall(
    name: string,
    args: Record<string, unknown> | undefined,
  ): string {
    const argText = args && Object.keys(args).length
      ? JSON.stringify(args)
      : "(no args)";
    return `${name} ${argText}`;
  }
</script>

<svelte:head>
  <title>June — your personal AI</title>
</svelte:head>

<main class="app">
  <header class="top">
    <h1>June</h1>
    {#if system}
      <span
        class="runtime"
        title="{system.base_url || 'no endpoint'} · privacy: {system.privacy_label}"
      >
        <span class="dot" data-mode={system.mode}></span>
        {system.label} · {system.model}
        {#if system.provider === "gemma"}
          · {system.ollama_reachable ? (system.ollama_has_model ? "ready" : "model missing") : "Ollama offline"}
        {:else if !system.api_key_present}
          · key missing
        {/if}
      </span>
    {:else}
      <span class="runtime offline">
        <span class="dot" data-mode="api"></span>
        API unreachable at {apiUrl}
      </span>
    {/if}
  </header>

  <section class="transcript">
    {#if messages.length === 0}
      <div class="empty">
        <p>Hi, I'm June. I'll remember what matters so you don't have to.</p>
        <p class="muted">Type below to start.</p>
      </div>
    {:else}
      <MessageList {messages} />
    {/if}
  </section>

  <footer class="compose">
    <Composer
      {streaming}
      onSubmit={handleSubmit}
      onCancel={handleCancel}
    />
  </footer>
</main>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    max-width: 860px;
    margin: 0 auto;
    padding: 0 var(--space-4);
  }

  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-5) 0 var(--space-3);
    border-bottom: 1px solid var(--color-border);
  }

  h1 {
    margin: 0;
    font-size: var(--size-xl);
    font-weight: 600;
    letter-spacing: -0.01em;
  }

  .runtime {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-fg-muted);
  }
  .runtime.offline {
    color: var(--color-danger);
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    background: var(--color-success);
  }
  .dot[data-mode="api"] {
    background: var(--color-accent);
  }

  .transcript {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .empty {
    margin: auto;
    max-width: 40ch;
    text-align: center;
    color: var(--color-fg-muted);
  }
  .empty p {
    margin: var(--space-2) 0;
  }
  .empty .muted {
    color: var(--color-fg-subtle);
    font-size: var(--size-sm);
  }

  .compose {
    padding: var(--space-3) 0 var(--space-5);
  }
</style>
