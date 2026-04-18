<script lang="ts">
  /**
   * One chat bubble — user, assistant, or a tool marker.
   *
   * Tool bubbles collapse the raw JSON args/result into a single line
   * so the transcript stays readable; clicking expands the detail.
   */
  interface Props {
    role: "user" | "assistant" | "tool";
    content: string;
    toolName?: string;
  }

  const { role, content, toolName = "" }: Props = $props();
</script>

<article class="bubble" data-role={role}>
  {#if role === "tool"}
    <header class="tool-header">
      <span class="tool-dot" aria-hidden="true"></span>
      <span class="tool-label">{toolName || "tool"}</span>
    </header>
  {/if}
  <div class="body" class:tool={role === "tool"}>{content}</div>
</article>

<style>
  .bubble {
    max-width: 72ch;
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-lg);
    font-size: var(--size-md);
    line-height: var(--leading-relaxed);
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: var(--shadow-sm);
  }

  .bubble[data-role="user"] {
    align-self: flex-end;
    background: var(--color-bg-raised);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border);
  }

  .bubble[data-role="assistant"] {
    align-self: flex-start;
    background: transparent;
    color: var(--color-fg-primary);
    padding-left: 0;
    padding-right: 0;
    box-shadow: none;
  }

  .bubble[data-role="tool"] {
    align-self: flex-start;
    background: var(--color-bg-sunken);
    color: var(--color-fg-muted);
    border: 1px dashed var(--color-border);
    font-family: var(--font-mono);
    font-size: var(--size-xs);
  }

  .tool-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-1);
  }

  .tool-dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--color-accent);
  }

  .tool-label {
    font-family: var(--font-mono);
    font-size: var(--size-xs);
    color: var(--color-accent);
    letter-spacing: 0.02em;
  }

  .body.tool {
    color: var(--color-fg-muted);
  }
</style>
