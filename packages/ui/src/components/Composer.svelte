<script lang="ts">
  /**
   * Message composer. Enter and Cmd/Ctrl+Enter send, Shift+Enter inserts a
   * newline. While a turn is streaming, the submit button turns into a Stop
   * button that fires `onCancel`.
   *
   * The caller can bind `focus` to pull focus from a parent-level shortcut
   * (e.g. Cmd+K).
   */
  interface Props {
    streaming?: boolean;
    disabled?: boolean;
    placeholder?: string;
    onSubmit: (message: string) => void;
    onCancel?: () => void;
    focus?: () => void;
  }

  let {
    streaming = false,
    disabled = false,
    placeholder = "Message June...",
    onSubmit,
    onCancel,
    focus = $bindable(),
  }: Props = $props();

  let value = $state("");
  let textarea: HTMLTextAreaElement | undefined = $state();

  focus = () => {
    textarea?.focus();
  };

  function autoResize() {
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || streaming || disabled) return;
    onSubmit(trimmed);
    value = "";
    queueMicrotask(autoResize);
  }

  function handleKey(event: KeyboardEvent) {
    const isEnter = event.key === "Enter";
    const isShift = event.shiftKey;
    const isMod = event.metaKey || event.ctrlKey;
    if (isEnter && !isShift) {
      event.preventDefault();
      submit();
    } else if (isEnter && isMod) {
      event.preventDefault();
      submit();
    }
  }
</script>

<form
  class="composer"
  onsubmit={(event) => {
    event.preventDefault();
    submit();
  }}
>
  <textarea
    bind:this={textarea}
    bind:value
    oninput={autoResize}
    onkeydown={handleKey}
    {placeholder}
    {disabled}
    rows="1"
    aria-label="Message June"
  ></textarea>

  {#if streaming}
    <button type="button" class="stop" onclick={onCancel} aria-label="Stop">
      Stop
    </button>
  {:else}
    <button
      type="submit"
      class="send"
      disabled={disabled || value.trim().length === 0}
      aria-label="Send message"
    >
      Send
    </button>
  {/if}
</form>

<style>
  .composer {
    display: flex;
    align-items: flex-end;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
  }

  textarea {
    flex: 1;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--color-fg-primary);
    font-family: var(--font-sans);
    font-size: var(--size-md);
    line-height: var(--leading-normal);
    max-height: 240px;
    padding: var(--space-1) 0;
  }

  textarea::placeholder {
    color: var(--color-fg-subtle);
  }

  button {
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-md);
    border: none;
    font-family: var(--font-sans);
    font-size: var(--size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: background 120ms ease, opacity 120ms ease;
  }

  .send {
    background: var(--color-accent);
    color: var(--color-bg-base);
  }
  .send:hover:not(:disabled) {
    background: var(--color-accent-muted);
  }
  .send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .stop {
    background: var(--color-bg-sunken);
    color: var(--color-fg-primary);
    border: 1px solid var(--color-border-strong);
  }
  .stop:hover {
    background: var(--color-border);
  }
</style>
