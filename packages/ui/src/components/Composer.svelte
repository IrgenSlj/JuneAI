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
      <svg
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        aria-hidden="true"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M8 13V3.5M4.5 7L8 3.5 11.5 7"
          stroke="var(--color-accent-ink)"
          stroke-width="1.7"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </button>
  {/if}
</form>

<style>
  .composer {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 50px;
    padding: 0 8px 0 16px;
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-lg);
    box-shadow: 0 1px 0 var(--color-border), var(--shadow-md);
    transition: border-color var(--motion-base, 220ms) var(--ease, ease),
      box-shadow var(--motion-base, 220ms) var(--ease, ease);
  }
  .composer:focus-within {
    border-color: var(--color-accent);
    box-shadow: 0 1px 0 var(--color-border),
      0 0 0 1px color-mix(in srgb, var(--color-accent) 55%, transparent);
  }

  textarea {
    flex: 1;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--color-fg-primary);
    font-family: var(--font-sans);
    font-size: 15.5px;
    line-height: var(--leading-normal);
    max-height: 240px;
    padding: var(--space-3) 0;
  }

  textarea::placeholder {
    color: var(--color-fg-subtle);
  }

  button {
    flex-shrink: 0;
    border: none;
    cursor: pointer;
    transition: background 120ms ease, opacity 120ms ease, border-color 120ms ease;
  }

  .send {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: var(--radius-md);
    background: var(--color-accent);
  }
  .send:hover:not(:disabled) {
    background: var(--color-accent-muted);
  }
  .send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .stop {
    background: transparent;
    color: var(--color-fg-secondary);
    border: 1px solid var(--color-border-strong);
    border-radius: 9px;
    padding: 7px 13px;
    font-family: var(--font-sans);
    font-size: 12.5px;
  }
  .stop:hover {
    background: var(--color-bg-sunken);
    color: var(--color-fg-primary);
  }
</style>
