<script lang="ts">
  let {
    open = $bindable(false),
    title,
    message,
    confirmLabel = "Confirm",
    cancelLabel = "Cancel",
    danger = false,
    onConfirm,
    onCancel,
  }: {
    open: boolean;
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    onConfirm: () => void | Promise<void>;
    onCancel?: () => void;
  } = $props();

  let pending = $state(false);
  let dialogEl: HTMLDialogElement | undefined = $state();

  $effect(() => {
    if (!dialogEl) return;
    if (open) {
      if (!dialogEl.open) dialogEl.showModal();
    } else {
      if (dialogEl.open) dialogEl.close();
    }
  });

  function handleCancel() {
    if (pending) return;
    open = false;
    onCancel?.();
  }

  async function handleConfirm() {
    if (pending) return;
    pending = true;
    try {
      await onConfirm();
    } finally {
      pending = false;
      open = false;
    }
  }

  function handleBackdropClick(e: MouseEvent) {
    if (e.target === dialogEl) {
      handleCancel();
    }
  }

  function handleDialogCancel(e: Event) {
    e.preventDefault();
    handleCancel();
  }
</script>

<dialog
  bind:this={dialogEl}
  class="dialog"
  oncancel={handleDialogCancel}
  onclick={handleBackdropClick}
  aria-labelledby="cd-title"
  aria-describedby="cd-message"
>
  <div class="panel" role="presentation" onclick={(e) => e.stopPropagation()}>
    <h2 class="title" id="cd-title">{title}</h2>
    <p class="message" id="cd-message">{message}</p>
    <div class="actions">
      <button
        type="button"
        class="btn cancel"
        onclick={handleCancel}
        disabled={pending}
      >
        {cancelLabel}
      </button>
      <button
        type="button"
        class="btn confirm"
        class:danger
        onclick={handleConfirm}
        disabled={pending}
      >
        {pending ? "…" : confirmLabel}
      </button>
    </div>
  </div>
</dialog>

<style>
  .dialog {
    border: none;
    padding: 0;
    background: transparent;
    max-width: 420px;
    width: calc(100vw - var(--space-6));
  }

  .dialog::backdrop {
    background: color-mix(in srgb, var(--color-bg) 60%, transparent);
    backdrop-filter: blur(2px);
  }

  .panel {
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }

  .title {
    margin: 0;
    font-size: var(--size-md);
    font-weight: 600;
    color: var(--color-fg-primary);
  }

  .message {
    margin: 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
    line-height: var(--leading-normal);
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-2);
  }

  .btn {
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-4);
    font: inherit;
    font-size: var(--size-sm);
    cursor: pointer;
    color: var(--color-fg-primary);
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .btn.cancel:hover:not(:disabled) {
    border-color: var(--color-border);
    color: var(--color-fg-muted);
  }

  .btn.confirm {
    background: var(--color-bg-raised);
    font-weight: 500;
  }
  .btn.confirm:hover:not(:disabled) {
    border-color: var(--color-accent);
    color: var(--color-accent);
  }

  .btn.confirm.danger {
    color: var(--color-danger);
    border-color: color-mix(in srgb, var(--color-danger) 40%, transparent);
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }
  .btn.confirm.danger:hover:not(:disabled) {
    border-color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 18%, transparent);
  }
</style>
