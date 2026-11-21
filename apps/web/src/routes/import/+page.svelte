<script lang="ts">
  // Import a Claude export (PLAN.md M2). The Mac app hands June a real path
  // from its native picker, so even a large export is read from disk; the
  // browser uploads the chosen file instead. The report's totals add up to
  // the export, so the user can see nothing went missing.
  import { ApiError, type ImportReport } from "@june/ui/api";
  import { getPlatform } from "@june/ui/platform";
  import { session } from "#lib/session.svelte.js";

  const tauri = getPlatform().runtime === "tauri";
  let busy = $state(false);
  let report: ImportReport | undefined = $state();
  let error: string | undefined = $state();
  let typedPath = $state("");

  async function run(task: () => Promise<ImportReport>): Promise<void> {
    busy = true;
    error = undefined;
    report = undefined;
    try {
      report = await task();
    } catch (e) {
      error = e instanceof ApiError ? e.reason : e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function chooseFile(): Promise<void> {
    const path = await getPlatform().pickFile({ title: "Choose your Claude export", extensions: ["zip", "json", "jsonl"] });
    if (path !== null) await run(() => session.client.importPath(path));
  }
  async function chooseFolder(): Promise<void> {
    const path = await getPlatform().pickFolder({ title: "Choose the folder with your Claude export" });
    if (path !== null) await run(() => session.client.importPath(path));
  }
  async function upload(event: Event & { currentTarget: HTMLInputElement }): Promise<void> {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file !== undefined) await run(() => session.client.importFile(file));
  }

  const n = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`;
</script>

<main class="page">
  <h1>Import your history</h1>
  <p class="lede">Bring your Claude and ChatGPT conversations into your vault. Each becomes a Markdown file you own, searchable here and readable in any editor. Importing again later adds what is new and changes nothing else.</p>

  <section class="card" aria-label="Claude">
    <h2>Claude</h2>
    <ol class="steps">
      <li>On claude.ai, open Settings, then Privacy, and choose <strong>Export data</strong>.</li>
      <li>When the email arrives, download the files it links to (the links expire after a day). At least <code>conversations-000.zip</code>; add <code>projects-000.zip</code> to keep your projects as lanes.</li>
      <li>Choose the zip here, or the folder you saved them all in.</li>
    </ol>
    <div class="buttons">
      {#if tauri}
        <button type="button" class="primary" onclick={chooseFile} disabled={busy}>Choose export file</button>
        <button type="button" onclick={chooseFolder} disabled={busy}>Choose folder</button>
      {:else}
        <label class="button primary" class:disabled={busy}>
          Choose export file
          <input type="file" accept=".zip,.json,.jsonl" onchange={upload} disabled={busy} />
        </label>
        <form
          class="path"
          onsubmit={(e) => {
            e.preventDefault();
            if (typedPath.trim() !== "") void run(() => session.client.importPath(typedPath));
          }}
        >
          <input bind:value={typedPath} placeholder="or a folder on this Mac, like ~/Downloads/claude" aria-label="Export folder" disabled={busy} />
          <button type="submit" disabled={busy || typedPath.trim() === ""}>Import folder</button>
        </form>
      {/if}
    </div>
  </section>

  <section class="card" aria-label="ChatGPT">
    <h2>ChatGPT</h2>
    <ol class="steps">
      <li>In ChatGPT, open Settings, then Data controls, and choose <strong>Export data</strong>.</li>
      <li>When the email arrives, download the zip (the link expires after a day).</li>
      <li>Choose that zip here with the same button; June tells Claude and ChatGPT exports apart by their contents.</li>
    </ol>
    <p class="note">Nothing is sent anywhere: the export is read on this Mac. Imported text is marked as imported and never sent to a model unless you continue a conversation.</p>
  </section>

  {#if busy}
    <p class="status" role="status" aria-busy="true">Importing… a large history can take a minute.</p>
  {/if}
  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}
  {#if report}
    <section class="report" aria-label="Import report">
      <h2>{report.created + report.updated === 0 ? "Already up to date" : `Imported ${n(report.created + report.updated, "conversation")}`}</h2>
      <p class="from">From {report.sources.map((s) => (s === "claude-export" ? "Claude" : "ChatGPT")).join(" and ")}</p>
      <dl>
        <dt>In the export</dt>
        <dd>{n(report.conversations, "conversation")}</dd>
        <dt>Now in your vault</dt>
        <dd>{n(report.threads, "conversation")} with {n(report.messages, "message")}: {report.created.toLocaleString()} new, {report.updated.toLocaleString()} updated, {report.unchanged.toLocaleString()} unchanged</dd>
        {#if report.empty > 0}
          <dt>Skipped, empty</dt>
          <dd>{n(report.empty, "conversation")} with no messages</dd>
        {/if}
        {#if report.unreadable > 0}
          <dt>Skipped, unreadable</dt>
          <dd>{n(report.unreadable, "entry", "entries")} without an id or date</dd>
        {/if}
        {#if report.offPath > 0}
          <dt>Edited versions</dt>
          <dd>{n(report.offPath, "earlier version")} of edited messages, kept in each thread's original JSON</dd>
        {/if}
        {#if report.attachments + report.filesNotIncluded > 0}
          <dt>Attachments</dt>
          <dd>{report.attachments.toLocaleString()} with their text; {report.filesNotIncluded.toLocaleString()} named only (Claude's export leaves the files out)</dd>
        {/if}
        {#if report.ignored.length > 0}
          <dt>Not imported yet</dt>
          <dd>{report.ignored.join(", ")}</dd>
        {/if}
      </dl>
      <p class="check">{report.threads + report.empty + report.unreadable === report.conversations ? "Every conversation in the export is accounted for." : "The totals do not add up; please report this."} Receipt #{report.receipt}.</p>
      <a class="button primary" href="/chat">Open conversations</a>
    </section>
  {/if}
</main>

<style>
  .page {
    max-width: 44rem;
    width: 100%;
    margin: 0 auto;
    padding: var(--space-6) 1.25rem;
    overflow-y: auto;
  }
  h1 {
    margin: 0 0 var(--space-2);
    font-size: var(--size-xl);
  }
  .lede {
    margin: 0 0 var(--space-5);
    color: var(--color-fg-muted);
  }
  .card,
  .report {
    padding: var(--space-5);
    background: var(--color-bg-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-4);
  }
  h2 {
    margin: 0 0 var(--space-3);
    font-size: var(--size-md);
  }
  .steps {
    margin: 0 0 var(--space-4);
    padding-left: 1.25rem;
    color: var(--color-fg-secondary);
  }
  .steps li {
    margin-bottom: var(--space-1);
  }
  .buttons {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    align-items: center;
  }
  .button {
    display: inline-block;
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-md);
    background: var(--color-accent);
    color: var(--color-accent-ink);
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
  }
  .button.disabled {
    opacity: 0.5;
    pointer-events: none;
  }
  .button input {
    display: none;
  }
  .path {
    display: flex;
    gap: var(--space-2);
    flex: 1;
    min-width: 16rem;
  }
  .path input {
    flex: 1;
  }
  .note {
    margin: var(--space-4) 0 0;
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }
  .status {
    color: var(--color-fg-muted);
  }
  .error {
    padding: var(--space-2) var(--space-3);
    border-left: 3px solid var(--color-danger);
  }
  dl {
    display: grid;
    grid-template-columns: 10rem 1fr;
    gap: var(--space-2) var(--space-4);
    margin: 0 0 var(--space-4);
  }
  dt {
    color: var(--color-fg-muted);
  }
  dd {
    margin: 0;
  }
  .from {
    margin: calc(-1 * var(--space-2)) 0 var(--space-3);
    color: var(--color-fg-muted);
    font-size: var(--size-sm);
  }
  .check {
    font-size: var(--size-sm);
    color: var(--color-fg-muted);
  }
  @media (max-width: 40rem) {
    dl {
      grid-template-columns: 1fr;
    }
  }
</style>
