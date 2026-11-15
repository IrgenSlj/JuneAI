<script lang="ts">
  /**
   * Model output and imported text as Markdown, rendered without HTML: the
   * text is parsed into a tree and drawn with plain elements, so nothing in it
   * can run or load anything. Headings are shifted down two levels so a reply
   * never outranks the page's own headings.
   */
  import Blocks from "./Blocks.svelte";
  import { parseMarkdown } from "./parse.js";

  const { text }: { text: string } = $props();
  const blocks = $derived(parseMarkdown(text));
</script>

<div class="markdown"><Blocks {blocks} /></div>

<style>
  .markdown :global(:first-child) {
    margin-top: 0;
  }
  .markdown :global(:last-child) {
    margin-bottom: 0;
  }
  .markdown :global(p),
  .markdown :global(ul),
  .markdown :global(ol),
  .markdown :global(blockquote),
  .markdown :global(pre),
  .markdown :global(.table) {
    margin: 0 0 0.75em;
  }
  .markdown :global(ul),
  .markdown :global(ol) {
    padding-left: 1.4em;
  }
  .markdown :global(li > p) {
    margin: 0 0 0.25em;
  }
  .markdown :global(h3),
  .markdown :global(h4),
  .markdown :global(h5),
  .markdown :global(h6) {
    font-size: 1em;
    margin: 1em 0 0.5em;
  }
  .markdown :global(code) {
    font-family: var(--font-mono, monospace);
    font-size: 0.9em;
  }
  .markdown :global(pre) {
    overflow-x: auto;
    padding: 0.75em;
    border-radius: 6px;
    background: var(--color-bg-sunken, rgba(127, 127, 127, 0.1));
    white-space: pre;
  }
  .markdown :global(blockquote) {
    padding-left: 0.75em;
    border-left: 3px solid var(--color-border, rgba(127, 127, 127, 0.3));
    opacity: 0.85;
  }
  .markdown :global(.table) {
    overflow-x: auto;
  }
  .markdown :global(table) {
    border-collapse: collapse;
  }
  .markdown :global(th),
  .markdown :global(td) {
    padding: 0.3em 0.6em;
    border: 1px solid var(--color-border, rgba(127, 127, 127, 0.3));
    text-align: left;
  }
  .markdown :global(hr) {
    border: none;
    border-top: 1px solid var(--color-border, rgba(127, 127, 127, 0.3));
  }
</style>
