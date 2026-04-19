<script lang="ts">
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import {
    createJuneClient,
    type SetupApplyResponse,
    type SetupStatus,
  } from "@june/ui";

  const DEFAULT_API = "http://localhost:8000";
  const apiUrl =
    (import.meta.env.PUBLIC_JUNE_API_URL as string | undefined) ?? DEFAULT_API;
  const client = createJuneClient({ baseUrl: apiUrl });

  type Provider = "gemma" | "gemini";
  type Step = "loading" | "choose" | "verifying" | "error" | "done";

  let status: SetupStatus | null = $state(null);
  let provider: Provider = $state("gemma");
  let geminiKey = $state("");
  let step: Step = $state("loading");
  let result: SetupApplyResponse | null = $state(null);

  onMount(async () => {
    await refreshStatus();
    if (status?.is_configured) {
      step = "done";
    } else {
      provider = (status?.provider as Provider) || "gemma";
      step = "choose";
    }
  });

  async function refreshStatus() {
    try {
      status = await client.getSetupStatus();
    } catch (err) {
      console.warn("June: /setup/status unreachable", err);
    }
  }

  async function handleApply() {
    if (provider === "gemini" && !geminiKey.trim()) return;
    step = "verifying";
    result = null;
    try {
      result = await client.applySetup({
        provider,
        gemini_api_key: provider === "gemini" ? geminiKey.trim() : null,
      });
    } catch (err) {
      console.error(err);
      result = {
        ok: false,
        provider,
        model: "",
        verified: false,
        message: "Couldn't reach the API. Is it running on " + apiUrl + "?",
        hint: "Start the API with ./tools/dev.sh and reload this page.",
      };
    }
    await refreshStatus();
    step = result?.ok ? "done" : "error";
  }

  function continueToChat() {
    goto("/");
  }
</script>

<div class="min-h-screen bg-slate-50 px-6 py-16 text-slate-900">
  <div class="mx-auto max-w-xl">
    <header class="mb-10">
      <p class="text-xs uppercase tracking-[0.2em] text-slate-500">June</p>
      <h1 class="mt-2 text-3xl font-semibold">Set up your assistant</h1>
      <p class="mt-3 text-slate-600">
        Pick a model provider. Your conversations and memories stay on this
        machine either way.
      </p>
    </header>

    {#if step === "loading"}
      <p class="text-slate-500">Checking your environment…</p>
    {:else}
      <section class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <fieldset class="space-y-4">
          <legend class="text-sm font-semibold text-slate-900">Provider</legend>

          <label class="flex cursor-pointer items-start gap-3 rounded-xl border p-4 transition hover:border-slate-400"
                 class:border-slate-900={provider === "gemma"}
                 class:bg-slate-50={provider === "gemma"}>
            <input
              type="radio"
              value="gemma"
              bind:group={provider}
              class="mt-1"
            />
            <span class="flex-1">
              <span class="block font-medium">Gemma 4 (local)</span>
              <span class="mt-1 block text-sm text-slate-600">
                Runs entirely on this machine via Ollama. No network calls for
                inference.
              </span>
              {#if status}
                <span class="mt-2 block text-xs">
                  {#if status.ollama_reachable && status.ollama_has_model}
                    <span class="text-emerald-700">Ollama is ready with {status.model || "gemma4:e4b"}.</span>
                  {:else if status.ollama_reachable}
                    <span class="text-amber-700">
                      Ollama is running but the model isn't pulled. Run
                      <code class="rounded bg-slate-100 px-1 py-0.5">ollama pull gemma4:e4b</code>.
                    </span>
                  {:else}
                    <span class="text-amber-700">
                      Ollama isn't running. Install it with
                      <code class="rounded bg-slate-100 px-1 py-0.5">brew install ollama</code>
                      then run <code class="rounded bg-slate-100 px-1 py-0.5">ollama serve</code>.
                    </span>
                  {/if}
                </span>
              {/if}
            </span>
          </label>

          <label class="flex cursor-pointer items-start gap-3 rounded-xl border p-4 transition hover:border-slate-400"
                 class:border-slate-900={provider === "gemini"}
                 class:bg-slate-50={provider === "gemini"}>
            <input
              type="radio"
              value="gemini"
              bind:group={provider}
              class="mt-1"
            />
            <span class="flex-1">
              <span class="block font-medium">Gemini (cloud)</span>
              <span class="mt-1 block text-sm text-slate-600">
                Calls Google's API. Faster and smarter, but prompts leave your
                machine.
              </span>
            </span>
          </label>
        </fieldset>

        {#if provider === "gemini"}
          <div class="mt-6">
            <label for="gemini-key" class="block text-sm font-medium">
              Gemini API key
            </label>
            <input
              id="gemini-key"
              type="password"
              autocomplete="off"
              spellcheck="false"
              bind:value={geminiKey}
              placeholder="AIza…"
              class="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-sm focus:border-slate-900 focus:outline-none"
            />
            <p class="mt-2 text-xs text-slate-500">
              Get a free key at
              <a class="underline" href="https://aistudio.google.com" target="_blank" rel="noreferrer">aistudio.google.com</a>.
              Stored locally with file mode 0600 — never sent anywhere except Google.
            </p>
          </div>
        {/if}

        <div class="mt-8 flex items-center justify-between gap-4">
          <p class="text-xs text-slate-500">
            June will send one short verification prompt to confirm the provider works.
          </p>
          <button
            type="button"
            onclick={handleApply}
            disabled={step === "verifying" || (provider === "gemini" && !geminiKey.trim())}
            class="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {step === "verifying" ? "Verifying…" : "Verify and continue"}
          </button>
        </div>
      </section>

      {#if step === "error" && result}
        <div class="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <p class="font-medium">{result.message || "Verification failed."}</p>
          {#if result.hint}
            <p class="mt-1 text-red-800">{result.hint}</p>
          {/if}
        </div>
      {/if}

      {#if step === "done"}
        <div class="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
          <p class="font-medium">Ready to chat.</p>
          <p class="mt-1">June is set up with {result?.model || status?.model || "your chosen model"}.</p>
          <button
            type="button"
            onclick={continueToChat}
            class="mt-4 rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-800"
          >
            Continue to chat
          </button>
        </div>
      {/if}
    {/if}
  </div>
</div>
