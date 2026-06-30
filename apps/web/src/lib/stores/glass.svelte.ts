/**
 * Glass Box store — persisted turn traces for the /system/glass view.
 *
 * Reactive state for the turn-tree timeline. Each GlassTurn starts with
 * events: null and is expanded lazily via loadTurn(). The syncLiveTurn helper
 * mirrors the current in-flight chat turn into the glass store in real time.
 */
import { type TraceEventView, type ActivityStep } from "@june/ui";
import { client } from "$lib/api.js";

export type GlassTurn = {
  turn_id: string;
  started_at: number;
  event_count: number;
  events: TraceEventView[] | null;
  loading: boolean;
  error: string | null;
  live?: boolean;
};

export const glass = $state({
  turns: [] as GlassTurn[],
  loading: false,
  error: null as string | null,
});

/**
 * Mirror the current in-flight turn into the glass store.
 * Called after every chat event; re-syncing the full array makes the
 * in-place reasoning accumulation just work.
 *
 * CLOUD RULE: for provenance steps, the summary includes "cloud" when
 * step.cloud === true, else "local", so TraceEventList renders the
 * boundary tag correctly.
 */
export function syncLiveTurn(
  turn_id: string,
  steps: ActivityStep[],
  startedAtSec: number,
): void {
  const mapped: TraceEventView[] = steps.map((step, i) => ({
    seq: i,
    ts: Math.floor(step.ts / 1000),
    kind: step.kind,
    summary:
      step.kind === "provenance"
        ? `${step.cloud ? "cloud" : "local"} · ${step.label}`
        : step.label,
    detail: step.detail ?? "",
  }));

  const turn: GlassTurn = {
    turn_id,
    started_at: startedAtSec,
    event_count: steps.length,
    events: mapped,
    loading: false,
    error: null,
    live: true,
  };

  const idx = glass.turns.findIndex((t) => t.turn_id === turn_id);
  if (idx >= 0) {
    // Replace in place, preserving position.
    glass.turns = glass.turns.map((t) => (t.turn_id === turn_id ? turn : t));
  } else {
    // Prepend as new live turn.
    glass.turns = [turn, ...glass.turns];
  }
}

/** Mark the live turn as finished. */
export function endLiveTurn(turn_id: string): void {
  glass.turns = glass.turns.map((t) =>
    t.turn_id === turn_id ? { ...t, live: false } : t,
  );
}

export async function loadTurns(): Promise<void> {
  glass.loading = true;
  glass.error = null;
  try {
    const res = await client.getTraces(50);
    const fetched = res.traces ?? [];
    const fetchedIds = new Set(fetched.map((t) => t.turn_id));

    // Build the new list from fetched summaries, merging already-loaded turns.
    const newList: GlassTurn[] = fetched.map((t) => {
      const existing = glass.turns.find((e) => e.turn_id === t.turn_id);
      if (existing && existing.events !== null) {
        // Keep the already-loaded/live turn; just refresh metadata and clear live.
        return {
          ...existing,
          started_at: t.started_at,
          event_count: t.event_count,
          live: false,
        };
      }
      return {
        turn_id: t.turn_id,
        started_at: t.started_at,
        event_count: t.event_count,
        events: null,
        loading: false,
        error: null,
      };
    });

    // Prepend any in-memory turns not in the fetched set (e.g. still-streaming
    // live turns that are not yet persisted).
    const liveOnly = glass.turns.filter((t) => !fetchedIds.has(t.turn_id));
    glass.turns = [...liveOnly, ...newList];
  } catch (err) {
    glass.error = err instanceof Error ? err.message : String(err);
  } finally {
    glass.loading = false;
  }
}

export async function loadTurn(turn_id: string): Promise<void> {
  const existing = glass.turns.find((t) => t.turn_id === turn_id);
  // No-op if already loaded or not found.
  if (!existing || existing.events !== null) return;
  glass.turns = glass.turns.map((t) =>
    t.turn_id === turn_id ? { ...t, loading: true, error: null } : t,
  );
  try {
    const view = await client.getTrace(turn_id);
    glass.turns = glass.turns.map((t) =>
      t.turn_id === turn_id
        ? { ...t, events: view.events ?? [], loading: false }
        : t,
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    glass.turns = glass.turns.map((t) =>
      t.turn_id === turn_id ? { ...t, loading: false, error: msg } : t,
    );
  }
}

/**
 * Upsert a turn by turn_id. If the turn exists, replace it; otherwise prepend
 * it so the newest-first order is maintained. Intended for GB-2 live streaming.
 */
export function upsertTurn(turn: GlassTurn): void {
  const idx = glass.turns.findIndex((t) => t.turn_id === turn.turn_id);
  if (idx >= 0) {
    glass.turns = glass.turns.map((t) => (t.turn_id === turn.turn_id ? turn : t));
  } else {
    glass.turns = [turn, ...glass.turns];
  }
}
