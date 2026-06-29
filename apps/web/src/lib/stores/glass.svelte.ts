/**
 * Glass Box store — persisted turn traces for the /system/glass view.
 *
 * Reactive state for the turn-tree timeline. Each GlassTurn starts with
 * events: null and is expanded lazily via loadTurn(). The upsertTurn helper
 * is a seam for GB-2 (live streaming of the current turn).
 */
import { type TraceEventView } from "@june/ui";
import { client } from "$lib/api.js";

export type GlassTurn = {
  turn_id: string;
  started_at: number;
  event_count: number;
  events: TraceEventView[] | null;
  loading: boolean;
  error: string | null;
};

export const glass = $state({
  turns: [] as GlassTurn[],
  loading: false,
  error: null as string | null,
});

export async function loadTurns(): Promise<void> {
  glass.loading = true;
  glass.error = null;
  try {
    const res = await client.getTraces(50);
    // API returns newest-first; preserve that order.
    glass.turns = (res.traces ?? []).map((t) => ({
      turn_id: t.turn_id,
      started_at: t.started_at,
      event_count: t.event_count,
      events: null,
      loading: false,
      error: null,
    }));
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
