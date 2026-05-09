const MIN = 60_000;
const HOUR = 3_600_000;
const DAY = 86_400_000;

export function formatRelative(input: string | Date): string {
  const t = typeof input === "string" ? Date.parse(input) : input.getTime();
  if (Number.isNaN(t)) return typeof input === "string" ? input : String(input);
  const diff = Date.now() - t;
  const abs = Math.abs(diff);
  if (abs < MIN) return "just now";
  if (abs < HOUR) {
    const m = Math.round(diff / MIN);
    return m > 0 ? `${m}m ago` : `in ${-m}m`;
  }
  if (abs < DAY) {
    const h = Math.round(diff / HOUR);
    return h > 0 ? `${h}h ago` : `in ${-h}h`;
  }
  if (abs < DAY * 30) {
    const d = Math.round(diff / DAY);
    return d > 0 ? `${d}d ago` : `in ${-d}d`;
  }
  return new Date(t).toLocaleDateString();
}
