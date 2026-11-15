import type { VaultLock } from "../adapters.js";

/** A VaultLock for one process: tasks with the same name run one at a time, in call order. */
export function memoryLock(): VaultLock {
  const queues = new Map<string, Promise<unknown>>();
  return {
    exclusive(name, task) {
      const result = (queues.get(name) ?? Promise.resolve()).then(task);
      const settled = result.catch(() => undefined);
      queues.set(name, settled);
      void settled.then(() => {
        if (queues.get(name) === settled) queues.delete(name);
      });
      return result;
    },
  };
}
