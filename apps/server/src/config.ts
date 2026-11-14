// Where the daemon remembers which vault is open. Outside the vault, because
// the vault is the user's and may be moved; this is just June's bookmark.
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export interface JuneConfig {
  vault?: string;
}

/** Same folder the Mac app uses (Tauri's config dir for "ai.june.app"), so browser mode and the app share one vault bookmark. */
export function defaultConfigDir(home = homedir()): string {
  return process.platform === "darwin" ? join(home, "Library", "Application Support", "ai.june.app") : join(home, ".config", "june");
}

export async function readConfig(dir: string): Promise<JuneConfig> {
  try {
    const parsed: unknown = JSON.parse(await readFile(join(dir, "config.json"), "utf8"));
    return typeof parsed === "object" && parsed !== null ? (parsed as JuneConfig) : {};
  } catch {
    return {};
  }
}

/** Write atomically: a crash mid-write leaves the old file, never half of one. */
export async function writeConfig(dir: string, config: JuneConfig): Promise<void> {
  await mkdir(dir, { recursive: true });
  const tmp = join(dir, `config.json.${process.pid}.tmp`);
  await writeFile(tmp, `${JSON.stringify(config, null, 2)}\n`);
  await rename(tmp, join(dir, "config.json"));
}
