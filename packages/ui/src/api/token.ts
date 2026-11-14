// The per-launch API token reaches the page in the URL fragment
// (http://127.0.0.1:<port>/#token=...). Browsers never send the fragment to a
// server, so it stays out of logs and referrers. On first load the page moves
// it into session storage (gone when the window closes) and strips it from the
// address bar so it is not bookmarked or shared by accident.

const KEY = "june.token";

export interface TokenEnvironment {
  hash: string;
  storage: Pick<Storage, "getItem" | "setItem">;
  /** Replace the current URL without the fragment. */
  clearHash(): void;
}

export function takeApiToken(env: TokenEnvironment): string | undefined {
  const fromHash = new URLSearchParams(env.hash.replace(/^#/, "")).get("token");
  if (fromHash) {
    env.storage.setItem(KEY, fromHash);
    env.clearHash();
    return fromHash;
  }
  return env.storage.getItem(KEY) ?? undefined;
}

/** The browser's own environment, for app code. */
export function browserTokenEnvironment(): TokenEnvironment {
  return {
    hash: window.location.hash,
    storage: window.sessionStorage,
    clearHash: () => history.replaceState(null, "", window.location.pathname + window.location.search),
  };
}
