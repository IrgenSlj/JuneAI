const STORAGE_KEY = "june:user_id";
const DEFAULT_USER = "local";

function readInitial(): string {
  if (typeof localStorage === "undefined") return DEFAULT_USER;
  try {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_USER;
  } catch {
    return DEFAULT_USER;
  }
}

export const profileName = $state<{ value: string }>({ value: readInitial() });

export function setProfileName(id: string): void {
  profileName.value = id || DEFAULT_USER;
  try {
    localStorage.setItem(STORAGE_KEY, profileName.value);
  } catch {}
}
