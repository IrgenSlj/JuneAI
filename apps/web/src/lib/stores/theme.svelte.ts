/**
 * Theme preference (dark / light) with localStorage persistence.
 *
 * The initial `data-theme` attribute is written by an inline script in
 * app.html to avoid a flash of the wrong theme on load. This module
 * handles runtime toggling after hydration.
 */

export type Theme = "dark" | "light";
const STORAGE_KEY = "june:theme";

function readInitial(): Theme {
  if (typeof document === "undefined") return "light";
  const attr = document.documentElement.getAttribute("data-theme");
  return attr === "light" ? "light" : "dark";
}

export const theme = $state<{ value: Theme }>({ value: readInitial() });

export function toggleTheme(): void {
  setTheme(theme.value === "dark" ? "light" : "dark");
}

export function setTheme(next: Theme): void {
  theme.value = next;
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // localStorage can be disabled (private mode); the theme just
      // won't persist across reloads.
    }
  }
}
