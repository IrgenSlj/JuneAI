/**
 * June design tokens.
 *
 * Every shell (web, desktop, mobile) consumes these values — never
 * hard-coded hex or pixel values elsewhere. When a token changes, the
 * change propagates everywhere, which is the whole point of having
 * them centralized.
 *
 * The CSS custom-property names in `tokens.css` mirror these keys so
 * Svelte components can reference the same values in stylesheets
 * without importing TS.
 */

export const color = {
  // Surfaces
  bgBase: "#0b0d10",
  bgRaised: "#14181d",
  bgSunken: "#08090b",

  // Foregrounds
  fgPrimary: "#f5f6f7",
  fgMuted: "#a3acb6",
  fgSubtle: "#6a727d",

  // Accents — June's identity color is a warm amber that reads as
  // attentive rather than clinical. The assistant-bubble uses it.
  accent: "#f5a524",
  accentMuted: "#b87917",

  // Semantic
  success: "#3ecf8e",
  danger: "#ef4146",
  border: "#20262e",
  borderStrong: "#2e3641",
} as const;

export const typography = {
  fontSans:
    '"Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  fontMono:
    '"JetBrains Mono", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
  sizeXs: "0.75rem",
  sizeSm: "0.875rem",
  sizeMd: "1rem",
  sizeLg: "1.125rem",
  sizeXl: "1.375rem",
  leadingTight: "1.25",
  leadingNormal: "1.5",
  leadingRelaxed: "1.7",
  weightRegular: "400",
  weightMedium: "500",
  weightSemibold: "600",
} as const;

export const space = {
  x0: "0",
  x1: "0.25rem",
  x2: "0.5rem",
  x3: "0.75rem",
  x4: "1rem",
  x5: "1.5rem",
  x6: "2rem",
  x7: "3rem",
  x8: "4rem",
} as const;

export const radius = {
  sm: "4px",
  md: "8px",
  lg: "14px",
  pill: "999px",
} as const;

export const shadow = {
  sm: "0 1px 2px rgba(0, 0, 0, 0.3)",
  md: "0 4px 16px rgba(0, 0, 0, 0.35)",
  lg: "0 12px 48px rgba(0, 0, 0, 0.45)",
} as const;

export const breakpoint = {
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px",
} as const;

export const tokens = {
  color,
  typography,
  space,
  radius,
  shadow,
  breakpoint,
} as const;

export type Tokens = typeof tokens;
