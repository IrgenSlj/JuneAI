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
  bgBase: "#13110E",
  bgRaised: "#1A1814",
  bgSunken: "#22201B",

  // Foregrounds
  fgPrimary: "#F1EEE7",
  fgSecondary: "#D8D3C9",
  fgMuted: "#8E8778",
  fgSubtle: "#5F5A50",

  // Accents — June's identity color is a warm clay-orange that reads as
  // attentive rather than clinical. The assistant-bubble uses it.
  accent: "#F2AC6E",
  accentMuted: "#E0945A",
  accentInk: "#2A1B0E",
  accentSoft: "rgba(242, 172, 110, 0.16)",

  // Semantic
  success: "#8AA884",
  danger: "#C88080",
  warn: "#C8A260",
  border: "rgba(255, 250, 240, 0.08)",
  borderStrong: "rgba(255, 250, 240, 0.16)",

  // UI chrome
  termBg: "#0E0C09",
  userBubble: "rgba(242, 172, 110, 0.16)",
  userBubbleLine: "rgba(255, 255, 255, 0.04)",
} as const;

export const typography = {
  fontSans: '"Helvetica Neue", Helvetica, Arial, sans-serif',
  fontMono:
    'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
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
  sm: "6px",
  md: "10px",
  lg: "14px",
  xl: "22px",
  bubble: "16px",
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

export const motion = {
  fast: "120ms",
  base: "220ms",
  slow: "420ms",
  enter: "320ms",
  breath: "5200ms",
  spin: "14s",
  pulse: "1100ms",
  ease: "cubic-bezier(0.4,0,0.2,1)",
  easeOut: "cubic-bezier(0,0,0.2,1)",
  easeIn: "cubic-bezier(0.4,0,1,1)",
} as const;

export const tokens = {
  color,
  typography,
  space,
  radius,
  shadow,
  breakpoint,
  motion,
} as const;

export type Tokens = typeof tokens;
