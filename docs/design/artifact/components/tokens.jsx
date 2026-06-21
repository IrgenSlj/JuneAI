// Design tokens for June
// Two variants: "tight" (quiet precision) and "warm" (editorial reading-app)
// Each with light + dark modes.

const TOKENS = {
  tight: {
    light: {
      // Near-white with the faintest warm tint — not sterile
      bg:        '#FAF9F7',
      surface:   '#FFFFFF',
      surface2:  '#F3F1ED',
      line:      'rgba(20, 16, 10, 0.08)',
      lineStrong:'rgba(20, 16, 10, 0.14)',
      ink:       '#141410',
      ink2:      '#2A2824',
      muted:     '#6B665D',
      muted2:    '#9A948A',
      accent:    '#E8965A',   // pastel orange
      accentInk: '#4A2912',   // deep cocoa — legible on orange
      accentSoft:'rgba(232, 150, 90, 0.12)',
      ok:        '#4E6B4A',
      warn:      '#8A6A2F',
      err:       '#8A3B3B',
    },
    dark: {
      bg:        '#13110E',
      surface:   '#1A1814',
      surface2:  '#22201B',
      line:      'rgba(255, 250, 240, 0.08)',
      lineStrong:'rgba(255, 250, 240, 0.16)',
      ink:       '#F1EEE7',
      ink2:      '#D8D3C9',
      muted:     '#8E8778',
      muted2:    '#5F5A50',
      accent:    '#F2AC6E',   // lifted pastel orange for dark
      accentInk: '#2A1B0E',
      accentSoft:'rgba(242, 172, 110, 0.16)',
      ok:        '#8AA884',
      warn:      '#C8A260',
      err:       '#C88080',
    },
  },
  warm: {
    light: {
      bg:        '#F4EFE6',    // warm oatmeal
      surface:   '#FBF7EE',
      surface2:  '#EDE6D8',
      line:      'rgba(60, 40, 20, 0.10)',
      lineStrong:'rgba(60, 40, 20, 0.18)',
      ink:       '#22190F',
      ink2:      '#3A2E20',
      muted:     '#75675A',
      muted2:    '#A59787',
      accent:    '#E8965A',   // pastel orange
      accentInk: '#4A2912',
      accentSoft:'rgba(232, 150, 90, 0.12)',
      ok:        '#4E6B4A',
      warn:      '#8A6A2F',
      err:       '#8A3B3B',
    },
    dark: {
      bg:        '#1A140D',
      surface:   '#221A12',
      surface2:  '#2B2218',
      line:      'rgba(245, 232, 210, 0.08)',
      lineStrong:'rgba(245, 232, 210, 0.18)',
      ink:       '#F2E9D8',
      ink2:      '#D6C9B2',
      muted:     '#9A8C78',
      muted2:    '#6B5E4E',
      accent:    '#F2AC6E',
      accentInk: '#2A1B0E',
      accentSoft:'rgba(242, 172, 110, 0.16)',
      ok:        '#92AE88',
      warn:      '#C8A260',
      err:       '#C88080',
    },
  },
};

const RADII = { sm: 6, md: 10, lg: 14, xl: 22, bubble: 16 };
const SPACE = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 40, xxxl: 64 };

// Motion scale — exported to tokens.ts as `motion`.
// Calm by default; the only continuous motion lives in the mascot busy state.
const MOTION = {
  fast:   '120ms',   // hover, toggle thumb
  base:   '220ms',   // terminal collapse/expand, tab change
  slow:   '420ms',   // mascot settle-to-rest
  enter:  '320ms',   // activity line fade/slide in
  breath: '5200ms',  // mascot idle breathing
  spin:   '14s',     // mascot busy ray rotation
  pulse:  '1100ms',  // streaming last-line pulse / busy corona
  ease:   'cubic-bezier(0.4, 0, 0.2, 1)',
  easeOut:'cubic-bezier(0, 0, 0.2, 1)',
  easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
};

// Type scale — Helvetica throughout.
// Warm variant adds a serif only for the wordmark, nothing else.
const TYPE = {
  tight: {
    family: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    mono:   'ui-monospace, "SF Mono", Menlo, monospace',
    wordmark: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    wordmarkWeight: 500,
    wordmarkLetter: '-0.04em',
  },
  warm: {
    family: '"Helvetica Neue", Helvetica, Arial, sans-serif',
    mono:   'ui-monospace, "SF Mono", Menlo, monospace',
    // The only typographic difference in Warm: a serif wordmark, set lowercase
    wordmark: '"Cormorant Garamond", "Times New Roman", serif',
    wordmarkWeight: 400,
    wordmarkLetter: '-0.02em',
  },
};

// Helper: pull the active palette
function palette(variant, mode) {
  return TOKENS[variant][mode];
}

// Derived bubble + terminal colors so we don't fork the palettes.
function chrome(variant, mode) {
  const p = palette(variant, mode);
  const dark = mode === 'dark';
  return {
    juneBubble:  dark ? p.surface  : p.surface,
    juneBubbleLine: p.line,
    userBubble:  p.accentSoft,
    userBubbleLine: dark ? 'rgba(255,255,255,0.04)' : 'rgba(20,16,10,0.04)',
    termBg:      dark ? '#0E0C09' : '#F0ECE3',
    termLine:    p.line,
    termInk:     p.muted,
    termInkStrong: p.ink2,
    termDim:     p.muted2,
  };
}

Object.assign(window, { TOKENS, RADII, SPACE, TYPE, MOTION, palette, chrome });
