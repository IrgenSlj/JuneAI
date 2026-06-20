# Responsive and Touch Plan

This document covers how June's UI looks and behaves across every screen size and input method. It applies to all three shells equally because all three serve the same SvelteKit build. It is referenced from [desktop-shell-plan.md](desktop-shell-plan.md) Phase 5.

## The Principle

One UI, three input methods, four screen sizes. The UI is the same code; the differences are CSS media queries and a few capability checks. We do not maintain a "mobile site" and a "desktop site"; we maintain one site that knows which environment it is in.

## Current State (2026-04-27)

- Existing breakpoints: 520px (settings), 640px (layout, forms, help, ollama).
- Max content widths: 640px (forms), 760px (memory and skills), 860px (chat), 980px (outer layout).
- One `prefers-reduced-motion` query in `app.css` disables animations.
- Hover styles are cosmetic only (color and border changes); no affordances are revealed by hover. Delete buttons are always visible.
- No tablet-specific breakpoint between 640px and 980px.
- No `(pointer: coarse)` queries; touch targets rely on default padding.
- No iOS PWA safe-area handling.

The current state is acceptable for desktop and small-phone web; it falls short on tablets, on touch laptops, and on iPad in PWA mode. This plan closes those gaps.

## The Five Form Factors

We design for these explicit form factors. Anything in between scales by interpolation.

| Form Factor | Width | Input | Notes |
|---|---|---|---|
| Phone (small) | 320–480px | touch | iPhone SE through iPhone 13 mini; rare but real. |
| Phone (large) | 481–640px | touch | iPhone 14/15/16, all Android phones. The PWA on iOS lives here. |
| Tablet (portrait) | 641–1024px | touch | iPad portrait, large Android tablets. Most underserved today. |
| Tablet (landscape) and small laptop | 1025–1280px | touch or pointer | iPad landscape, 13" laptops, Surface in tablet mode. Mixed input. |
| Desktop | 1281px+ | pointer (mostly) | All real laptops and desktops. Touch on Surface and touch laptops. |

## The Breakpoints

We standardize on five breakpoints expressed in `em` so they scale with the user's font size, not hard-coded pixels.

```css
--bp-phone-large:    30em;  /* 480px  — small phone → large phone */
--bp-tablet-portrait: 40em;  /* 640px  — large phone → tablet portrait */
--bp-tablet-landscape: 64em;  /* 1024px — tablet portrait → tablet landscape */
--bp-desktop:        80em;  /* 1280px — tablet landscape → desktop */
```

Existing media queries are migrated to use these tokens. New media queries always use them. No raw pixel values for breakpoints in component CSS.

## Touch Target Size

Apple's Human Interface Guidelines call for 44×44pt. Material's guideline is 48×48dp. We adopt **48×48px minimum on touch input** and accept the desktop default (which is smaller and tighter) on pointer input.

The mechanism is `(pointer: coarse)`:

```css
@media (pointer: coarse) {
  button, .icon-btn, .toggle, [role="button"], a.button {
    min-height: 3rem;     /* 48px */
    min-width: 3rem;
  }
}
```

This is added to a single global CSS file (`apps/web/src/routes/app.css`) so every component picks it up. Components that need different behavior override locally.

## Hover and Focus Behavior

Touch devices fire a "synthesized" hover on first tap, which causes the "first tap reveals, second tap clicks" pattern. We avoid this by following two rules:

1. **No hover-only affordances.** Every interactive element is visible without hover. Hover is decoration only.
2. **Use `@media (hover: hover) and (pointer: fine)` for hover styles** that should not apply on touch. Existing cosmetic hover styles get this guard.

Focus rings stay visible everywhere. They are the keyboard equivalent of hover and they matter for accessibility.

## Tablet Layout

The 640–1024px range currently scales the desktop layout to a narrower width. That is acceptable but not great. The fix is intentional: the chat surface keeps its 860px max width and centers; the memory and skills pages get a two-column treatment in landscape (sidebar + content) and stay single-column in portrait.

The system header drops the runtime text below 640px (already implemented at `apps/web/src/routes/+layout.svelte:246`). Above 640px the runtime text is visible. Below 480px the privacy label collapses to its dot only.

## iOS PWA Specifics

The PWA installed on iOS (Add to Home Screen) has known quirks the web version does not:

- **Safe-area insets.** The notch and home-indicator areas need padding. Add `env(safe-area-inset-*)` to the layout's outer padding.
- **Dynamic viewport.** Safari's address bar grows and shrinks. Use `100dvh` instead of `100vh` for full-height containers.
- **Input zoom.** iOS Safari zooms the page when an input with font-size below 16px is focused. Composer textarea must be at least 16px.
- **Pull-to-refresh.** Disabled on the chat route via `overscroll-behavior: contain` to avoid accidental refresh mid-scroll.
- **Status bar style.** The `apple-mobile-web-app-status-bar-style` meta is `default` for light mode and `black-translucent` for dark mode; we toggle it from JS.

## Composer on Touch

The composer is the most-used interactive element. Its touch-mode behavior:

- Visible keyboard hint (`Cmd+Enter to send`) hides on `(pointer: coarse)`.
- Send button grows to 56×56px on touch.
- The textarea autosizes from 1 line to 6 lines, then scrolls.
- `enterkeyhint="send"` on the textarea so iOS shows a Send key on the soft keyboard.
- No autocorrect or autocapitalize on user-typed messages? Decision deferred. Default behavior (autocapitalize sentences, autocorrect on) is fine for now.

## Memory and Skills on Touch

- **Delete affordance.** Already always-visible. No change needed.
- **Toggle (Skills).** Already a real `<button>` with text and visual state. Already 44px+ on touch via the new global rule.
- **Search input (Memory).** Sticky to the top of the scroll container so it stays visible while scrolling cards.

## Setup Flow on Touch

The `/setup` flow is now the most important screen for non-technical users. Each step:

- Single-column. Inputs are 48px tall. Buttons are 56px tall.
- "Test connection" and "Save and continue" are stacked vertically on phone, side-by-side on tablet+.
- The Gemini key input is `type="password"` and has a "show" toggle; the toggle is keyboard-reachable.

## Accessibility Reaffirmation

The accessibility pass shipped (commit `02ab8152`). The responsive work cannot regress it. Specifically:

- All new touch-target rules respect existing focus-visible styles.
- Reduced-motion users continue to see no animation.
- Screen-reader landmarks (`<main>`, `<nav>`, `<header>`) stay intact.
- Color contrast survives every breakpoint.

## Acceptance Checklist

The responsive work is done when every box below is checked.

- [ ] All new and migrated media queries use `--bp-*` tokens, not raw pixels.
- [ ] `(pointer: coarse)` global rule enforces 48px minimum touch targets.
- [ ] Every hover-only style guards itself with `@media (hover: hover)`.
- [ ] iPhone SE (375×667) — chat, memory, skills, settings, setup, help/ollama all usable without horizontal scroll.
- [ ] iPhone 16 Pro Max (430×932) — same, plus PWA installed mode tested with safe-area insets.
- [ ] iPad mini portrait (744×1133) — single-column, comfortable, not stretched mobile.
- [ ] iPad Pro 12.9" landscape (1366×1024) — chat centered at 860px, memory and skills get two-column treatment.
- [ ] Surface Pro 9 in tablet mode (2880×1920 logical 1440×960) — touch input works, hover styles do not flash.
- [ ] 13" MacBook Air (1280×832) — fits without compromise.
- [ ] 4K external display (3840×2160) — content stays bounded, does not stretch.
- [ ] Dark mode works at every breakpoint.
- [ ] Reduced-motion users see static state at every breakpoint.
- [ ] Composer, send button, and tray-icon controls all reachable via keyboard at every breakpoint.

## Tooling

- **Visual regression.** Playwright screenshots at the five form factors above for each route. Run on PR. Fail on diff over a threshold. Stretch goal; not blocking the responsive ship.
- **Manual matrix.** Until visual regression lands, every change to a layout file requires the author to spot-check the five form factors using browser devtools' device emulation.

## Out of Scope For This Plan

- TV and ultra-wide layouts. Not on the roadmap.
- Foldables. Treat as small tablets; revisit if a user asks.
- Watch surfaces. Not a target.
- Voice-only navigation. Different problem; covered by the voice feature plan when it lands.
