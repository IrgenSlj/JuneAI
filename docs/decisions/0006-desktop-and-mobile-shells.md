# ADR 0006: Tauri for Desktop, Capacitor for Mobile

**Status:** Accepted
**Date:** 2026-04-17

## Context

The v2 vision requires native-feeling apps on macOS and iOS. Users must be able to download a `.dmg`, double-click it, and have June running with a system tray icon and a global hotkey. They must be able to install a June app from the iPhone App Store that feels indistinguishable from a native app.

The core constraint: one codebase. The SvelteKit app in `apps/web/` must serve all three surfaces. Otherwise we end up maintaining three UIs and the multi-platform vision collapses into "three separate products that kind of resemble each other."

## Decision

- **Desktop (macOS, and Linux/Windows for free):** Tauri. A Rust host process wrapping the system webview, with small Rust commands for native features (system tray, global hotkey, notifications, filesystem, autostart). Binary size is ~10 MB versus Electron's ~150 MB. Memory footprint is a fraction of Electron's.

- **Mobile (iOS first, Android for free):** Capacitor. A Swift/Kotlin host that serves the same SvelteKit build to a webview, plus a plugin surface for iOS-specific features (push notifications, share extensions, voice input, Siri Shortcuts).

- **Web:** SvelteKit built as a PWA. Installable to the home screen on iOS and Android, installable via Chrome on desktop. The PWA is a first-class surface, not a fallback.

The same `apps/web/` build artifact is used by all three. Platform-specific code lives in `apps/desktop/src-tauri/` (Rust) and `apps/mobile/ios/` (Swift) respectively. A small runtime capability detector exposes platform features to the Svelte code (e.g., `showNativeNotification(title, body)` works natively on desktop and mobile, falls back to Web Notifications in the browser).

## Consequences

**Positive:**

- One frontend codebase, three platforms. This is the critical property.
- Tauri binaries are small and fast. macOS users get a real Mac app, not a 200 MB Electron bundle.
- Capacitor's plugin ecosystem covers the iOS features June needs on day one.
- The PWA is free once SvelteKit is in place.
- Rust in the desktop shell is a forcing function for doing systems work correctly (Ollama process management, filesystem permissions).

**Negative:**

- Tauri's webview is the system webview (WKWebView on macOS, WebView2 on Windows). CSS and JS differ slightly per platform. Mitigated by targeting modern evergreen webviews only.
- Capacitor adds an iOS build pipeline. A one-time setup cost. Apple developer account required for distribution.
- Two extra languages in the repo (Rust and Swift), though exposure is minimal — both are thin shells.

## Alternatives Considered

**Electron.** Rejected because bundle size and memory footprint are objectively worse, and because users can tell the difference between an Electron app and a Tauri app on macOS.

**React Native.** Rejected because adopting it would force the UI to be written twice (once for web, once for RN) unless we also adopt React Native for Web, which brings its own complications. Capacitor lets us reuse the SvelteKit build directly.

**Flutter.** Rejected because it replaces the entire UI stack. June's web app would have to be thrown away. The cost is not justified.

**Native apps (Swift for iOS/macOS, no cross-platform layer).** Rejected because it requires two-to-three full UI implementations. Not viable for a small team.

**Progressive Web App only, no native shells.** Considered seriously. Rejected because iOS PWAs are second-class (no push notifications before iOS 16.4, limited share extensions, awkward install flow). A true iPhone app unlocks real product value. The desktop case is similar — tray icons and global hotkeys matter.
