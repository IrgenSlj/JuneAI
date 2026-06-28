# June — design artifact

The realized June visual system, imported from the Claude Design project
(`claude.ai/design`, project `JuneAI`). `June.html` is the host shell: it loads
React + Babel from a CDN, then the JSX components under `components/` and
`frames/`, and renders a tabbed explorer of every surface.

## Run it

`June.html` loads its components with Babel's `src=` (which fetches over the
network), so it must be served over **HTTP** — opening the file directly as a
`file://` URL will fail (the browser blocks the fetches). It also needs network
access for the React/Babel CDN scripts.

```
cd docs/design/artifact
python3 -m http.server 8080
# then open http://localhost:8080/June.html
```

Use the top tab strip to switch surfaces (Mascot, Chat, Mobile, Memory, Promises,
Skills, Trust, Settings, First run, Tokens), the sun/moon for light/dark, and —
on the Chat tab — the scenario strip (Greeting / Active turn / Cloud turn /
Approval gate) and the activity toggle.

## Layout

- `June.html` — host shell, tab router, light/dark + accent state.
- `components/tokens.jsx` — the design tokens (color, type, radii, motion scale).
- `components/primitives.jsx` — Wordmark, StatusDot, ModelStatus, QuietButton.
- `components/Mascot.jsx` — the abstract sun/solstice mark + busy/idle motion.
- `components/ProductHeader.jsx` — the shared slim header.
- Surfaces: `ChatStage`, `MemoryScreen`, `TasksScreen` (Promises), `SkillsScreen`,
  `SystemScreen` (Trust), `SettingsScreen`, `FirstRun`, plus `ApprovalGate`, `Bubble`,
  `ActivityTerminal`, `CenteredComposer` (the two-register chat), `MobileChat2`
  (phone), `MascotBoard`, `TokensView`, `DesktopFrame`, `frames/ios-frame.jsx`.

This is a design prototype (React + Babel in the browser), not the production
SvelteKit app. It is the visual source of truth to port from. The brief it
realizes is `docs/design/master-brief.md`.
