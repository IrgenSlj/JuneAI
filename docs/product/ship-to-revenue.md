# June AI - Ship-to-Revenue Plan

> **SUPERSEDED (2026-07-06).** Superseded by the milestones in
> [`JUNE_V02_BRIEF.md`](../../JUNE_V02_BRIEF.md) §11. Retained for history only.

Prepared 2026-06-30. Synthesizes a deployment-readiness audit and a monetization/
GTM study into one plan: get June from "runs in a dev terminal" to "a stranger
pays for it." This sits beside `development-plan.md` (the engineering checklist)
and is governed by `overview.md` / `vision.md` (where they conflict, the vision
wins).

## The strategic read

We do not have a product problem. June already has a rich, differentiated
product: the local-first spine, the continuity engine (Promises), inspectable
Memory, and a glass box that shows every agentic and LLM step - which no
competitor offers. The bottleneck to revenue is two things, neither of them
"more features":

1. **Distribution.** `June.app` today is a UI-only shell. It does not start the
   brain, Ollama, or any model. A non-technical user who double-clicks it gets a
   window that fails every API call plus a Gatekeeper warning. Nobody can buy
   what they can't run.
2. **A reason to pay, wired in honestly.** There is no licensing or tier today,
   and June's invariants (no account, no cloud memory by default, local-only
   blocks egress, BYO cloud key, no paid hosting dependency) delete the easy
   levers (metered/usage billing, "your data on our cloud").

## Positioning

One line: **"A private AI that lives on your Mac, remembers what matters, and
shows you every step it takes - nothing leaves your machine unless you say so."**

Sharper, for the moment: **"The personal AI we'll never sell to Meta."** (The
Dec 2025 Limitless -> Meta acquisition spooked the exact privacy audience June
courts; that backlash is a GTM gift.)

The glass box is the wedge. No competitor shows the user the full reasoning,
tool calls, token counts, and cloud boundary live. A 20-second screen recording -
June reasons through a turn, runs a skill, stamps "0 bytes left your machine,"
then flips to cloud and surfaces the egress before and after - is the artifact
that spreads on its own.

## Monetization - the Obsidian model

Obsidian is the canonical template for exactly June's constraints: free
local-first core, paid OPTIONAL cloud conveniences (Sync, Publish), a one-time
supporter tip, and a commercial license. June's roadmap D.8 (encrypted backup)
and D.9 (Google skills) are the Obsidian-Sync equivalents - already planned.

Market evidence: indie Mac one-time licenses cluster at **$39-79** (lifetime
tiers up to ~$249); privacy-first subscriptions land at **~$8/mo or ~$90/yr**
(Raycast Pro $8/mo, Standard Notes ~$90/yr, MacWhisper $79 one-time, Superwhisper
offers $84.99/yr or $249.99 lifetime).

### Recommended model (B, with A as the launch wedge)

1. **Now -> first revenue: one-time license.** $59 (3 activations). The fastest
   honest path; the truest expression of "June is installed, not subscribed to."
   Add a **$149 Founder/Lifetime** tier that includes Pro Sync forever, to pull
   early-adopter cash forward (lifetime tiers convert).
2. **When D.8/D.9 land: optional June Pro subscription.** $8/mo or $79/yr, exactly
   like Obsidian Sync. The app stays free-and-whole offline forever; we charge
   only for conveniences that cost us money to run.
3. **Commercial-use license** $60/yr (Obsidian's exact move).

Why not a flat subscription for the base app: charging a recurring fee for an app
whose cloud is BYOK and whose data never touches our servers is the kind of thing
June's "honesty is not adjustable" invariant would flag as a tell. We charge for
**encrypted storage + bandwidth + a managed relay we actually operate** - the
user is transparently paying our hosting bill, not a toll.

### The non-negotiable licensing rule (vision-critical)

Activation must be **offline-verifiable**: Ed25519-signed license files validated
locally with a vetted crypto library (never hand-rolled - the one allowed crypto
exception), with **no phone-home**. A silent activation ping would itself violate
"local-only provably blocks egress / no silent network calls." Any optional online
check (e.g. seat count) must be surfaced in the per-turn provenance frame like any
other egress. This is both a hard requirement and a marketing asset: *even our
licensing respects local-only mode.*

### Free vs Pro

**Free (the entire product soul - never gated):** local Gemma chat, the full
three-store memory (inspect / edit / export - "your memory is yours" must be
free or the pitch collapses), Promises, the glass-box trace, local-only mode,
BYOK Gemini. No gating of memory, honesty, or transparency - those are the
product, not the upsell.

**Pro (recurring-cost conveniences only):** encrypted backup + multi-device sync
(D.8, the headline); a managed cloud relay (Gemini without managing your own key/
billing, capped allowance, still surfaced per call); Google skills polish (D.9,
Gmail/Calendar/Drive, granted once, revocable); supporter status; commercial
license.

Illustrative shape (not a forecast): ~2,000 launch sales at $59 ~= ~$118k gross
spike; then ~1,000 Pro subs at $79/yr ~= ~$79k ARR that compounds. Spike from
licenses, compounding floor from sync.

## Go-to-market

**Wedge audience:** privacy-conscious power users and developers who already own
an M-series Mac with the RAM, already run Ollama/LM Studio locally, and already
pay for indie Mac tools (Raycast Pro, MacWhisper, Obsidian Sync). They have the
hardware, the paying habit, and an active grievance about cloud-AI privacy. They
live on Hacker News, r/LocalLLaMA, r/privacy, Lobsters, Privacy Guides.

**Launch motion:**
1. **Show HN first** (weekday ~9-11am PT) - highest-value channel for deep dev
   tools; open-core + local-first + a glass box is HN's taste. Founder answers in
   comments.
2. **The glass-box recording** as the viral artifact (see Positioning).
3. **Same-week multi-channel:** Product Hunt + Uneed/OpenHunts day one; cross-post
   to r/LocalLLaMA, r/privacy, Lobsters; ride the post-Limitless narrative.
4. **Waitlist** captured on the landing page pre-launch, leaned on hard day-of.

## Deployment gap - prioritized (from the readiness audit)

| # | Gap | Approach | Effort | Owner |
|---|-----|----------|--------|-------|
| 1 | Brain not bundled/launched (THE blocker) | Freeze `june-api` (PyInstaller; fallback python-build-standalone) -> Tauri `externalBin`; spawn + health-wait + shutdown in `lib.rs`, mirroring `start_ollama` | L | AI (spike first to flush sqlite-vec/pynacl native-ext issues) |
| 2 | Ollama not bundled | Keep detect-and-launch; add in-process download or gate the window on readiness | M-L | AI |
| 3 | Embedding model (`nomic-embed-text`) never pulled in onboarding | Add a second pull step to the one-click flow + setup gate | S | AI |
| 4 | No notarization / signing config / release CI | `bundle.macOS` signing + entitlements; CI with `notarytool` + `stapler` | M | AI config; **$99 enrollment + cert human-gated** |
| 5 | Loopback API unauthenticated | Desktop-generated local token injected into the webview | M | AI |
| 6 | Terminal-oriented error hints in setup/help | Reword for a shipped-app audience; link the one-click flow | S | AI |
| 7 | No update / crash-recovery flow | Tauri updater plugin + brain-restart + corrupt-datadir recovery | M-L | AI |
| 8 | Model-tag inconsistency (`e2b` scripts vs `e4b` UI) | One default, one source of truth | S | AI |

## Execution sequence (what the AI builds, in order)

1. **Ship-1: Python sidecar** (gap 1) - the unlock. Spike -> integrate. In flight.
2. **First-run that just works** (gaps 3, 6, 8; then 2) - bundle/guide Ollama,
   pull both models, fix hints + the model-tag split.
3. **Offline license + entitlement gate** - Ed25519-signed license files (no
   phone-home), a single `pro_entitlement` gate defaulting to the free local
   experience, any online check surfaced as provenance.
4. **Landing page** in the SvelteKit stack - pricing, the glass-box demo embed,
   waitlist, FAQ that leads with the privacy/honesty story.
5. **Notarization + signing CI** (gap 4) - everything except the certs.
6. **Loopback token** (gap 5) + **update/recovery** (gap 7).
7. **Pro features** - D.8 encrypted backup, then D.9 Google skills.

## Human-gated decisions (batched - one decision each, not projects)

- **Open-core vs closed-source, and which license** (AGPL vs permissive). Gates
  the HN narrative and everything below. Founder-only.
- **Apple Developer Program enrollment ($99/yr)** - required for Developer ID
  signing + notarization. Needs the founder's legal identity.
- **Payment processor.** Recommend a merchant-of-record (Paddle or Lemon Squeezy,
  ~5%+50c) so we never touch global VAT/sales tax; Gumroad (10%+50c) is the
  fastest path to ship this week. Founder's bank/tax/identity.
- **Price points** ($59 one-time / $79-96 yr Pro / lifetime tier). Founder's call.
- **Whether to operate sync + a cloud relay at all** - introduces a real hosting
  cost and a new privacy surface; must stay strictly optional. Founder decision.
- **Domain + business entity/trademark.**

The AI cannot enroll the Apple account, open the processor account, pick prices,
choose the OSS license, stand up paid hosting, or sign legal/tax paperwork. It
prepares everything up to those lines.

## Sources

Obsidian, Raycast, Standard Notes/Proton, Superwhisper, MacWhisper, EmberType,
Trace, LM Studio pricing pages and teardowns; Rewind/Limitless + Meta acquisition;
Stripe/Paddle/Lemon Squeezy/Gumroad fee comparisons; Apple Developer Program +
Developer ID notarization docs. (Full URLs in the 2026-06-30 monetization study.)
