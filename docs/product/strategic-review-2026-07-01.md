# Strategic review — the wedge question (2026-07-01)

**This is a decision brief for the founder, not a decision I made.** A sharpen
pass on `ship-to-revenue.md`, backed by fresh market research, surfaced a
finding that challenges that plan's central recommendation. I did not change the
plan or the product's direction. I'm laying out the evidence and the options so
you can decide. Meanwhile the overnight build continues on work that serves any
of these options (see "What proceeds regardless").

## The finding

`ship-to-revenue.md` leads with: **"$59 one-time private Mac personal assistant,
launched via Show HN."** The research says that is the *weakest* of the credible
wedges. Three evidence points:

1. **Transparency is a paid feature nowhere.** The local-AI space (Jan, LM Studio,
   GPT4All, Msty, Reor, Khoj, Enchanted, Ollama's own UI) is saturated at the
   free/open-source layer. Where anyone monetizes, it's for cloud convenience,
   sync, or teams — never for privacy or "show your work." June's live glass box
   is genuinely differentiated, but differentiation is not willingness-to-pay, and
   no competitor has validated anyone paying for transparency.

2. **People pay for a job done privately, not for "a private assistant."**
   MacWhisper ($59/$69 one-time, ~1,900 reviews at 4.8) proves the exact $59
   one-time Mac model *converts* — because it does a concrete task (dictation)
   with privacy as the sweetener. Superwhisper ($249 lifetime / $8.49-mo) and
   Simple Analytics (~$39K MRR) follow the same pattern. There is no evidence a
   *generic* private assistant monetizes; it competes with free Jan/Ollama.

3. **The compliance wedge is real but narrower than it looks.** Regulatory
   pressure is fresh and genuine (a Feb 2026 SDNY ruling and a UK tribunal both
   held that consumer-AI use can waive attorney-client privilege; consumer LLMs
   are unsafe for PHI without a BAA; press-freedom groups call local the "gold
   standard" for source protection). BUT: for *most* regulated work a BAA-backed
   cloud (Supanote, BastionGPT, Twofold) is already compliant. "Local-only is
   *required*" is true only for the highest-sensitivity subset — and those buyers
   need vendor trust, references, and maybe certifications a solo unknown can't
   quickly manufacture.

## The three wedges, ranked (research verdict)

1. **Vertical compliance tool for one regulated ICP — most credible.** Reframe
   from "assistant" to a job-to-be-done: e.g. *private session-notes for
   therapists*, or *privileged-document analysis for solo/small-firm lawyers*.
   Sold at $30-100/mo with "provably offline" as the compliance proof. Highest
   willingness-to-pay, real differentiation, reachable through professional
   channels — needs no pre-existing audience. **Biggest risk:** BAA cloud covers
   most compliance, so the "cloud is barred" pitch only lands for the most
   sensitive tier, and winning regulated buyers demands trust/credibility that
   takes time to build.

2. **$59 one-time generic assistant via Show HN — weak.** The HN crowd already
   runs free Jan/Ollama and won't pay for a wrapper. The $59 model is proven only
   when the tool does a task, not for a generic assistant.

3. **Open-core + paid Pro sync (Obsidian path) — worst fit now.** Obsidian's sync
   revenue works *because* a massive free userbase existed first. Chicken-and-egg
   with no audience; a multi-year community play, not a near-term revenue path.

## What this means

June's *technology* is strong and the local-first + glass-box + memory + promises
stack is a real asset. The question the research reframes is **not "how do we
package and sell June-the-assistant"** but **"whose specific, painful, compliance-
bound job does June do better than anything they can legally use today?"** The
same engine, pointed at a vertical, has a far more credible path to revenue than
the same engine sold as a general-purpose private assistant.

This is a positioning/business-model decision — yours. It changes GTM, landing-
page copy, possibly onboarding and a skill or two — but **not the core engine**,
which is why it's safe to keep building while you decide.

## The decision (for you)

- **Option A — Hold course:** ship June as the general private assistant, $59
  one-time + Founder/lifetime, Show HN launch (current `ship-to-revenue.md`).
- **Option B — Pivot to a vertical:** pick one compliance ICP (therapists'
  session notes is the cleanest first candidate — high pain, clear "no cloud"
  rule, reachable, less credibility-gated than law/finance), and reshape GTM +
  a thin vertical skill on top of the same engine. Keep the general app as the
  free/OSS base if useful.
- **Option C — Hybrid:** ship the general app free (community + glass-box demo as
  top-of-funnel), and sell a paid vertical layer for one ICP. Captures the
  differentiation *and* the willingness-to-pay, at the cost of two audiences.

My read as your technical co-founder: **B or C, with therapists' private session
notes as the first vertical.** It's the only option the evidence says a solo
founder can convert on, and it turns "provably local" from a nice-to-have into a
purchase requirement. But I'm not making this call unilaterally — it resets GTM,
and that's founder territory.

## What proceeds regardless (no decision needed)

Every wedge needs the same foundations, so the overnight build is unaffected:

- **First-run that actually works** (in progress) — any wedge needs the app to run.
- **Standalone distribution** (the sidecar spike) — any wedge needs a
  double-clickable, installable app; a vertical ICP needs it *more* (non-technical
  buyers).
- **Product hardening** — a compliance buyer is *less* forgiving of rough edges
  than an HN tinkerer.
- **The offline license primitive** (`license-design.md`, Slice 1) — every paid
  wedge needs offline entitlement; it's ICP-agnostic.

If you choose B/C, the incremental work is GTM + one vertical skill + copy — not
an engine rebuild. Nothing built tonight is wasted under any option.
