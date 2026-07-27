# ADR 0031 — The update check, and June's first automatic network call

## Status

Accepted. Implements slice 7.4 of
[`v0.3-development-plan.md`](../product/v0.3-development-plan.md). Constrained by
ADR 0009 (private by default), ADR 0016 (event-driven, no heartbeat), and
ADR 0022 (Trust Ledger).

## Context

Every network call June makes today has one of two causes: the user said
something that needed a model, or the user asked for something from the web.
There is no third kind. That is the substance behind "no silent network calls",
and it is checkable — every egress passes one chokepoint and lands in the ledger.

An update check breaks that pattern. Nobody asks for it. It fires because time
passed and a process is running, which is the shape ADR 0016 exists to forbid.

We need it anyway, for a reason that is specific rather than convenient:
**June's threat model promises a security posture, and there is currently no way
to deliver a security fix.** A user who installs `v0.3.0` and never revisits the
repository stays on `v0.3.0` forever, including through a disclosed
vulnerability. Publishing a threat model while having no channel to act on one is
the sort of gap that makes the rest of the document worth less.

So the question is not whether to add an automatic call. It is how to add exactly
one, on terms that do not erode the claim.

## Decision

June checks for updates against the GitHub Releases API, at most once every 24
hours, under the following constraints. Each is a property of the code, not a
promise in a document.

### 1. It is not a timer

ADR 0016 forbids clock-driven behaviour, and this does not get an exception. The
check is evaluated **at read time**, when something already happening asks — the
same discipline as the temporal block and the Silence Model. No thread, no
scheduler, no wakeup. If June is not running, nothing happens; if June is running
and nobody interacts with it, nothing happens.

The 24-hour interval is therefore a *floor on frequency*, not a schedule. A user
who opens June once a month gets one check a month.

### 2. Local-only mode blocks it, hard

The privacy dial's `local_only` setting is the strongest promise June makes.
Adding an automatic call that ignored it would make that promise conditional, and
a conditional promise about egress is not one. The check is refused before the
request is constructed, in the same way a cloud model call is.

### 3. It is ledgered as egress, like everything else

The check appends an `egress` entry with kind, destination host, and outcome. It
appears in Receipts alongside model calls. A user auditing "what left this
machine" sees it without having to know it exists.

### 4. What leaves is stated, not minimised

The request is an unauthenticated `GET` to a single public endpoint. It carries
**no user data, no memory, no identifiers June generates**. What it unavoidably
reveals is what any HTTP request reveals: the machine's IP address, a User-Agent
naming June and its version, and the timing of the request.

That is not nothing. It tells GitHub that an instance of June exists at that
address and was running at that moment. Users who consider that unacceptable have
`local_only`, and a setting to disable the check on its own.

### 5. It can be turned off without turning off everything else

A dedicated setting, defaulting to on. Refusing updates should not require
refusing cloud models.

### 6. It never installs anything on its own

The check reports; the user decides. An automatic download would be a second,
much larger decision — arbitrary code arriving without an explicit act — and this
ADR does not authorise it.

## Consequences

**Good.** A disclosed vulnerability can reach installed users. The threat model's
"no update channel" residual closes. The mechanism is auditable by the same tools
as everything else, so the new call does not need a new kind of trust.

**Bad, and accepted.** June now makes a network request the user did not directly
ask for. The honest framing is that "no silent network calls" was always the
claim — *silent*, not *none* — and this one is neither silent nor unlogged. But
the sentence "June only talks to the network when you ask it to" stops being
literally true, and documentation that says so must change rather than be left
to age into a falsehood.

**Rejected alternatives.**

- *No update channel at all.* Purest, and the position until now. It fails the
  first time a real vulnerability is found, which for a security-positioned
  product is the one failure that matters.
- *Check on every launch.* Simpler, no persistence. Multiplies the signal sent to
  GitHub for no benefit, and a user who restarts often would be pinging
  constantly.
- *A background timer.* Directly contradicts ADR 0016 and buys nothing a
  read-time check does not.
- *Self-hosted update endpoint.* Removes GitHub from the picture but adds
  infrastructure to run and secure, and points users at a server with no
  independent reputation. GitHub already hosts the releases.
- *Automatic download and install.* Rejected under decision 6.

## Verification

- Local-only blocks the check before any request is built — tested.
- Two checks inside 24 hours make one request — tested.
- The ledger gains one `egress` entry per check — tested.
- Disabling the setting stops it while cloud models still work — tested.
- Failure is silent to the user and never blocks a turn: no network, a rate
  limit, or a malformed response all degrade to "no update information".
