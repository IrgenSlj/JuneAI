# Repo audit — 2026-08-18

Line-by-line audit of `main @ 78885078`, run against the four inversions, the
seven invariants in `CLAUDE.md`, and `../security/threat-model.md`. It is the
source of Stream D in [`v0.4-development-plan.md`](v0.4-development-plan.md).

Supersedes nothing. The [2026-07-26 audit](repo-audit-2026-07-26.md) concluded
"the codebase is clean; the open items are a 240MB git history and untested
first-run paths." Both remain true. This audit went a layer deeper and found
three defects on the live chat path plus one structural problem larger than all
of them.

## Method, and its limits

Every finding below was checked against source. Three were then checked again by
writing a test that fails on `main` — those are marked **proven**. The rest are
counts from source or reasoning from reading, and are marked as such, because a
finding that has not been executed is a hypothesis with good evidence.

Three claims from the first pass of this audit did not survive that second check
and were withdrawn. They are recorded in §6 rather than deleted, because the
reason they were wrong is more useful than the fact that they were.

## 1. What holds up

- **The gate.** 1261 tests in 14 seconds, with a written rationale for why each
  expensive check is opt-in rather than gated. This is the best asset in the repo
  and the reason Stream D can move quickly.
- **One model chokepoint.** Every cloud call routes through `record_cloud_call`;
  no raw HTTP model call exists elsewhere. This decision, made early, is what
  makes a provable egress claim possible at all.
- **The threat model.** It opens with what June does not stop, "because a threat
  model that leads with its strengths is marketing", and corrects its own prior
  severity rating after building the fix taught it better. It caught an error in
  the first pass of this audit (§6.1).
- **Graceful degradation is real** — deep-tier fallback, heuristic classifier
  fallback, prose-JSON tool fallback. Wired and tested.
- **`june-verify`** — the ledger is checkable without June running, and
  exportable to someone who does not run June at all. This is the thesis,
  executed.

## 2. Defects on the live chat path

### 2.1 Native tool calls are dropped by `stream_turn` (proven)

`Provider.stream()` yields `str`, so it structurally cannot carry a tool call.
`GemmaProvider.stream()` passes `tools=` to the model and reads only
`delta.content`; `delta.tool_calls` is never touched. `run_turn` prefers native
calls via `_resolve_tool_calls`; `stream_turn` builds a text-only
`pseudo_result` and calls `_extract_tool_calls`.

Every production caller uses `stream_turn`: the chat route, the promise runtime,
the Telegram consumer. `run_turn` is reached only by `loop/experiment.py` and
`tools/reliability_harness.py`. `test_loop_native_tools.py` opens with the
docstring "*run_turn* prefers provider-native tool calls" — the feature is tested
precisely on the path nobody uses.

Given a provider that emits a native tool call and no content — what Ollama does
for a tool-calling turn — the tool is never dispatched and the user receives an
empty reply. Conditional on the model taking the native path, but advertising the
tools is what invites it.

Slice: **D.4**.

### 2.2 Local-only does not block outbound network writes (proven)

`classify_action()` derives `write_network` from a prefix table (`send_`, `post_`,
`publish_`, `email_`, `notify_`, `sms_`, `tweet_`). `NETWORK_TOOLS` holds the
three *read*-network tools. `loop/wiring.is_network_tool()` tests membership in
`NETWORK_TOOLS`, so it answers "is this one of three named read tools" when it
means "does this reach the network." The two disagree on all seven write
prefixes.

`is_network_tool` drives both the Local-only partition in `stream_turn` — which
is the only local-only gate for tools anywhere in the loop, guard or dispatch
layers — and `provenance.egress`. So an outbound write is neither blocked under
Local-only nor reported in the provenance frame. `guard/actions.py` names
`write_network` as the primary exfiltration vector in its own module docstring.

`requires_approval()` does catch these calls ("Sends data off your device"), but
that is a consent gate, not the privacy dial. A user in Local-only who approves
an action still reasonably believes Local-only holds.

Nothing shipped trips this today: the only `send_` tool is in the Telegram skill,
which is `enabled=False` and needs a bot token. It goes live when that skill is
switched on.

Slice: **D.3**.

### 2.3 The privacy dial fails open in two of three places (inconsistency proven; trigger not demonstrated)

`updates.py` returns `True` when the dial read raises, with the comment "a config
failure must not open the gate." `providers/provenance.py` and
`loop/handwritten.py` return `False`. The correct answer and the correct comment
are already in the repo; the predicate exists in three copies and drifted in two.

Honest limit: `load_stored_config()` catches `OSError` and `JSONDecodeError`
around the file read, and `load_secret()` is guarded twice over. No realistic
path that makes `get_privacy_dial()` raise was constructed. The lazy imports
inside the `try` are the most plausible candidate. Defense-in-depth on the
central claim, not a live hole.

Slice: **D.2**.

## 3. The structural problem

The 2025 life-coach product was never deleted. Measured: **30 of the 54 tools in
`JUNE_TOOLS`, spanning 919 of the 1,484 lines of `tools.py`**. The trimmed set
the default local model receives, `JUNE_TOOLS_GEMMA`, is **24 tools of which none
belong to the current product** — no recall, no promises, no search, no skills.

On the default local path June's advertised vocabulary describes a
fitness-and-journaling app while `overview.md` describes a trusted continuity
engine. The model cannot be more June than its toolbox allows.

This is a live cost, not only an aesthetic one. Thirty irrelevant tools in the
prompt is a plausible cause of a small local model picking the wrong one, and
wrong picks are why `tool_aliases.py` exists as 391 lines of alias and parameter
normalisation.

The "do NOT reintroduce" list in `CLAUDE.md` is a good instrument aimed at the
wrong risk: nothing is being reintroduced, because nothing was removed.

Slice: **D.5**.

## 4. Coherence and hygiene

| Finding | Evidence | Slice |
|---|---|---|
| Proactivity runs through the scheduler prompt — day-of-year rotation, detected patterns, "JUNE'S SUGGESTION FOR TODAY" | `skills/prompts.py:287`, `patterns.py` | D.6 |
| "One `MemoryManager`" is not what the code does — 18 modules import the private `_get_connection`; 338 raw SQL sites, 62 outside `memory/` | count from source | D.1 (doc), later |
| 203 `except Exception` handlers, ~29 to a bare `pass`; 34 in the loop alone | count from source | D.7 |
| Token double-count in `stream_turn`'s generate-fallback | read from source | D.8 |
| ~35-line copy-paste of the suppression buffer, already diverged | read from source | D.8 |
| `_assert_protocol()` never called | read from source | D.8 |
| Docstrings contradicting code (`make_tools_block`, `check.sh` mypy note) | read from source | D.8 |
| 243MB `.git` for 34k lines — committed `junevenv/`, `venv/`, a 68MB `.pt`, MNIST | `git rev-list` | B.1 |

## 5. Scale

| | |
|---|---|
| Source (excl. tests, build artifacts) | 33,929 lines Python |
| Tests | 19,591 lines Python |
| Frontend | 20,329 lines TS/Svelte |
| Suite | 1261 tests, 14.06s, green |
| Commits | 862 |
| Docs | 62 files, 9,241 lines |

## 6. Withdrawn claims

Recorded because the reason each was wrong is instructive.

**6.1 "`send_telegram_message` escapes classification."** It does not.
`classify_action()` matches the `send_` prefix and correctly returns
`write_network`. The first pass also claimed two competing sources of truth for
network-ness; in fact `NETWORK_TOOLS` has a single definition in the guard,
re-exported to the loop, with a comment explaining that consolidation. The real
defect is narrower and worse — §2.2.

**6.2 "Classification should be derived from the manifest."** Proposed as if
new. `threat-model.md` §2.1 considered exactly this, rejected it deliberately,
and explains why: the class is what the UI and ledger *display*, so it must
describe what a call is, not how cautious June is being. Caution is applied
separately from the contract, and `is_network_capable()` already exists. The
reasoning is correct and the audit should have engaged with it before proposing
a replacement.

**6.3 "The privacy fail-open is Critical."** Downgraded to High. The handlers are
wrong and the inconsistency is proven, but no natural trigger was demonstrated —
see §2.3. Rating a latent defect as live spends credibility that the findings
which *are* live then have to borrow back.
