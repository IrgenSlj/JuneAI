# June — threat model

**Version:** 1.1, 2026-07-27. Covers `main` at the time of writing.
**Changed in 1.1:** §2.1 was partly wrong and is corrected — see the note there.
**Status:** June is alpha. Treat this as a description of a work in progress.

June's positioning is that it can prove what it did. A claim like that earns
scrutiny, so this document starts with what it does *not* stop, names the gaps
specifically enough to be checked, and only then describes the defences. Where
a defence exists, it points at the file and the test, so you do not have to take
prose for evidence.

It is not a comparison with other products. Other agents' vulnerabilities are
their business; the useful question here is what happens when someone attacks
this one.

---

## 1. Scope

June runs on the user's own machine: a Python "Brain", a FastAPI server bound to
loopback, a SvelteKit UI in a Tauri shell, local models via Ollama, optional
cloud models, and MCP skill servers as child processes. Memory lives in one
SQLite file under the user's data directory.

**In scope:** prompt injection through content June reads; exfiltration of
memory or credentials; tampering with the audit trail; a hostile or careless
skill; a third-party MCP client reading memory; a malicious web page attacking
the local API.

**Out of scope:** an attacker who already has code execution as the user's OS
account. At that point they can read the database, edit the binaries, and
replace the verifier. Nothing in a local-first application survives that, and
claiming otherwise would be the first dishonest sentence in this document.

---

## 2. What June does not stop

First, and in detail, because a threat model that leads with its strengths is
marketing.

### 2.1 A skill you install can do anything you can

**This is the most important gap in the system, and most of it is not fixable
at this layer.**

Version 1.0 of this document filed this as "tool classification is by naming
convention" and rated it High, implying the action gate could be made to stop
it. Building the fix showed that framing was wrong, so it is corrected here.

A skill is a subprocess, spawned with the user's privileges, with no sandbox.
**It does not need June to call it in order to act.** A malicious skill can open
a socket and exfiltrate the moment it starts. Every gate June has governs
whether *June* invokes a tool — which is worth a great deal against a *hijacked*
agent, and worth nothing against a *hostile skill*. No classification scheme
changes that; only OS-level sandboxing would.

So: **install skills the way you install programs.** They are visible in
`/skills` and their calls appear in the trace, which is supply-chain trust, not
enforcement.

Two parts of the original concern *were* fixable, and are now fixed:

- **Contract violation.** A skill declares its action classes in the manifest.
  That contract used to be *reported* — `check_scope_drift()` showed the UI a
  warning and the call went through. It is now enforced at dispatch
  (`exceeds_declared_scopes`), and an "always allow" cannot waive it. This
  closes the update attack: a skill granted `read_local` that ships a new
  version advertising `send_report` is blocked until the user widens the
  contract deliberately. All six bundled skills now declare contracts, pinned by
  `test_skill_scope_contracts.py`.
- **Network reads hiding behind local-sounding names.** A tool named
  `get_page_content` that fetches URLs classifies as `read_local` and escaped
  taint gating entirely. Reads from a skill whose contract permits network
  access are now gated like network reads whenever the arguments are tainted or
  a prior result looked hostile.

The classification itself remains name-based, deliberately: it is what the UI
and the ledger display, so it has to describe what a call *is*, not how cautious
June is being about it. Caution is applied separately, from the contract.

### 2.2 The MCP client's identity is self-declared

An MCP client identifies itself with `JUNE_MCP_CLIENT`. Nothing proves it. Any
program on the machine that can run `june-mcp` can claim to be `claude-desktop`
and inherit whatever grants that name holds.

MCP has no client authentication to build on. Closing this needs OS-level
attestation of the calling process, and is tracked for Phase 6. A grant narrows
blast radius and creates an audit trail; it does not authenticate.

### 2.3 Injection detection is a heuristic, and a shallow one

`guard/injection.py` catches 30 of 30 documented attack shapes in its corpus at
a 3% false-positive rate ([benchmark](../product/injection-benchmark.md)). It
does not catch:

- **Paraphrase.** "The previous guidance no longer applies to this request"
  contains no keyword and scores zero. This is the fundamental limit of a
  pattern matcher.
- **Non-English attacks.** Every pattern is English.
- **Homoglyphs.** `іgnore` with a Cyrillic і is not normalised.
- **Encoded payloads.** Base64 and ROT13 are not decoded before scanning.
- **CSS-hidden text.** `display: none` around instructions needs HTML parsing.

Detection is defence in depth. The layers that hold regardless of what the
content says — framing, action classes, taint, approval — are the actual
defence, and they do not care what language the attack is in.

### 2.4 The ledger is tamper-evident, not tamper-proof

Anyone with write access to `june.db` and a copy of the published scheme can
delete entries and recompute every hash after them. Optional Ed25519 signing
raises the bar to needing the device key, which is stored on the same machine.

The ledger also proves the integrity of what was written, never the
completeness of it. Completeness rests on the write seams being the only paths
to egress and consequential action — a property of the code you check by reading
`loop/wiring.py` and `mcp/server.py`, not one the verifier can establish.

### 2.5 The human is the last line, and humans click through

Every gate in June ends at a person approving something. Approval fatigue is
real and well documented, and the false-positive rate of the injection heuristic
feeds directly into it: every unnecessary prompt trains the reflex to approve
without reading. That is why the false-positive rate is measured and published
rather than assumed, and why a single strong signal is deliberately *not* enough
to trigger a prompt.

### 2.6 Everything on the machine has the user's privileges

There is no account system and no multi-tenant authorization. Skills run as
child processes with the user's privileges — no sandbox, no seccomp, no
separate user. Local-only mode blocks egress at *June's* provider seam
(`providers/provenance.py`); it does not prevent a skill subprocess from opening
a socket.

The secret store prefers the OS keychain and falls back to a file when the
keychain is unavailable or unresponsive (which happens in the packaged app —
see `secret_store.py`). File-fallback secrets are protected by filesystem
permissions alone.

### 2.7 Distribution is not notarized

The macOS DMG is ad-hoc signed, not Developer ID signed and not notarized. Users
must bypass Gatekeeper to run it, which is exactly the habit an attacker
distributing a fake June would want them to have. There is no update channel
yet, so there is no mechanism to reach users with a security fix.

---

## 3. Assets

| Asset | Why an attacker wants it | Where it lives |
|---|---|---|
| Memory (facts, entities, journal) | The point of the product: an intimate, structured record of a person | `june.db` |
| API credentials | Resale, and access to the user's cloud accounts | OS keychain, file fallback |
| The conversation | Whatever the user is currently doing | `june.db`, model context |
| June's tools | Sending, fetching, executing on the user's behalf | The dispatch seam |
| The Trust Ledger | Its destruction hides everything above | `june.db` |

---

## 4. Threat actors

| Actor | Capability | Primary risk |
|---|---|---|
| **Content author** | Controls a web page, email, README or MCP response June reads | Prompt injection → exfiltration or execution |
| **Skill author** | Ships an MCP skill the user installs | Anything, per §2.1 |
| **Local program** | Runs as the user, no June privileges | Reads memory via a spoofed MCP client identity (§2.2) |
| **Malicious web page** | The user visits it in a browser | Reaching the loopback API via the browser |
| **Network attacker** | On the same network | Very little — nothing listens beyond loopback |
| **Machine owner** | Full local access | Out of scope (§1) |

---

## 5. Trust boundaries

```
  user  ──trusted──▶  June  ──untrusted──▶  tool results, web, email, MCP responses
                       │
                       ├──gated──▶  network egress, code execution
                       ├──logged──▶ Trust Ledger (append-only, hash-chained)
                       └──granted──▶ MCP clients (read-only, per-tool, expiring)
```

The load-bearing rule: **instructions come from the system prompt and the user;
everything else is data.** Every defence below is an attempt to make that true
in code rather than in a prompt.

---

## 6. What June does stop

Each row names the code and the test, so the claim is checkable.

| Threat | Defence | Code | Tests |
|---|---|---|---|
| Instructions inside tool results | Every result wrapped in an untrusted-content envelope, applied centrally at dispatch so no skill can bypass it | `guard/framing.py` | `test_guard_framing.py` |
| Injection shapes in content | Deterministic scan; a detection revokes standing approvals | `guard/injection.py` | `test_guard_injection.py` |
| Exfiltration via a tool argument | Taint tracking: an argument derived from a prior tool result always asks, even under "always allow" | `guard/actions.py` | `test_guard_actions.py` |
| Exfiltration via a *described* target | Injection detection gates network reads even when nothing was copied | `guard/actions.py` | `test_guard_actions.py` |
| Silent egress | Every cloud call surfaced in the UI before and after, and appended to the ledger | `providers/provenance.py` | `test_provenance*.py` |
| Egress in local-only mode | Blocked at the provider seam; the promise blocks with `local_only` rather than degrading silently | `providers/provenance.py` | `test_routing*.py` |
| Secrets reaching the ledger | Payloads redacted centrally in the writer, not by callers | `trust/ledger.py`, `guard/redaction.py` | `test_guard_redaction.py` |
| History rewritten | blake2b hash chain plus an AUTOINCREMENT high-water check for tail truncation | `trust/verify.py` | `test_trust_verify_cli.py` |
| Audit only checkable via June | `june-verify` reads the DB directly and exports for third-party checking | `trust/cli.py` | `test_trust_verify_cli.py` |
| Memory read over MCP | Read-only surface, per-tool consent, 90-day expiry, rate limited, every access ledgered | `mcp/server.py` | `test_mcp_server.py` |
| Malicious page hitting the API | Loopback bind, Host-header validation, CORS allow-list, loopback token | `api/app.py`, `middleware/auth.py` | `test_auth*.py` |
| A skill crashing June | Subprocess isolation with respawn | `skills/supervisor.py` | `test_skill_supervisor*.py` |
| A skill exceeding its permission contract | Declared scopes enforced at dispatch; "always allow" cannot waive a breach | `guard/actions.py` | `test_skill_scope_contracts.py` |

### The property worth stating separately

The gates are **structural**. They classify the action and the provenance of its
arguments, not the meaning of the content. A defence that depends on
understanding the attack fails on the attack it did not understand; a defence
that says "this call sends data off the device, and its argument came from a web
page, so ask" holds regardless of how the page was worded, or in what language.

That is the whole design bet, and §2.1 is where it is currently weakest — the
classification is structural, but its *input* is a name a skill chooses.

---

## 7. Residual risk

| Risk | Severity | Status |
|---|---|---|
| A hostile skill acts without June invoking it (§2.1) | **High** | Accepted at this layer. Only OS sandboxing changes it; the defence is install-time trust |
| A skill exceeding its declared contract (§2.1) | Medium | **Fixed.** Enforced at dispatch, not waivable by "always allow" |
| Network reads named like local reads (§2.1) | Medium | **Fixed.** Gated under taint or injection when the skill's contract permits network access |
| MCP client identity spoofable (§2.2) | Medium | Open, Phase 6. Needs OS process attestation |
| Injection heuristic evaded by paraphrase (§2.3) | Medium | Accepted. Structural layers are the defence |
| Ledger rewritten wholesale (§2.4) | Medium | Accepted. Out of scope per §1 |
| Approval fatigue (§2.5) | Medium | Partly mitigated: false-positive rate measured and published |
| Unsandboxed skills (§2.6) | Medium | Open. Needs OS sandboxing |
| No notarization, no update channel (§2.7) | Medium | Phase 8 |
| Secrets in file fallback (§2.6) | Low | Accepted; keychain preferred, fallback documented |

---

## 8. Checking this yourself

None of the above requires trusting the author.

```
june-verify --json                      # is the audit trail intact?
june-verify --export chain.jsonl        # check it with your own code
june-mcp list                           # which programs may read memory?
packages/brain/.venv/bin/python tools/injection_bench.py --sweep
./tools/check.sh                        # every test named in §6
```

Read `packages/brain/src/june_brain/loop/wiring.py` for the dispatch seam — it
is the one place tool calls are gated, framed, and recorded, and it is about a
hundred lines. If a claim in §6 is not true, it is not true there.

---

## 9. Reporting

See [`../../SECURITY.md`](../../SECURITY.md). Reports about anything in §2 are
welcome but already known — what is most useful is a way past §6, or a gap §2
does not mention.

---

## Related

- [`injection-benchmark.md`](../product/injection-benchmark.md) — the detector's measured numbers and limits
- [`trust-ledger-verification.md`](../product/trust-ledger-verification.md) — the chain scheme, in full
- ADRs [0021](../decisions/0021-guard-layer.md) (guard), [0022](../decisions/0022-trust-ledger.md) (ledger), [0030](../decisions/0030-june-as-mcp-memory-server.md) (MCP)
