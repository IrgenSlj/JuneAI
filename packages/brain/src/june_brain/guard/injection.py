"""Injection detection — the heuristic layer of the guard (ADR 0021, Phase 5).

Framing tells the model that tool results are data; the action gates limit what
happens when the model is fooled anyway. Both are *structural*: they hold
regardless of what the content says. This module is the third layer, and it is
the only one that reads the content — a deterministic scan for the shapes that
published personal-agent attacks actually take.

It is deliberately not a classifier and not a blocker:

- **Not a classifier.** No model, no network, no state. A model-based detector
  would be one more thing an attacker can talk to, would cost a round trip on
  every tool result, and could not run when June is offline. This is regexes
  over normalised text, and it either fires or it does not.
- **Not a blocker.** A page *about* prompt injection contains prompt injection;
  so does a security advisory, and so does this repository. Dropping content on
  a match would make June useless on exactly the material a security-minded
  user reads. Detection instead *revokes trust*: a suspicious result voids
  standing approvals for consequential actions, so the next network write or
  code execution asks the user again even if they said "always allow" earlier.

The layer that stops the attack is still the gate. This one decides when the
gate stops taking the user's earlier word for it.

Thresholds are measured, not assumed — see
``packages/brain/tests/fixtures/injection_corpus/`` for the scored corpus and
`docs/product/injection-benchmark.md` for the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tool results are already truncated to 4000 chars at the dispatch seam, but the
# scanner is also called on untruncated content (skills, MCP), so it caps its
# own work. Attacks live at the head or tail of a document, not at 20k.
MAX_SCAN_CHARS = 20_000

# Invisible characters used to smuggle text past both the reader and a naive
# regex: "i<ZWSP>gnore all previous instructions" defeats a literal match.
#
# Two sets, because they answer different questions. Everything invisible is
# *stripped* before the phrase signals run, so padding can never be an evasion.
# Only the subset with no legitimate use in prose is *reported*: the zero-width
# joiner builds every family and profession emoji, the soft hyphen is ordinary
# German and Dutch typography, a BOM is a mis-decoded file, and the bidi
# isolates are the correct way to write mixed-direction text. Flagging those
# would mean flagging emoji, which is not a security posture.
_SMUGGLING = "​‌⁠"  # ZWSP, ZWNJ, word joiner
_BIDI_OVERRIDE = "‪‫‬‭‮"  # LRE, RLE, PDF, LRO, RLO — Trojan Source
_TAG_BLOCK = "\U000e0000-\U000e007f"
_INNOCUOUS = "‍­﻿᠎"  # ZWJ, soft hyphen, BOM, Mongolian vowel separator

_HIDDEN_CHARS = re.compile(f"[{_SMUGGLING}{_BIDI_OVERRIDE}{_TAG_BLOCK}]")
_STRIP_CHARS = re.compile(f"[{_SMUGGLING}{_TAG_BLOCK}{_INNOCUOUS}]")


@dataclass(frozen=True)
class Signal:
    """One named detector: what it looks for, and what it is worth."""

    name: str
    weight: int
    why: str
    pattern: re.Pattern[str]


_F = re.IGNORECASE | re.MULTILINE

# Weights are tiers, and the tiers came from the corpus rather than from taste.
#
#   4 — decisive. Zero hits across the benign corpus, including its hardest
#       cases (a tracking URL carrying a 20-char opaque token, emoji built from
#       zero-width joiners, "do not share this link with anyone"). These have no
#       benign explanation, so one is enough.
#   3 — strong. Each of these *does* fire alone on legitimate content: security
#       advisories quote instruction overrides, API docs tell you to send your
#       API key, ordinary mail says "send the contract to legal@". Two together
#       is the evidence; one is a coincidence.
#   2/1 — weak. Install docs pipe curl into a shell; job ads say "you must".
#       They corroborate, they never conclude.
#
# Each pattern targets a shape seen in a published attack rather than a word
# that sounds alarming. The `[^.\n]{0,N}` gaps let the parts of a phrase sit
# apart ("ignore, for this task only, all of your earlier instructions") without
# matching across sentence boundaries, which is where false positives come from.
SIGNALS: tuple[Signal, ...] = (
    Signal(
        name="instruction_override",
        weight=3,
        why="Tells the assistant to discard its own instructions",
        # The middle group must be genuinely self-referential — "previous",
        # "your", "system". Allowing "all"/"any"/"the" there matched a terms-of
        # -service clause about bypassing *access* restrictions, which is the
        # opposite of an attack.
        pattern=re.compile(
            r"(\b(ignore|disregard|forget|override|bypass|drop|skip)\b[^.\n]{0,50}?"
            r"\b(previous|prior|above|earlier|preceding|your|system)\b[^.\n]{0,50}?"
            r"\b(instructions?|prompts?|rules?|directives?|guidelines?|"
            r"restrictions?|constraints?|training)\b"
            # The declarative form, which names no verb for the model to refuse:
            # "your previous instructions are superseded".
            r"|\byour\s+(previous|prior|earlier|original)\s+"
            r"(instructions?|prompts?|rules?|directives?)\s+"
            r"(are|is|have\s+been|has\s+been)\s+"
            r"(superseded|revoked|replaced|void|cancelled|canceled|no\s+longer))",
            _F,
        ),
    ),
    Signal(
        name="role_hijack",
        weight=3,
        why="Forges a system turn or reassigns the assistant's identity",
        pattern=re.compile(
            r"(<\|?\s*(im_start|im_end|system|assistant|endoftext)\s*\|?>"
            r"|^\s*(system|assistant)\s*:"
            r"|\[\s*(system|admin|developer)\s*(message|prompt|note|instruction)?\s*\]"
            r"|#{2,}\s*(system|admin|developer)\b"
            r"|\bnew\s+(system\s+)?(prompt|instructions?|persona|role)\b"
            r"|\byou\s+are\s+now\b"
            r"|\bfrom\s+now\s+on,?\s+you\b"
            r"|\bact\s+as\s+(if\s+you\s+are\s+)?an?\s+unrestricted\b)",
            _F,
        ),
    ),
    Signal(
        name="concealment",
        weight=4,
        why="Asks the assistant to act without telling the user",
        pattern=re.compile(
            # The target list stays tight on purpose: "we agreed not to tell the
            # customer" is a normal work note, "do not tell the user" is not.
            r"(\b(do\s+not|don'?t|never|no\s+need\s+to)\b[^.\n]{0,40}?"
            r"\b(tell|inform|mention|show|reveal|alert|notify|display|ask|surface)\b"
            r"[^.\n]{0,40}?\b(the\s+user|the\s+human|the\s+operator|them|anyone)\b"
            r"|\bwithout\s+(the\s+)?(user'?s?\s+|their\s+)?"
            r"(knowledge|permission|approval|consent|asking|confirmation|notifying)\b"
            # Either order: "silently send it" and "send it, silently".
            r"|\b(silently|quietly|secretly|covertly|discreetly)\b[^.\n]{0,50}?"
            r"\b(send|post|run|execute|delete|fetch|forward|call|do|proceed)\b"
            r"|\b(send|post|run|execute|delete|fetch|forward|call|do\s+this|proceed)\b"
            r"[^.\n]{0,50}?\b(silently|quietly|secretly|covertly|discreetly)\b"
            # Attacking the gate itself. Kept to the approval vocabulary — a
            # terms-of-service "bypass any access restrictions" must not match.
            r"|\b(disable|bypass|skip|suppress|turn\s+off|deprecate|waive|ignore)\b"
            r"[^.\n]{0,40}?\b(approvals?|confirmations?|consent|guardrails?|"
            r"permission\s+prompts?|approval\s+(step|requirement|gate)s?)\b)",
            _F,
        ),
    ),
    Signal(
        name="exfiltration",
        weight=3,
        why="Directs data to an external destination",
        pattern=re.compile(
            r"\b(send|post|upload|forward|transmit|exfiltrate|leak|deliver|report|"
            r"submit|e-?mail|mail|message|dm|copy|relay|sync)\b"
            r"[^.\n]{0,70}?"
            r"(https?://|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b|\bwebhook\b)",
            _F,
        ),
    ),
    Signal(
        name="secret_solicitation",
        weight=3,
        why="Asks for credentials or configuration",
        # "access" is deliberately not a verb here: it is almost always a noun
        # in this vocabulary ("access token", "access restrictions"), and
        # including it matched a terms-of-service clause.
        #
        # `.env` and `document.cookie` sit outside the `\b(...)\b` group because
        # there is no word boundary before a leading dot — inside it, they could
        # never match at all.
        pattern=re.compile(
            r"\b(reveal|show|print|output|include|append|attach|dump|expose|"
            r"send|give|list|repeat|echo|read|fetch|retrieve|collect|"
            r"gather|extract|enumerate)\b[^.\n]{0,60}?"
            r"(\b(api[\s_-]?keys?|secret[\s_-]?keys?|access[\s_-]?tokens?|"
            r"auth[\s_-]?tokens?|bearer\s+tokens?|passwords?|passphrases?|"
            r"credentials?|private\s+keys?|ssh\s+keys?|session\s+tokens?|"
            r"system\s+prompt|localstorage)\b"
            r"|\.env\b|document\.cookie)",
            _F,
        ),
    ),
    Signal(
        name="link_payload",
        weight=4,
        why="A URL shaped to carry data out in a link or image preview",
        pattern=re.compile(
            # A markdown image auto-loads: the request leaves on render, with no
            # click. The CVSS 8.8 cookie-exfiltration shape. The payload must be
            # long AND contain a word break or run of capitals — a plain 24-char
            # hex tracking token stays below this, as the benign corpus checks.
            r"(!\[[^\]]{0,80}\]\(\s*https?://[^)\s]{0,200}[?&][\w-]{1,24}="
            r"[A-Za-z0-9%+/=_.~-]{24,}"
            # Or any URL with a placeholder where the secret is meant to be
            # substituted: ?d={{api_key}}, ?q=<TOKEN>, ?v=$SECRET. Placeholders
            # do not occur in URLs that were actually meant to be fetched.
            r"|https?://[^\s)\"']{0,160}[?&][\w-]{1,24}="
            r"(\{\{?[^}\s]{1,60}\}?\}|<[^<>\n]{1,70}>|\$[A-Za-z_]\w{0,40}))",
            _F,
        ),
    ),
    Signal(
        name="tool_coercion",
        weight=2,
        why="Names a tool or command for the assistant to run",
        pattern=re.compile(
            r"(\b(call|invoke|use|execute|run|trigger)\b[^.\n]{0,40}?"
            r"\b(tool|function|command|script|shell|terminal|browser)\b"
            r"|\bcurl\b[^\n|]{0,200}\|\s*(ba|z|fi)?sh\b"
            r"|\b(os\.system|subprocess\.(run|call|Popen)|child_process)\s*\("
            r"|\beval\s*\(\s*(atob|base64|decode|request|fetch))",
            _F,
        ),
    ),
    Signal(
        name="urgency_authority",
        weight=1,
        why="Borrows authority or manufactures urgency",
        pattern=re.compile(
            r"(\b(important|urgent|critical|attention|warning)\b\s*[:!]"
            r"[^.\n]{0,60}?\b(you\s+must|you\s+should|immediately|required|comply)\b"
            r"|\b(this\s+is\s+)?(an?\s+)?(official|authorised|authorized|admin|"
            r"system|developer)\s+(instruction|directive|override|request)\b"
            r"|\byour\s+(new\s+)?(task|objective|goal)\s+is\s+to\b)",
            _F,
        ),
    ),
)

# Hidden characters are handled separately because the finding is the character
# itself, not a phrase — and because it is also an evasion tell in its own right.
HIDDEN_TEXT = Signal(
    name="hidden_text",
    weight=4,
    why="Carries text the user cannot see",
    # An HTML comment is the other invisible channel: rendered pages drop it,
    # scraped text keeps it. Only comments carrying instruction-shaped words count.
    pattern=re.compile(
        r"<!--(?:(?!-->)[\s\S]){0,800}?\b(ignore|instruction|system\s+prompt|"
        r"you\s+must|do\s+not\s+tell|api[\s_-]?key|send\s+(it|this|them))\b",
        _F,
    ),
)

_WEIGHTS = {s.name: s.weight for s in (*SIGNALS, HIDDEN_TEXT)}

# One knob, because the weights already encode the shape of the decision: a
# single decisive signal reaches it, two strong signals reach it, and no amount
# of weak corroboration gets there alone. Measured, not chosen — see
# docs/product/injection-benchmark.md for the sweep this came from.
SUSPICIOUS_SCORE = 4


@dataclass(frozen=True)
class InjectionScan:
    """The result of scanning one piece of untrusted content."""

    score: int
    signals: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        """Whether this content should revoke standing approvals.

        A single strong signal is usually prose *about* an attack rather than an
        attack — security advisories quote the payloads. It is recorded, and it
        changes nothing on its own.
        """
        return self.score >= SUSPICIOUS_SCORE

    def describe(self) -> str:
        """A short user-facing summary. Empty when nothing fired."""
        if not self.signals:
            return ""
        why = {s.name: s.why for s in (*SIGNALS, HIDDEN_TEXT)}
        return "; ".join(why[name] for name in self.signals if name in why)


_CLEAN = InjectionScan(score=0, signals=())


def scan(content: str) -> InjectionScan:
    """Scan untrusted content for injection shapes. Pure and deterministic.

    Invisible characters are both reported and removed before the phrase
    signals run, so zero-width padding cannot be used to break a match.
    """
    if not content:
        return _CLEAN
    text = content[:MAX_SCAN_CHARS]

    score = 0
    fired: list[str] = []

    hidden = bool(_HIDDEN_CHARS.search(text))
    text = _STRIP_CHARS.sub("", text)
    if hidden or HIDDEN_TEXT.pattern.search(text):
        score += HIDDEN_TEXT.weight
        fired.append(HIDDEN_TEXT.name)

    for signal in SIGNALS:
        if signal.pattern.search(text):
            score += signal.weight
            fired.append(signal.name)

    if not fired:
        return _CLEAN
    return InjectionScan(score=score, signals=tuple(fired))


def scan_all(contents: list[str]) -> InjectionScan:
    """Scan several results as one body of evidence.

    Signals are unioned rather than summed per document, so a page that fires
    ``exfiltration`` and a later page that fires ``concealment`` together reach
    the threshold. Splitting an attack across two fetches is otherwise free.
    """
    score = 0
    fired: list[str] = []
    for content in contents:
        result = scan(content)
        for name in result.signals:
            if name in fired:
                continue
            fired.append(name)
            score += _WEIGHTS[name]
    if not fired:
        return _CLEAN
    return InjectionScan(score=score, signals=tuple(fired))
