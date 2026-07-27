"""Fail if product-surface docs re-mention removed tech (W0.4).

Scans README.md and docs/CURRENT.md for banned stale *tech-stack* tokens: names
of components that were removed from the codebase (Chroma, LangGraph/LangChain,
sentence-transformers) and must never reappear in the two reader-facing surfaces
(brief W0.3: "delete every reference to ChromaDB, LangGraph").

Only unambiguous tech names are banned. Abandoned *product directions* (Quick
Capture, heartbeat-as-cron, operating layer) are deliberately NOT token-banned:
those words legitimately appear in these docs when stating the invariant that the
direction was abandoned (e.g. "No heartbeat, no timer-driven proactivity"). A
substring check cannot tell "we use X" from "we removed X"; reintroducing an
abandoned direction is caught by review + ADRs, not by this check.

Usage:
    python tools/check_doc_hygiene.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCOPED_FILES = ("README.md", "docs/CURRENT.md")

# June is software and is referred to by name, or as "it" — never "she".
# Enforced here because a voice convention applied once decays; the next person
# writing a paragraph has no way to know the rule exists unless something says
# so. Word-boundary matched, so "other", "where" and "gather" are untouched.
BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE),
        "June is 'June' or 'it', never 'she' — prefer the name",
    ),
]

# The product-voice surfaces: everywhere June is described to a reader. Source
# and tests are deliberately out of scope — they legitimately contain people
# ("She lives in Berlin" in a memory fixture, Marie Curie in a capability probe)
# and English stop-word lists.
VOICE_FILES: tuple[str, ...] = (
    "README.md",
    "SECURITY.md",
    "ROADMAP.md",
    "docs/CURRENT.md",
    "docs/vision.md",
    "docs/product/overview.md",
    "docs/product/roadmap.md",
    "docs/security/threat-model.md",
    "apps/landing/index.html",
)
VOICE_GLOBS: tuple[str, ...] = ("apps/web/src/routes/**/*.svelte",)

# Module-level, easy to extend: (token, reason). Matching is case-insensitive
# substring matching, done line-by-line.
BANNED_TOKENS: list[tuple[str, str]] = [
    ("chromadb", "vector backend removed (ADR 0019, use sqlite-vec)"),
    ("langgraph", "loop engine removed (ADR 0018)"),
    ("langchain", "loop engine removed (ADR 0018)"),
    ("sentence-transformers", "dropped with Chroma (ADR 0019)"),
    ("sentence_transformers", "dropped with Chroma (ADR 0019)"),
]


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return (1-based lineno, token, reason) for every banned token found."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for token, reason in BANNED_TOKENS:
            if token in lowered:
                hits.append((lineno, token, reason))
    return hits


def scan_voice(text: str) -> list[tuple[int, str, str]]:
    """Return (1-based lineno, matched word, reason) for pronoun violations."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, reason in BANNED_PATTERNS:
            match = pattern.search(line)
            if match:
                hits.append((lineno, match.group(0), reason))
    return hits


def _voice_paths() -> list[Path]:
    paths = [REPO_ROOT / rel for rel in VOICE_FILES]
    for glob in VOICE_GLOBS:
        paths.extend(sorted(REPO_ROOT.glob(glob)))
    return [p for p in paths if p.exists()]


def main() -> int:
    ok = True
    for rel_path in SCOPED_FILES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, token, reason in scan_text(text):
            ok = False
            print(
                f"docs-hygiene: {rel_path}:{lineno}: banned token '{token}' found ({reason})"
            )

    voice_paths = _voice_paths()
    for path in voice_paths:
        rel = path.relative_to(REPO_ROOT)
        for lineno, word, reason in scan_voice(path.read_text(encoding="utf-8")):
            ok = False
            print(f"docs-hygiene: {rel}:{lineno}: '{word}' — {reason}")

    if not ok:
        return 1

    print(
        f"docs-hygiene: OK (README.md, docs/CURRENT.md clean; "
        f"June's voice consistent across {len(voice_paths)} surfaces)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
