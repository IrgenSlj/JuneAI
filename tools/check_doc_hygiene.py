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

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCOPED_FILES = ("README.md", "docs/CURRENT.md")

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

    if not ok:
        return 1

    print("docs-hygiene: OK (README.md, docs/CURRENT.md clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
