#!/usr/bin/env python3
"""Generate the light-theme twin of a dark-theme diagram.

GitHub picks between two files with <picture media="(prefers-color-scheme...)">,
so every diagram ships as a pair. Maintaining both by hand guarantees they drift.
This maps the dark token values onto their light counterparts from
``packages/design/src/tokens.css`` and changes nothing else, so the two files
differ only in colour and stay diffable against each other.

Usage:
    python3 tools/svg_light_variant.py docs/architecture/diagrams/*-dark.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ordered longest-first where prefixes overlap, so a replacement cannot eat a
# substring of another token.
TOKEN_MAP: list[tuple[str, str]] = [
    # surfaces
    ("#13110E", "#FAF9F7"),  # bg-base
    ("#1A1814", "#FFFFFF"),  # bg-raised
    ("#22201B", "#F3F1ED"),  # bg-sunken
    ("#0E0C09", "#F0ECE3"),  # terminal / ledger surface
    # text
    ("#F1EEE7", "#141410"),  # fg-primary
    ("#D8D3C9", "#2A2824"),  # fg-secondary
    ("#8E8778", "#6B665D"),  # fg-muted
    ("#5F5A50", "#9A948A"),  # fg-subtle
    # accent + status
    ("#F2AC6E", "#E8965A"),
    ("#E0945A", "#D17F3F"),
    ("#8AA884", "#4E6B4A"),  # success
    ("#C88080", "#8A3B3B"),  # danger
    ("#C8A260", "#8A6A2F"),  # warn
    # borders (alpha-on-dark becomes alpha-on-light)
    ("rgba(255,250,240,0.16)", "rgba(20,16,10,0.14)"),
    ("rgba(255,250,240,0.08)", "rgba(20,16,10,0.08)"),
    ("rgba(242,172,110,0.16)", "rgba(232,150,90,0.12)"),
]

# Sentinels keep an already-substituted value from being matched again by a
# later rule (e.g. dark #F1EEE7 -> #141410 must not then be treated as an input).
def convert(text: str) -> str:
    out = text
    for i, (dark, _light) in enumerate(TOKEN_MAP):
        out = out.replace(dark, f"\x00{i}\x00")
    for i, (_dark, light) in enumerate(TOKEN_MAP):
        out = out.replace(f"\x00{i}\x00", light)
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv:
        src = Path(arg)
        if "-dark" not in src.name:
            print(f"skip {src}: not a -dark.svg source", file=sys.stderr)
            continue
        dst = src.with_name(src.name.replace("-dark", "-light"))
        dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"{src.name} -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
