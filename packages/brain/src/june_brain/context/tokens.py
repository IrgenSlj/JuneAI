"""Language-aware token estimation.

Budget decisions (context assembly, compaction triggers) depend on a token
count, but June must not pull a tokenizer dependency (`tiktoken` / HF) for it —
invariant 4. Instead we approximate with a calibrated chars-per-token ratio per
script class:

    Latin / default ~4.0   (English, Dutch, code, punctuation, whitespace)
    Greek / Cyrillic ~2.5   (Gemma splits these scripts into more sub-word pieces)
    CJK ~1.0                (one token per character is the right order of magnitude)

The old estimator was a flat ``len // 4``, which under-counted Greek by ~1.6x
and CJK by ~4x — a real problem for a multilingual assistant whose owner writes
Greek. Latin text keeps the same ~len/4 behavior (every char, including spaces,
counts at 4.0), so existing Latin-tuned budgets and compaction thresholds are
unchanged; only non-Latin scripts are corrected.

The ratios are reasoned defaults; ``tools/calibrate_token_ratios.py`` measures
them against the live Gemma tokenizer (via Ollama usage counts) and prints
updated constants. Keep this dependency-free and right-signed, not exact.
"""

from __future__ import annotations

# Chars per token, per script bucket. Lower = the script tokenizes denser.
_RATIOS: dict[str, float] = {
    "latin": 4.0,
    "greek": 2.5,
    "cyrillic": 2.5,
    "cjk": 1.0,
}


def _script_of(code: int) -> str:
    """Bucket a Unicode codepoint into a script class for token estimation."""
    # CJK: Unified Ideographs, Hiragana/Katakana, Hangul.
    if (
        0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7A3
        or 0x3400 <= code <= 0x4DBF
    ):
        return "cjk"
    # Greek and Coptic, plus Greek Extended.
    if 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
        return "greek"
    # Cyrillic (+ supplement).
    if 0x0400 <= code <= 0x052F:
        return "cyrillic"
    # Latin, ASCII, punctuation, whitespace, and everything else.
    return "latin"


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` with per-script calibrated ratios.

    Always returns at least 1. Latin-script text matches the prior ~chars/4
    behavior; Greek/Cyrillic/CJK are counted denser so budgets hold across
    languages.
    """
    if not text:
        return 1
    counts: dict[str, int] = {}
    for ch in text:
        bucket = _script_of(ord(ch))
        counts[bucket] = counts.get(bucket, 0) + 1
    total = sum(n / _RATIOS[bucket] for bucket, n in counts.items())
    return max(1, round(total))
