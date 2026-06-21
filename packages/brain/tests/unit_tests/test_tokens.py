"""Unit tests for the language-aware token estimator (S4.1)."""

from __future__ import annotations

from june_brain.context.tokens import estimate_tokens


def test_empty_is_one():
    assert estimate_tokens("") == 1


def test_latin_matches_chars_over_four():
    # Latin keeps the prior ~len/4 behavior so existing budgets are unchanged.
    text = "a" * 40
    assert estimate_tokens(text) == 10


def test_greek_counts_denser_than_latin():
    greek = "καλημέρα" * 5  # 40 Greek chars
    latin = "a" * 40
    # Same character count, but Greek tokenizes denser -> more estimated tokens.
    assert estimate_tokens(greek) > estimate_tokens(latin)
    # ~2.5 chars/token -> ~16 for 40 chars.
    assert 14 <= estimate_tokens(greek) <= 18


def test_cjk_roughly_one_token_per_char():
    cjk = "東京都庁" * 5  # 20 CJK chars
    assert 18 <= estimate_tokens(cjk) <= 22


def test_mixed_script_sums_per_bucket():
    # 4 Latin (->1) + 4 CJK (->4) ≈ 5 tokens.
    assert estimate_tokens("test" + "日本語だ") == 5


def test_always_at_least_one():
    assert estimate_tokens(".") == 1
