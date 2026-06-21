#!/usr/bin/env python
"""Calibrate per-script chars-per-token ratios against the live Gemma tokenizer.

June estimates tokens with a dependency-free per-script heuristic
(``june_brain.context.tokens``). This script measures the real ratios by asking
the local provider to echo script-pure samples and reading the ``input_tokens``
it reports — Ollama's indirect tokenization. Run it once when the chat model
changes and paste the printed constants into ``context/tokens.py``.

Usage (Ollama must be running with the chat model pulled):
    packages/brain/.venv/bin/python tools/calibrate_token_ratios.py
"""

from __future__ import annotations

import asyncio

# Script-pure samples (no whitespace) so the ratio reflects one script class.
_SAMPLES = {
    "latin": "thequickbrownfoxjumpsoverthelazydogneartheriverbankatdawn",
    "greek": "οιπολλοίάνθρωποιπερπατούσανστηνπαραλίακάτωαπότονήλιοτοκαλοκαίρι",
    "cyrillic": "быстраякоричневаялисапрыгаетчерезленивуюсобакууреки",
    "cjk": "東京都庁舎前広場朝日昇川岸静寂",
}


async def _main() -> int:
    from june_brain.providers.base import GenerateRequest, Message
    from june_brain.providers.registry import get_registry

    provider = get_registry().get("local-fast")
    print(f"calibrating against {provider.model_id}\n")
    print("script      chars  tokens  chars/token")
    for script, sample in _SAMPLES.items():
        req = GenerateRequest(
            messages=[Message(role="user", content=sample)],
            max_tokens=1,
            temperature=0.0,
        )
        result = await provider.generate(req)
        toks = result.input_tokens or 1
        ratio = len(sample) / toks
        print(f"{script:<10} {len(sample):>6} {toks:>7} {ratio:>11.2f}")
    print("\nUpdate _RATIOS in packages/brain/src/june_brain/context/tokens.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
