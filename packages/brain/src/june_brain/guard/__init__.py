"""The guard layer — June's injection-defense surface (ADR 0021).

June holds intimate context and runs local Gemma-class models, which are more
susceptible to prompt injection than frontier models. Visibility (the glass box)
is necessary but not sufficient: the blast radius must be limited in code. The
guard layer is where that happens — untrusted-content framing on every tool
result, action classification + approval gates, and skill permission manifests.

This is the anti-OpenClaw position made real (S6). Start with framing.
"""

from .framing import UNTRUSTED_CONTENT_RULE, is_framed, wrap_untrusted

__all__ = ["UNTRUSTED_CONTENT_RULE", "is_framed", "wrap_untrusted"]
