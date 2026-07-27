"""The guard layer — June's injection-defense surface (ADR 0021).

June holds intimate context and runs local Gemma-class models, which are more
susceptible to prompt injection than frontier models. Visibility (the glass box)
is necessary but not sufficient: the blast radius must be limited in code. The
guard layer is where that happens — untrusted-content framing on every tool
result, action classification + approval gates, and skill permission manifests.

This is the anti-OpenClaw position made real (S6). Start with framing.
"""

from .actions import (
    ActionClass,
    classify_action,
    evaluate_call,
    exceeds_declared_scopes,
    is_network_capable,
    is_tainted,
    is_waivable,
    requires_approval,
)
from .framing import UNTRUSTED_CONTENT_RULE, is_framed, wrap_untrusted
from .injection import InjectionScan, scan, scan_all
from .redaction import redact_secrets
from .ssrf import SsrfBlocked, check_url, fetch_guarded

__all__ = [
    "ActionClass",
    "InjectionScan",
    "SsrfBlocked",
    "UNTRUSTED_CONTENT_RULE",
    "check_url",
    "classify_action",
    "evaluate_call",
    "fetch_guarded",
    "exceeds_declared_scopes",
    "is_framed",
    "is_network_capable",
    "is_tainted",
    "is_waivable",
    "redact_secrets",
    "requires_approval",
    "scan",
    "scan_all",
    "wrap_untrusted",
]
