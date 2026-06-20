"""Three-tier model router.

Per ADR 0009, model selection is no longer a binary preset switch. Every model
invocation is resolved by combining two inputs — the user's privacy dial and
the calling skill's model policy — into one of three outcomes: run locally,
run in the cloud, or refuse the request as unavailable under the current
policy.

This module owns the routing decision. It does not own the actual model call;
that stays in the LangGraph agent and the existing provider clients in
``models.py``. The router returns the resolved tier and a ``RuntimeConfig``
that callers feed to the provider stack. Provenance is recorded around the
call by the caller and emitted on the chat-event stream in a separate slice.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from .config import (
    DEFAULT_RUNTIME_PRESET,
    RUNTIME_PRESETS,
    RuntimeConfig,
    _resolve_runtime_config_for_preset,
)


class SkillModelPolicy(StrEnum):
    """A skill's declared model requirement.

    Skills that touch private memory and need no remote intelligence should
    declare ``LOCAL_ONLY``. Skills whose useful work needs frontier reasoning
    (browser automation, long-context extraction, vision) declare
    ``CLOUD_REQUIRED``. Most skills are ``CLOUD_ALLOWED`` and let the router
    decide based on the user's dial and whether the turn is agentic.
    """

    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"
    CLOUD_REQUIRED = "cloud_required"


class UserPrivacyDial(StrEnum):
    """The user's persistent privacy preference."""

    LOCAL_ONLY = "local_only"
    PRIVATE_BY_DEFAULT = "private_by_default"
    CLOUD_FIRST = "cloud_first"


class ResolvedTier(StrEnum):
    """The tier the router picked for one call.

    ``UNAVAILABLE`` means the request cannot be satisfied under the active
    dial and skill policy. Callers must surface a useful explanation rather
    than silently switching tiers.
    """

    LOCAL = "local"
    CLOUD = "cloud"
    UNAVAILABLE = "unavailable"


class ModelUnavailableError(RuntimeError):
    """Raised when the router cannot satisfy a request under the current policy."""

    def __init__(self, *, dial: UserPrivacyDial, skill_policy: SkillModelPolicy) -> None:
        super().__init__(
            f"No model tier available: dial={dial.value} + skill_policy={skill_policy.value}. "
            "Change the privacy dial or relax the skill's model policy."
        )
        self.dial = dial
        self.skill_policy = skill_policy


@dataclass
class ModelProvenance:
    """Per-call record of which model handled a turn.

    Populated incrementally: ``provider``, ``model``, ``tier``, and ``call_id``
    are known before the call; ``latency_ms``, ``prompt_tokens``, and
    ``completion_tokens`` are filled in after the response.
    """

    provider: str
    model: str
    tier: ResolvedTier
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def record(
        self,
        *,
        started_at: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> ModelProvenance:
        """Stamp latency and token counts after a model call returns."""
        self.latency_ms = max(0, int((time.monotonic() - started_at) * 1000))
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        return self


_LOCAL_PRESET = "gemma"
_CLOUD_PRESET = "gemini"


class ModelRouter:
    """Decides which tier handles a given call and produces its provenance.

    The router is intentionally stateless except for an injected check for
    whether the cloud provider is configured (a Gemini key is present). That
    check is used only by the ``CLOUD_FIRST`` fallback path; the resolution
    matrix itself is pure.
    """

    def __init__(self, *, cloud_available: bool | None = None) -> None:
        self._cloud_available_override = cloud_available

    def resolve(
        self,
        *,
        skill_policy: SkillModelPolicy,
        dial: UserPrivacyDial,
        is_agentic: bool = False,
    ) -> ResolvedTier:
        """Return the tier for one call. See the table in ADR 0009."""
        if skill_policy == SkillModelPolicy.LOCAL_ONLY:
            return ResolvedTier.LOCAL

        if skill_policy == SkillModelPolicy.CLOUD_REQUIRED:
            if dial == UserPrivacyDial.LOCAL_ONLY:
                return ResolvedTier.UNAVAILABLE
            if dial == UserPrivacyDial.CLOUD_FIRST and not self._cloud_available():
                return ResolvedTier.UNAVAILABLE
            return ResolvedTier.CLOUD

        # CLOUD_ALLOWED — depends on dial and whether the call is agentic.
        if dial == UserPrivacyDial.LOCAL_ONLY:
            return ResolvedTier.LOCAL
        if dial == UserPrivacyDial.PRIVATE_BY_DEFAULT:
            return ResolvedTier.CLOUD if is_agentic else ResolvedTier.LOCAL
        # CLOUD_FIRST — prefer cloud, fall back to local if cloud is offline.
        return ResolvedTier.CLOUD if self._cloud_available() else ResolvedTier.LOCAL

    def select_runtime(self, tier: ResolvedTier) -> RuntimeConfig:
        """Resolve the ``RuntimeConfig`` for a tier."""
        if tier == ResolvedTier.UNAVAILABLE:
            raise ValueError("Cannot select a runtime for ResolvedTier.UNAVAILABLE")
        preset_key = _LOCAL_PRESET if tier == ResolvedTier.LOCAL else _CLOUD_PRESET
        preset = RUNTIME_PRESETS.get(preset_key, RUNTIME_PRESETS[DEFAULT_RUNTIME_PRESET])
        return _resolve_runtime_config_for_preset(preset)

    def new_provenance(self, tier: ResolvedTier) -> ModelProvenance:
        """Open a new provenance record for an outbound call."""
        runtime = self.select_runtime(tier)
        return ModelProvenance(provider=runtime.preset_key, model=runtime.model, tier=tier)

    def route(
        self,
        *,
        skill_policy: SkillModelPolicy,
        dial: UserPrivacyDial,
        is_agentic: bool = False,
    ) -> tuple[RuntimeConfig, ModelProvenance]:
        """Convenience: resolve, raise on unavailable, return runtime + open provenance."""
        tier = self.resolve(skill_policy=skill_policy, dial=dial, is_agentic=is_agentic)
        if tier == ResolvedTier.UNAVAILABLE:
            raise ModelUnavailableError(dial=dial, skill_policy=skill_policy)
        runtime = self.select_runtime(tier)
        provenance = ModelProvenance(provider=runtime.preset_key, model=runtime.model, tier=tier)
        return runtime, provenance

    def _cloud_available(self) -> bool:
        if self._cloud_available_override is not None:
            return self._cloud_available_override
        return _detect_cloud_available()


def _detect_cloud_available() -> bool:
    """The cloud tier is available if a Gemini key is configured anywhere."""
    # config_store is imported lazily to avoid a routing <-> config_store
    # cycle at module load. ``os`` is at module top — no cycle risk.
    from .config_store import load_stored_config

    if os.getenv("GEMINI_API_KEY"):
        return True
    stored = load_stored_config()
    return bool(stored.gemini_api_key)
