"""Tests for the three-tier model router (ADR 0009)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from june_brain.routing import (
    ModelProvenance,
    ModelRouter,
    ModelUnavailableError,
    ResolvedTier,
    SkillModelPolicy,
    UserPrivacyDial,
)

# --- Resolution matrix (ADR 0009) ----------------------------------------- #


@pytest.mark.parametrize(
    ("dial", "skill_policy", "is_agentic", "expected"),
    [
        # dial = LOCAL_ONLY
        (UserPrivacyDial.LOCAL_ONLY, SkillModelPolicy.LOCAL_ONLY, False, ResolvedTier.LOCAL),
        (UserPrivacyDial.LOCAL_ONLY, SkillModelPolicy.CLOUD_ALLOWED, False, ResolvedTier.LOCAL),
        (UserPrivacyDial.LOCAL_ONLY, SkillModelPolicy.CLOUD_ALLOWED, True, ResolvedTier.LOCAL),
        (UserPrivacyDial.LOCAL_ONLY, SkillModelPolicy.CLOUD_REQUIRED, False, ResolvedTier.UNAVAILABLE),
        # dial = PRIVATE_BY_DEFAULT
        (UserPrivacyDial.PRIVATE_BY_DEFAULT, SkillModelPolicy.LOCAL_ONLY, False, ResolvedTier.LOCAL),
        (UserPrivacyDial.PRIVATE_BY_DEFAULT, SkillModelPolicy.CLOUD_ALLOWED, False, ResolvedTier.LOCAL),
        (UserPrivacyDial.PRIVATE_BY_DEFAULT, SkillModelPolicy.CLOUD_ALLOWED, True, ResolvedTier.CLOUD),
        (UserPrivacyDial.PRIVATE_BY_DEFAULT, SkillModelPolicy.CLOUD_REQUIRED, False, ResolvedTier.CLOUD),
        # dial = CLOUD_FIRST (cloud available — covers fallback in separate tests)
        (UserPrivacyDial.CLOUD_FIRST, SkillModelPolicy.LOCAL_ONLY, False, ResolvedTier.LOCAL),
        (UserPrivacyDial.CLOUD_FIRST, SkillModelPolicy.CLOUD_ALLOWED, False, ResolvedTier.CLOUD),
        (UserPrivacyDial.CLOUD_FIRST, SkillModelPolicy.CLOUD_ALLOWED, True, ResolvedTier.CLOUD),
        (UserPrivacyDial.CLOUD_FIRST, SkillModelPolicy.CLOUD_REQUIRED, False, ResolvedTier.CLOUD),
    ],
)
def test_resolve_matches_adr_matrix(
    dial: UserPrivacyDial,
    skill_policy: SkillModelPolicy,
    is_agentic: bool,
    expected: ResolvedTier,
) -> None:
    router = ModelRouter(cloud_available=True)
    assert router.resolve(skill_policy=skill_policy, dial=dial, is_agentic=is_agentic) == expected


def test_cloud_first_falls_back_to_local_when_cloud_offline() -> None:
    router = ModelRouter(cloud_available=False)
    tier = router.resolve(
        skill_policy=SkillModelPolicy.CLOUD_ALLOWED,
        dial=UserPrivacyDial.CLOUD_FIRST,
    )
    assert tier == ResolvedTier.LOCAL


def test_cloud_first_cloud_required_unavailable_when_cloud_offline() -> None:
    router = ModelRouter(cloud_available=False)
    tier = router.resolve(
        skill_policy=SkillModelPolicy.CLOUD_REQUIRED,
        dial=UserPrivacyDial.CLOUD_FIRST,
    )
    assert tier == ResolvedTier.UNAVAILABLE


def test_route_raises_model_unavailable_on_local_only_dial_with_cloud_required_skill() -> None:
    router = ModelRouter(cloud_available=True)
    with pytest.raises(ModelUnavailableError) as exc:
        router.route(
            skill_policy=SkillModelPolicy.CLOUD_REQUIRED,
            dial=UserPrivacyDial.LOCAL_ONLY,
        )
    assert exc.value.dial == UserPrivacyDial.LOCAL_ONLY
    assert exc.value.skill_policy == SkillModelPolicy.CLOUD_REQUIRED


# --- select_runtime / provenance ----------------------------------------- #


def test_select_runtime_returns_gemma_preset_for_local_tier() -> None:
    router = ModelRouter()
    runtime = router.select_runtime(ResolvedTier.LOCAL)
    assert runtime.preset_key == "gemma"


def test_select_runtime_returns_gemini_preset_for_cloud_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    router = ModelRouter()
    runtime = router.select_runtime(ResolvedTier.CLOUD)
    assert runtime.preset_key == "gemini"


def test_select_runtime_raises_on_unavailable() -> None:
    router = ModelRouter()
    with pytest.raises(ValueError):
        router.select_runtime(ResolvedTier.UNAVAILABLE)


def test_new_provenance_carries_provider_model_and_uuid_call_id() -> None:
    router = ModelRouter(cloud_available=True)
    provenance = router.new_provenance(ResolvedTier.LOCAL)
    assert provenance.provider == "gemma"
    assert provenance.model  # populated by RUNTIME_PRESETS resolution
    assert provenance.tier == ResolvedTier.LOCAL
    assert re.fullmatch(r"[0-9a-f]{32}", provenance.call_id), "call_id should be a uuid4 hex"
    assert provenance.latency_ms == 0
    assert provenance.prompt_tokens is None
    assert provenance.completion_tokens is None


def test_provenance_record_stamps_latency_and_tokens() -> None:
    import time

    provenance = ModelProvenance(provider="gemma", model="gemma4:e4b", tier=ResolvedTier.LOCAL)
    started = time.monotonic() - 0.123  # simulate a 123 ms call
    provenance.record(started_at=started, prompt_tokens=42, completion_tokens=7)
    assert provenance.latency_ms >= 100  # tolerant of scheduling jitter
    assert provenance.prompt_tokens == 42
    assert provenance.completion_tokens == 7


# --- Persistence of the dial through config_store ------------------------ #


def test_privacy_dial_round_trips_via_config_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUNE_DATA_DIR", str(tmp_path))
    # The config module reads JUNE_DATA_DIR at import time, so patch the resolved value too.
    import june_brain.config as config_mod
    import june_brain.config_store as store_mod

    monkeypatch.setattr(config_mod, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(store_mod, "MEMORY_DIR", str(tmp_path))

    # Default when nothing is persisted yet.
    assert store_mod.get_privacy_dial() == UserPrivacyDial.PRIVATE_BY_DEFAULT

    store_mod.set_privacy_dial(UserPrivacyDial.LOCAL_ONLY)
    assert store_mod.get_privacy_dial() == UserPrivacyDial.LOCAL_ONLY

    store_mod.set_privacy_dial(UserPrivacyDial.CLOUD_FIRST)
    assert store_mod.get_privacy_dial() == UserPrivacyDial.CLOUD_FIRST


def test_set_privacy_dial_rejects_non_enum() -> None:
    import june_brain.config_store as store_mod

    with pytest.raises(TypeError):
        store_mod.set_privacy_dial("local_only")  # type: ignore[arg-type]


# --- Skill manifest default ---------------------------------------------- #


def test_skill_manifest_defaults_to_cloud_allowed_when_policy_absent(tmp_path: Path) -> None:
    """A manifest entry with no explicit model_policy resolves to cloud_allowed."""
    from june_brain.skills.manifest import (
        DEFAULT_MODEL_POLICY,
        SkillManifest,
        SkillManifestEntry,
        load_manifest,
        save_manifest,
    )

    assert DEFAULT_MODEL_POLICY == "cloud_allowed"

    # An entry constructed without specifying model_policy uses the default.
    entry = SkillManifestEntry(key="example")
    assert entry.model_policy == DEFAULT_MODEL_POLICY
    assert entry.policy_enum() == SkillModelPolicy.CLOUD_ALLOWED

    # Round-tripping a manifest with no model_policy on disk yields the default.
    target = tmp_path / "skills.toml"
    save_manifest(
        SkillManifest(entries={"example": SkillManifestEntry(key="example", args=["-m", "x"])}),
        target,
    )
    reloaded = load_manifest(target)
    # The default manifest was merged in too; check the on-disk entry specifically.
    assert reloaded.entries["example"].model_policy == DEFAULT_MODEL_POLICY


def test_first_party_default_manifest_policies_are_intentional() -> None:
    """First-party skill defaults: most are cloud_allowed; files is local_only."""
    from june_brain.skills.manifest import DEFAULT_MANIFEST

    assert DEFAULT_MANIFEST.entries["files"].model_policy == "local_only"
    assert DEFAULT_MANIFEST.entries["calendar"].model_policy == "cloud_allowed"
    assert DEFAULT_MANIFEST.entries["research"].model_policy == "cloud_allowed"


def test_skill_manifest_round_trips_non_default_policy(tmp_path: Path) -> None:
    from june_brain.skills.manifest import load_manifest, save_manifest

    target = tmp_path / "skills.toml"
    manifest = load_manifest(target)
    manifest.entries["calendar"].model_policy = "cloud_required"
    save_manifest(manifest, target)

    reloaded = load_manifest(target)
    assert reloaded.entries["calendar"].model_policy == "cloud_required"
    # Other skills keep the default.
    assert reloaded.entries["research"].model_policy == "cloud_allowed"


def test_skill_manifest_invalid_policy_falls_back_to_default(tmp_path: Path) -> None:
    from june_brain.skills.manifest import load_manifest

    target = tmp_path / "skills.toml"
    target.write_text(
        '[skill.research]\nenabled = true\ncommand = "python"\n'
        'args = ["-m", "june_skill_research"]\nmodel_policy = "nonsense"\n',
        encoding="utf-8",
    )
    manifest = load_manifest(target)
    assert manifest.entries["research"].model_policy == "cloud_allowed"
