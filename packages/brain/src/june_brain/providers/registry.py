"""Provider registry — maps roles to concrete Provider instances.

Resolution order for providers.toml:
  1. <config_dir>/providers.toml  (user override — allows per-machine tuning)
  2. The packaged default (providers/providers.toml next to this file)

A broken or missing user override silently falls back to the packaged default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..failure import degrade_quietly
from .base import Provider
from .gemini import GeminiProvider
from .gemma import GemmaProvider


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _packaged_toml() -> dict[str, Any]:
    """Load the providers.toml that is distributed with the package."""
    pkg_dir = Path(__file__).parent
    toml_path = pkg_dir / "providers.toml"
    return _load_toml(toml_path)


def _build_provider(name: str, cfg: dict[str, Any]) -> Provider:
    ptype = cfg.get("type", "")
    if ptype == "gemma":
        return GemmaProvider(
            model_id=cfg["model"],
            base_url=cfg["base_url"],
            tier=cfg["tier"],
        )
    if ptype == "gemini":
        return GeminiProvider(
            model_id=cfg["model"],
            base_url=cfg["base_url"],
        )
    raise ValueError(f"Unknown provider type {ptype!r} for provider {name!r}")


class ProviderRegistry:
    """Maps role names to Provider instances.

    Roles are the brain's stable vocabulary (local-fast, local-deep,
    cloud-capable). Provider names and concrete models live in the toml and
    are invisible to the rest of the brain.
    """

    def __init__(self, toml_data: dict[str, Any] | None = None) -> None:
        if toml_data is None:
            toml_data = self._load_toml_with_fallback()

        self._role_to_name: dict[str, str] = dict(toml_data.get("roles", {}))
        providers_cfg: dict[str, dict] = toml_data.get("providers", {})

        self._providers: dict[str, Provider] = {}
        for role, name in self._role_to_name.items():
            if name not in providers_cfg:
                raise KeyError(f"Role {role!r} maps to unknown provider {name!r}")
            self._providers[role] = _build_provider(name, providers_cfg[name])

    @staticmethod
    def _load_toml_with_fallback() -> dict[str, Any]:
        try:
            from june_brain.datadir.layout import config_dir

            user_path = config_dir() / "providers.toml"
            if user_path.exists():
                return _load_toml(user_path)
        except Exception:  # noqa: BLE001
            degrade_quietly("provider registration")
        return _packaged_toml()

    def get(self, role: str) -> Provider:
        """Return the Provider for *role*; raise KeyError with available roles on miss."""
        if role not in self._providers:
            available = ", ".join(sorted(self._providers))
            raise KeyError(f"Unknown role {role!r}. Available: {available}")
        return self._providers[role]

    def roles(self) -> dict[str, str]:
        """Return role → model_id mapping (the resolution table)."""
        return {role: p.model_id for role, p in self._providers.items()}

    def register(self, role: str, provider: Provider) -> None:
        """Override or insert a provider for *role*.

        This is the zero-brain-change swap point: tests inject a mock provider,
        and the C.2 loop experiment swaps in an alternate without touching graph.py.
        """
        self._providers[role] = provider


_default_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the cached default ProviderRegistry (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProviderRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the cached singleton — for use in tests only."""
    global _default_registry
    _default_registry = None
