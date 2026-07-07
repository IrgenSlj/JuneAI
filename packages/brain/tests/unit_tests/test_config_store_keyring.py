from __future__ import annotations

from pathlib import Path


def test_local_provider_does_not_probe_keyring_for_gemini_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import june_brain.config_store as store_mod

    monkeypatch.setattr(store_mod, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PROVIDER", "gemma")

    calls: list[str] = []
    monkeypatch.setattr(
        store_mod,
        "load_secret",
        lambda name: calls.append(name) or "should-not-be-read",
    )

    stored = store_mod.load_stored_config()

    assert stored.provider is None
    assert stored.gemini_api_key is None
    assert calls == []


def test_gemini_provider_reads_keyring_when_needed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import june_brain.config_store as store_mod

    monkeypatch.setattr(store_mod, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    monkeypatch.setattr(store_mod, "load_secret", lambda name: "stored-key")

    stored = store_mod.load_stored_config()

    assert stored.gemini_api_key == "stored-key"


def test_disable_keyring_env_makes_keyring_unavailable(monkeypatch) -> None:
    import june_brain.secret_store as secret_store

    monkeypatch.setenv("JUNE_DISABLE_KEYRING", "1")

    assert secret_store.keyring_available() is False
    assert secret_store.load_secret("gemini_api_key") is None
    assert secret_store.save_secret("gemini_api_key", "secret") == "file"
