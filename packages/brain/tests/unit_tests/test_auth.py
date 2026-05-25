"""Tests for API key authentication."""
from __future__ import annotations

import pytest
from june_brain.auth import get_api_key, get_api_key_hash, validate_api_key


@pytest.fixture(autouse=True)
def _fresh_state():
    """Reset auth state by clearing the '_api_key' extra before each test."""
    from june_brain.config_store import load_stored_config, save_stored_config

    stored = load_stored_config()
    stored.extras.pop("_api_key", None)
    save_stored_config(stored)


def test_generates_key_on_first_access():
    key = get_api_key()
    assert isinstance(key, str)
    assert len(key) == 64  # 32 bytes → 64 hex chars
    assert key.isalnum()


def test_key_is_persisted():
    key1 = get_api_key()
    key2 = get_api_key()
    assert key1 == key2  # same key on repeated calls


def test_valid_key_passes_validation():
    key = get_api_key()
    assert validate_api_key(key) is True


def test_none_key_fails():
    assert validate_api_key(None) is False


def test_empty_key_fails():
    assert validate_api_key("") is False


def test_wrong_key_fails():
    assert validate_api_key("wrong-key") is False


def test_hash_is_short():
    h = get_api_key_hash()
    assert isinstance(h, str)
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)
