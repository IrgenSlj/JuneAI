"""Test configuration for API tests.

API key auth is opt-in (off unless ``JUNE_API_AUTH_ENABLED`` is set), so the
default test environment needs no ``X-June-Api-Key`` header. Tests that
exercise auth set the flag explicitly via ``monkeypatch``.
"""

from __future__ import annotations
