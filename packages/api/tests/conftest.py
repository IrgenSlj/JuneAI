"""Test configuration for API tests.

Disables API key auth so tests don't need to send ``X-June-Api-Key`` headers.
"""

from __future__ import annotations

import os

os.environ.setdefault("JUNE_API_AUTH_DISABLED", "1")
