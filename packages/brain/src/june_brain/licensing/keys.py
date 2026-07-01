"""Ed25519 public keys shipped with the application.

The map is intentionally empty until the founder generates the production
keypair and pastes the public hex here.  An empty map means every license
file degrades to FREE — the correct safe pre-launch default.

Key IDs are opaque strings (e.g. "prod-1", "prod-2") that match the
``key_id`` field in the signed license envelope.  Multiple entries support
key rotation without invalidating older licenses signed under the prior key.

NEVER commit a private key here.  Private keys are founder-held and
human-gated; only public (verify) keys belong in this file.
"""

from __future__ import annotations

PUBLIC_KEYS: dict[str, str] = {}
