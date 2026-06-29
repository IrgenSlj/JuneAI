"""Optional Ed25519 signing for the Trust Ledger (ADR 0022).

The hash chain gives tamper-*evidence* without any key: mutate one entry and
``verify_chain`` reports the break. Signing adds *authenticity* on top — proof
that an entry was produced by this device's key, so rewriting the whole chain is
also detectable (the signatures no longer verify).

Signing is strictly optional. The brain must run whether or not a crypto library
is installed, so PyNaCl is imported lazily (the same pattern as ``keyring`` in
``secret_store``). When PyNaCl is absent or no device key exists, the ledger runs
unsigned and chain integrity still holds.

Per the no-hand-rolled-crypto invariant, the signature scheme is Ed25519 from a
vetted library (libsodium via PyNaCl); nothing here invents a primitive.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from june_brain import secret_store

# Keyring names for the device signing keypair (stored via secret_store, i.e. the
# OS keychain when available). Hex-encoded.
SECRET_NAME_SK = "trust_ledger_ed25519_sk"
SECRET_NAME_PK = "trust_ledger_ed25519_pk"


@runtime_checkable
class Signer(Protocol):
    """Minimal signing seam so the ledger writer can be tested without a keychain."""

    def sign(self, entry_hash: str) -> str:
        """Return a hex Ed25519 signature over the ``entry_hash`` hex string."""
        ...

    @property
    def public_key_hex(self) -> str:
        """The hex Ed25519 public (verify) key matching this signer."""
        ...


def _nacl():  # type: ignore[no-untyped-def]
    """Import PyNaCl lazily; return the signing module or ``None`` if absent."""
    try:
        import nacl.encoding
        import nacl.signing

        return nacl
    except ImportError:
        return None


class Ed25519Signer:
    """Ed25519 signer built from a hex private key. No keychain dependency.

    Constructed directly (e.g. in tests) or via :func:`device_signer`, which
    loads the key from ``secret_store``.
    """

    def __init__(self, signing_key_hex: str) -> None:
        nacl = _nacl()
        if nacl is None:  # pragma: no cover - exercised only without PyNaCl
            raise RuntimeError("PyNaCl is not installed; cannot sign ledger entries.")
        from nacl.encoding import HexEncoder
        from nacl.signing import SigningKey

        self._key = SigningKey(signing_key_hex.encode("ascii"), encoder=HexEncoder)

    def sign(self, entry_hash: str) -> str:
        signed = self._key.sign(entry_hash.encode("ascii"))
        return signed.signature.hex()

    @property
    def public_key_hex(self) -> str:
        from nacl.encoding import HexEncoder

        return self._key.verify_key.encode(encoder=HexEncoder).decode("ascii")


def generate_keypair() -> tuple[str, str]:
    """Generate a fresh Ed25519 keypair. Returns ``(private_hex, public_hex)``.

    Raises if PyNaCl is unavailable — callers that want graceful degradation
    should check :func:`signing_available` first.
    """
    nacl = _nacl()
    if nacl is None:
        raise RuntimeError("PyNaCl is not installed; cannot generate a signing key.")
    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey

    key = SigningKey.generate()
    sk_hex = key.encode(encoder=HexEncoder).decode("ascii")
    pk_hex = key.verify_key.encode(encoder=HexEncoder).decode("ascii")
    return sk_hex, pk_hex


def signing_available() -> bool:
    """True when a crypto backend is importable (signing *can* be enabled)."""
    return _nacl() is not None


def verify_sig(public_key_hex: str, entry_hash: str, sig_hex: str) -> bool:
    """Verify a hex Ed25519 signature over ``entry_hash``. Never raises."""
    nacl = _nacl()
    if nacl is None:
        return False
    try:
        from nacl.encoding import HexEncoder
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey

        vk = VerifyKey(public_key_hex.encode("ascii"), encoder=HexEncoder)
        try:
            vk.verify(entry_hash.encode("ascii"), bytes.fromhex(sig_hex))
            return True
        except BadSignatureError:
            return False
    except Exception:  # noqa: BLE001 - malformed key/sig must not crash verification
        return False


def ensure_device_key() -> str | None:
    """Ensure a device signing keypair exists in ``secret_store``; return its public key hex.

    Generates and stores a keypair on first call. Returns ``None`` (and stores
    nothing) when no crypto backend is available — unsigned remains a valid mode.
    Not called automatically; a setup step or wiring opts in.
    """
    if not signing_available():
        return None
    existing_pk = secret_store.load_secret(SECRET_NAME_PK)
    existing_sk = secret_store.load_secret(SECRET_NAME_SK)
    if existing_sk and existing_pk:
        return existing_pk
    sk_hex, pk_hex = generate_keypair()
    secret_store.save_secret(SECRET_NAME_SK, sk_hex)
    secret_store.save_secret(SECRET_NAME_PK, pk_hex)
    return pk_hex


def device_signer() -> Ed25519Signer | None:
    """Return a signer built from the stored device key, or ``None`` if unavailable.

    ``None`` when no crypto backend is installed or no device key has been
    provisioned — the ledger then runs unsigned.
    """
    if not signing_available():
        return None
    sk_hex = secret_store.load_secret(SECRET_NAME_SK)
    if not sk_hex:
        return None
    try:
        return Ed25519Signer(sk_hex)
    except Exception:  # noqa: BLE001 - a malformed stored key must not break egress
        return None


def device_public_key() -> str | None:
    """The stored device public key hex, or ``None`` if no key is provisioned."""
    return secret_store.load_secret(SECRET_NAME_PK)
