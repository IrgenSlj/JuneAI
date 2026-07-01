# June Offline License + Entitlement — design

Status: **design only, partially greenlit.** Prepared 2026-07-01 as the buildable
spec for the monetization mechanism in `ship-to-revenue.md`. Sequencing decision
(from a sharpen pass on the ship-to-revenue plan): the license *build* waits
behind distribution (the app must be installable before a license gates anything
buyable), with one exception — **Slice 1 is zero-regret and greenlit** because it
is pure, fully tested, adds no dependency, changes no schema, and ships with an
empty public-key map so every code path degrades to FREE. Slices 2-3 (API surface,
gate wiring) wait until the standalone build exists.

## Anchoring findings

- **Crypto is already present.** `packages/brain/pyproject.toml` declares
  `pynacl>=1.5.0` (libsodium), used by `trust/signing.py`. Reuse PyNaCl for Ed25519
  verification — no new dependency, satisfying the "vetted library / never
  hand-roll crypto" invariant. Import lazily; degrade to FREE if absent.
  Verification pattern (from `trust/signing.py`):
  ```python
  from nacl.encoding import HexEncoder
  from nacl.exceptions import BadSignatureError
  from nacl.signing import VerifyKey
  vk = VerifyKey(pubkey_hex.encode("ascii"), encoder=HexEncoder)
  vk.verify(message_bytes, sig_bytes)  # raises BadSignatureError on mismatch
  ```
- **Canonical JSON has a house convention** (`trust/ledger.py`):
  `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
  encoded UTF-8. The license signature MUST reuse exactly this so signer and
  verifier agree. Keep a local one-line copy in `licensing/verify.py` rather than
  importing the ledger internal (keeps modules independent).
- **License home:** `<datadir>/config/license.json`. Add a `license_path()`
  accessor to `datadir/layout.py` mirroring the existing lazy `MEMORY_DIR`-based
  accessors so tests that monkeypatch `MEMORY_DIR` keep working. Support a
  `JUNE_LICENSE_PATH` env override for the desktop shell / tests.
- **House module style** (`trust/`, `silence/`): a package with
  `models`/`store`/domain files, a flat `__init__.py` re-export, a process-wide
  singleton guarded by a lock, and a `reset_for_tests()` registered in the brain
  `conftest.py` autouse reset.
- **Status is surfaced best-effort** via `SystemStatus` (`schemas/system.py`),
  built with helpers like `_ledger_summary()` wrapped in `except Exception` so the
  badge never breaks. Entitlement follows the same pattern.
- **Greenfield:** no existing licensing/entitlement/pro references in the brain or
  API. The safe gate seam today is `POST /skills/registry/{key}/install`
  (`routes/skills.py`) gated only for entries that opt in via a new optional
  `pro_feature` field — no entry declares it, so nothing free is gated; the
  primitive is wired and ready for D.9 (Google skills).

## License file format

A signed envelope: an inner `license` body (the signed material) plus outer `sig`
+ `key_id` (not signed).

```json
{
  "license": {
    "schema_version": 1,
    "license_id": "JUNE-2026-000042",
    "tier": "pro",
    "holder": "Jane Doe <jane@example.com>",
    "seats": 3,
    "issued_at": "2026-07-01T00:00:00Z",
    "expires_at": "2027-07-01T00:00:00Z",
    "features": ["backup_sync", "cloud_relay", "google_skills", "supporter"]
  },
  "key_id": "prod-1",
  "sig": "<hex ed25519 signature over canonical(license)>"
}
```

- `schema_version` (int) — bump-guard; unknown version => FREE.
- `license_id` (str) — opaque, for support/revocation; not used in verification.
- `tier` (str) — `"pro" | "founder"` in a real file (`"free"` needs no file).
  Unknown tier => FREE.
- `holder` (str) — display name/email; shown honestly in status.
- `seats` (int) — informational offline (activation-count enforcement needs an
  online check, out of scope, and would have to be surfaced as provenance).
- `issued_at` (ISO-8601 UTC) — informational.
- `expires_at` (ISO-8601 UTC or `null`) — `null` = lifetime/founder; past => FREE.
- `features` (list[str]) — pro-gated keys; on load intersected with
  `KNOWN_FEATURES` so an unknown key can never silently grant anything.

Signature message: `canonical_json(body).encode("utf-8")` where `body` is the
inner `license` object only. `sig` is the hex Ed25519 signature; `key_id` selects
the verifying public key.

**Keys.** The signing private key is founder-held and human-gated; the app embeds
only public keys, in `licensing/keys.py` as an id->hex map (supports rotation and
dev/prod separation). The map ships **empty** until the founder generates the
production keypair — with an empty map every license degrades to FREE, the correct
safe pre-launch default. A DEV keypair is never committed: tests generate an
ephemeral keypair in-process and inject the public half via a seam
`entitlement_from_raw(raw, *, public_keys=PUBLIC_KEYS, now=...)`, so a dev key can
never grant Pro in a shipped build.

## Module layout — `packages/brain/src/june_brain/licensing/`

| File | Responsibility |
|------|----------------|
| `__init__.py` | Re-export the flat API: `get_entitlement()`, `reset_for_tests()`, `FREE`, feature/tier constants, `entitlement_from_raw`. |
| `models.py` | `Entitlement` + `LicenseBody` dataclasses, tier + `FEATURE_*` constants, `KNOWN_FEATURES`, `FREE` default. |
| `keys.py` | `PUBLIC_KEYS: dict[str, str]` (ships empty). |
| `verify.py` | Canonical serialization + lazy PyNaCl Ed25519 verify + expiry -> `Entitlement`; never raises. |
| `store.py` | `license_path()` + env override, load raw JSON, cached `get_entitlement()` + `reset_for_tests()`. |

`Entitlement` (frozen dataclass): `tier`, `features: frozenset[str]`, `holder`,
`valid_until`, `status` (honest, non-alarming, UI-ready), `.has(feature)`.
`FREE = Entitlement(tier="free", features=frozenset(), status="free — no license installed")`.

## Verification flow (no network, ever)

Load `license.json` -> parse envelope -> look up embedded public key by `key_id`
-> verify Ed25519 over canonical body -> check `schema_version`, `tier`,
`expires_at` vs local UTC now -> build `Entitlement`. **Every failure path
(missing / unreadable / malformed / unknown key / bad signature / expired /
unknown tier) returns a FREE `Entitlement` with an honest status. Nothing raises.
No socket opens.** Any future online seat check must be wired through the existing
egress/provenance path (Trust Ledger `egress` entry), exactly like a cloud call.

## API surface

`EntitlementView` in `schemas/system.py` (`tier`, `status`, `holder`,
`valid_until`, `features`). Embed on `SystemStatus` as nullable
`entitlement: EntitlementView | None` (degrades like `ledger_summary`). Add
`GET /system/entitlement` and a best-effort `_entitlement_view()` helper mirroring
`_ledger_summary()`. Honest examples: `"free — no license installed"`,
`"pro — licensed to Jane Doe, valid until 2027-07-01"`,
`"founder — lifetime, licensed to Jane Doe"`,
`"free — license expired on 2026-06-01"`,
`"free — license signature did not verify"`.

## Free vs Pro mapping (from ship-to-revenue.md)

**Free — never gated, never in the feature set:** local Gemma chat, full
three-store memory (inspect/edit/export), Promises, glass-box trace, local-only
mode, BYOK Gemini. `Entitlement.has(x)` is never consulted on these paths.

**Pro — the only gated keys:**

| Feature key | Maps to | Roadmap |
|---|---|---|
| `backup_sync` | encrypted backup + multi-device sync (headline) | D.8 |
| `cloud_relay` | managed Gemini relay (no BYOK), capped, still surfaced per call | ship-to-revenue Pro |
| `google_skills` | Gmail/Calendar/Drive polish, granted once, revocable | D.9 |
| `supporter` | supporter status (cosmetic) | ship-to-revenue Pro |
| `commercial_use` | commercial-use license | ship-to-revenue Pro |

A `pro` license typically carries `[backup_sync, cloud_relay, google_skills,
supporter]`; a `founder` license carries the same with `expires_at: null`.

## Test plan

Brain `test_licensing.py` (use `pytest.importorskip("nacl")`, generate an
ephemeral keypair, sign a fixture body, inject the public half): valid pro
verifies; tampered body -> free; expired -> free; lifetime founder valid; missing
file -> FREE; unreadable -> free; unknown key_id -> free; unknown feature dropped;
nacl-absent -> free; `FREE.has(x)` False.

API `test_system_entitlement.py`: `/system` includes free entitlement by default;
`/system/entitlement` free default; pro with a fixture license
(`JUNE_LICENSE_PATH` + monkeypatched `PUBLIC_KEYS` + `licensing.reset_for_tests()`).

## Slices (independently committable)

- **Slice 1 — verification core + brain tests. GREENLIT (zero-regret).** New
  `licensing/` package, `license_path()` accessor, `test_licensing.py`, register
  `reset_for_tests` in conftest. `PUBLIC_KEYS` empty. No schema/route change ->
  **no codegen.** Brain pytest + ruff + mypy only.
- **Slice 2 — API surface (deferred until distribution).** `EntitlementView`,
  embed in `SystemStatus`, `GET /system/entitlement`, export from
  `schemas/__init__.py`, `test_system_entitlement.py`. Changes a Pydantic model +
  adds a route -> **run `./tools/codegen.sh` and stage `openapi.json` +
  `types.ts`** or the drift check fails.
- **Slice 3 — wire one real gate (deferred until D.9).** Optional
  `pro_feature: str | None` on the skills registry entry; in
  `install_from_registry`, if declared, `get_entitlement().has(...)` and return an
  honest non-error payload when absent. No entry declares it today -> nothing free
  is gated. Adding `pro_feature` to a wire schema -> codegen again.

## Human-gated (founder only)

- Production Ed25519 keypair generation (private key stays secret and out of the
  repo; public hex pasted into `licensing/keys.py`).
- The license-issuance tool (private-key signer) + merchant-of-record integration
  (Paddle / Lemon Squeezy / Gumroad) that emits a signed `license.json` per sale.
- Price/tier definitions and which features each SKU grants.
- Whether to operate sync + a cloud relay at all (hosting cost + privacy surface).
- Apple Developer enrollment, OSS license choice, payment processor, domain/entity.
