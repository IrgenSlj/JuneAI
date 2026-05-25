# ADR 0012 — API Key Authentication for Local API

## Status

Accepted

## Context

The June API currently has no authentication. Any process on localhost can make requests to the API with any `user_id`. While the primary deployment is local-only (user's own machine), we need:

1. A boundary so that other local applications cannot impersonate the user
2. Future compatibility with remote access (optional, opt-in)
3. A way to distinguish between different clients (web PWA, desktop shell, Telegram bot, CLI tools)

The product is single-user for now, so a full auth system (registration, login, sessions, JWTs) is overkill.

## Decision

- Add a simple **API key authentication** scheme
- On first run, generate a random 32-byte hex API key and store it in the config store (`config.json`)
- The web PWA reads the key from a local file (not env var) during startup
- Desktop shell reads the same key from the config store
- All API requests must include `X-June-Api-Key` header
- The API validates the key via middleware
- User-facing endpoints (`/setup/status`, `/healthz`) are exempt
- CORS remains restricted to localhost origins
- The key can be rotated from settings page
- Optional: allow setting `JUNE_API_KEY` env var to override (for Docker/headless use)

## Consequences

**Positive:**
- Simple, effective boundary that doesn't require a database
- No user management complexity (still single-user)
- The key is generated once and "just works" for legitimate clients
- Easy to audit (every request carries the key header)
- Forward-compatible with remote access (could be extended to TLS + API key)

**Negative:**
- One more thing that must work on first-run (key generation + distribution)
- If the key file is world-readable, security is minimal (acceptable for local-first design)
- Legacy clients without the key header will get 401 until updated

## Implementation

```python
# Middleware pseudocode
API_KEY = None  # loaded on first request

def _load_api_key():
    global API_KEY
    if API_KEY is not None:
        return API_KEY
    store = ConfigStore()
    if store.get("api_key"):
        API_KEY = store["api_key"]
    else:
        API_KEY = secrets.token_hex(32)
        store.set("api_key", API_KEY)
    return API_KEY

# On first-run: generate and store
# Per-request: check X-June-Api-Key header against stored key
```

## References

- ADR 0001 — Monorepo Structure
- ADR 0009 — Private-by-default and Model Routing
