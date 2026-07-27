# Security Policy

June is local-first alpha software. Treat it as a developer preview, not as a
hardened production service.

## The threat model

[`docs/security/threat-model.md`](docs/security/threat-model.md) is the real
document: what June stops, what she does not, and the residual risks — gaps
first, each specific enough to check. Read it before deciding what June should
be allowed to see.

The short version of the gaps:

- **Skills are executables.** A skill runs as a subprocess with your privileges
  and does not need June to call it in order to act, so no gate of June's can
  stop a hostile one. Skills declare a capability contract that is enforced, and
  that stops a skill exceeding what you granted it — not a skill that was never
  honest. Install skills the way you install programs.
- **MCP client identity is self-declared.** A grant limits blast radius and
  creates an audit trail; it does not authenticate the caller.
- **The ledger is tamper-evident, not tamper-proof.** It makes silent revision
  impossible, not revision impossible.
- **Nothing survives an attacker who already runs code as your OS account.**

## Supported versions

`main` and the most recent tagged release. There is no in-app update channel
yet, so a security fix reaches you only if you download a new build.

## Verifying June yourself

```
june-verify --json                    # is the audit trail intact?
june-verify --export chain.jsonl      # check it with your own code
june-mcp list                         # which programs may read your memory?
```

The chain format is documented in full, with a twelve-line standard-library
verifier, in
[`docs/product/trust-ledger-verification.md`](docs/product/trust-ledger-verification.md).
You do not have to trust June's verifier.

## Reporting a vulnerability

Report privately through GitHub Security Advisories. If advisories are not
enabled, open a minimal public issue saying you have a security report, without
exploit details, and we will arrange a private channel.

Useful details:

- Affected route, package, or command.
- Steps to reproduce.
- Whether it needs local machine access, same-network access, a malicious web
  page, or only content June reads.
- Logs or stack traces, with secrets removed.

Reports about anything already listed in the threat model are welcome but
known. The most valuable report is a way past a defence that document claims
works, or a gap it does not mention.

## Current security posture

- The API binds to `127.0.0.1`, validates the `Host` header, uses a CORS
  allow-list, and requires a loopback token. Nothing listens beyond loopback.
- No account system and no multi-tenant authorization: everything runs with the
  privileges of the OS user.
- Memory, the vector index, the entity graph, and the Trust Ledger live in one
  local SQLite file. Embeddings are computed locally via Ollama.
- Cloud model calls are surfaced in the UI before and after, and appended to the
  Trust Ledger. Local-only mode blocks them at the provider seam.
- Secrets prefer the OS keychain, with a file fallback protected by filesystem
  permissions when the keychain is unavailable.
- macOS builds are ad-hoc signed, not notarized.

Do not expose the API to a public network. If you deliberately run it beyond
localhost, put your own authentication in front of it.
