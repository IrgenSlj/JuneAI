# Security Policy

June is local-first alpha software. Please treat it as a developer preview, not as a hardened production service.

## Supported Versions

Only the current `main` branch receives security fixes until the project starts publishing tagged releases.

## Reporting a Vulnerability

Please report security issues privately through GitHub Security Advisories when available. If advisories are not enabled for the repository, open a minimal public issue that says you have a security report without posting exploit details.

Useful details to include:

- Affected route, package, or command.
- Steps to reproduce.
- Whether the issue requires local machine access, same-network access, or a malicious web page.
- Any logs or stack traces with secrets removed.

## Current Security Model

- The API is intended to bind to `127.0.0.1` by default.
- There is no account system and no multi-tenant authorization model.
- Memory, Chroma indexes, config, and fallback secrets are stored on the local machine.
- Gemini mode sends the current prompt and relevant recalled memory context to Google's API for inference.
- The bundled research/files skills can access network URLs. The files skill restricts PDF reads to paths inside the user's home directory.

Do not expose the API directly to a public network. Put it behind your own authentication layer if you intentionally run it beyond localhost.
