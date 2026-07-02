# Releasing June desktop

This document covers how to produce a signed, notarized, and distributable
`.dmg` of the June desktop app. The build itself is proven unsigned (see
`docs/product/tauri-build-report.md`). Signing and notarization are wired via
CI env vars and a `--config` override so `tauri.conf.json` stays untouched.

## Prerequisites

1. **Apple Developer Program membership** ($99/yr at developer.apple.com).

2. **Developer ID Application certificate** — create it in Xcode or on
   developer.apple.com under Certificates. Download and install it into your
   login keychain. The certificate name looks like:
   `Developer ID Application: Your Name (TEAMID)`

3. **Team ID** — the 10-character identifier visible in your Developer account
   (developer.apple.com/account, or `security find-certificate -c "Developer ID Application" | grep "alis"` on your Mac).

4. **Notarization credentials** — choose one:
   - **Apple ID (simpler):** your Apple ID email + an app-specific password
     generated at appleid.apple.com (under "Sign-In and Security > App-Specific
     Passwords"). Do NOT use your real Apple ID password.
   - **App Store Connect API key (preferred for CI):** create a key under
     Users and Access > Integrations > App Store Connect API in App Store
     Connect. Download the `.p8` file. Note the Key ID and Issuer ID.

## GitHub secrets to set

Go to your GitHub repo > Settings > Secrets and variables > Actions > New
repository secret, and add the following:

| Secret name | Value |
| --- | --- |
| `APPLE_CERTIFICATE` | base64-encoded `.p12` export of your Developer ID Application cert (see below) |
| `APPLE_CERTIFICATE_PASSWORD` | passphrase you chose when exporting the `.p12` |
| `APPLE_SIGNING_IDENTITY` | full string, e.g. `Developer ID Application: Your Name (AB12CD34EF)` |
| `APPLE_TEAM_ID` | 10-char Team ID, e.g. `AB12CD34EF` |
| `APPLE_ID` | your Apple ID email (Apple ID notarization route) |
| `APPLE_PASSWORD` | app-specific password (Apple ID notarization route) |

If using the App Store Connect API key route instead of Apple ID + password,
set `APPLE_API_ISSUER`, `APPLE_API_KEY`, and `APPLE_API_KEY_PATH` instead of
(or in addition to) `APPLE_ID` / `APPLE_PASSWORD`. The API key file must be
written to disk before the build step if using that route; update the workflow
accordingly.

**How to obtain `APPLE_CERTIFICATE` (base64 .p12):**

```
# In Keychain Access: right-click the Developer ID Application cert,
# Export, choose .p12 format, set a passphrase.
# Then base64-encode it:
base64 -i /path/to/certificate.p12 | pbcopy
# Paste the clipboard contents as the APPLE_CERTIFICATE secret.
```

## Cutting a release

Push a version tag to trigger the release workflow:

```
git tag v0.1.0
git push origin v0.1.0
```

The workflow (`.github/workflows/release.yml`) will:
1. Build the web frontend and freeze the Python sidecar (via `beforeBuildCommand`).
2. Compile the Tauri shell.
3. Sign with your Developer ID Application cert (Hardened Runtime enabled,
   entitlements from `src-tauri/entitlements.plist`).
4. Submit to Apple for notarization and staple the ticket to the `.dmg`.
5. Upload the `.dmg` as a workflow artifact and attach it to a GitHub Release.

If the signing secrets are absent (e.g. a fork) the workflow falls back to an
unsigned build, which is useful for validating the build pipeline without a
Developer ID cert.

## Building signed locally

```
cd apps/desktop
APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
APPLE_TEAM_ID=TEAMID \
APPLE_ID=you@example.com \
APPLE_PASSWORD=xxxx-xxxx-xxxx-xxxx \
pnpm exec tauri build \
  --config '{"bundle":{"macOS":{"hardenedRuntime":true,"entitlements":"entitlements.plist"}}}'
```

Tauri will sign the `.app`, submit it to Apple for notarization, wait for
approval, then staple the notarization ticket to the `.dmg`.

## Verifying a signed build

```
# Signature validity
codesign --verify --deep --strict --verbose=2 \
  src-tauri/target/release/bundle/macos/June.app

# Gatekeeper acceptance (expect "accepted, source=Notarized Developer ID")
spctl -a -vvv -t install \
  src-tauri/target/release/bundle/macos/June.app

# Notarization ticket stapled to the .dmg
xcrun stapler validate \
  src-tauri/target/release/bundle/dmg/June_0.1.0_aarch64.dmg
```

## How signing is injected without touching tauri.conf.json

`tauri.conf.json` has no `bundle.macOS` block so the proven unsigned local build
continues to work as-is. The CI workflow injects the signing configuration
at build time via a `--config` override:

```
--config '{"bundle":{"macOS":{"hardenedRuntime":true,"entitlements":"entitlements.plist"}}}'
```

This JSON deep-merges with `tauri.conf.json` at invocation time. Tauri 2.x
also reads `APPLE_SIGNING_IDENTITY` from the environment for the codesign
call. The three entitlements in `src-tauri/entitlements.plist` are required
for the PyInstaller-frozen sidecar to run under the Hardened Runtime:

- `allow-unsigned-executable-memory` — PyInstaller maps writable+executable pages.
- `allow-jit` — companion to the above; required by some Python C extensions.
- `disable-library-validation` — `sqlite_vec/vec0.dylib` and `libsodium` in the
  onedir are not signed with the app's Team ID; validation would reject them
  at `dlopen()` time.

Trim entitlements to the minimum set that still notarizes once signing is
validated on a real Developer ID cert.
