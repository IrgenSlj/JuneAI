# Dependency audit — August 2026 (B.5)

Run against `main` on 2026-08-20 with `uvx pip-audit` (Python) and `pnpm audit`
(JavaScript). Reproduce with:

```
packages/brain/.venv/bin/python -m pip freeze > /tmp/reqs.txt
uvx pip-audit --requirement /tmp/reqs.txt
pnpm audit
```

## Result

| | before | after |
|---|---|---|
| Python advisories | 32 across 8 packages | **0** |
| JavaScript advisories | 35 (20 high, 12 moderate, 3 low) | **1 low** |

## Python — what mattered and why

Two of the eight are meaningfully different from the rest, and they are the
reason this slice was worth doing before launch rather than after.

**`pypdf` 6.10.2 → 6.16.1 (13 advisories)** and **`lxml-html-clean` 0.4.4 →
0.4.5 (1)** are what `skills/files` uses to parse PDFs and web pages. That is
input the user did not write — a document or a page someone else made — which is
precisely the untrusted surface the guard layer exists for
([threat model](threat-model.md)). A parser CVE there is reachable in a way most
dependency advisories are not: the guard's structural defences gate what a tool
may *do*, and say nothing about a parser being exploited while reading its
input.

The rest are ordinary currency:

| Package | From | To | Note |
|---|---|---|---|
| `starlette` | 1.0.0 | 1.6.0 | 7 advisories; the API's own framework |
| `urllib3` | 2.6.3 | 2.7.0 | transitive |
| `idna` | 3.11 | 3.19 | transitive |
| `pydantic-settings` | 2.13.1 | 2.15.0 | transitive |
| `click` | 8.3.2 | 8.4.2 | transitive |
| `setuptools` | 81.0.0 | 83.0.0 | build only |

**Floors are declared only for dependencies the repo actually names.**
`starlette` in `packages/api`, `pypdf` and `lxml-html-clean` in `skills/files`.
The transitive ones are deliberately *not* pinned: adding `urllib3` or `idna` as
direct dependencies to hold a floor would fail
`test_the_runtime_dependency_set_is_deliberate` — correctly, since the invariant
is that June does not take dependencies it does not need. There is no Python
lockfile, so `./tools/bootstrap.sh` resolves current releases and picks the
fixed versions up on its own; the risk being managed is a stale venv, not a
pinned-vulnerable graph.

## JavaScript — all build-chain, one left

Every advisory was in the frontend build chain (vite, SvelteKit, the
`vite-plugin-pwa` → `workbox-build` tree, `sharp`), not in shipped browser code.
Closed by updating `vite` (7.3.6), `@sveltejs/kit` (2.70.3) and `sharp`
(0.35.3 — the libvips CVEs), plus one `pnpm.overrides` entry for
`serialize-javascript` (>=7.0.5), which `workbox-build` reaches through
`@rollup/plugin-terser` and cannot be updated directly.

**One remains, and is left deliberately:** `cookie` <0.7.0, severity low,
reached only through `@sveltejs/kit`'s own pin. Overriding a framework's pinned
version of its cookie parser to close a low-severity out-of-bounds-character
issue trades a real compatibility risk for a marginal one. Revisit when
SvelteKit moves.

## Note for whoever runs this next

Updating `@playwright/test` within its existing range moves the browser build it
expects, and the e2e suite fails with "Executable doesn't exist" until
`pnpm exec playwright install chromium` runs. That is not a regression in the
specs; it caught one here and cost a confusing ten minutes.
