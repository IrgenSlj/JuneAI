# e2e tests

Playwright smoke tests for the June web app. Tests are hermetic: every
backend call is intercepted by `page.route()` mocks so no real API, no
Ollama, and no network egress is required.

## Run

```
pnpm --filter @june/web test:e2e
```

Or from `apps/web` directly:

```
pnpm test:e2e
```

Playwright starts the Vite dev server automatically (`pnpm dev`) with
`PUBLIC_JUNE_API_URL=http://127.0.0.1:8099`. Nothing listens at that
address; all requests are intercepted in-test before they leave the browser.

## Mocked API approach

`e2e/_mocks.ts` exports `mockApi(page, overrides?)`. It registers
`page.route()` interceptors for every endpoint the Home page fetches on
mount:

| Endpoint | What it guards |
|---|---|
| `GET /setup/status` | `is_configured: true` — prevents redirect to `/setup` |
| `GET /system` | Runtime badge (local provider, ready) |
| `GET /home/{user}/holdings` | Zero open promises — renders the "clear" state |
| `GET /tasks/{user}` | Empty task list |
| `GET /greeting/{user}` | Static greeting text |

Call `mockApi(page)` before `page.goto()` in every test.

## Gate

These tests are **excluded from the default `./tools/check.sh` gate**. They
run only when invoked explicitly via `pnpm --filter @june/web test:e2e` or
when `JUNE_E2E=1` is set (wired into check.sh in a later slice).
