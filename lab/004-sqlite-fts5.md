# 004 — Is FTS5 in the pinned Node's SQLite?

## Question
Can the index use Node's built-in `node:sqlite` (no native module to rebuild
for the sidecar), or does it need `better-sqlite3`?

## Hypothesis
Node's bundled SQLite is compiled with FTS5, so `node:sqlite` is enough.

## Cheapest experiment
With the pinned Node binary: open an in-memory database, `CREATE VIRTUAL TABLE t USING fts5(body)`, insert accented and CJK text, run a `MATCH` query with `bm25()` ranking. Record the Node and SQLite versions. Repeat on the bundled sidecar binary, not just the dev machine's Node.

## Result
Pending.

## Decision
Pending. Updates the Index line of PLAN.md's daemon rules.
