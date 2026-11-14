# 004 — Is FTS5 in the pinned Node's SQLite?

## Question
Can the index use Node's built-in `node:sqlite` (no native module to rebuild
for the sidecar), or does it need `better-sqlite3`?

## Hypothesis
Node's bundled SQLite is compiled with FTS5, so `node:sqlite` is enough.

## Cheapest experiment
With the pinned Node binary: open an in-memory database, `CREATE VIRTUAL TABLE t USING fts5(body)`, insert accented and CJK text, run a `MATCH` query with `bm25()` ranking. Record the Node and SQLite versions. Repeat on the bundled sidecar binary, not just the dev machine's Node.

## Result (2026-11-14, Node v26.10.0, SQLite 3.53.4, dev Mac)
- FTS5 is compiled in. `unicode61 remove_diacritics 2` folds accents: "cafe" finds "café", "kuchenbudget" finds "Küchenbudget".
- Surprise: word tokenising fails European compound words. "budget" does not find "Küchenbudget". It also treats a run of Japanese as one token, so "テキスト" alone would not match.
- The `trigram` tokenizer (with `remove_diacritics 1`) fixes both: "budget" and "kuchen" find "Küchenbudget", "テキスト" finds the Japanese line. Its limit: queries under three characters return nothing ("vu" found no match).
- Not yet repeated on the bundled sidecar binary; do that when the sidecar exists (M0).

## Decision
Use `node:sqlite`; no native module, so no rebuild per Node version. Search uses a trigram index so German, Dutch and Nordic compounds work; queries under three characters fall back to a word index or a plain scan. Measure both against the 100 ms target on the founder's real history in M2 before settling the schema.
