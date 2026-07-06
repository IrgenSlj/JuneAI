# ADR 0024 — Retrieval v2: multi-signal fusion and bi-temporal facts

- **Status:** Accepted (implementation in progress)
- **Date:** 2026-07-06
- **Workstream:** W2 (JUNE_V02_BRIEF.md)
- **Supersedes:** nothing. Extends ADR 0019 (single-engine sqlite-vec storage) and
  the salience recall introduced with the Tier 1 spine.

## Context

June's recall today (`memory/recall.py::gather_hits`) already fuses three stores —
the sqlite-vec `vec0` semantic index, the entity graph, and a structured-table
keyword scan — dedupes by text, applies user-feedback multipliers, and reranks the
semantic candidates by *salience* (recency × frequency × relevance). This is good,
but it has two gaps the 2026 memory-system state of the art has closed:

1. **No real lexical channel.** The "keyword" path is a naive Python
   substring-in-lowercase scan over *structured* tables (goals, open loops,
   preferences, relationships, journal). It does not touch `semantic_facts.text`
   and has no term-frequency weighting. Exact-term and rare-term queries
   ("the Varna library invoice number") under-retrieve.
2. **No temporal validity.** A fact that stopped being true ("I work at X") sits in
   the index with equal standing to current facts. There is no notion of *when a
   fact was true in the world* vs. *when June learned it*, and contradiction leaves
   both rows live.

The reconciliation (`docs/RECONCILIATION.md`) also records the concrete shape we
must build against, which differs from the brief's illustrative SQL:

- The fact table is **`semantic_facts`**, primary key **`(user_id, fact_id TEXT)`**,
  content column **`text`**, learned-at column **`created_at`**. There is no integer
  `id`. `access_count` / `last_accessed` already exist (migration 5).
- There is **no FTS5** anywhere yet.
- Schema migrations are a versioned registry in `memory/migration.py`; latest
  applied version is **6**.

## Decision

Add a fourth, real signal and a temporal model, fused behind the existing single
facade, with all weights in one config object and a benchmark harness to keep us honest.

### 1. Bi-temporal facts (migration v7)

Add to `semantic_facts` (adapting the brief's `memories` DDL to reality):

```sql
ALTER TABLE semantic_facts ADD COLUMN valid_from   TEXT;    -- when true in the world (nullable)
ALTER TABLE semantic_facts ADD COLUMN valid_to     TEXT;    -- when it stopped (nullable = still valid)
ALTER TABLE semantic_facts ADD COLUMN observed_at  TEXT;    -- when June learned it; backfilled from created_at
ALTER TABLE semantic_facts ADD COLUMN superseded_by TEXT;   -- a fact_id (NOT an integer FK)
CREATE INDEX idx_semantic_facts_validity ON semantic_facts(user_id, valid_to, valid_from);
```

Rules:
- **Never hard-delete a superseded fact.** Contradiction/merge sets `valid_to` +
  `superseded_by`. (Forgetting keeps its existing tombstone path via `forgotten_facts`;
  Night Shift, W4, is the only sanctioned removal path and it too tombstones.)
- **Retrieval default: currently-valid only** (`valid_to IS NULL`). Explicit temporal
  queries may include superseded rows, clearly marked.
- **Relative dates are absolutized at write time** using the session timestamp; store
  absolute only. (A sweep for legacy relative strings is a Night Shift op, W4.)
- `observed_at` is backfilled from `created_at` for existing rows.

### 2. Lexical channel: FTS5 over fact text (migration v7)

Because `semantic_facts` has a **composite TEXT primary key and no integer rowid
surrogate**, we do **not** use an external-content FTS table (which needs an integer
`content_rowid`). Instead: a **standalone** FTS5 table keyed by `fact_id`, kept in
sync by triggers on `semantic_facts`.

```sql
CREATE VIRTUAL TABLE semantic_facts_fts USING fts5(
  fact_id UNINDEXED, user_id UNINDEXED, text,
  tokenize='unicode61 remove_diacritics 2'
);
-- AFTER INSERT/UPDATE/DELETE triggers on semantic_facts keep it in sync.
```

BM25 via `bm25(semantic_facts_fts)` (lower is better — inverted during fusion).
The user operates in EN/NL/GR; `unicode61` is adequate — **no language-specific
stemmers this phase**. The existing structured-table keyword scan is retained (it
covers goals/loops/prefs, which are not in `semantic_facts`).

### 3. Four-signal fusion

Per query, over a candidate pool of top-50 per channel:

- `s_vec` — cosine similarity from sqlite-vec (existing).
- `s_bm25` — normalized, inverted BM25 over the FTS candidate set.
- `s_entity` — entity-overlap boost from the graph (+per matched entity edge).
- `s_time` — validity/recency prior: currently-valid = 1.0; superseded decays by a
  90-day half-life from `valid_to`, floor 0.1.

**Fusion:** Reciprocal Rank Fusion (k=60) across the vec and bm25 ranked lists, then
`× (1 + w_e · s_entity) × s_time`. To avoid double-counting recency, `s_time`
**replaces** the recency term inside the current salience score rather than stacking
on it; frequency (`access_count`) and feedback multipliers are preserved. Return top-k
(default 8).

All weights live in one `RetrievalConfig` dataclass (`w_e=0.15`, `rrf_k=60`,
`half_life_days=90`, `pool=50`, `k=8`), overridable via the settings file and logged
(values only) at startup. **Graceful degradation:** if FTS5 is unavailable (like the
existing sqlite-vec fallback), fusion drops the bm25 channel and continues on
vec+graph+structured — never a hard failure.

### 4. Facade discipline

The single facade function (`MemoryManager.recall` → `gather_hits`) gets the
signature change; **no caller queries FTS/vec tables directly.** Add a test (import
graph / grep guard) that fails if any module outside `memory/` imports the FTS or
vec internals.

### 5. Benchmarks (`benchmarks/`, `make bench-memory`)

LongMemEval-subset + LoCoMo runners, checksummed download scripts (datasets **not**
vendored), an adapter piping conversations through the real facade (ingest → retrieve
→ answer with `local-deep`), output JSON + a markdown table under
`benchmarks/results/DATE/`. Plus an internal 100-case golden retrieval test
(query → expected fact_ids).

## Consequences

- **Migrations v7 (temporal columns + FTS + triggers)** are forward-only, idempotent,
  tested against a seeded v0.1-shaped DB and an empty DB; lossless on existing data.
- **Perf budget is a test:** p95 retrieval < 120 ms at 50k facts on M1/16GB, via a
  synthetic 50k-corpus fixture and capped candidate pools.
- **Acceptance:** fusion beats vec-only by ≥10% recall@8 on the golden set.
- Writes cost two more index maintenances (vec + FTS) per fact; acceptable at our scale.
- Temporal columns are the substrate W3 (provenance) and W4 (Night Shift
  contradiction/merge/forget) build on — this ADR lands first by dependency.

## Alternatives considered

- **External-content FTS5** (`content='semantic_facts'`): rejected — needs an integer
  `content_rowid` the composite-TEXT-PK table doesn't have; adding a surrogate rowid
  is a larger, riskier migration than a standalone trigger-synced table.
- **Weighted-sum fusion instead of RRF:** rejected — RRF is scale-free across
  heterogeneous scorers (cosine vs. BM25) and needs no per-channel normalization tuning.
- **Language stemmers now:** deferred — EN/NL/GR mix makes a single stemmer net-negative;
  `unicode61` diacritic folding is the right scope this phase.
