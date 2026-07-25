# Retrieval benchmark — results, 2026-07-26

Phase 2 of the [v0.3 execution plan](v0.3-execution-plan.md). Retrieval v2
(ADR 0024) shipped four-signal fusion with unit tests proving each part runs and
nothing proving the whole beats the vector similarity it replaced. This is that
measurement.

Reproduce with:

```sh
packages/brain/.venv/bin/python tools/retrieval_bench.py --k 8
packages/brain/.venv/bin/python tools/retrieval_bench.py --k 8 --scale 50000
```

Corpus: 100 cases over 133 facts about one fictional user, across eight recall
shapes, with twelve bi-temporal supersession pairs and fourteen distractors.
Fixtures in `packages/brain/tests/fixtures/retrieval_golden/`.

---

## Headline

**The quality gate passed by a wide margin. The latency gate failed.**

| Gate | Target | Result | |
|---|---|---|---|
| recall@8 vs. vector-only | >= +10% | **+29% to +60%** depending on pool size | pass |
| p95 retrieval latency, 50k facts | < 120 ms | **252 ms** | fail |

Both numbers are reported because the second one is the interesting one.

## Results (50,000 facts, k=8)

| Configuration | recall@8 | vs baseline | MRR | rank discipline | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector only (baseline) | 0.588 | +0.0% | 0.456 | 1.00 | 231 | 256 |
| BM25 only | 0.630 | +7.1% | 0.608 | 0.74 | 68 | 75 |
| **fusion (shipped)** | **0.760** | **+29.2%** | **0.701** | 0.79 | 229 | 253 |
| fusion, no entity signal | 0.750 | +27.5% | 0.700 | 0.79 | 237 | 293 |
| fusion, flat temporal | 0.760 | +29.2% | 0.701 | 0.79 | 236 | 253 |

Latency is retrieval only, with the query embedding cache warm. A query June has
never embedded before pays a further **~510 ms p50** in the local embedding call.

### recall@8 by category (133-fact corpus)

| Category | vector only | BM25 only | fusion |
| --- | ---: | ---: | ---: |
| direct | 0.80 | 0.70 | 0.80 |
| paraphrase | 0.42 | 0.28 | 0.56 |
| lexical_rare | 0.43 | 1.00 | 1.00 |
| entity_linked | 0.30 | 0.96 | 0.92 |
| temporal_supersession | 0.75 | 0.42 | 0.75 |
| negation_contrast | 0.12 | 0.69 | 0.69 |
| multi_hop | 0.19 | 0.75 | 0.75 |
| distractor_heavy | 0.19 | 0.25 | 0.25 |

---

## What the measurement changed

### 1. Stop words were poisoning the lexical channel

`_query_terms` kept every token of two characters or more, so *"What is my
diet?"* became `"what" OR "is" OR "my" OR "diet"`. The FTS query ORs its terms
and the candidate pool is capped, so the pool filled with facts that merely
contain "my" — and because RRF credits a document once per channel it appears
in, those flooded facts then outranked the fact the user asked for. The answer to
"What is my diet?" was ranked below "Ana is learning Dutch".

Filtering function words moved **MRR from 0.582 to 0.676 (+16%)** and restored
the supersession category from 0.67 to 0.75. It costs recall in three categories
(distractor_heavy, negation_contrast, multi_hop) whose expected facts were being
reached incidentally through common words, leaving overall recall@8 flat — the
right answers moved *up* rather than *in*. For a system that feeds a limited
context window, rank beats presence.

### 2. The candidate pool was twice the size it should be

recall@8 rises monotonically as the pool shrinks: 50 -> 0.735, 30 -> 0.740,
20 -> 0.760, 15 -> 0.790. RRF credits everything in the pool, so a wide pool
feeds the fusion more noise to rank rather than more signal. The default is now
**20**, taken over the corpus-best 15 to keep margin against overfitting a
100-case set. Latency is indifferent to it — sqlite-vec scans the whole index
regardless of `k` — so this is a quality-only change.

---

## The latency gate, honestly

p95 is **252 ms at 50k facts against a 120 ms target**, and no configuration
change fixes it. The shape of the cost is clear from the ablation: BM25-only runs
at 75 ms, vector-only at 256 ms. The vec0 index is a brute-force KNN scan —
sqlite-vec has no approximate index — so vector search cost grows linearly with
memory size and does not respond to `candidate_pool`.

Two things follow.

**The target was aimed at the wrong cost.** A query June has not embedded before
spends ~510 ms in the local embedding call before retrieval starts. Getting the
scan from 252 ms to 120 ms would move a ~760 ms total to ~630 ms — real, but not
what a user would notice, and not where the next hour of work belongs.

**The scan is still the thing that will eventually break.** At 50k facts it is
250 ms; the growth is linear, so a heavy user in year two is looking at seconds.
Options, none of them free, in rough order of preference:

1. **Pre-filter before the scan** — restrict the vec0 scan by recency or validity
   so the KNN runs over live facts rather than all history. Cheapest, and fits
   the existing bi-temporal columns.
2. **Quantize the index** — sqlite-vec supports binary and int8 vectors; a coarse
   pass over compact vectors followed by an exact rescoring of the survivors.
3. **A real ANN index** — the largest change and a new dependency, which the
   working agreement is deliberately hostile to.

None of this is a v0.3 blocker: 250 ms sits underneath a ~510 ms embedding call
that dominates it. It is filed here so the next person does not rediscover it
from scratch, and so the "p95 < 120 ms" line in the plan is not quietly treated
as met.

---

## What the corpus says about the local embedder

The vector channel alone reaches recall@8 of 0.46-0.59 and loses to plain BM25
on five of eight categories. Cosine similarities cluster in a narrow 0.50-0.63
band whether or not the fact is relevant.

This was checked directly rather than assumed. `nomic-embed-text` documents
task-instruction prefixes (`search_query:` / `search_document:`), and June does
not use them — an obvious suspect. Adding them **did not help**: it improved one
probe query and made two others worse. The local embedder is simply weak at this
kind of personal-fact retrieval.

That is the strongest argument for the fusion work. Fusion is not a marginal
gain layered on a good vector search; it is what makes recall usable given a
small local embedding model. It also means the highest-leverage retrieval
improvement available is a better embedding model, not a better ranker.

---

## Caveats

- One synthetic corpus, one fictional user, 100 cases. Directionally sound,
  precise to maybe a few points, not more.
- Filler facts at `--scale` are seeded with synthetic unit vectors rather than
  real embeddings (embedding 50k facts would take hours). They make the index
  realistically large for latency; they are never scored for quality.
- Rank discipline is only counted when a case declares `outranks`, and
  vector-only's perfect 1.00 partly reflects that it retrieves fewer competing
  facts at all rather than that it orders them better.
- Latency is one machine (Apple Silicon, local Ollama), warm cache.
