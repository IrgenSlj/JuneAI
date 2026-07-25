#!/usr/bin/env python3
"""Measure June's recall quality against the golden corpus.

Retrieval v2 fuses four signals (vector, BM25, entity overlap, temporal
validity) with reciprocal rank fusion. That machinery shipped with unit tests
proving each part runs — and nothing proving the whole is better than the vector
similarity it replaced. This script is that proof, or the refutation.

It seeds a throwaway data directory with the golden fact universe, runs every
case through ``gather_hits`` under several channel configurations, and reports
recall@k, MRR, supersession accuracy, and latency per configuration.

Usage:

    packages/brain/.venv/bin/python tools/retrieval_bench.py
    packages/brain/.venv/bin/python tools/retrieval_bench.py --k 8 --markdown out.md
    packages/brain/.venv/bin/python tools/retrieval_bench.py --scale 50000

Requires a running Ollama with the embedding model pulled — without embeddings
the vector channel is empty and every number below is meaningless, so the script
refuses to run rather than report a flattering lie.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "packages/brain/tests/fixtures/retrieval_golden"

# The data dir is read from the environment at import time, so it must be set
# before june_brain comes in.
_SCRATCH = tempfile.mkdtemp(prefix="june-retrieval-bench-")
os.environ["JUNE_DATA_DIR"] = _SCRATCH

# isort: off
# These must import *after* JUNE_DATA_DIR is set above: june_brain.config reads
# the data directory at import time, so a sorted-to-the-top import block would
# bind the benchmark to the developer's real memory instead of the scratch dir.
from june_brain.memory import recall as recall_mod  # noqa: E402
from june_brain.memory.manager import MemoryManager  # noqa: E402
from june_brain.memory.recall import RetrievalConfig, gather_hits  # noqa: E402
from june_brain.memory.sqlite import _get_connection  # noqa: E402
from june_brain.memory import vec_index  # noqa: E402
from june_brain.memory.vector import _db_path  # noqa: E402
# isort: on


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def load_fixtures() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    facts_doc = json.loads((FIXTURES / "facts.json").read_text(encoding="utf-8"))
    cases_doc = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    facts = {f["id"]: f for f in facts_doc["facts"]}
    return facts, cases_doc["cases"], facts_doc["user_id"]


def seed(mgr: MemoryManager, facts: dict[str, Any], scale: int) -> None:
    """Write the fact universe, then pad to ``scale`` with synthetic filler.

    Filler exists to make latency honest: recall over 130 facts says nothing
    about recall over a real memory. It is deliberately in-domain-ish so it
    competes for candidate slots instead of being trivially separable.
    """
    conn = _get_connection(_db_path())
    for fact_id, fact in facts.items():
        mgr.vector.upsert(
            text=fact["text"],
            fact_id=fact_id,
            source="golden",
            metadata={"kind": "fact", "golden": True},
        )
        if any(k in fact for k in ("valid_from", "valid_to", "superseded_by")):
            conn.execute(
                "UPDATE semantic_facts SET valid_from=?, valid_to=?, superseded_by=? "
                "WHERE user_id=? AND fact_id=?",
                (
                    fact.get("valid_from"),
                    fact.get("valid_to"),
                    fact.get("superseded_by"),
                    mgr.user_id,
                    fact_id,
                ),
            )
    conn.commit()

    # Graph nodes + edges so the entity channel has something to link against.
    for fact_id, fact in facts.items():
        for label in fact.get("entities", []):
            node = mgr.graph.add_node(label=label, kind="entity", props={})
            mgr.graph.add_edge(src=node["node_id"], dst=fact_id, kind="mentioned_in", props={})

    filler = max(0, scale - len(facts))
    if not filler:
        return
    rng = random.Random(20260726)
    # Filler is seeded with synthetic unit vectors written straight to the vec0
    # index rather than through the embedder. Embedding 50k facts at ~0.5s each
    # would take hours, and filler exists only to make the *index* realistically
    # large for a latency measurement — it is never scored for quality. The
    # golden facts above always go through the real embedder.
    probe = mgr.vector._embedder().embed_one("dimension probe")
    dim = len(probe or [])
    vec_index.ensure_table(conn, dim)
    subjects = ["the client", "a supplier", "the team", "a customer", "the auditor",
                "a contractor", "the landlord", "a colleague", "the vendor", "a partner"]
    verbs = ["asked about", "confirmed", "postponed", "reviewed", "queried",
             "escalated", "approved", "rejected", "revised", "logged"]
    objects = ["the maintenance window", "the invoice", "the shipping estimate",
               "the calibration report", "the service contract", "the firmware notes",
               "the site visit", "the export format", "the warranty terms", "the pilot"]
    print(f"seeding {filler} filler facts (this is the slow part)...", file=sys.stderr)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for i in range(filler):
        fact_id = f"filler_{i}"
        text = (f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} "
                f"in week {rng.randint(1, 52)} of note {i}.")
        conn.execute(
            "INSERT OR REPLACE INTO semantic_facts "
            "(user_id, fact_id, text, source, metadata, created_at) VALUES (?,?,?,?,?,?)",
            (mgr.user_id, fact_id, text, "filler", '{"kind": "fact"}', now),
        )
        if dim:
            vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vec_index.upsert(conn, fact_id, [v / norm for v in vec])
        if i and i % 5000 == 0:
            conn.commit()
            print(f"  {i}/{filler}", file=sys.stderr)
    conn.commit()


# ---------------------------------------------------------------------------
# Channel ablation
# ---------------------------------------------------------------------------


class Ablation:
    """Disable one retrieval channel for the duration of a block.

    Patches the module-level entry points ``gather_hits`` calls, which is how
    the channels are separable without threading flags through production code
    that has no reason to carry them.
    """

    def __init__(self, *, vector: bool = True, bm25: bool = True) -> None:
        self.vector = vector
        self.bm25 = bm25
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> Ablation:
        if not self.bm25:
            self._saved["bm25"] = recall_mod.semantic_bm25_hits
            recall_mod.semantic_bm25_hits = lambda *a, **k: []
        if not self.vector:
            self._saved["rerank"] = recall_mod._salience_rerank
            recall_mod._salience_rerank = lambda *a, **k: []
        return self

    def __exit__(self, *exc: object) -> None:
        if "bm25" in self._saved:
            recall_mod.semantic_bm25_hits = self._saved["bm25"]
        if "rerank" in self._saved:
            recall_mod._salience_rerank = self._saved["rerank"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def warm_query_cache(mgr: MemoryManager, cases: list[dict[str, Any]]) -> None:
    """Embed every case query once before any timing runs.

    Embeddings are cached in SQLite keyed by (model, sha256(text)). Without this
    pass the first configuration measured pays the embedding cost for all 100
    queries and every later configuration reads them back from cache, which made
    the first draft of this script report a 160x latency difference between
    configurations that do identical embedding work. Warm first, then the
    numbers compare retrieval to retrieval.
    """
    embedder = mgr.vector._embedder()
    for case in cases:
        embedder.embed_one(case["query"])


def measure_embedding_cost(mgr: MemoryManager, samples: int = 20) -> dict[str, float]:
    """Time embedding of never-before-seen text, so the cache cannot help.

    This is the cost a genuinely new query pays on top of retrieval, and it is
    reported separately rather than folded in: it is dominated by the local
    model server, not by anything in June's retrieval path.
    """
    embedder = mgr.vector._embedder()
    timings: list[float] = []
    for i in range(samples):
        text = f"uncached probe {i} {time.time_ns()}"
        started = time.perf_counter()
        embedder.embed_one(text)
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    return {
        "p50_ms": statistics.median(timings),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def hit_ids(hits: list[dict[str, Any]]) -> list[str]:
    """Fact ids in rank order, deduped, ignoring non-fact sources."""
    out: list[str] = []
    for h in hits:
        ref = str(h.get("ref") or "")
        fid = ref.split(":")[-1] if ref else ""
        if fid and fid not in out:
            out.append(fid)
    return out


def score_case(case: dict[str, Any], ranked: list[str], k: int) -> dict[str, float]:
    expected = case["expected"]
    top = ranked[:k]

    found = [e for e in expected if e in top]
    recall = len(found) / len(expected)

    rr = 0.0
    for pos, fid in enumerate(top, start=1):
        if fid in expected:
            rr = 1.0 / pos
            break

    # Supersession / distractor discipline: every id in `outranks` must sit
    # below every expected id that was retrieved at all. Retrieving a stale
    # fact is tolerable; ranking it above the current one is the failure.
    outranks = case.get("outranks") or []
    discipline = 1.0
    if outranks:
        best_expected = min((top.index(e) for e in found), default=None)
        for bad in outranks:
            if bad in top:
                if best_expected is None or top.index(bad) < best_expected:
                    discipline = 0.0
                    break

    return {"recall": recall, "mrr": rr, "discipline": discipline,
            "has_discipline": 1.0 if outranks else 0.0}


def run_config(
    mgr: MemoryManager,
    cases: list[dict[str, Any]],
    user_id: str,
    k: int,
    config: RetrievalConfig,
    ablation: Ablation,
) -> dict[str, Any]:
    per_category: dict[str, list[float]] = {}
    recalls: list[float] = []
    mrrs: list[float] = []
    disciplines: list[float] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    with ablation:
        recall_mod._LOGGED_CONFIG = config
        for case in cases:
            started = time.perf_counter()
            hits = gather_hits(mgr.vector, mgr.graph, mgr.sqlite, user_id, case["query"], k=k)
            latencies.append((time.perf_counter() - started) * 1000)

            ranked = hit_ids(hits)
            s = score_case(case, ranked, k)
            recalls.append(s["recall"])
            mrrs.append(s["mrr"])
            if s["has_discipline"]:
                disciplines.append(s["discipline"])
            per_category.setdefault(case["category"], []).append(s["recall"])
            if s["recall"] < 1.0 or s["discipline"] < 1.0:
                failures.append({
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "expected": case["expected"],
                    "missing": [e for e in case["expected"] if e not in ranked[:k]],
                    "top": ranked[:k],
                    "discipline": s["discipline"] if s["has_discipline"] else None,
                })

    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "recall": statistics.mean(recalls),
        "mrr": statistics.mean(mrrs),
        "discipline": statistics.mean(disciplines) if disciplines else float("nan"),
        "p50_ms": statistics.median(latencies),
        "p95_ms": p95,
        "per_category": {c: statistics.mean(v) for c, v in sorted(per_category.items())},
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=8, help="cut-off for recall@k (default 8)")
    ap.add_argument("--scale", type=int, default=0,
                    help="pad the corpus to this many facts for a latency-realistic run")
    ap.add_argument("--markdown", type=str, default="",
                    help="also write the results table to this path")
    ap.add_argument("--failures", type=str, default="",
                    help="print failing cases for the named configuration "
                         "(substring match, e.g. 'fusion (shipped)')")
    ap.add_argument("--category", type=str, default="",
                    help="with --failures, restrict output to one category")
    args = ap.parse_args()

    facts, cases, user_id = load_fixtures()
    mgr = MemoryManager(user_id)

    # An empty vector channel would make every comparison below meaningless.
    probe = mgr.vector._embedder().embed_one("connectivity probe")
    if not probe:
        print("ERROR: no embeddings available. Start Ollama and pull the embedding "
              "model; this benchmark is not meaningful without the vector channel.",
              file=sys.stderr)
        return 2

    print(f"seeding {len(facts)} golden facts into {_SCRATCH} ...", file=sys.stderr)
    seed(mgr, facts, args.scale)

    print("warming the query embedding cache ...", file=sys.stderr)
    warm_query_cache(mgr, cases)
    embed_cost = measure_embedding_cost(mgr)

    base = RetrievalConfig.load()
    configs: list[tuple[str, RetrievalConfig, Ablation]] = [
        ("vector only (baseline)", base, Ablation(bm25=False)),
        ("BM25 only", base, Ablation(vector=False)),
        ("fusion (shipped)", base, Ablation()),
        ("fusion, no entity signal",
         RetrievalConfig(base.candidate_pool, base.rrf_k, 0.0, base.temporal_half_life_days),
         Ablation()),
        ("fusion, flat temporal",
         RetrievalConfig(base.candidate_pool, base.rrf_k, base.entity_weight, 36500.0),
         Ablation()),
    ]

    results: list[tuple[str, dict[str, Any]]] = []
    for name, config, ablation in configs:
        print(f"running: {name}", file=sys.stderr)
        results.append((name, run_config(mgr, cases, user_id, args.k, config, ablation)))

    baseline = results[0][1]["recall"]
    lines = [
        f"### Retrieval benchmark — {len(cases)} cases, "
        f"{len(facts) if not args.scale else args.scale} facts, k={args.k}",
        "",
        f"| Configuration | recall@{args.k} | vs baseline | MRR | rank discipline | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results:
        delta = ((r["recall"] - baseline) / baseline * 100) if baseline else float("nan")
        disc = "n/a" if r["discipline"] != r["discipline"] else f"{r['discipline']:.2f}"
        lines.append(
            f"| {name} | {r['recall']:.3f} | {delta:+.1f}% | {r['mrr']:.3f} | "
            f"{disc} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        )

    lines += [
        "",
        f"Latency is retrieval only, measured with the query embedding cache warm. "
        f"A query whose text June has never embedded before pays an additional "
        f"{embed_cost['p50_ms']:.0f} ms (p50) / {embed_cost['p95_ms']:.0f} ms (p95) "
        f"in the local embedding call, which is model-server cost rather than "
        f"retrieval cost.",
    ]
    lines += ["", f"#### recall@{args.k} by category", "",
              "| Category | " + " | ".join(n for n, _ in results) + " |",
              "| --- | " + " | ".join("---:" for _ in results) + " |"]
    for cat in results[0][1]["per_category"]:
        row = " | ".join(f"{r['per_category'].get(cat, float('nan')):.2f}" for _, r in results)
        lines.append(f"| {cat} | {row} |")

    table = "\n".join(lines)
    print()
    print(table)

    if args.failures:
        for name, r in results:
            if args.failures.lower() not in name.lower():
                continue
            print(f"\n--- failing cases: {name} ---")
            for f in r["failures"]:
                if args.category and f["category"] != args.category:
                    continue
                flag = "" if f["discipline"] in (None, 1.0) else "  [RANK DISCIPLINE FAIL]"
                print(f"\n{f['id']} ({f['category']}){flag}\n  q: {f['query']}")
                print(f"  missing: {f['missing'] or '-'}")
                print(f"  top{args.k}: {f['top']}")
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
