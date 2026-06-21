# ADR 0019 — Single-Engine Storage (sqlite-vec) and Ollama-Served Embeddings

## Status

Accepted. Supersedes — in scope of the vector backend only — the ChromaDB choice
in [ADR 0004](0004-memory-architecture.md); the three-store memory *model*
(structured rows, semantic vectors, entity graph behind one `MemoryManager`) is
unchanged. Anchored by the [rebuild plan](../product/rebuild-plan.md), S2.

## Context

June's semantic-recall layer ran on ChromaDB (embedded) with an in-process
`sentence-transformers` embedder (`all-MiniLM-L6-v2`). For a single-user,
local-first app this was the heaviest thing in the install by far: chromadb +
sentence-transformers pull `torch` (408 MB), `transformers` (98 MB),
`onnxruntime` (69 MB), `tokenizers`, and `safetensors` — about 575 MB of the
1.3 GB brain venv, and the main obstacle to a small, reliable desktop bundle
(S8). It also meant two storage engines (Chroma's own files plus the SQLite
database), which complicates backup, export, and the single-file portability
story.

A prior decision already anticipated this: every semantic fact is shadow-copied
into the SQLite `semantic_facts` table (text, source, metadata), with Chroma
holding only the embeddings and serving ANN search. That shadow is the
authoritative copy, which makes the vector index rebuildable and the embedder
swappable without data loss.

Before committing, a load probe confirmed `sqlite-vec` (a small loadable C
extension) loads into the stdlib `sqlite3` and serves a `vec0` KNN query on this
target (Python 3.14 / Apple Silicon), so no platform pivot was needed.

## Decision

**One storage engine: SQLite owns everything, including vectors.**

- **Vector index.** A `vec0` virtual table (`semantic_vectors`) in the same
  `june.db` holds the embeddings, keyed by `fact_id`, cosine distance. The
  `vec_index` module wraps load/create/upsert/search/delete and degrades to a
  no-op if the extension cannot load on a platform. `VectorStore` keeps its
  public interface and its authoritative `semantic_facts` shadow writes; search
  over-fetches from `vec0` then filters to the user via the shadow rows.
- **Embeddings.** Come from a local Ollama model (default `nomic-embed-text`,
  `JUNE_EMBED_MODEL` override) through the provider registry — `EmbeddingService`
  with a SQLite hash cache (`embedding_cache`, keyed by `(model, sha256(text))`)
  so re-embeds are free. No in-process model, no model weights in the install.
- **Graceful degradation (invariant 6, ships in the same change).** If no local
  embedder is reachable or the model is not pulled, `embed` returns `None`,
  `upsert` still writes the authoritative shadow, and `search` returns `[]` — the
  caller falls back to the SQLite keyword scan (`recall.sqlite_keyword_hits`).
  Nothing here ever triggers a model download, so an uncached model degrades
  rather than egressing (privacy invariant).
- **Migration.** The data-dir manifest bumps v1 -> v2 and archives the legacy
  `chroma` directory to `chroma.bak` (kept until the user clears it). Vectors are
  rebuilt into `vec0` from the shadow rows by an opportunistic startup backfill
  (best-effort, never blocks startup, idempotent); new writes index lazily on
  upsert regardless. `tools/migrate_chroma_to_sqlitevec.py` forces a full
  backfill for power users. The old Chroma embeddings are not reused — the new
  model has a different dimension, which is exactly why re-embedding from the
  shadow copies is the safe path.

## Alternatives Considered

- **Keep ChromaDB.** Rejected: the dependency mass is the single biggest barrier
  to the installable-desktop goal, and a second storage engine works against the
  one-copyable-file portability story.
- **FAISS / hnswlib as a standalone index.** Rejected: still a separate index
  file to keep in sync with SQLite, and FAISS pulls its own heavyweight wheels.
  `sqlite-vec` keeps vectors *inside* the database June already owns.
- **pgvector / a server.** Rejected: a local single-user app must not require a
  database server; it breaks the no-server, no-account product boundary.
- **Reuse the in-process sentence-transformer, only swap the index.** Rejected:
  the embedder is the bulk of the weight (torch/transformers). Ollama is already
  a runtime dependency for chat, so serving embeddings from it removes the heavy
  tree and keeps embeddings local.

## Consequences

Positive: brain venv 1.3 GB -> 653 MB (~647 MB freed), the largest single drop in
the rebuild; one storage engine means backup/export/portability is a single
`june.db`; embeddings are cached and local; the install is now small enough to
make the S8 desktop bundle credible.

Negative / accepted: semantic recall now depends on Ollama having an embedding
model pulled — handled by `run.sh` and the S8 managed-Ollama path, and degraded
to keyword recall (visibly, via provenance) when absent. The embedding model
differs from the old one, so recall "feel" can shift; the shadow copies make a
full re-embed safe and `chroma.bak` is retained until the user clears it. The
`vec0` table is global rather than per-user; June is single-user per data dir, so
search filters by user via the shadow join at negligible cost.
