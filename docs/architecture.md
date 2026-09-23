# Architecture

A short tour of the engine, meant for someone about to modify it.

---

## Data model

Two stores, deliberately separate:

**ChromaDB** — one collection, one entry per chunk.

| Field | Meaning |
|---|---|
| `id` | `sha256("<filepath>::<chunk index>")` — stable across runs |
| `embedding` | Vector from the configured embedding model |
| `document` | The chunk text |
| `metadata.source` | Absolute file path |
| `metadata.chunk` | Chunk ordinal within the file |
| `metadata.ext` | Lowercase file extension |

The id is derived from path + index rather than from content so it is stable
and cheap; **content** changes are detected through the SQLite table below, not
through the vector id.

**SQLite** — one table, one row per indexed file.

| Column | Meaning |
|---|---|
| `filepath` | Primary key |
| `content_hash` | `sha256` of the file text |
| `indexed_at` | Unix timestamp |
| `chunk_count` | Chunks produced last time |

This table is what makes indexing incremental: if the stored hash equals the
current one, the file is skipped without ever touching the embedding server.

> The two stores can drift if a process is killed mid-write. Recovering is
> cheap and non-destructive in the other direction: `reindex_folder` rewrites
> both. There is no repair command that deletes data implicitly.

---

## The indexing pipeline

```
for each candidate file
  ├─ extension allowed?        no → skip
  ├─ in an excluded dir?       no → skip
  ├─ under max_file_kb?        no → skip
  ├─ read (utf-8, replace)     failure → count as error, continue
  ├─ hash unchanged?           yes → skip  ◄── the incremental fast path
  ├─ chunk_text(size, overlap)
  ├─ embed_many(chunks)        ◄── ONE request for the whole file
  ├─ remove_file(path)         ◄── clears previous chunks AND the hash row
  ├─ add_chunks(...)
  └─ upsert_file(hash, n)
```

**Why remove before add.** Chunk ids are positional. If a file shrinks from 10
chunks to 3, adding the new ones leaves chunks 3–9 orphaned and still
retrievable. Deleting first makes the replacement atomic from a reader's point
of view.

**Why one bad file does not stop the run.** An unreadable file, a binary
masquerading as `.txt`, or a transient embedding failure is counted in
`stats["errors"]` with a detail string. Indexing a large tree must not abort
because of one bad leaf.

### Embedding in batches

`embed_many` sends every chunk of a file in a single request to Ollama's
current `/api/embed` endpoint, which takes a list under `input`. Measured
against a local `bge-m3`, this is ~2x faster end to end than one request per
chunk, and the gap widens with file size because the per-request overhead stops
dominating.

The legacy `/api/embeddings` endpoint (one `prompt` per call) is kept as a
fallback for servers that predate `/api/embed` or never implemented it. Three
things trigger that fallback, and each is deliberate:

| Trigger | Why it must fall back |
|---|---|
| 404 without mentioning a model | The server may simply lack the endpoint; claiming the model is missing would be wrong. |
| A batch shorter than the input | A truncated response would silently pair vectors with the wrong chunks. |
| Any transport or HTTP error | Batching is an optimisation, never a requirement. |

`embed()` is a thin wrapper over `embed_many([text], cfg)[0]`, so there is a
single code path to maintain. Note that it resolves `embed_many` through the
module global: patching `embed` alone in a test does **not** replace the
underlying behaviour.

---

## Retrieval

### Dense

Cosine similarity over the embeddings. ChromaDB returns distances; the store
converts them to `similarity = 1 - distance` because the reranker and callers
want a similarity, not a distance.

### Sparse

Okapi BM25, implemented in `sparse.py` with no dependencies. It is built from
the chunks already in the vector store — nothing is re-embedded — and rebuilt
lazily whenever the store mutates. For a corpus of a few thousand chunks this
is fast enough that an external index would be pure overhead.

Tokenization normalizes accents (`configuración` matches `configuracion`) and
drops stopwords in English and Spanish.

### Fusion

Reciprocal Rank Fusion:

```
score(doc) = Σ_runs 1 / (k + rank(doc))     k = 60
```

RRF uses only **ranks**, never raw scores. That matters because cosine
similarity and BM25 live on incomparable scales — any weighted sum of them
would need per-corpus tuning that silently breaks when the corpus changes.
RRF needs no weights: a document both retrievers rank highly wins, and a
document only one finds still gets a fair shot.

`dense_limit` (default 30) is the size of each candidate run, deliberately
wider than `top_n` so the fusion and the optional reranker have something to
work with.

---

## Threading

An MCP server may dispatch tool calls on a different thread than the one that
constructed the store, so:

- the SQLite connection uses `check_same_thread=False`;
- a `threading.Lock` serializes every SQLite mutation and read;
- the BM25 index is rebuilt lazily and invalidated by a dirty flag, so no lock
  is held during the (potentially slow) rebuild of the document list.

The sparse index itself is not independently locked: it is rebuilt as a fresh
object and swapped in, so a concurrent reader sees either the old or the new
index, never a half-built one.

---

## Why there is no `requirements.txt` duplication

`pyproject.toml` is the source of truth. `requirements.txt` exists for people
who prefer `pip install -r`, and mirrors the core dependencies. Keep them in
sync when adding a dependency; core dependencies belong in
`[project].dependencies`, optional ones in an extra.

---

## Dependencies worth pinning deliberately

| Package | Floor | Why the floor matters |
|---|---|---|
| `mcp` | `>=2.0.0` | v2 renamed `FastMCP` to `MCPServer` and removed `mcp.server.fastmcp`. The server imports the v2 path, so allowing v1 would resolve to a version whose import fails at startup. |
| `chromadb` | `>=0.5.0` | Collection names are validated by the store because the library rejects malformed ones with an opaque error. |
| `onnxruntime` | `>=1.17.0` | Needs the execution-provider API used to select CPU/GPU. |

`embed_many` calls Ollama's `/api/embed`, which is the current endpoint;
`/api/embeddings` is deprecated but still served, which is what makes the
fallback viable rather than dead code.

---

## Extending

**A new embedding provider.** Add a tier inside `embed()` in `rag.py`, keeping
the "raise if everything fails" contract — never return a zero vector, which
would silently poison the index.

**A different retriever.** Return `[{"id", "source", "text"}, ...]` ranked
best-first and it can join the RRF fusion unchanged.

**A different chunker.** `chunk_text` is a pure function; swap it as long as
chunks stay non-empty strings. Token-based chunking would need a tokenizer and
a per-file token budget instead of a character budget.
