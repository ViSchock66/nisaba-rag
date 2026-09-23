"""Nisaba RAG — core engine: embeddings, chunking, incremental indexing, search.

Design:

* **Vectors** live in ChromaDB (cosine space).
* **Content hashes** live in SQLite, which makes re-indexing incremental: an
  unchanged file is skipped, a changed file is re-chunked and re-embedded.
* **Embeddings** come from a local Ollama server by default. A second tier
  (any OpenAI-compatible ``/v1/embeddings`` endpoint) is used when its API key
  is present in the environment.

Nothing here is tied to a specific agent framework.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from typing import Optional

import chromadb
import httpx

from .config import load_config, resolve_path, validate_collection_name  # noqa: F401  (re-exported)
from .sparse import SparseIndex


class EmbeddingModelMissing(RuntimeError):
    """The embedding server is reachable but does not serve the configured model.

    Distinct from a bare connection failure on purpose: this one is fixed by a
    single command (`ollama pull <model>`), so it is reported on its own instead
    of being buried in the generic "no provider available" message.
    """


def _error_detail(response) -> str:
    """Pull the human-readable message out of an error response body."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - not every server answers with JSON
        return f"HTTP {response.status_code}"
    if isinstance(body, dict):
        return str(body.get("error") or body.get("message") or f"HTTP {response.status_code}")
    return f"HTTP {response.status_code}"


def embed(text: str, cfg: dict) -> list[float]:
    """Embed a single `text`, trying each configured provider in order.

    Raises ``RuntimeError`` if every provider fails, so callers never silently
    index a document with a zero vector.

    Prefer :func:`embed_many` when embedding several texts: it batches them
    into one request, which measured ~5x faster than one call per chunk.
    """
    return embed_many([text], cfg)[0]


def embed_many(texts: list[str], cfg: dict) -> list[list[float]]:
    """Embed several texts at once, preserving their order.

    Ollama's current endpoint (``/api/embed``) accepts a list under ``input``
    and returns all vectors from a single request. The older
    ``/api/embeddings`` endpoint takes one ``prompt`` at a time, so it is used
    as a fallback for servers that predate ``/api/embed`` — including the
    non-Ollama servers that only ever implemented the old contract.
    """
    if not texts:
        return []

    embed_cfg = cfg.get("embed", {})
    base_url = embed_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
    model = embed_cfg.get("model", "bge-m3")
    timeout = float(embed_cfg.get("timeout_s", 30.0))
    errors: list[str] = []

    # Tier 1a — batched request against the current endpoint.
    try:
        response = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        if response.status_code == 404:
            # Could be a missing model or a server without /api/embed at all.
            # Inspect the body to tell them apart before giving up on batching.
            detail = _error_detail(response)
            if "not found" in detail.lower() and "model" in detail.lower():
                raise EmbeddingModelMissing(
                    f"the embedding server at {base_url} does not have the model "
                    f"{model!r} ({detail}). Install it with:  ollama pull {model}"
                )
            errors.append(f"local batch: HTTP 404")
        elif response.status_code == 200:
            embeddings = response.json().get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
            errors.append(f"local batch: expected {len(texts)} embeddings, got {len(embeddings or [])}")
        else:
            errors.append(f"local batch: HTTP {response.status_code}")
    except EmbeddingModelMissing:
        raise
    except Exception as exc:  # noqa: BLE001 - fall back to the per-text endpoint
        errors.append(f"local batch: {exc}")

    # Tier 1b — legacy endpoint, one request per text.
    try:
        out: list[list[float]] = []
        for text in texts:
            response = httpx.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
            if response.status_code == 404:
                detail = _error_detail(response)
                raise EmbeddingModelMissing(
                    f"the embedding server at {base_url} does not have the model "
                    f"{model!r} ({detail}). Install it with:  ollama pull {model}"
                )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                raise RuntimeError("empty embedding in response")
            out.append(embedding)
        return out
    except EmbeddingModelMissing:
        raise
    except Exception as exc:  # noqa: BLE001 - fall through to the next tier
        errors.append(f"local: {exc}")

    # Tier 2 — optional hosted fallback, only when a key is configured.
    api_key = os.getenv("NISABA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
    if api_key:
        try:
            response = httpx.post(
                embed_cfg.get("nim_url", "https://integrate.api.nvidia.com/v1/embeddings"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": text,
                    "model": embed_cfg.get("nim_model", "nvidia/nv-embedqa-e5-v5"),
                    "encoding_format": "float",
                },
                timeout=timeout,
            )
            if response.status_code == 200:
                embedding = response.json().get("data", [{}])[0].get("embedding")
                if embedding:
                    return embedding
            errors.append(f"remote: HTTP {response.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"remote: {exc}")

    raise RuntimeError(
        "nisaba embed: no provider available. "
        f"Tried -> {'; '.join(errors)}. "
        "Start a local embedding server (see README) or set NISABA_API_KEY."
    )


def check_embedding_backend(cfg: dict) -> dict:
    """Report whether the configured embedding backend is usable.

    Intended for diagnostics (the ``nisaba-rag doctor`` command and the
    ``get_index_status`` MCP tool), so a caller can tell "Ollama is down" apart
    from "Ollama is up but the model was never pulled" without guessing.
    """
    embed_cfg = cfg.get("embed", {})
    base_url = embed_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
    model = embed_cfg.get("model", "bge-m3")
    timeout = min(float(embed_cfg.get("timeout_s", 30.0)), 10.0)

    try:
        response = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": ["ping"]},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        return {
            "ok": False,
            "base_url": base_url,
            "model": model,
            "reason": "server unreachable",
            "detail": str(exc),
            "hint": "start the embedding server (ollama serve) or fix embed.base_url",
        }

    if response.status_code == 404:
        detail = _error_detail(response)
        # A 404 means either the model is absent or the server predates
        # /api/embed. Only the former is fixed by pulling a model.
        if "model" in detail.lower():
            return {
                "ok": False,
                "base_url": base_url,
                "model": model,
                "reason": "model not installed",
                "detail": detail,
                "hint": f"ollama pull {model}",
            }
        return {
            "ok": False,
            "base_url": base_url,
            "model": model,
            "reason": "server does not support /api/embed",
            "detail": detail,
            "hint": "upgrade the embedding server; the legacy endpoint will still work but is slower",
        }

    if response.status_code != 200:
        return {
            "ok": False,
            "base_url": base_url,
            "model": model,
            "reason": f"unexpected HTTP {response.status_code}",
            "detail": _error_detail(response),
            "hint": "check the embedding server logs",
        }

    embeddings = response.json().get("embeddings") or []
    vector = embeddings[0] if embeddings else []
    declared_dims = embed_cfg.get("dims")
    result = {
        "ok": True,
        "base_url": base_url,
        "model": model,
        "dimensions": len(vector),
    }
    # A mismatch here means the model was changed without re-indexing, which
    # makes every stored vector unusable against new queries.
    if declared_dims and vector and len(vector) != declared_dims:
        result["ok"] = False
        result["reason"] = "dimension mismatch"
        result["hint"] = (
            f"config declares dims={declared_dims} but {model} returns {len(vector)}; "
            "update embed.dims and re-index with --force"
        )
    return result


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Split `text` into overlapping character chunks.

    Character-based chunking keeps the engine dependency-free and behaves the
    same for prose, code and config files. Guard rails: ``size`` must be
    positive and ``overlap`` strictly smaller than ``size``.
    """
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be > 0")

    overlap = max(0, min(overlap, size - 1))
    step = size - overlap

    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


class RagStore:
    """ChromaDB (vectors) + SQLite (incremental hash tracking).

    The SQLite connection is opened with ``check_same_thread=False`` because an
    MCP server may call tools from a different thread than the one that built
    the store; a lock serializes the SQLite mutations.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or load_config()
        self.chroma_path = str(resolve_path(self.cfg["store"]["path"]))
        self.collection_name = validate_collection_name(self.cfg["store"]["collection"])
        self.db_path = str(resolve_path(self.cfg["index_db"]))

        # ChromaDB is happy to create the directory itself, but SQLite is not.
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS files ("
            " filepath TEXT PRIMARY KEY, content_hash TEXT, indexed_at REAL, chunk_count INTEGER)"
        )
        self._conn.commit()

        # Sparse (BM25) index, built lazily from the chunks already in ChromaDB
        # and invalidated on any mutation.
        self._sparse: Optional[SparseIndex] = None
        self._sparse_dirty = True

    # ── hash tracking ──────────────────────────────────────────────

    @staticmethod
    def hash_of(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def stored_hash(self, filepath: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM files WHERE filepath=?", (filepath,)
            ).fetchone()
        return row[0] if row else None

    def upsert_file(self, filepath: str, content_hash: str, chunk_count: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO files(filepath, content_hash, indexed_at, chunk_count) VALUES(?,?,?,?) "
                "ON CONFLICT(filepath) DO UPDATE SET content_hash=excluded.content_hash, "
                "indexed_at=excluded.indexed_at, chunk_count=excluded.chunk_count",
                (filepath, content_hash, time.time(), chunk_count),
            )
            self._conn.commit()

    def remove_file(self, filepath: str) -> None:
        existing = self._collection.get(where={"source": filepath}, include=[])["ids"]
        if existing:
            self._collection.delete(ids=existing)
        with self._lock:
            self._conn.execute("DELETE FROM files WHERE filepath=?", (filepath,))
            self._conn.commit()
        self._sparse_dirty = True

    def list_sources(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT filepath FROM files ORDER BY filepath").fetchall()
        return [row[0] for row in rows]

    # ── vector ops ─────────────────────────────────────────────────

    @staticmethod
    def chunk_id(filepath: str, index: int) -> str:
        return hashlib.sha256(f"{filepath}::{index}".encode()).hexdigest()

    def add_chunks(
        self,
        filepath: str,
        chunks: list[str],
        embeddings: list[list[float]],
        ext: str = "",
    ) -> int:
        if not chunks:
            return 0
        ids = [self.chunk_id(filepath, i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=list(embeddings),
            documents=list(chunks),
            metadatas=[
                {"source": filepath, "chunk": i, "ext": ext}
                for i in range(len(chunks))
            ],
        )
        self._sparse_dirty = True
        return len(ids)

    # ── sparse (BM25) ──────────────────────────────────────────────

    def _all_chunks(self) -> list[dict]:
        """Every chunk in the vector store, in the shape BM25 expects."""
        result = self._collection.get(include=["documents", "metadatas"])
        out: list[dict] = []
        for i, chunk_id in enumerate(result.get("ids") or []):
            metadata = (result["metadatas"] or [])[i] or {}
            out.append({
                "id": chunk_id,
                "source": metadata.get("source"),
                "text": (result["documents"] or [])[i],
            })
        return out

    def sparse_index(self) -> SparseIndex:
        """Return the BM25 index, rebuilding it lazily when dirty."""
        if self._sparse is None or self._sparse_dirty:
            self._sparse = SparseIndex()
            self._sparse.build(self._all_chunks())
            self._sparse_dirty = False
        return self._sparse

    def sparse_search(self, query: str, k: int = 20) -> list[dict]:
        return self.sparse_index().search(query, k=k)

    # ── vector search ──────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        where = {"source": source_filter} if source_filter else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict] = []
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return out

        for i, chunk_id in enumerate(ids):
            metadata = (result["metadatas"] or [[]])[0][i] or {}
            distance = (result["distances"] or [[]])[0][i]
            out.append({
                "id": chunk_id,
                "text": (result["documents"] or [[]])[0][i],
                "source": metadata.get("source"),
                "chunk": metadata.get("chunk"),
                "distance": distance,
                # The collection uses cosine space, so similarity = 1 - distance.
                # Exposed explicitly because the reranker and downstream consumers
                # lose the dense signal otherwise.
                "similarity": 1.0 - distance,
            })
        return out

    def stats(self) -> dict:
        with self._lock:
            files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"files": files, "chunks": self._collection.count()}

    def reset(self) -> None:
        """Drop every chunk and hash entry from this collection."""
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        with self._lock:
            self._conn.execute("DELETE FROM files")
            self._conn.commit()
        self._sparse_dirty = True

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
