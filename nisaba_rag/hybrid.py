"""Nisaba RAG — hybrid retrieval: dense + sparse fused with RRF.

Reciprocal Rank Fusion combines the semantic ranking (embeddings) with the
lexical ranking (BM25) without tuning fragile score weights:

    score(doc) = Σ_over_runs 1 / (k + rank_in_run)

A document ranked high by both retrievers is reinforced; ``k`` (60 is the
canonical value) damps the weight of the top few positions.

RRF needs a shared document identity across runs, which is why both retrievers
key on the same stable chunk id.
"""

from __future__ import annotations

from typing import Any

RRF_K = 60.0


def rrf_fuse(dense: list[dict], sparse: list[dict], top_n: int = 10) -> list[dict]:
    """Fuse two already-ranked result lists into one.

    Both inputs must be ordered best-first and carry a stable ``id``.
    """
    scores: dict[str, float] = {}
    registry: dict[str, dict] = {}

    def accumulate(ranked: list[dict]) -> None:
        for rank, hit in enumerate(ranked, start=1):
            chunk_id = hit.get("id")
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank))
            registry.setdefault(chunk_id, hit)

    accumulate(dense)
    accumulate(sparse)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    fused: list[dict] = []
    for chunk_id in ordered[:top_n]:
        item = dict(registry[chunk_id])
        item["rrf_score"] = scores[chunk_id]
        fused.append(item)
    return fused


def hybrid_search(
    store: Any,
    query: str,
    cfg: dict,
    top_n: int = 10,
    dense_limit: int = 30,
    rerank: bool = False,
) -> list[dict]:
    """Run dense + sparse retrieval, fuse with RRF, optionally rerank.

    Returns at most ``top_n`` documents. Requires a ``RagStore`` whose sparse
    index is available (it is built lazily from the vector store).
    """
    from .rag import embed

    query_embedding = embed(query, cfg)
    dense = store.search(query_embedding, limit=dense_limit)
    sparse = store.sparse_search(query, k=dense_limit)

    # Fuse over the wider candidate pool, then cut down to top_n.
    fused = rrf_fuse(dense, sparse, top_n=max(top_n, dense_limit))

    if rerank and fused:
        # Reranking improves quality but is never required: if no model is
        # installed, fall back to the fused ordering instead of failing the
        # whole search.
        try:
            from .rerank import get_reranker

            return get_reranker(cfg).rerank(query, fused, top_n=top_n)
        except Exception:  # noqa: BLE001 - degrade to the fused order
            pass

    if len(fused) > top_n:
        fused = fused[:top_n]
    return fused
